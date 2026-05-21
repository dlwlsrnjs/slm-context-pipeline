"""Evaluate a single (model, LoRA) on GSM8K test — used by Tier 3 launcher.

Each invocation runs as its own Python subprocess so vLLM resources are freed
on exit. Caller iterates over multiple (model, LoRA, label) configs.

Usage:
  python eval_single_lora.py --model unsloth/Qwen2.5-3B-Instruct-bnb-4bit \
      --lora .../stage3/lora_final --label phase_c_seed1234 \
      --out-dir .../eval_results/tier3
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "0")
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("HF_HOME", "/home/jklee/ondevice/slm-context-pipeline/icrl_math/.hf-cache")

import argparse, json, re, time
from pathlib import Path

from datasets import load_dataset
from unsloth import FastLanguageModel
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

SYSTEM_PROMPT_BASE = """You are a careful step-by-step math reasoner.

Respond in EXACTLY this format and nothing else:

<reasoning>
your step-by-step reasoning here
</reasoning>
<answer>
the final numeric answer here, e.g. 42
</answer>
"""

_HASH = re.compile(r"####\s*(\-?[0-9\.,]+)")
_ANS = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def gold(t):
    m = _HASH.search(t or "")
    return m.group(1).replace(",", "").strip() if m else None


def pred(t):
    m = _ANS.search(t or "")
    if not m:
        nums = re.findall(r"\-?\d+(?:\.\d+)?", t or "")
        return nums[-1] if nums else ""
    raw = m.group(1).strip()
    nums = re.findall(r"\-?\d+(?:\.\d+)?", raw)
    return nums[-1] if nums else raw


def ok(p, g):
    if not p or not g:
        return False
    if p.replace(",", "").strip() == g.replace(",", "").strip():
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lora", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    if not Path(args.lora).exists():
        print(f"LoRA not found: {args.lora}")
        return

    print(f"loading {args.model}")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=2048, load_in_4bit=True,
        fast_inference=True, max_lora_rank=args.lora_rank,
        gpu_memory_utilization=0.30,
    )

    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    exs = [{"q": e["question"], "g": gold(e["answer"])} for e in ds]
    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens,
                        stop=["</answer>"], include_stop_str_in_output=True)
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT_BASE},
         {"role": "user", "content": e["q"]}],
        tokenize=False, add_generation_prompt=True) for e in exs]

    lr = LoRARequest(args.label, abs(hash(args.label)) % 10000 + 1, args.lora)
    t0 = time.time()
    outs = model.fast_generate(prompts, sampling_params=sp, lora_request=lr)
    elapsed = time.time() - t0

    n_ok = sum(int(ok(pred(o.outputs[0].text), e["g"])) for e, o in zip(exs, outs))
    em = n_ok / len(exs)
    print(f"[{args.label}] EM = {em:.3f}  ({n_ok}/{len(exs)})  elapsed={elapsed:.1f}s")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        summary = json.load(open(summary_path))
        summary["results"][f"{args.label}__base"] = {"em": em, "correct": n_ok}
    else:
        summary = {"n": args.n, "results": {f"{args.label}__base": {"em": em, "correct": n_ok}}}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / f"records_{args.label}.jsonl", "w") as f:
        for e, o in zip(exs, outs):
            text = o.outputs[0].text
            f.write(json.dumps({"gold": e["g"], "pred": pred(text), "ok": ok(pred(text), e["g"])}) + "\n")
    print(f"saved -> {summary_path}")


if __name__ == "__main__":
    main()
