"""CoT baseline — no LoRA, no demos, but with an explicit Chain-of-Thought
prompt. This is a standard comparison baseline reviewers will expect.

We use the canonical "Let's think step by step" CoT framing.

Usage: python eval_cot_baseline.py --n 100
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


SYSTEM_PROMPT_COT = """You are a careful step-by-step math reasoner.

Solve the problem by thinking step by step. After your reasoning, give the final
numeric answer on a new line in the form `#### <answer>` (e.g. `#### 42`).
"""

_HASH = re.compile(r"####\s*(\-?[0-9\.,]+)")


def gold(t):
    m = _HASH.search(t or "")
    return m.group(1).replace(",", "").strip() if m else None


def extract_pred(t):
    if not t:
        return ""
    m = _HASH.search(t)
    if m:
        return m.group(1).replace(",", "").strip()
    # fallback: last number
    nums = re.findall(r"\-?\d+(?:\.\d+)?", t)
    return nums[-1] if nums else ""


def is_correct(p, g):
    if not p or not g:
        return False
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return p.strip() == g.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    p.add_argument("--out-dir",
                   default="/home/jklee/ondevice/slm-context-pipeline/icrl_math/eval_results/cot_baseline")
    args = p.parse_args()

    print(f"loading {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=2048, load_in_4bit=True,
        fast_inference=True, max_lora_rank=8, gpu_memory_utilization=0.30,
    )

    print(f"loading GSM8K (n={args.n})")
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    examples = [{"q": e["question"], "g": gold(e["answer"])} for e in ds]

    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)

    prompts = [tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT_COT},
         {"role": "user", "content": e["q"]}],
        tokenize=False, add_generation_prompt=True) for e in examples]

    t0 = time.time()
    outs = model.fast_generate(prompts, sampling_params=sp)
    elapsed = time.time() - t0

    n_correct = 0; records = []
    for e, o in zip(examples, outs):
        text = o.outputs[0].text
        pred = extract_pred(text)
        ok = is_correct(pred, e["g"])
        n_correct += int(ok)
        records.append({"gold": e["g"], "pred": pred, "ok": ok})
    em = n_correct / max(1, len(examples))
    print(f"[CoT baseline] EM = {em:.3f}  ({n_correct}/{len(examples)})  elapsed={elapsed:.1f}s")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir)/"records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(Path(args.out_dir)/"summary.json", "w") as f:
        json.dump({"n": len(examples), "results": {"cot_baseline": {"em": em, "correct": n_correct}}}, f, indent=2)
    print(f"saved -> {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
