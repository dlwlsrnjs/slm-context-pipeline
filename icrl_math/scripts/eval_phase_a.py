"""Quick eval for Phase A: baseline (Qwen2.5-3B-Instruct 4bit) vs +LoRA on GSM8K test.

Uses the same XML-tag prompt format as train_grpo_unsloth_l40s.py so the LoRA
adapter is actually scored under the regime it was trained for.

Single Unsloth model load, then we toggle the LoRA adapter on/off between the
two passes (saves ~30 s of model reload).

Usage:
    python eval_phase_a.py --n 100 --max-new-tokens 256
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "0")
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("HF_HOME", "/home/jklee/ondevice/slm-context-pipeline/icrl_math/.hf-cache")

import argparse
import json
import re
import time
from pathlib import Path

from datasets import load_dataset
from unsloth import FastLanguageModel

SYSTEM_PROMPT = """You are a careful step-by-step math reasoner.

Respond in EXACTLY this format and nothing else:

<reasoning>
your step-by-step reasoning here
</reasoning>
<answer>
the final numeric answer here, e.g. 42
</answer>
"""

_HASH_RE = re.compile(r"####\s*(\-?[0-9\.,]+)")
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def gold(text):
    m = _HASH_RE.search(text or "")
    return m.group(1).replace(",", "").strip() if m else None


def extract_pred(text):
    m = _ANSWER_RE.search(text or "")
    if not m:
        # fallback: last number in the response
        nums = re.findall(r"\-?\d+(?:\.\d+)?", text or "")
        return nums[-1] if nums else ""
    raw = m.group(1).strip()
    nums = re.findall(r"\-?\d+(?:\.\d+)?", raw)
    return nums[-1] if nums else raw


def is_correct(pred, gold):
    if not pred or not gold:
        return False
    p, g = pred.replace(",", "").strip(), gold.replace(",", "").strip()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False


def run(model, tokenizer, examples, max_new_tokens, label, save_dir, sampling_params):
    from vllm import SamplingParams
    sp = SamplingParams(
        n=1, temperature=0.0, top_p=1.0, max_tokens=max_new_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    chat_prompts = []
    for ex in examples:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ]
        chat_prompts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )

    t0 = time.time()
    outs = model.fast_generate(chat_prompts, sampling_params=sp)
    elapsed = time.time() - t0

    n_correct = 0
    records = []
    for ex, out in zip(examples, outs):
        text = out.outputs[0].text
        pred = extract_pred(text)
        ok = is_correct(pred, ex["gold"])
        n_correct += int(ok)
        records.append({"q": ex["question"][:120], "gold": ex["gold"], "pred": pred, "ok": ok})

    em = n_correct / len(examples)
    print(f"[{label}] EM = {em:.3f}  ({n_correct}/{len(examples)})  elapsed={elapsed:.1f}s")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(save_dir) / f"records_{label}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return em, n_correct, len(examples)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    p.add_argument("--lora-path", default="/home/jklee/ondevice/slm-context-pipeline/icrl_math/checkpoints/grpo_l40s_full/lora_final")
    p.add_argument("--out-dir", default="/home/jklee/ondevice/slm-context-pipeline/icrl_math/eval_results/phase_a")
    args = p.parse_args()

    print(f"loading {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=1024,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=8,
        gpu_memory_utilization=0.55,
    )

    print(f"loading GSM8K test (n={args.n})")
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    examples = [{"question": ex["question"], "gold": gold(ex["answer"])} for ex in ds]
    print(f"  {len(examples)} examples")

    # 1) baseline
    print("\n=== baseline (no LoRA) ===")
    em_base, c_base, n = run(model, tokenizer, examples, args.max_new_tokens, "baseline", args.out_dir, None)

    # 2) load LoRA and re-run.
    # Unsloth's FastLanguageModel uses vLLM under the hood; the right API is to
    # build a vLLM LoRARequest and pass it to fast_generate.
    print(f"\n=== +LoRA ({args.lora_path}) ===")
    from vllm.lora.request import LoRARequest
    lora_request = LoRARequest(lora_name="phase_a", lora_int_id=1,
                               lora_path=args.lora_path)
    from vllm import SamplingParams
    sp = SamplingParams(
        n=1, temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens,
        stop=["</answer>"], include_stop_str_in_output=True,
    )
    chat_prompts = []
    for ex in examples:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["question"]},
        ]
        chat_prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    t0 = time.time()
    outs = model.fast_generate(chat_prompts, sampling_params=sp, lora_request=lora_request)
    elapsed = time.time() - t0
    n_correct = 0
    records = []
    for ex, out in zip(examples, outs):
        text = out.outputs[0].text
        pred = extract_pred(text)
        ok = is_correct(pred, ex["gold"])
        n_correct += int(ok)
        records.append({"q": ex["question"][:120], "gold": ex["gold"], "pred": pred, "ok": ok})
    em_lora = n_correct / len(examples)
    print(f"[+LoRA] EM = {em_lora:.3f}  ({n_correct}/{len(examples)})  elapsed={elapsed:.1f}s")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / "records_lora.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== summary ===")
    print(f"  baseline EM:  {em_base:.3f}  ({c_base}/{n})")
    print(f"  +LoRA EM:     {em_lora:.3f}  ({n_correct}/{n})")
    print(f"  delta:        {em_lora - em_base:+.3f}")
    summary = {
        "n": n,
        "baseline_em": em_base, "baseline_correct": c_base,
        "lora_em": em_lora, "lora_correct": n_correct,
        "delta": em_lora - em_base,
        "model": args.model, "lora": args.lora_path,
    }
    with open(Path(args.out_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved -> {Path(args.out_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
