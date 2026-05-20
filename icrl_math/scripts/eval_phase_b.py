"""3-way eval: baseline (no LoRA) vs Phase A LoRA vs Phase B LoRA.

Important: each LoRA was trained under its own system prompt:
  - Phase A: simple SYSTEM_PROMPT_BASE (no demos)
  - Phase B: SYSTEM_PROMPT_BASE + 5 prepended math demos

We honor that — Phase A is evaluated with the base prompt, Phase B with the
demos-prepended prompt. baseline is evaluated under BOTH prompts so we can
separate "demos help out-of-the-box" from "GRPO LoRA helps".

Usage:
    python eval_phase_b.py --n 100
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


def build_demo_prompt(demos_file):
    text = Path(demos_file).read_text(encoding="utf-8").strip()
    return (
        SYSTEM_PROMPT_BASE
        + "\nHere are some worked examples you can refer to:\n\n"
        + text
        + "\n\nNow solve the new problem using the same format. Do NOT reuse problems from the examples."
    )


_HASH_RE = re.compile(r"####\s*(\-?[0-9\.,]+)")
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def gold(text):
    m = _HASH_RE.search(text or "")
    return m.group(1).replace(",", "").strip() if m else None


def extract_pred(text):
    m = _ANSWER_RE.search(text or "")
    if not m:
        nums = re.findall(r"\-?\d+(?:\.\d+)?", text or "")
        return nums[-1] if nums else ""
    raw = m.group(1).strip()
    nums = re.findall(r"\-?\d+(?:\.\d+)?", raw)
    return nums[-1] if nums else raw


def is_correct(pred, gold_val):
    if not pred or not gold_val:
        return False
    p, g = pred.replace(",", "").strip(), gold_val.replace(",", "").strip()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return False


def run_one(model, tokenizer, examples, system_prompt, sp, lora_request, label, out_dir):
    chat_prompts = []
    for ex in examples:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ex["question"]},
        ]
        chat_prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    t0 = time.time()
    if lora_request:
        outs = model.fast_generate(chat_prompts, sampling_params=sp, lora_request=lora_request)
    else:
        outs = model.fast_generate(chat_prompts, sampling_params=sp)
    elapsed = time.time() - t0

    n_correct = 0
    records = []
    for ex, out in zip(examples, outs):
        text = out.outputs[0].text
        pred = extract_pred(text)
        ok = is_correct(pred, ex["gold"])
        n_correct += int(ok)
        records.append({"gold": ex["gold"], "pred": pred, "ok": ok})

    em = n_correct / max(1, len(examples))
    print(f"[{label}] EM = {em:.3f}  ({n_correct}/{len(examples)})  elapsed={elapsed:.1f}s")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / f"records_{label}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return em, n_correct


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    p.add_argument("--lora-phase-a",
                   default="/home/jklee/ondevice/slm-context-pipeline/icrl_math/checkpoints/grpo_l40s_full/lora_final")
    p.add_argument("--lora-phase-b",
                   default="/home/jklee/ondevice/slm-context-pipeline/icrl_math/checkpoints/grpo_l40s_phase_b/lora_final")
    p.add_argument("--demos-file",
                   default="/home/jklee/ondevice/slm-context-pipeline/icrl_math/example/math_demos_simple.txt")
    p.add_argument("--out-dir", default="/home/jklee/ondevice/slm-context-pipeline/icrl_math/eval_results/phase_b")
    args = p.parse_args()

    print(f"loading {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=16,
        gpu_memory_utilization=0.55,
    )

    print(f"loading GSM8K test (n={args.n})")
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    examples = [{"question": ex["question"], "gold": gold(ex["answer"])} for ex in ds]
    print(f"  {len(examples)} examples")

    sp = SamplingParams(
        n=1, temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens,
        stop=["</answer>"], include_stop_str_in_output=True,
    )

    demos_prompt = build_demo_prompt(args.demos_file)
    print(f"  demos_prompt len = {len(demos_prompt)} chars")

    results = {}

    # 1) baseline (no LoRA) under base prompt
    print("\n=== baseline (no LoRA, base prompt) ===")
    em, c = run_one(model, tokenizer, examples, SYSTEM_PROMPT_BASE, sp, None, "baseline_base", args.out_dir)
    results["baseline_base"] = {"em": em, "correct": c}

    # 2) baseline (no LoRA) with demos prompt
    print("\n=== baseline (no LoRA, demos prompt) ===")
    em, c = run_one(model, tokenizer, examples, demos_prompt, sp, None, "baseline_demos", args.out_dir)
    results["baseline_demos"] = {"em": em, "correct": c}

    # 3) Phase A LoRA + base prompt (its training-time setup)
    print(f"\n=== Phase A LoRA + base prompt ===  ({args.lora_phase_a})")
    lr_a = LoRARequest(lora_name="phase_a", lora_int_id=1, lora_path=args.lora_phase_a)
    em, c = run_one(model, tokenizer, examples, SYSTEM_PROMPT_BASE, sp, lr_a, "phase_a", args.out_dir)
    results["phase_a"] = {"em": em, "correct": c}

    # 4) Phase B LoRA + demos prompt (its training-time setup)
    print(f"\n=== Phase B LoRA + demos prompt ===  ({args.lora_phase_b})")
    lr_b = LoRARequest(lora_name="phase_b", lora_int_id=2, lora_path=args.lora_phase_b)
    em, c = run_one(model, tokenizer, examples, demos_prompt, sp, lr_b, "phase_b", args.out_dir)
    results["phase_b"] = {"em": em, "correct": c}

    print("\n" + "=" * 60)
    print(f"{'condition':30s}  EM       correct/n")
    print("-" * 60)
    for k, v in results.items():
        print(f"{k:30s}  {v['em']:.3f}    {v['correct']}/{args.n}")
    print("=" * 60)

    # deltas
    bb, bd = results["baseline_base"]["em"], results["baseline_demos"]["em"]
    pa, pb = results["phase_a"]["em"], results["phase_b"]["em"]
    print(f"\nΔ(phase_a vs baseline_base):   {pa - bb:+.3f}")
    print(f"Δ(baseline_demos vs baseline_base):  {bd - bb:+.3f}   (demo-only effect)")
    print(f"Δ(phase_b vs baseline_demos):  {pb - bd:+.3f}   (GRPO-on-top effect)")
    print(f"Δ(phase_b vs baseline_base):   {pb - bb:+.3f}   (total Phase B effect)")

    summary = {
        "n": args.n,
        "results": results,
        "deltas": {
            "phase_a_vs_baseline_base": pa - bb,
            "demo_only_effect": bd - bb,
            "phase_b_grpo_on_top": pb - bd,
            "phase_b_total": pb - bb,
        },
    }
    with open(Path(args.out_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> {Path(args.out_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
