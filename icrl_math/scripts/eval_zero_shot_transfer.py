"""Critical experiment: does Phase B's LoRA generalize when demos are removed at inference?

Phase B trained with demos always in the system prompt. Two interpretations:
  (a) "ICL internalized" — the LoRA absorbed the demo pattern; should still work in 0-shot
  (b) "demo dependency" — the LoRA only knows how to behave WITH demos; collapses 0-shot

This script runs every (LoRA, prompt) cross-condition on GSM8K test:

                          | base prompt (no demos) | demos prompt
    no LoRA               |  baseline_base         |  baseline_demos
    Phase A LoRA          |  phase_a__base         |  phase_a__demos
    Phase B LoRA          |  phase_b__base   ★     |  phase_b__demos

The ★ row is the test of generalization / forgetting.

Usage:  python eval_zero_shot_transfer.py --n 100
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


def gold(t):
    m = _HASH_RE.search(t or ""); return m.group(1).replace(",","").strip() if m else None


def extract_pred(t):
    m = _ANSWER_RE.search(t or "")
    if not m:
        nums = re.findall(r"\-?\d+(?:\.\d+)?", t or "")
        return nums[-1] if nums else ""
    raw = m.group(1).strip()
    nums = re.findall(r"\-?\d+(?:\.\d+)?", raw)
    return nums[-1] if nums else raw


def is_correct(pred, gold_val):
    if not pred or not gold_val: return False
    p, g = pred.replace(",","").strip(), gold_val.replace(",","").strip()
    if p == g: return True
    try: return abs(float(p)-float(g)) < 1e-6
    except: return False


def run_one(model, tokenizer, examples, system_prompt, sp, lora_request, label, out_dir):
    chat_prompts = []
    for ex in examples:
        msgs = [{"role":"system","content":system_prompt},{"role":"user","content":ex["question"]}]
        chat_prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    t0 = time.time()
    outs = model.fast_generate(chat_prompts, sampling_params=sp, lora_request=lora_request) if lora_request \
        else model.fast_generate(chat_prompts, sampling_params=sp)
    elapsed = time.time() - t0
    n_correct = 0; records = []
    for ex, out in zip(examples, outs):
        text = out.outputs[0].text
        pred = extract_pred(text)
        ok = is_correct(pred, ex["gold"])
        n_correct += int(ok)
        records.append({"gold": ex["gold"], "pred": pred, "ok": ok})
    em = n_correct / max(1, len(examples))
    print(f"[{label:25s}] EM = {em:.3f}  ({n_correct}/{len(examples)})  elapsed={elapsed:.1f}s")
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
    p.add_argument("--lora-phase-c", default=None,
        help="optional: Phase C (curriculum) LoRA")
    p.add_argument("--lora-phase-d", default=None,
        help="optional: Phase D (2-shot only, no curriculum) LoRA")
    p.add_argument("--lora-phase-e", default=None,
        help="optional: Phase E (5-shot, longer training) LoRA")
    p.add_argument("--demos-file",
        default="/home/jklee/ondevice/slm-context-pipeline/icrl_math/example/math_demos_simple.txt")
    p.add_argument("--out-dir",
        default="/home/jklee/ondevice/slm-context-pipeline/icrl_math/eval_results/cross")
    args = p.parse_args()

    print(f"loading {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=2048, load_in_4bit=True,
        fast_inference=True, max_lora_rank=16, gpu_memory_utilization=0.30,
    )

    print(f"loading GSM8K test (n={args.n})")
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    examples = [{"question": ex["question"], "gold": gold(ex["answer"])} for ex in ds]

    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens,
                        stop=["</answer>"], include_stop_str_in_output=True)

    demos_prompt = build_demo_prompt(args.demos_file)
    lr_a = LoRARequest(lora_name="phase_a", lora_int_id=1, lora_path=args.lora_phase_a)
    lr_b = LoRARequest(lora_name="phase_b", lora_int_id=2, lora_path=args.lora_phase_b)

    cases = [
        ("baseline_base",   SYSTEM_PROMPT_BASE, None),
        ("baseline_demos",  demos_prompt,       None),
        ("phase_a__base",   SYSTEM_PROMPT_BASE, lr_a),
        ("phase_a__demos",  demos_prompt,       lr_a),
        ("phase_b__base",   SYSTEM_PROMPT_BASE, lr_b),  # ★ Phase B 0-shot transfer test
        ("phase_b__demos",  demos_prompt,       lr_b),
    ]
    if args.lora_phase_c:
        lr_c = LoRARequest(lora_name="phase_c", lora_int_id=3, lora_path=args.lora_phase_c)
        cases.append(("phase_c__base",  SYSTEM_PROMPT_BASE, lr_c))   # ★★ key cell
        cases.append(("phase_c__demos", demos_prompt,       lr_c))
    if args.lora_phase_d:
        lr_d = LoRARequest(lora_name="phase_d", lora_int_id=4, lora_path=args.lora_phase_d)
        cases.append(("phase_d__base",  SYSTEM_PROMPT_BASE, lr_d))
        cases.append(("phase_d__demos", demos_prompt,       lr_d))
    if args.lora_phase_e:
        lr_e = LoRARequest(lora_name="phase_e", lora_int_id=5, lora_path=args.lora_phase_e)
        cases.append(("phase_e__base",  SYSTEM_PROMPT_BASE, lr_e))
        cases.append(("phase_e__demos", demos_prompt,       lr_e))
    results = {}
    for label, prompt, lora in cases:
        print(f"\n=== {label} ===")
        em, c = run_one(model, tokenizer, examples, prompt, sp, lora, label, args.out_dir)
        results[label] = {"em": em, "correct": c}

    print("\n" + "="*60)
    print(f"{'condition':30s}  EM       correct")
    print("-"*60)
    for k, v in results.items():
        print(f"{k:30s}  {v['em']:.3f}    {v['correct']}/{args.n}")
    print("="*60)

    bb = results["baseline_base"]["em"];   bd = results["baseline_demos"]["em"]
    pab = results["phase_a__base"]["em"];  pad = results["phase_a__demos"]["em"]
    pbb = results["phase_b__base"]["em"];  pbd = results["phase_b__demos"]["em"]
    print(f"\nDemo-only effect (baseline):           {bd-bb:+.3f}")
    print(f"Phase A GRPO (base prompt):            {pab-bb:+.3f}")
    print(f"Phase A LoRA + extra demos at infer:   {pad-pab:+.3f}    (does LoRA stack with demos?)")
    print(f"Phase B 0-shot transfer (★ key test):  {pbb-bb:+.3f}    (did LoRA internalize?)")
    print(f"Phase B GRPO-on-top of demos:          {pbd-bd:+.3f}")

    summary = {"n": args.n, "results": results,
               "deltas": {
                   "demo_only": bd-bb,
                   "phase_a_grpo_only": pab-bb,
                   "phase_a_with_demos_extra": pad-pab,
                   "phase_b_zero_shot_transfer": pbb-bb,
                   "phase_b_grpo_on_top_of_demos": pbd-bd,
               }}
    with open(Path(args.out_dir)/"summary.json","w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> {Path(args.out_dir)/'summary.json'}")


if __name__ == "__main__":
    main()
