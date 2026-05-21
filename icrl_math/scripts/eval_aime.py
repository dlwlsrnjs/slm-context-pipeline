"""AIME 2024 + 2025 evaluation — 3rd benchmark (super-hard OOD).

AIME = American Invitational Mathematics Examination, 15 problems/year, integer
answers 0-999. Much harder than GSM8K/MATH500. baseline 3B-Instruct typically
solves ~5-15% of these.

Reuses the same 7-cell grid (no LoRA + Phase A/B/C × {base, demos}) but extends
to as many phases as adapters are provided.

Usage:  python eval_aime.py --n 30
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "0")
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("HF_HOME", "/home/jklee/ondevice/slm-context-pipeline/icrl_math/.hf-cache")

import argparse, json, re, time
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

SYSTEM_PROMPT_BASE = """You are a careful step-by-step math reasoner.

Respond in EXACTLY this format and nothing else:

<reasoning>
your step-by-step reasoning here
</reasoning>
<answer>
the final numeric answer here (integer 0-999), e.g. 42
</answer>
"""


def build_demo_prompt(demos_file):
    text = Path(demos_file).read_text(encoding="utf-8").strip()
    return (SYSTEM_PROMPT_BASE + "\nHere are some worked examples you can refer to:\n\n"
            + text + "\n\nNow solve the new problem using the same format. Do NOT reuse problems from the examples.")


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_pred(t):
    """AIME answer is integer 0-999. Look in <answer>, then \\boxed, then last int."""
    if not t: return None
    m = _ANSWER_RE.search(t)
    if m:
        raw = m.group(1).strip()
        b = _BOXED_RE.search(raw)
        if b:
            raw = b.group(1).strip()
        nums = re.findall(r"\-?\d+", raw)
        if nums: return nums[-1]
    b = _BOXED_RE.search(t)
    if b:
        nums = re.findall(r"\-?\d+", b.group(1))
        if nums: return nums[-1]
    nums = re.findall(r"\-?\d+", t)
    return nums[-1] if nums else None


def is_correct(pred, gold):
    if pred is None or gold is None: return False
    try:
        return int(str(pred).strip()) == int(str(gold).strip())
    except: return False


def load_aime():
    """Load AIME 2024 + 2025 from HF. Different mirrors — try several."""
    out = []
    for repo, year in [("Maxwell-Jia/AIME_2024", 2024), ("yentinglin/aime_2025", 2025)]:
        try:
            d = load_dataset(repo, split="train")
            for ex in d:
                prob_k = next((k for k in ("Problem", "problem", "question") if k in ex), None)
                ans_k = next((k for k in ("Answer", "answer", "solution") if k in ex), None)
                if prob_k and ans_k:
                    out.append({"problem": str(ex[prob_k]).strip(),
                                "gold": str(ex[ans_k]).strip(), "year": year})
            print(f"  loaded {repo} → {len(d)} problems")
        except Exception as e:
            print(f"  skip {repo}: {e}")
    return out


def run_one(model, tokenizer, examples, sys_prompt, sp, lora, label, out_dir):
    chat_prompts = []
    for ex in examples:
        msgs = [{"role":"system","content":sys_prompt},{"role":"user","content":ex["problem"]}]
        chat_prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    t0 = time.time()
    outs = (model.fast_generate(chat_prompts, sampling_params=sp, lora_request=lora)
            if lora else model.fast_generate(chat_prompts, sampling_params=sp))
    elapsed = time.time() - t0
    n_correct = 0; records = []
    for ex, out in zip(examples, outs):
        text = out.outputs[0].text
        pred = extract_pred(text); ok = is_correct(pred, ex["gold"])
        n_correct += int(ok)
        records.append({"gold": ex["gold"], "pred": pred, "ok": ok, "year": ex["year"]})
    em = n_correct / max(1, len(examples))
    print(f"[{label:30s}] EM = {em:.3f}  ({n_correct}/{len(examples)})  elapsed={elapsed:.1f}s")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir)/f"records_{label}.jsonl","w") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    return em, n_correct


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    p.add_argument("--lora-phase-a", default="/home/jklee/ondevice/slm-context-pipeline/icrl_math/checkpoints/grpo_l40s_full/lora_final")
    p.add_argument("--lora-phase-b", default="/home/jklee/ondevice/slm-context-pipeline/icrl_math/checkpoints/grpo_l40s_phase_b/lora_final")
    p.add_argument("--lora-phase-c", default=None)
    p.add_argument("--lora-phase-d", default=None)
    p.add_argument("--lora-phase-e", default=None)
    p.add_argument("--demos-file", default="/home/jklee/ondevice/slm-context-pipeline/icrl_math/example/math_demos_simple.txt")
    p.add_argument("--out-dir", default="/home/jklee/ondevice/slm-context-pipeline/icrl_math/eval_results/aime")
    args = p.parse_args()

    print(f"loading {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=2048, load_in_4bit=True,
        fast_inference=True, max_lora_rank=16, gpu_memory_utilization=0.30)

    print("loading AIME 2024 + 2025")
    examples = load_aime()
    if args.n and args.n < len(examples):
        examples = examples[:args.n]
    print(f"  {len(examples)} total")
    if not examples:
        print("no AIME problems loaded; abort"); return

    sp = SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens,
                        stop=["</answer>"], include_stop_str_in_output=True)
    demos = build_demo_prompt(args.demos_file)
    lr_a = LoRARequest("phase_a", 1, args.lora_phase_a)
    lr_b = LoRARequest("phase_b", 2, args.lora_phase_b)
    cases = [
        ("baseline_base",  SYSTEM_PROMPT_BASE, None),
        ("baseline_demos", demos,              None),
        ("phase_a__base",  SYSTEM_PROMPT_BASE, lr_a),
        ("phase_a__demos", demos,              lr_a),
        ("phase_b__base",  SYSTEM_PROMPT_BASE, lr_b),
        ("phase_b__demos", demos,              lr_b),
    ]
    if args.lora_phase_c:
        lc = LoRARequest("phase_c", 3, args.lora_phase_c)
        cases += [("phase_c__base", SYSTEM_PROMPT_BASE, lc), ("phase_c__demos", demos, lc)]
    if args.lora_phase_d:
        ld = LoRARequest("phase_d", 4, args.lora_phase_d)
        cases += [("phase_d__base", SYSTEM_PROMPT_BASE, ld), ("phase_d__demos", demos, ld)]
    if args.lora_phase_e:
        le = LoRARequest("phase_e", 5, args.lora_phase_e)
        cases += [("phase_e__base", SYSTEM_PROMPT_BASE, le), ("phase_e__demos", demos, le)]

    results = {}
    for label, prompt, lora in cases:
        print(f"\n=== {label} ===")
        em, c = run_one(model, tokenizer, examples, prompt, sp, lora, label, args.out_dir)
        results[label] = {"em": em, "correct": c}
    print("\n" + "="*60)
    for k, v in results.items():
        print(f"{k:30s}  {v['em']:.3f}    {v['correct']}/{len(examples)}")
    with open(Path(args.out_dir)/"summary.json","w") as f:
        json.dump({"n": len(examples), "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
