#!/usr/bin/env python3
"""
6-baseline evaluation for math ICL context generators.

Conditions (all use Qwen2.5-1.5B-Instruct as the answer model):
  NO           - answer model sees only the question
  BASE-1B      - vanilla Qwen2.5-1.5B-Instruct generates 4 ICL demos
  BASE-7B      - vanilla Qwen2.5-7B-Instruct generates 4 ICL demos
  TRAIN-1B     - fine-tuned 1B generates 4 ICL demos
  TRAIN-7B     - fine-tuned 7B generates 4 ICL demos

Flow (memory-efficient, one generator at a time):
  Stage 1. Load each generator, produce & cache 4 ICL demos per test question.
  Stage 2. Load answer model, run all 5 conditions.
  Stage 3. Score & aggregate.

Usage:
    python scripts/eval_math_icl_baselines.py \
        --n-per-dataset 200 \
        --answer-model Qwen/Qwen2.5-1.5B-Instruct \
        --base-1b Qwen/Qwen2.5-1.5B-Instruct \
        --base-7b Qwen/Qwen2.5-7B-Instruct \
        --train-1b experiment_results/math_icl_sft/qwen1b_run2/final \
        --train-7b experiment_results/math_icl_sft/qwen7b_run2/final \
        --output-dir eval_results/math_icl_baselines
"""
import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent

SEED_POOL_PATHS = {
    "gsm8k": REPO_ROOT / "slm_context_pipeline/data/math_5k/gsm8k_5k.jsonl",
    "orcamath": REPO_ROOT / "slm_context_pipeline/data/math_5k/orcamath_5k.jsonl",
    "metamath": REPO_ROOT / "slm_context_pipeline/data/math_5k/metamath_5k.jsonl",
}


# ───────────────────────── CLI ──────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-dataset", type=int, default=200)
    p.add_argument("--answer-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--base-1b", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--base-7b", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train-1b",
                   default="experiment_results/math_icl_sft/qwen1b_run2/final")
    p.add_argument("--train-7b",
                   default="experiment_results/math_icl_sft/qwen7b_run2/final")
    p.add_argument("--rl-1b", default="",
                   help="LoRA adapter dir for RL-trained 1B (applied on train-1b base)")
    p.add_argument("--rl-7b", default="",
                   help="LoRA adapter dir for RL-trained 7B (applied on train-7b base)")
    p.add_argument("--output-dir", default="eval_results/math_icl_baselines")
    p.add_argument("--n-shots", type=int, default=4)
    p.add_argument("--gen-max-tokens", type=int, default=600)
    p.add_argument("--ans-max-tokens", type=int, default=256)
    p.add_argument("--gen-batch", type=int, default=4)
    p.add_argument("--ans-batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage", choices=["contexts", "answers", "score", "all"],
                   default="all")
    p.add_argument("--only-tags", default="",
                   help="Comma-separated tags to run in stage 1 "
                        "(base_1b,base_7b,train_1b,train_7b). Empty = all.")
    p.add_argument("--only-conditions", default="",
                   help="Comma-separated conditions for stage 2 "
                        "(no,base_1b,base_7b,train_1b,train_7b). Empty = all.")
    p.add_argument("--cluster-dir", default="",
                   help="If set, use cluster representatives as seeds (Auto-CoT). "
                        "Path to math_5k_clusters dir.")
    p.add_argument("--use-prl-prompt", action="store_true",
                   help="Use PRL-style prompt with <think>...</think><answer>...</answer>.")
    return p.parse_args()


# ──────────────────────── data loading ────────────────────────

def load_test_questions(n: int, rng: random.Random) -> List[dict]:
    tests = []

    gsm = load_from_disk(str(REPO_ROOT / "local_datasets/GSM8k"))["test"]
    gsm_indices = rng.sample(range(len(gsm)), min(n, len(gsm)))
    for i in gsm_indices:
        row = gsm[i]
        ans_text = str(row["answer"])
        m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", ans_text)
        final = m.group(1).replace(",", "") if m else ""
        tests.append({
            "id": f"gsm8k_{i:05d}",
            "source": "gsm8k",
            "question": str(row["question"]).strip(),
            "gold": final,
        })

    for src, fname in [("orcamath", "orcamath_test_1k.jsonl"),
                       ("metamath", "metamath_test_1k.jsonl")]:
        path = REPO_ROOT / "slm_context_pipeline/data/math_test" / fname
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        picks = rng.sample(rows, min(n, len(rows)))
        for row in picks:
            tests.append({
                "id": row["id"],
                "source": src,
                "question": str(row["question"]).strip(),
                "gold": str(row["answer"]).replace(",", "").strip(),
            })

    return tests


def load_cluster_reps_for_eval(cluster_dir: str, k: int = 8):
    """Load cluster representatives keyed by short source name."""
    cdir = REPO_ROOT / cluster_dir
    out = {}
    for short, full in [("gsm8k", "gsm8k_5k"),
                        ("orcamath", "orcamath_5k"),
                        ("metamath", "metamath_5k")]:
        path = cdir / f"{full}_clusters_k{k}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        out[short] = data["representatives"]
    print(f"[clusters] loaded reps from {cdir} (k={k})")
    return out


def load_seed_pools() -> Dict[str, List[dict]]:
    pools = {}
    for src, path in SEED_POOL_PATHS.items():
        with path.open(encoding="utf-8") as f:
            pools[src] = [json.loads(line) for line in f if line.strip()]
    return pools


# ──────────────────── prompt builders ────────────────────

def format_seed_examples(rows: List[dict], shots: int) -> str:
    lines = []
    for row in rows[:shots]:
        q = str(row["question"]).strip()
        sol = str(row.get("solution", row.get("answer", ""))).strip()
        sol = re.sub(r"<<[^>]*>>", "", sol)
        sol = re.sub(r"####\s*[\d,]+", "", sol).strip()
        final = str(row.get("answer", "")).strip()
        if final and not re.search(r"[Tt]he answer is", sol):
            sol = sol.rstrip(".") + f"\nThe answer is {final}."
        lines.append(f"Q: {q}\nA: {sol}")
    return "\n\n".join(lines)


def build_gen_prompt(seed_examples: str, dataset_name: str, shots: int) -> str:
    format_rule = (
        "Q: <math word problem>\n"
        "A: <step-by-step reasoning ending with 'The answer is [number].'>"
    )
    return (
        f"You are generating in-context demonstrations for math word problems "
        f"({dataset_name}).\n"
        f"Generate exactly {shots} demonstrations with this format:\n"
        f"{format_rule}\n\n"
        "Rules:\n"
        "- Each demo must be self-contained (no external tables or formulas needed).\n"
        "- Show clear arithmetic steps.\n"
        "- End every answer with 'The answer is [number].'\n"
        "- Do NOT add explanations outside the Q/A format.\n\n"
        "Candidate examples:\n"
        f"{seed_examples}\n\n"
        "Now output demonstrations only:\n"
    )


def build_prl_gen_prompt(seed_examples: str, dataset_name: str, shots: int) -> str:
    """PRL-style prompt with reasoning step."""
    return (
        f"You are generating in-context demonstrations for math word problems "
        f"({dataset_name}).\n"
        f"First, think briefly inside <think>...</think> about which patterns "
        f"and difficulty level are useful. Then output exactly {shots} "
        f"self-contained demonstrations inside <answer>...</answer>.\n\n"
        "Demonstration format inside <answer>:\n"
        "Q: <math word problem>\n"
        "A: <step-by-step reasoning ending with 'The answer is [number].'>\n\n"
        "Rules:\n"
        "- Each demo must be self-contained.\n"
        "- Show clear arithmetic steps.\n"
        "- End every answer with 'The answer is [number].'\n"
        "- Use exactly the tags <think>, </think>, <answer>, </answer> "
        "(each appears once).\n\n"
        "Candidate examples (for reference, do NOT copy verbatim):\n"
        f"{seed_examples}\n\n"
        "Now output:\n"
    )


def build_answer_prompt(question: str, demos: Optional[str]) -> str:
    header = ("Solve the following math word problem. "
              "Show your reasoning and end with 'The answer is [number].'\n\n")
    if demos:
        return header + demos + "\n\n" + f"Q: {question}\nA:"
    return header + f"Q: {question}\nA:"


# ──────────────────── demo post-processing ────────────────────

DEMO_PATTERN = re.compile(
    r"Q:\s*(.+?)\nA:\s*(.+?)(?=\nQ:\s|\Z)",
    re.DOTALL,
)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_first_n_demos(text: str, n: int) -> str:
    # If <answer>...</answer> wrapper present (PRL-style), extract from inside.
    m = ANSWER_PATTERN.search(text)
    payload = m.group(1) if m else text
    blocks = DEMO_PATTERN.findall(payload)
    clean = []
    for q, a in blocks:
        q = q.strip()
        a = a.strip()
        m = re.search(r"[Tt]he answer is[^.\n]*\.?", a)
        if m:
            a = a[: m.end()].strip()
        if q and a:
            clean.append(f"Q: {q}\nA: {a}")
        if len(clean) >= n:
            break
    return "\n\n".join(clean)


# ──────────────────── scoring ────────────────────

NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def extract_final_number(text: str) -> Optional[str]:
    m = re.search(r"[Tt]he answer is[:\s]+([^\n]+)", text)
    if m:
        nums = NUMBER_RE.findall(m.group(1))
        if nums:
            return nums[0].replace(",", "")
    nums = NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def numeric_match(pred: Optional[str], gold: str) -> bool:
    if pred is None or not gold:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred.strip() == gold.strip()


# ──────────────────── batched inference ────────────────────

def load_model(path: str, dtype=torch.bfloat16, adapter_path: str = ""):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print(f"  [load_model] merged adapter from {adapter_path}")
    model.eval()
    return tok, model


def generate_batched(tok, model, prompts: List[str],
                     max_new_tokens: int, batch_size: int,
                     temperature: float = 0.4) -> List[str]:
    outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=0.9,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
        outputs.extend(decoded)
        if (i // batch_size) % 5 == 0:
            print(f"  [gen] {i + len(batch)}/{len(prompts)}", flush=True)
    return outputs


# ──────────────────── stages ────────────────────

def stage_generate_contexts(args, tests: List[dict], seed_pools: Dict[str, List[dict]]):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        ("base_1b", args.base_1b, ""),
        ("base_7b", args.base_7b, ""),
        ("train_1b", args.train_1b, ""),
        ("train_7b", args.train_7b, ""),
    ]
    if args.rl_1b:
        generators.append(("rl_1b", args.train_1b, args.rl_1b))
    if args.rl_7b:
        generators.append(("rl_7b", args.train_7b, args.rl_7b))
    if args.only_tags:
        wanted = set(args.only_tags.split(","))
        generators = [g for g in generators if g[0] in wanted]
        print(f"[filter] only_tags={args.only_tags} → {[g[0] for g in generators]}")

    rng = random.Random(args.seed)
    # pre-build per-question seed examples (same across generators for fairness)
    seeds_for_q = []
    if args.cluster_dir:
        cluster_reps = load_cluster_reps_for_eval(args.cluster_dir, k=8)
        print(f"[seeds] using AUTO-COT cluster reps as seed pool (k=8 → 4 of 8)")
        for t in tests:
            reps = cluster_reps[t["source"]]
            # deterministic per-question 4-of-8 selection via id hash
            local_rng = random.Random(hash(t["id"]) & 0xFFFFFFFF)
            picks = local_rng.sample(reps, args.n_shots)
            seeds_for_q.append(picks)
    else:
        for t in tests:
            pool = seed_pools[t["source"]]
            picks = rng.sample(pool, args.n_shots)
            seeds_for_q.append(picks)

    prompts = []
    prompt_builder = build_prl_gen_prompt if args.use_prl_prompt else build_gen_prompt
    for t, seeds in zip(tests, seeds_for_q):
        seed_text = format_seed_examples(seeds, args.n_shots)
        prompts.append(prompt_builder(seed_text, f"{t['source']}_5k", args.n_shots))

    for tag, model_path, adapter in generators:
        out_file = out_dir / f"contexts_{tag}.jsonl"
        if out_file.exists():
            print(f"[skip] {out_file} exists")
            continue
        adapter_msg = f" + adapter {adapter}" if adapter else ""
        print(f"\n[STAGE 1] Generating contexts with {tag} ← {model_path}{adapter_msg}")
        tok, model = load_model(model_path, adapter_path=adapter)
        raw = generate_batched(tok, model, prompts,
                               max_new_tokens=args.gen_max_tokens,
                               batch_size=args.gen_batch)
        with out_file.open("w", encoding="utf-8") as f:
            for t, r in zip(tests, raw):
                demos = extract_first_n_demos(r, args.n_shots)
                f.write(json.dumps({
                    "id": t["id"],
                    "source": t["source"],
                    "raw": r,
                    "demos": demos,
                }, ensure_ascii=False) + "\n")
        del model
        torch.cuda.empty_cache()
        print(f"[saved] {out_file}")


def stage_generate_answers(args, tests: List[dict]):
    out_dir = Path(args.output_dir)
    tok, model = load_model(args.answer_model)

    conditions = [("no", None)] + [
        (tag, out_dir / f"contexts_{tag}.jsonl")
        for tag in ["base_1b", "base_7b", "train_1b", "train_7b", "rl_1b", "rl_7b"]
        if (out_dir / f"contexts_{tag}.jsonl").exists() or tag in ["base_1b", "base_7b", "train_1b", "train_7b"]
    ]
    if args.only_conditions:
        wanted = set(args.only_conditions.split(","))
        conditions = [(tag, p) for tag, p in conditions if tag in wanted]
        print(f"[filter] only_conditions={args.only_conditions} → {[t for t,_ in conditions]}")

    demo_by_id = {tag: {} for tag, _ in conditions if tag != "no"}
    for tag, path in conditions:
        if path is None:
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                demo_by_id[tag][row["id"]] = row["demos"]

    for tag, _ in conditions:
        out_file = out_dir / f"answers_{tag}.jsonl"
        if out_file.exists():
            print(f"[skip] {out_file} exists")
            continue
        print(f"\n[STAGE 2] Answering with condition = {tag}")
        prompts = []
        for t in tests:
            demos = None if tag == "no" else demo_by_id[tag].get(t["id"], "")
            demos = demos if demos else None  # empty string → no demos
            prompts.append(build_answer_prompt(t["question"], demos))
        raw = generate_batched(tok, model, prompts,
                               max_new_tokens=args.ans_max_tokens,
                               batch_size=args.ans_batch,
                               temperature=0.0)
        with out_file.open("w", encoding="utf-8") as f:
            for t, r in zip(tests, raw):
                f.write(json.dumps({
                    "id": t["id"], "source": t["source"],
                    "answer_raw": r,
                    "pred": extract_final_number(r),
                    "gold": t["gold"],
                }, ensure_ascii=False) + "\n")
        print(f"[saved] {out_file}")

    del model
    torch.cuda.empty_cache()


def stage_score(args, tests: List[dict]):
    out_dir = Path(args.output_dir)
    summary = {}
    candidate_tags = ["no", "base_1b", "base_7b", "train_1b", "train_7b", "rl_1b", "rl_7b"]
    tags = [t for t in candidate_tags if (out_dir / f"answers_{t}.jsonl").exists()]
    for tag in tags:
        path = out_dir / f"answers_{tag}.jsonl"
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        by_src = {}
        for r in rows:
            ok = numeric_match(r["pred"], r["gold"])
            by_src.setdefault(r["source"], []).append(ok)
        summary[tag] = {
            src: {"n": len(v), "acc": 100 * sum(v) / len(v)}
            for src, v in by_src.items()
        }
        total = [ok for v in by_src.values() for ok in v]
        summary[tag]["all"] = {"n": len(total), "acc": 100 * sum(total) / len(total)}

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # table
    cols = ["gsm8k", "orcamath", "metamath", "all"]
    print("\n" + "=" * 72)
    print(f"{'condition':<12} | " + " | ".join(f"{c:>9}" for c in cols))
    print("-" * 72)
    for tag in tags:
        cells = []
        for c in cols:
            if c in summary[tag]:
                cells.append(f"{summary[tag][c]['acc']:>9.2f}")
            else:
                cells.append(f"{'-':>9}")
        print(f"{tag:<12} | " + " | ".join(cells))
    print("=" * 72)
    print(f"[saved] {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    tests_path = out_dir / "tests.jsonl"
    if tests_path.exists():
        with tests_path.open(encoding="utf-8") as f:
            tests = [json.loads(line) for line in f if line.strip()]
        print(f"[loaded] {len(tests)} tests from cache")
    else:
        tests = load_test_questions(args.n_per_dataset, rng)
        with tests_path.open("w", encoding="utf-8") as f:
            for t in tests:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"[built] {len(tests)} test questions")

    if args.stage in ("contexts", "all"):
        stage_generate_contexts(args, tests, load_seed_pools())
    if args.stage in ("answers", "all"):
        stage_generate_answers(args, tests)
    if args.stage in ("score", "all"):
        stage_score(args, tests)


if __name__ == "__main__":
    main()
