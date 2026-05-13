#!/usr/bin/env python3
"""Re-generate failed (Q<n_shots) contexts with larger token budget, merge back.

Usage:
    python scripts/regen_failed_contexts.py \
        --context-file eval_results/math_icl_baselines_full/contexts_base_7b.jsonl \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --tests-file eval_results/math_icl_baselines_full/tests.jsonl \
        --max-new-tokens 1200
"""
import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_POOL_PATHS = {
    "gsm8k": REPO_ROOT / "slm_context_pipeline/data/math_5k/gsm8k_5k.jsonl",
    "orcamath": REPO_ROOT / "slm_context_pipeline/data/math_5k/orcamath_5k.jsonl",
    "metamath": REPO_ROOT / "slm_context_pipeline/data/math_5k/metamath_5k.jsonl",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--context-file", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--tests-file", required=True)
    p.add_argument("--n-shots", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=1200)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_seed_examples(rows, shots):
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


def build_gen_prompt(seed_examples, dataset_name, shots):
    format_rule = ("Q: <math word problem>\n"
                   "A: <step-by-step reasoning ending with 'The answer is [number].'>")
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


DEMO_PATTERN = re.compile(r"Q:\s*(.+?)\nA:\s*(.+?)(?=\nQ:\s|\Z)", re.DOTALL)


def extract_first_n_demos(text, n):
    clean = []
    for q, a in DEMO_PATTERN.findall(text):
        q, a = q.strip(), a.strip()
        m = re.search(r"[Tt]he answer is[^.\n]*\.?", a)
        if m:
            a = a[: m.end()].strip()
        if q and a:
            clean.append(f"Q: {q}\nA: {a}")
        if len(clean) >= n:
            break
    return "\n\n".join(clean)


def main():
    args = parse_args()

    ctx_path = Path(args.context_file)
    tests = load_jsonl(args.tests_file)
    contexts = load_jsonl(ctx_path)
    assert len(tests) == len(contexts), "tests/contexts length mismatch"

    # identify failed positions
    failed_idx = [i for i, c in enumerate(contexts)
                  if c["demos"].count("Q:") < args.n_shots]
    print(f"[regen] total={len(contexts)}  failed(Q<{args.n_shots})={len(failed_idx)}")
    if not failed_idx:
        print("[regen] nothing to do")
        return

    # load seed pools
    seed_pools = {}
    for src, p in SEED_POOL_PATHS.items():
        seed_pools[src] = load_jsonl(p)

    # rebuild seeds deterministically matching original order (seed=42)
    rng = random.Random(args.seed)
    seeds_for_q = [rng.sample(seed_pools[t["source"]], args.n_shots) for t in tests]

    prompts = []
    for i in failed_idx:
        t = tests[i]
        seed_text = format_seed_examples(seeds_for_q[i], args.n_shots)
        prompts.append(build_gen_prompt(seed_text, f"{t['source']}_5k", args.n_shots))

    # load model
    print(f"[regen] loading {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    # batched generation
    raws = []
    for i in range(0, len(prompts), args.batch):
        batch = prompts[i:i + args.batch]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.9,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        raws.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))
        if (i // args.batch) % 5 == 0:
            print(f"  [regen] {i + len(batch)}/{len(prompts)}", flush=True)

    # merge back: replace failed entries only if new extraction yields more Q
    updated = 0
    for pos, raw in zip(failed_idx, raws):
        new_demos = extract_first_n_demos(raw, args.n_shots)
        old_q = contexts[pos]["demos"].count("Q:")
        new_q = new_demos.count("Q:")
        if new_q > old_q:
            contexts[pos]["raw"] = raw
            contexts[pos]["demos"] = new_demos
            updated += 1
    print(f"[regen] improved {updated}/{len(failed_idx)} entries")

    # write back
    backup = ctx_path.with_suffix(".jsonl.bak")
    if not backup.exists():
        backup.write_bytes(ctx_path.read_bytes())
        print(f"[regen] backup → {backup}")
    with ctx_path.open("w", encoding="utf-8") as f:
        for c in contexts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[regen] merged → {ctx_path}")


if __name__ == "__main__":
    main()
