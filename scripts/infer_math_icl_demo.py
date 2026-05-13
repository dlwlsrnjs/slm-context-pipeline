#!/usr/bin/env python3
"""Quick inference demo for the math ICL generator."""
import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_seed_examples(rows, shots=4):
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


def build_student_prompt(seed_examples, dataset_name="math problems", shots=4):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--seed-jsonl", default="slm_context_pipeline/data/math_5k/gsm8k_5k.jsonl")
    p.add_argument("--dataset-name", default="gsm8k_5k")
    p.add_argument("--n-seeds", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    with open(args.seed_jsonl, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    seeds = rng.sample(rows, args.n_seeds)

    seed_examples = format_seed_examples(seeds, shots=args.n_seeds)
    prompt = build_student_prompt(seed_examples, args.dataset_name, shots=args.n_seeds)

    print("=" * 70)
    print("[SEED EXAMPLES PASSED TO THE MODEL]")
    print("=" * 70)
    print(seed_examples)
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    inputs = tokenizer(prompt + "\n", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(gen_ids, skip_special_tokens=True)

    print("=" * 70)
    print("[GENERATED ICL DEMONSTRATIONS]")
    print("=" * 70)
    print(generated)


if __name__ == "__main__":
    main()
