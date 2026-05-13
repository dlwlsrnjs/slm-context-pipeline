#!/usr/bin/env python3
"""
Hold-out test set splitter for OrcaMath and MetaMathQA.

Excludes all sample IDs used in the 5k training JSONL,
then samples N items for evaluation.

GSM8k uses the official test split — no action needed.

Usage:
    python scripts/split_math_test_sets.py \
        --output-dir slm_context_pipeline/data/math_test
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_from_disk

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_KW = [
    "atomic weight", "molecular weight", "standard deviation",
    "normal distribution", "z-score", "inscribed", "circumscribed",
    "probability density", "atomic mass",
]
GSM_TYPES = {"GSM_Rephrased", "GSM_AnsAug", "GSM_SV", "GSM_FOBAR"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir",
                   default="slm_context_pipeline/data/math_test")
    p.add_argument("--n-orca", type=int, default=1000)
    p.add_argument("--n-meta", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_train_ids(jsonl_path: str) -> set:
    ids = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.add(row["id"])
    return ids


def extract_orca_final(answer_text: str) -> str:
    import re
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()]
    for s in reversed(sentences):
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", s)
        if nums:
            return nums[-1].replace(",", "")
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", answer_text)
    return nums[-1].replace(",", "") if nums else answer_text.strip()


def write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def split_orcamath(train_ids: set, n: int, rng: random.Random) -> list:
    ds = load_from_disk(str(REPO_ROOT / "local_datasets" / "OrcaMath"))
    import re
    candidates = []
    for i in range(len(ds)):
        row = ds[i]
        row_id = f"orcamath_{i:06d}"
        if row_id in train_ids:
            continue
        q = str(row["question"]).lower()
        a = str(row["answer"])
        has_latex = ("\\(" in a or "\\[" in a or "\\frac" in a
                     or "\\sqrt" in a or "\\sum" in a or "\\int" in a)
        has_ext = any(kw in q for kw in EXTERNAL_KW)
        if not has_latex and not has_ext:
            candidates.append(i)

    sampled = rng.sample(candidates, min(n, len(candidates)))
    records = []
    for idx in sampled:
        row = ds[idx]
        records.append({
            "id": f"orcamath_{idx:06d}",
            "source": "OrcaMath",
            "question": str(row["question"]).strip(),
            "answer": extract_orca_final(str(row["answer"])),
            "solution": str(row["answer"]).strip(),
        })
    print(f"OrcaMath test: {len(records)} samples (from {len(candidates)} candidates)")
    return records


def split_metamathqa(train_ids: set, n: int, rng: random.Random) -> list:
    ds = load_from_disk(str(REPO_ROOT / "local_datasets" / "MetaMathQA"))["train"]
    import re

    def extract_final(text):
        m = re.search(r"[Tt]he answer is[:\s]+([^\n.]+)", text)
        if m:
            nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", m.group(1))
            if nums:
                return nums[0].replace(",", "")
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
        return nums[-1].replace(",", "") if nums else text.strip()

    candidates = []
    for i in range(len(ds)):
        row = ds[i]
        row_id = f"metamath_{i:06d}"
        if row_id in train_ids:
            continue
        if str(row["type"]) in GSM_TYPES:
            candidates.append(i)

    sampled = rng.sample(candidates, min(n, len(candidates)))
    records = []
    for idx in sampled:
        row = ds[idx]
        records.append({
            "id": f"metamath_{idx:06d}",
            "source": "MetaMathQA",
            "question": str(row["query"]).strip(),
            "answer": extract_final(str(row["response"])),
            "solution": str(row["response"]).strip(),
        })
    print(f"MetaMathQA test: {len(records)} samples (from {len(candidates)} candidates)")
    return records


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = REPO_ROOT / args.output_dir

    train_ids_orca = load_train_ids(
        str(REPO_ROOT / "slm_context_pipeline/data/math_5k/orcamath_5k.jsonl"))
    train_ids_meta = load_train_ids(
        str(REPO_ROOT / "slm_context_pipeline/data/math_5k/metamath_5k.jsonl"))

    orca_test = split_orcamath(train_ids_orca, args.n_orca, rng)
    meta_test = split_metamathqa(train_ids_meta, args.n_meta, rng)

    write_jsonl(out_dir / "orcamath_test_1k.jsonl", orca_test)
    write_jsonl(out_dir / "metamath_test_1k.jsonl", meta_test)
    print(f"\nSaved to {out_dir}")
    print("GSM8k test → local_datasets/GSM8k (official test split, no action needed)")


if __name__ == "__main__":
    main()
