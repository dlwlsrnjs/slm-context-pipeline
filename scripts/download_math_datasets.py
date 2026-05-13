#!/usr/bin/env python3
"""Download MATH (EleutherAI/hendrycks_math) and MetaMathQA datasets."""
import json
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, DatasetDict

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "local_datasets"

MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def download_math():
    target = OUT_DIR / "MATH"
    if target.exists():
        print("[SKIP] MATH: already exists")
        return True

    print("[DOWNLOAD] MATH (EleutherAI/hendrycks_math) — merging 7 subject configs")
    splits = {"train": [], "test": []}
    for cfg in MATH_CONFIGS:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg)
        for split in splits:
            if split in ds:
                splits[split].append(ds[split])
        print(f"  [{cfg}] train={len(ds.get('train', []))} test={len(ds.get('test', []))}")

    merged = DatasetDict({
        split: concatenate_datasets(parts)
        for split, parts in splits.items()
        if parts
    })
    merged.save_to_disk(str(target))
    total = sum(len(v) for v in merged.values())
    print(f"[OK] MATH -> {target}  (total {total} examples)")
    return True


def download_metamathqa():
    target = OUT_DIR / "MetaMathQA"
    if target.exists():
        print("[SKIP] MetaMathQA: already exists")
        return True

    print("[DOWNLOAD] MetaMathQA (meta-math/MetaMathQA)")
    ds = load_dataset("meta-math/MetaMathQA")
    ds.save_to_disk(str(target))
    total = sum(len(v) for v in ds.values())
    print(f"[OK] MetaMathQA -> {target}  (total {total} examples)")
    return True


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok_math = download_math()
    ok_meta = download_metamathqa()
    print("\n=== Done ===")
    print(f"MATH:       {'OK' if ok_math else 'FAIL'}")
    print(f"MetaMathQA: {'OK' if ok_meta else 'FAIL'}")
