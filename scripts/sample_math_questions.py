#!/usr/bin/env python3
"""
Stratified sampling of math reasoning datasets → questions JSONL.

Sampling strategy per dataset:
  GSM8k       : solution step count  (short / medium / long)
  OrcaMath    : answer length chars  (short / medium / long)
  MetaMathQA  : GSM subtype          (GSM_Rephrased / GSM_AnsAug / GSM_SV / GSM_FOBAR)

Output format (compatible with questions_real_max.jsonl):
  { "id": "...", "source": "...", "question": "...",
    "answer": "<final number>", "solution": "<full CoT>" }
"""
import json
import random
import re
from pathlib import Path

from datasets import load_from_disk

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "slm_context_pipeline" / "data"
OUT_DIR = DATA_DIR / "math_5k"

SEED = 42
N = 5000

# ── helpers ──────────────────────────────────────────────────────────────────

def extract_gsm8k_final(answer_text: str) -> str:
    """Extract numeric answer after #### marker."""
    m = re.search(r"####\s*([^\n]+)", answer_text)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", answer_text)
    return nums[-1].replace(",", "") if nums else answer_text.strip()


def extract_cot_final(answer_text: str) -> str:
    """Extract last number from a CoT solution (Orca / MetaMath style)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()]
    for sentence in reversed(sentences):
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", sentence)
        if nums:
            return nums[-1].replace(",", "")
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", answer_text)
    return nums[-1].replace(",", "") if nums else answer_text.strip()


def solution_steps(answer_text: str) -> int:
    """Estimate number of reasoning steps by sentence / line count."""
    lines = [l.strip() for l in answer_text.split("\n") if l.strip()]
    return max(len(lines), answer_text.count(". "))


def bucket_gsm8k(row) -> str:
    steps = solution_steps(str(row["answer"]))
    if steps <= 4:
        return "short"
    if steps <= 7:
        return "medium"
    return "long"


def bucket_orca(row) -> str:
    n = len(str(row["answer"]))
    if n < 200:
        return "short"
    if n < 500:
        return "medium"
    return "long"


def stratified_sample(groups: dict, total: int, rng: random.Random) -> list:
    """Sample evenly from each group; redistribute slots from undersized groups."""
    group_names = list(groups.keys())
    k = len(group_names)
    targets = {g: total // k for g in group_names}
    for g in sorted(group_names, key=lambda x: -len(groups[x]))[: total - (total // k) * k]:
        targets[g] += 1

    # iteratively redistribute slots from groups that can't fill their target
    changed = True
    while changed:
        changed = False
        overflow = 0
        eligible = []
        for g in group_names:
            if len(groups[g]) < targets[g]:
                overflow += targets[g] - len(groups[g])
                targets[g] = len(groups[g])
                changed = True
            else:
                eligible.append(g)
        if overflow and eligible:
            extra = overflow // len(eligible)
            rem = overflow - extra * len(eligible)
            for g in eligible:
                targets[g] += extra
            for g in sorted(eligible, key=lambda x: -len(groups[x]))[:rem]:
                targets[g] += 1

    result = []
    for g in group_names:
        result.extend(rng.sample(groups[g], targets[g]))
    rng.shuffle(result)
    return result


def write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── GSM8K ────────────────────────────────────────────────────────────────────

def sample_gsm8k(rng: random.Random) -> list:
    ds = load_from_disk(str(REPO_ROOT / "local_datasets" / "GSM8k"))["train"]
    groups: dict = {"short": [], "medium": [], "long": []}
    for i in range(len(ds)):
        row = ds[i]
        groups[bucket_gsm8k(row)].append(i)

    print(f"GSM8k buckets: { {g: len(v) for g, v in groups.items()} }")
    indices = stratified_sample(groups, N, rng)

    records = []
    for rank, idx in enumerate(indices):
        row = ds[idx]
        records.append({
            "id": f"gsm8k_{idx:05d}",
            "source": "GSM8k",
            "question": str(row["question"]).strip(),
            "answer": extract_gsm8k_final(str(row["answer"])),
            "solution": str(row["answer"]).strip(),
        })
    print(f"GSM8k sampled: {len(records)}")
    return records


# ── OrcaMath ─────────────────────────────────────────────────────────────────

EXTERNAL_KW = [
    "atomic weight", "molecular weight", "standard deviation",
    "normal distribution", "z-score", "inscribed", "circumscribed",
    "probability density", "atomic mass",
]


def sample_orcamath(rng: random.Random) -> list:
    ds = load_from_disk(str(REPO_ROOT / "local_datasets" / "OrcaMath"))
    groups: dict = {"short": [], "medium": [], "long": []}
    for i in range(len(ds)):
        row = ds[i]
        q = str(row["question"]).lower()
        a = str(row["answer"])
        has_latex = ("\\(" in a or "\\[" in a or "\\frac" in a
                     or "\\sqrt" in a or "\\sum" in a or "\\int" in a)
        has_ext = any(kw in q for kw in EXTERNAL_KW)
        if has_latex or has_ext:
            continue
        groups[bucket_orca(row)].append(i)

    print(f"OrcaMath buckets: { {g: len(v) for g, v in groups.items()} }")
    indices = stratified_sample(groups, N, rng)

    records = []
    for idx in indices:
        row = ds[idx]
        records.append({
            "id": f"orcamath_{idx:06d}",
            "source": "OrcaMath",
            "question": str(row["question"]).strip(),
            "answer": extract_cot_final(str(row["answer"])),
            "solution": str(row["answer"]).strip(),
        })
    print(f"OrcaMath sampled: {len(records)}")
    return records


# ── MetaMathQA ───────────────────────────────────────────────────────────────

GSM_TYPES = ["GSM_Rephrased", "GSM_AnsAug", "GSM_SV", "GSM_FOBAR"]


def sample_metamathqa(rng: random.Random) -> list:
    ds = load_from_disk(str(REPO_ROOT / "local_datasets" / "MetaMathQA"))["train"]
    groups: dict = {t: [] for t in GSM_TYPES}
    for i in range(len(ds)):
        t = str(ds[i]["type"])
        if t in groups:
            groups[t].append(i)

    print(f"MetaMathQA buckets: { {g: len(v) for g, v in groups.items()} }")
    indices = stratified_sample(groups, N, rng)

    records = []
    for idx in indices:
        row = ds[idx]
        records.append({
            "id": f"metamath_{idx:06d}",
            "source": "MetaMathQA",
            "question": str(row["query"]).strip(),
            "answer": extract_cot_final(str(row["response"])),
            "solution": str(row["response"]).strip(),
        })
    print(f"MetaMathQA sampled: {len(records)}")
    return records


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    rng = random.Random(SEED)

    datasets = [
        ("gsm8k_5k.jsonl",      sample_gsm8k),
        ("orcamath_5k.jsonl",   sample_orcamath),
        ("metamath_5k.jsonl",   sample_metamathqa),
    ]

    for filename, fn in datasets:
        print(f"\n=== {filename} ===")
        records = fn(rng)
        out_path = OUT_DIR / filename
        write_jsonl(out_path, records)
        print(f"Saved → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
