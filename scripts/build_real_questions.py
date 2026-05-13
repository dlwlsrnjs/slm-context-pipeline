import argparse
import json
import random
from pathlib import Path

from datasets import load_from_disk


def parse_gsm8k_answer(answer_text: str) -> str:
    if not answer_text:
        return ""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return answer_text.strip().split("\n")[-1].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="/workspace/local_datasets")
    parser.add_argument("--output", type=str, default="/workspace/slm_context_pipeline/data/questions_real.jsonl")
    parser.add_argument("--boolq", type=int, default=120)
    parser.add_argument("--piqa", type=int, default=120)
    parser.add_argument("--gsm8k", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    base = Path(args.base)

    records = []

    # BoolQ
    boolq = load_from_disk(str(base / "BoolQ" / "validation"))
    boolq_idx = list(range(len(boolq)))
    random.shuffle(boolq_idx)
    for i, idx in enumerate(boolq_idx[: args.boolq]):
        item = boolq[idx]
        question = item.get("question", "").strip()
        if not question:
            continue
        answer = "yes" if bool(item.get("answer")) else "no"
        records.append({
            "id": f"boolq_{i:04d}",
            "question": question,
            "answer": answer,
            "source": "BoolQ"
        })

    # PIQA
    piqa = load_from_disk(str(base / "PIQA" / "validation"))
    piqa_idx = list(range(len(piqa)))
    random.shuffle(piqa_idx)
    for i, idx in enumerate(piqa_idx[: args.piqa]):
        item = piqa[idx]
        goal = item.get("goal", "").strip()
        if not goal:
            continue
        label = int(item.get("label", 0))
        sol1 = item.get("sol1", "").strip()
        sol2 = item.get("sol2", "").strip()
        answer = sol1 if label == 0 else sol2
        question = (
            f"Goal: {goal}\n"
            f"Option A: {sol1}\n"
            f"Option B: {sol2}\n"
            "Which option better achieves the goal?"
        )
        records.append({
            "id": f"piqa_{i:04d}",
            "question": question,
            "answer": answer,
            "source": "PIQA"
        })

    # GSM8k
    gsm8k = load_from_disk(str(base / "GSM8k" / "test"))
    gsm_idx = list(range(len(gsm8k)))
    random.shuffle(gsm_idx)
    for i, idx in enumerate(gsm_idx[: args.gsm8k]):
        item = gsm8k[idx]
        question = item.get("question", "").strip()
        if not question:
            continue
        answer = parse_gsm8k_answer(item.get("answer", ""))
        records.append({
            "id": f"gsm8k_{i:04d}",
            "question": question,
            "answer": answer,
            "source": "GSM8k"
        })

    random.shuffle(records)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} questions to {out_path}")


if __name__ == "__main__":
    main()
