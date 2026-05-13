#!/usr/bin/env python3
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    datasets_file = root / "datasets.json"
    local_dir = root / "local_datasets"

    with open(datasets_file, "r", encoding="utf-8") as handle:
        all_datasets = [item["name"] for item in json.load(handle)["datasets"]]

    available = {p.name for p in local_dir.iterdir() if p.is_dir()}

    # RL reward stability heuristic for current GRPO trainer:
    # 1) clear closed-form label spaces
    # 2) short deterministic answer targets
    # 3) lower formatting ambiguity in prompt parsing
    ranked = [
        ("BoolQ", "binary labels (True/False), most stable reward signal"),
        ("PIQA", "2-choice format, robust numeric answer parsing"),
        ("WinoGrande", "2-choice format, concise context"),
        ("Hellaswag", "multiple-choice but longer contexts"),
        ("CommonsenseQA", "5-choice, more parsing ambiguity"),
        ("ARC-Easy", "5-choice, variable option formatting"),
    ]

    picks = []
    notes = []
    for name, reason in ranked:
        if name in all_datasets and name in available:
            picks.append(name)
            notes.append({"dataset": name, "reason": reason})
        if len(picks) == 3:
            break

    payload = {
        "recommended_for_grpo": picks,
        "notes": notes,
        "available_local_count": len(available),
    }

    out = root / "experiment_results" / "grpo_icl" / "recommended_datasets.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("Recommended datasets for GRPO:", ", ".join(picks))
    print("Saved:", out)


if __name__ == "__main__":
    main()
