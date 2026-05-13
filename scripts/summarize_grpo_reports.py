#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GRPO reports")
    parser.add_argument("--reports-dir", default="experiment_results/grpo_icl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    files = sorted(reports_dir.glob("grpo_*_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    latest_by_dataset = {}
    for path in files:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        ds = payload.get("dataset_name", "unknown")
        if ds not in latest_by_dataset:
            latest_by_dataset[ds] = (path, payload)

    summary_rows = []
    for ds, (path, payload) in sorted(latest_by_dataset.items()):
        summary_rows.append(
            {
                "dataset": ds,
                "pre_reward": payload.get("pre_reward", 0.0),
                "post_reward": payload.get("post_reward", 0.0),
                "delta_reward": payload.get("delta_reward", 0.0),
                "best_step": payload.get("best_step", 0),
                "fallback_to_initial": payload.get("fallback_to_initial", False),
                "report": str(path),
            }
        )

    out_path = reports_dir / "grpo_6_datasets_summary.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"rows": summary_rows}, handle, indent=2)

    print("Summary saved:", out_path)
    for row in summary_rows:
        print(
            f"- {row['dataset']}: pre={row['pre_reward']:.4f}, "
            f"post={row['post_reward']:.4f}, delta={row['delta_reward']:+.4f}, "
            f"fallback={row['fallback_to_initial']}"
        )


if __name__ == "__main__":
    main()
