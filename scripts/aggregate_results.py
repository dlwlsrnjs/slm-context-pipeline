#!/usr/bin/env python3
"""Aggregate SLM-Bench experiment JSON outputs into summary files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate SLM-Bench results")
    parser.add_argument("--results-dir", default="experiment_results")
    parser.add_argument("--summary-json", default="summary.json")
    parser.add_argument("--summary-csv", default="summary.csv")
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*_results.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["_path"] = str(path)
            items.append(payload)
        except Exception:
            continue
    return items


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in items:
        model = item.get("model", "unknown")
        dataset = item.get("dataset", "unknown")
        by_model[model].append(item)
        by_dataset[dataset].append(item)

    def avg_metric(rows: list[dict[str, Any]], key: str) -> float | None:
        vals = []
        for row in rows:
            metrics = row.get("metrics", {})
            val = metrics.get(key)
            f = safe_float(val)
            if f is not None:
                vals.append(f)
        if not vals:
            return None
        return sum(vals) / len(vals)

    summary = {
        "total_results": len(items),
        "by_model": {},
        "by_dataset": {},
    }

    for model, rows in sorted(by_model.items()):
        summary["by_model"][model] = {
            "count": len(rows),
            "avg_training_loss": sum(float(r.get("training_loss", 0.0)) for r in rows) / len(rows),
            "avg_eval_loss": avg_metric(rows, "eval_loss"),
            "avg_eval_accuracy": avg_metric(rows, "eval_accuracy"),
        }

    for dataset, rows in sorted(by_dataset.items()):
        summary["by_dataset"][dataset] = {
            "count": len(rows),
            "avg_training_loss": sum(float(r.get("training_loss", 0.0)) for r in rows) / len(rows),
            "avg_eval_loss": avg_metric(rows, "eval_loss"),
            "avg_eval_accuracy": avg_metric(rows, "eval_accuracy"),
        }

    return summary


def write_csv(items: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "model",
        "dataset",
        "training_loss",
        "eval_loss",
        "eval_accuracy",
        "eval_runtime",
        "epoch",
        "timestamp",
        "path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            metrics = item.get("metrics", {})
            writer.writerow(
                {
                    "model": item.get("model"),
                    "dataset": item.get("dataset"),
                    "training_loss": item.get("training_loss"),
                    "eval_loss": metrics.get("eval_loss"),
                    "eval_accuracy": metrics.get("eval_accuracy"),
                    "eval_runtime": metrics.get("eval_runtime"),
                    "epoch": metrics.get("epoch"),
                    "timestamp": item.get("timestamp"),
                    "path": item.get("_path"),
                }
            )


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    items = load_results(results_dir)
    summary = aggregate(items)

    summary_json = results_dir / args.summary_json
    summary_csv = results_dir / args.summary_csv

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    write_csv(items, summary_csv)

    print(f"[OK] aggregated {len(items)} result files")
    print(f"[OK] wrote {summary_json}")
    print(f"[OK] wrote {summary_csv}")


if __name__ == "__main__":
    main()
