#!/usr/bin/env python3
"""Utility script to orchestrate SLM-Bench experiments in manageable batches."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


def load_entities(config_path: Path, key: str) -> List[dict]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return list(data.get(key, []))


def ensure_names_available(entities: Sequence[dict], allowed: Sequence[str] | None, entity_label: str) -> List[dict]:
    if not allowed:
        return list(entities)
    allowed_set = set(allowed)
    filtered = [item for item in entities if item.get("name") in allowed_set]
    missing = allowed_set.difference({item.get("name") for item in filtered})
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Unknown {entity_label} requested: {names}")
    return filtered


def chunk_list(items: Sequence[dict], chunk_size: int | None) -> List[List[dict]]:
    if not items:
        return []
    if not chunk_size or chunk_size <= 0 or chunk_size >= len(items):
        return [list(items)]
    return [list(items[i : i + chunk_size]) for i in range(0, len(items), chunk_size)]


def write_chunk_file(folder: Path, prefix: str, key: str, chunk_index: int, payload: List[dict]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    chunk_path = folder / f"{prefix}_{chunk_index:03d}.json"
    with chunk_path.open("w", encoding="utf-8") as handle:
        json.dump({key: payload}, handle, indent=2)
        handle.write("\n")
    return chunk_path


def build_command(
    python_executable: Path,
    train_script: Path,
    email: str,
    models_file: Path,
    datasets_file: Path,
    extra_args: Sequence[str],
) -> List[str]:
    cmd = [str(python_executable), str(train_script), "--email", email]
    cmd += ["--models-file", str(models_file), "--datasets-file", str(datasets_file)]
    cmd.extend(extra_args)
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch runner for SLM-Bench experiments. Automatically chunks models and datasets into smaller runs and logs their output.",
    )
    parser.add_argument("--email", required=True, help="Notification email forwarded to train.py")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (defaults to the current interpreter)",
    )
    parser.add_argument(
        "--models-file",
        default="models.json",
        help="Path to the models.json file (relative to repo root unless absolute)",
    )
    parser.add_argument(
        "--datasets-file",
        default="datasets.json",
        help="Path to the datasets.json file (relative to repo root unless absolute)",
    )
    parser.add_argument("--models", nargs="+", help="Optional subset of model names to run")
    parser.add_argument("--datasets", nargs="+", help="Optional subset of dataset names to run")
    parser.add_argument(
        "--models-per-run",
        type=int,
        default=0,
        help="Number of models per train.py invocation (0 means all)",
    )
    parser.add_argument(
        "--datasets-per-run",
        type=int,
        default=3,
        help="Number of datasets per train.py invocation (0 means all)",
    )
    parser.add_argument(
        "--output-dir",
        default="experiment_runs",
        help="Directory (relative to repo root) where run logs are stored",
    )
    parser.add_argument(
        "--train-args",
        default="",
        help="Extra arguments forwarded verbatim to train.py (example: --train-args=\"--max-models 2\")",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue scheduling runs even if one invocation fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    train_script = repo_root / "train.py"
    # Preserve symlinks (e.g., virtualenv python) so subprocess inherits the expected site-packages
    python_path = Path(os.path.abspath(str(Path(args.python).expanduser())))
    python_executable = python_path

    models_config = (repo_root / args.models_file).resolve() if not Path(args.models_file).is_absolute() else Path(args.models_file)
    datasets_config = (repo_root / args.datasets_file).resolve() if not Path(args.datasets_file).is_absolute() else Path(args.datasets_file)

    models = ensure_names_available(load_entities(models_config, "models"), args.models, "models")
    datasets = ensure_names_available(load_entities(datasets_config, "datasets"), args.datasets, "datasets")

    if not models:
        raise ValueError("No models available to schedule.")
    if not datasets:
        raise ValueError("No datasets available to schedule.")

    model_chunks = chunk_list(models, args.models_per_run)
    dataset_chunks = chunk_list(datasets, args.datasets_per_run)

    tmp_dir = repo_root / ".slmbench" / "chunks"
    logs_dir = (repo_root / args.output_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    extra_train_args = shlex.split(args.train_args)

    model_chunk_paths = [
        write_chunk_file(tmp_dir, "models", "models", idx, chunk)
        for idx, chunk in enumerate(model_chunks)
    ]
    dataset_chunk_paths = [
        write_chunk_file(tmp_dir, "datasets", "datasets", idx, chunk)
        for idx, chunk in enumerate(dataset_chunks)
    ]

    run_counter = 0
    failures = 0

    for model_idx, model_file in enumerate(model_chunk_paths):
        for dataset_idx, dataset_file in enumerate(dataset_chunk_paths):
            run_counter += 1
            log_file = logs_dir / f"run_{run_counter:04d}.log"
            cmd = build_command(
                python_executable=python_executable,
                train_script=train_script,
                email=args.email,
                models_file=model_file,
                datasets_file=dataset_file,
                extra_args=extra_train_args,
            )
            print(f"[Run {run_counter:04d}] models chunk {model_idx:02d}, datasets chunk {dataset_idx:02d}")
            print(f"           Command: {' '.join(cmd)}")
            print(f"           Log file: {log_file}")

            if args.dry_run:
                continue

            with log_file.open("w", encoding="utf-8") as log_handle:
                process = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
            if process.returncode != 0:
                failures += 1
                print(
                    f"Run {run_counter:04d} failed with exit code {process.returncode}. Logs: {log_file}",
                    file=sys.stderr,
                )
                if not args.keep_going:
                    raise SystemExit(process.returncode)

    summary = f"Scheduled {run_counter} run(s). Failures: {failures}. Logs stored in {logs_dir}."
    print(summary)


if __name__ == "__main__":
    main()
