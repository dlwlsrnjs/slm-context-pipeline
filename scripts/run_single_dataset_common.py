#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train import ExperimentManager


def parse_common_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one model on one dataset")
    parser.add_argument("--email", required=True)
    parser.add_argument("--local-datasets-dir", default=str(ROOT / "local_datasets"))
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-eval-samples", type=int, default=32)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    return parser.parse_args()


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is required, but CUDA is not available. "
            "Check NVIDIA driver, Docker GPU runtime, and PyTorch CUDA build."
        )

    gpu_name = torch.cuda.get_device_name(0)
    print(f"[GPU-OK] Using CUDA device: {gpu_name}")


def run_single_dataset(models_file: Path, dataset: str, args: argparse.Namespace) -> None:
    require_cuda()
    manager = ExperimentManager(
        email=args.email,
        models_file=str(models_file),
        datasets_file=str(ROOT / "datasets.json"),
        local_datasets_dir=args.local_datasets_dir,
        dataset_filter=[dataset],
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_datasets=1,
        trust_remote_code=not args.no_trust_remote_code,
    )
    manager.run_all_experiments()
