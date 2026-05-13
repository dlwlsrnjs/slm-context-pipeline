#!/usr/bin/env python3
from pathlib import Path
from scripts.run_single_dataset_common import parse_common_args, run_single_dataset


def main() -> None:
    args = parse_common_args()
    run_single_dataset(Path(__file__).resolve().parent.parent / "model_configs" / "models_qwen2_5_1_5b_instruct.json", "Hellaswag", args)


if __name__ == "__main__":
    main()
