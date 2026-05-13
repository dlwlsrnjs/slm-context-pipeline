#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from datasets import get_dataset_config_names, load_dataset


REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_JSON = REPO_ROOT / "datasets.json"
OUT_DIR = REPO_ROOT / "local_datasets"
SUMMARY_PATH = OUT_DIR / "download_summary.json"

CONFIG_HINTS = {
    "ARC-Easy": ["ARC-Easy"],
    "ARC-Challenge": ["ARC-Challenge"],
    "RACE-Middle": ["middle"],
    "RACE-High": ["high"],
    "glue_qnli": ["qnli"],
}

REPO_HINTS = {
    "ARC-Easy": ["ai2_arc", "allenai/ai2_arc"],
    "ARC-Challenge": ["ai2_arc", "allenai/ai2_arc"],
    "Hellaswag": ["hellaswag", "Rowan/hellaswag"],
    "CommonsenseQA": ["commonsense_qa", "tau/commonsense_qa"],
    "CoQA": ["stanfordnlp/coqa", "coqa"],
    "viggo": ["GEM/viggo", "viggo"],
    "drop": ["ucinlp/drop", "drop"],
    "MATH": ["lighteval/MATH", "hendrycks/competition_mathematics"],
    "SVAMP": ["ChilleD/SVAMP"],
    "MathQA": ["math_qa"],
    "TheoremQA": ["TIGER-Lab/TheoremQA"],
}


def repo_id_from_url(url: str) -> str:
    marker = "huggingface.co/datasets/"
    if marker not in url:
        raise ValueError(f"Unsupported dataset URL format: {url}")
    return url.split(marker, 1)[1].strip("/")


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def try_load_dataset(repo_id: str, preferred_configs: List[Optional[str]]):
    attempted = []
    for cfg in preferred_configs:
        key = cfg if cfg is not None else "<default>"
        if key in attempted:
            continue
        attempted.append(key)
        try:
            if cfg is None:
                return load_dataset(repo_id, trust_remote_code=True), cfg, attempted
            return load_dataset(repo_id, cfg, trust_remote_code=True), cfg, attempted
        except Exception:
            pass

    try:
        available = get_dataset_config_names(repo_id, trust_remote_code=True)
    except Exception:
        available = []

    for cfg in available:
        if cfg in attempted:
            continue
        attempted.append(cfg)
        try:
            return load_dataset(repo_id, cfg, trust_remote_code=True), cfg, attempted
        except Exception:
            continue

    raise RuntimeError(f"Failed to load dataset: {repo_id}; attempted configs={attempted}")


def try_load_with_repo_fallbacks(name: str, primary_repo_id: str, preferred_configs: List[Optional[str]]):
    repo_candidates = []
    seen = set()
    for candidate in [primary_repo_id] + REPO_HINTS.get(name, []):
        if candidate in seen:
            continue
        seen.add(candidate)
        repo_candidates.append(candidate)

    errors = []
    for repo_id in repo_candidates:
        try:
            dataset, used_config, attempted = try_load_dataset(repo_id, preferred_configs)
            return dataset, repo_id, used_config, attempted
        except Exception as exc:
            errors.append(f"{repo_id}: {exc}")
    raise RuntimeError(" | ".join(errors))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with DATASETS_JSON.open("r", encoding="utf-8") as f:
        items = json.load(f)["datasets"]

    success = []
    failed = []

    for item in items:
        name = item["name"]
        url = item["huggingface_url"]
        repo_id = repo_id_from_url(url)

        target_dir = OUT_DIR / safe_name(name)
        if target_dir.exists():
            success.append(
                {
                    "name": name,
                    "repo_id": repo_id,
                    "config": "<already_downloaded>",
                    "path": str(target_dir),
                }
            )
            print(f"[SKIP] {name}: already exists")
            continue

        hints = CONFIG_HINTS.get(name, [])
        preferred = [None] + hints

        print(f"[DOWNLOAD] {name} ({repo_id})")
        try:
            dataset, used_repo, used_config, attempted = try_load_with_repo_fallbacks(name, repo_id, preferred)
            dataset.save_to_disk(str(target_dir))
            success.append(
                {
                    "name": name,
                    "repo_id": used_repo,
                    "config": used_config,
                    "path": str(target_dir),
                    "attempted": attempted,
                }
            )
            print(f"[OK] {name} -> {target_dir}")
        except Exception as e:
            failed.append(
                {
                    "name": name,
                    "repo_id": repo_id,
                    "error": str(e),
                }
            )
            print(f"[FAIL] {name}: {e}")

    summary = {
        "total": len(items),
        "success_count": len(success),
        "failed_count": len(failed),
        "success": success,
        "failed": failed,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========================")
    print(f"Total: {summary['total']}")
    print(f"Success: {summary['success_count']}")
    print(f"Failed: {summary['failed_count']}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
