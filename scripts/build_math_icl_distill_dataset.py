#!/usr/bin/env python3
"""
Math ICL distillation dataset builder — parallel version.

Generates 4 unified ICL demonstrations per math problem via GPT-4o-mini,
using concurrent workers for speed. Results are written to JSONL immediately
(thread-safe) so progress is never lost on interruption.

Format of each generated demonstration:
    Q: <math word problem>
    A: <step-by-step reasoning>
    The answer is <number>.

Usage:
    python scripts/build_math_icl_distill_dataset.py \
        --dataset-paths slm_context_pipeline/data/math_5k/gsm8k_5k.jsonl \
                        slm_context_pipeline/data/math_5k/orcamath_5k.jsonl \
                        slm_context_pipeline/data/math_5k/metamath_5k.jsonl \
        --samples-per-dataset 5000 \
        --workers 16 \
        --output-dir icl_distill_math
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import re
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-paths", nargs="+", required=True)
    p.add_argument("--samples-per-dataset", type=int, default=5000)
    p.add_argument("--shots", type=int, default=4)
    p.add_argument("--seed-pool-size", type=int, default=12)
    p.add_argument("--teacher-model", default="gpt-4o-mini")
    p.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--workers", type=int, default=16,
                   help="Concurrent API calls (default 16)")
    p.add_argument("--output-dir", default="icl_distill_math")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-refresh", action="store_true")
    return p.parse_args()


# ── prompt helpers ────────────────────────────────────────────────────────────

def format_seed_examples(rows: List[dict], shots: int) -> str:
    lines = []
    for row in rows[:shots]:
        q = str(row["question"]).strip()
        sol = str(row.get("solution", row.get("answer", ""))).strip()
        sol = re.sub(r"<<[^>]*>>", "", sol)
        sol = re.sub(r"####\s*[\d,]+", "", sol).strip()
        final = str(row.get("answer", "")).strip()
        if final and not re.search(r"[Tt]he answer is", sol):
            sol = sol.rstrip(".") + f"\nThe answer is {final}."
        lines.append(f"Q: {q}\nA: {sol}")
    return "\n\n".join(lines)


def build_student_prompt(seed_examples: str, shots: int, dataset_name: str) -> str:
    format_rule = (
        "Q: <math word problem>\n"
        "A: <step-by-step reasoning ending with 'The answer is [number].'>"
    )
    return (
        f"You are generating in-context demonstrations for math word problems "
        f"({dataset_name}).\n"
        f"Generate exactly {shots} demonstrations with this format:\n"
        f"{format_rule}\n\n"
        "Rules:\n"
        "- Each demo must be self-contained (no external tables or formulas needed).\n"
        "- Show clear arithmetic steps.\n"
        "- End every answer with 'The answer is [number].'\n"
        "- Do NOT add explanations outside the Q/A format.\n\n"
        "Candidate examples:\n"
        f"{seed_examples}\n\n"
        "Now output demonstrations only:\n"
    )


# ── quality check ─────────────────────────────────────────────────────────────

def quality_check(text: str, shots: int) -> Tuple[bool, str]:
    if not text.strip():
        return False, "empty"
    if len(re.findall(r"(?m)^Q:\s+", text)) < shots:
        return False, "format"
    if len(re.findall(r"(?m)^A:\s+", text)) < shots:
        return False, "format"
    if len(re.findall(r"[Tt]he answer is\s+[\d,.\-]+", text)) < shots:
        return False, "missing_final_answer"
    return True, ""


# ── OpenAI call ───────────────────────────────────────────────────────────────

def call_openai(api_key: str, model: str, prompt: str,
                max_tokens: int, temperature: float) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You are an expert ICL teacher for math reasoning. "
                        "Output only high-quality demonstrations."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


# ── thread-safe cache + writer ────────────────────────────────────────────────

class SafeWriter:
    def __init__(self, cache_file: Path):
        self._lock = threading.Lock()
        self._file = cache_file.open("a", encoding="utf-8")

    def write(self, item: dict) -> None:
        line = json.dumps(item, ensure_ascii=False) + "\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()

    def close(self) -> None:
        self._file.close()


def load_cache(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    cache = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                cache[item["prompt_hash"]] = item
    return cache


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── worker task ───────────────────────────────────────────────────────────────

def process_one(
    idx: int,
    rank: int,
    rows: List[dict],
    dataset_name: str,
    args: argparse.Namespace,
    api_key: str,
    cache: Dict[str, dict],
    writer: SafeWriter,
    stats_lock: threading.Lock,
    stats: dict,
    rng_seed: int,
) -> None:
    rng = random.Random(rng_seed)
    pool = [rows[j] for j in range(len(rows)) if j != idx]
    seed_rows = rng.sample(pool, min(args.seed_pool_size, len(pool)))
    seed_examples = format_seed_examples(seed_rows, shots=args.shots)
    prompt = build_student_prompt(seed_examples, args.shots, dataset_name)
    prompt_hash = hashlib.sha256(
        (dataset_name + "\n" + prompt).encode()).hexdigest()

    with stats_lock:
        stats["requested"] += 1

    if (not args.force_refresh) and prompt_hash in cache:
        with stats_lock:
            stats["cache_hit"] += 1
        return

    teacher_output = ""
    ok = False
    reason = ""

    for _ in range(max(1, args.retries)):
        with stats_lock:
            stats["teacher_calls"] += 1
        teacher_output = call_openai(
            api_key, args.teacher_model, prompt,
            args.max_tokens, args.temperature)
        ok, reason = quality_check(teacher_output, args.shots)
        if ok:
            break

    if not ok:
        with stats_lock:
            stats["rejected"] += 1
            stats["reject_reasons"][reason] = \
                stats["reject_reasons"].get(reason, 0) + 1
        return

    item = {
        "prompt_hash": prompt_hash,
        "dataset_name": dataset_name,
        "source_id": rows[idx].get("id", ""),
        "student_prompt": prompt,
        "teacher_output": teacher_output,
        "teacher_model": args.teacher_model,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "sample_index": rank,
    }
    writer.write(item)
    with stats_lock:
        stats["accepted"] += 1
        done = stats["accepted"] + stats["rejected"]
        if done % 100 == 0:
            print(f"  [{dataset_name}] done={done}  "
                  f"accepted={stats['accepted']}  "
                  f"rejected={stats['rejected']}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    api_key = os.getenv(args.openai_api_key_env, "")
    if not api_key:
        raise RuntimeError(f"env var {args.openai_api_key_env} is not set")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_file = out_dir / "distilled_icl_cache.jsonl"
    summary_file = out_dir / "distilled_icl_summary.json"
    hf_dir = out_dir / "hf_distilled_icl_dataset"

    cache = load_cache(cache_file)
    print(f"Loaded {len(cache)} cached items")

    writer = SafeWriter(cache_file)
    stats_lock = threading.Lock()
    stats = {
        "requested": 0, "cache_hit": 0,
        "teacher_calls": 0, "accepted": 0, "rejected": 0,
        "reject_reasons": {},
    }

    for dataset_path in args.dataset_paths:
        dataset_name = Path(dataset_path).stem
        rows = load_jsonl(dataset_path)
        n = min(args.samples_per_dataset, len(rows))

        rng = random.Random(args.seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        indices = indices[:n]

        print(f"\n[{dataset_name}] {n} samples  workers={args.workers}")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_one,
                    idx, rank, rows, dataset_name, args, api_key,
                    cache, writer, stats_lock, stats,
                    args.seed + rank,
                )
                for rank, idx in enumerate(indices)
            ]
            for _ in as_completed(futures):
                pass

        print(f"[{dataset_name}] complete  "
              f"accepted={stats['accepted']}  rejected={stats['rejected']}")

    writer.close()

    # Save HuggingFace dataset from final cache
    final_cache = load_cache(cache_file)
    from datasets import Dataset
    merged = sorted(final_cache.values(),
                    key=lambda x: (x["dataset_name"], x["created_at"]))
    if merged:
        hf_data = {
            "dataset_name": [x["dataset_name"] for x in merged],
            "source_id":    [x.get("source_id", "") for x in merged],
            "prompt":       [x["student_prompt"] for x in merged],
            "target":       [x["teacher_output"] for x in merged],
            "teacher_model":[x["teacher_model"] for x in merged],
            "created_at":   [x["created_at"] for x in merged],
        }
        Dataset.from_dict(hf_data).save_to_disk(str(hf_dir))

    summary = {
        "total_cached": len(final_cache),
        "last_run": stats,
        "output_dataset": str(hf_dir),
        "cache_file": str(cache_file),
    }
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[DONE] total={len(final_cache)}  "
          f"accepted={stats['accepted']}  rejected={stats['rejected']}")
    print(f"HF dataset → {hf_dir}")


if __name__ == "__main__":
    main()
