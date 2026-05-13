"""PRL-style distillation: GPT-4o-mini generates <think>+reasoning+<answer>+4 demos.

Differences from build_math_icl_distill_dataset.py:
  - Prompt asks for <think>...</think><answer>...</answer> structure
  - Quality check verifies both blocks + 4 demos inside <answer>
  - Larger max_tokens (think + 4 demos)
  - System prompt emphasizes reasoning quality
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-paths", nargs="+", required=True,
                   help="JSONL files with id/question/answer/solution")
    p.add_argument("--samples-per-dataset", type=int, default=5000)
    p.add_argument("--shots", type=int, default=4)
    p.add_argument("--seed-pool-size", type=int, default=12,
                   help="Number of nearby candidates from which we pick `shots`")
    p.add_argument("--teacher-model", default="gpt-4o-mini")
    p.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--max-tokens", type=int, default=900,
                   help="Larger than vanilla (need <think> + 4 demos)")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--workers", type=int, default=16,
                   help="Concurrent API calls (default 16)")
    p.add_argument("--output-dir", default="icl_distill_math",
                   help="Saves cache + hf_dataset under this folder")
    p.add_argument("--output-suffix", default="prl_real",
                   help="Suffix for cache+dataset names (e.g., prl_real)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap total prompts (0 = no cap, used for smoke test)")
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


def build_prl_student_prompt(seed_examples: str, shots: int,
                              dataset_name: str) -> str:
    return (
        f"You are generating in-context demonstrations for math word problems "
        f"({dataset_name}).\n\n"
        "First, inside <think>...</think>, briefly reason (2-4 sentences) about:\n"
        "  - what arithmetic operations and difficulty the candidate examples use\n"
        "  - what variations or styles you will use across your 4 demos\n"
        "  - any patterns to ensure variety so the demos cover different scenarios\n\n"
        f"Then, inside <answer>...</answer>, output exactly {shots} demonstrations.\n\n"
        "Inside <answer>, each demonstration MUST follow:\n"
        "Q: <math word problem>\n"
        "A: <step-by-step reasoning ending with 'The answer is [number].'>\n\n"
        "Rules:\n"
        "- Each demo self-contained (no external tables/formulas).\n"
        "- Show clear arithmetic steps.\n"
        f"- End every answer with 'The answer is [number].'.\n"
        f"- Output exactly {shots} Q/A pairs inside <answer>.\n"
        "- Do NOT add anything outside <think>/<answer> blocks.\n\n"
        "Candidate examples:\n"
        f"{seed_examples}\n\n"
        "Now output:"
    )


# ── quality check ─────────────────────────────────────────────────────────────

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def quality_check_prl(text: str, shots: int) -> Tuple[bool, str]:
    if not text.strip():
        return False, "empty"
    if text.count("<think>") != 1 or text.count("</think>") != 1:
        return False, "think_tags"
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return False, "answer_tags"
    th = THINK_RE.search(text)
    an = ANSWER_RE.search(text)
    if not th or not an:
        return False, "missing_blocks"
    if th.end() > an.start():
        return False, "wrong_order"
    if not th.group(1).strip():
        return False, "empty_think"
    payload = an.group(1)
    n_q = len(re.findall(r"(?m)^Q:\s+", payload))
    n_a = len(re.findall(r"(?m)^A:\s+", payload))
    n_final = len(re.findall(r"[Tt]he answer is\s+[\d,.\-]+", payload))
    if n_q < shots or n_a < shots or n_final < shots:
        return False, f"format(Q={n_q},A={n_a},final={n_final})"
    return True, ""


# ── OpenAI call ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert ICL teacher for math reasoning. "
    "You first reason step-by-step inside <think> tags about which patterns to "
    "match, then output high-quality, varied demonstrations inside <answer>."
)


def call_openai(api_key: str, model: str, prompt: str,
                max_tokens: int, temperature: float) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
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
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]["content"].strip()
    except Exception as e:
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
    cache: Dict[str, dict] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                cache[row["source_id"]] = row
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


# ── data load ─────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── per-row processing ────────────────────────────────────────────────────────

def process_one(
    target: dict,
    pool: List[dict],
    rng: random.Random,
    args: argparse.Namespace,
    api_key: str,
    cache: Dict[str, dict],
    writer: SafeWriter,
    progress: Dict[str, int],
    progress_lock: threading.Lock,
    dataset_name: str,
) -> None:
    sid = target["id"]
    if not args.force_refresh and sid in cache:
        with progress_lock:
            progress["cached"] += 1
        return

    candidates = [r for r in pool if r["id"] != sid]
    seed_pool = rng.sample(candidates,
                            min(args.seed_pool_size, len(candidates)))
    seed_rows = rng.sample(seed_pool, min(args.shots, len(seed_pool)))
    seed_examples = format_seed_examples(seed_rows, args.shots)
    prompt = build_prl_student_prompt(seed_examples, args.shots, dataset_name)

    teacher_output = ""
    accepted = False
    last_reason = ""
    for attempt in range(args.retries):
        teacher_output = call_openai(
            api_key, args.teacher_model, prompt,
            args.max_tokens, args.temperature)
        ok, reason = quality_check_prl(teacher_output, args.shots)
        if ok:
            accepted = True
            break
        last_reason = reason
        time.sleep(0.5 * (attempt + 1))    # polite backoff

    item = {
        "source_id": sid,
        "dataset_name": dataset_name,
        "prompt": prompt,
        "target": teacher_output,
        "teacher_model": args.teacher_model,
        "format_version": "prl_real_v1",
        "accepted": accepted,
        "reject_reason": "" if accepted else last_reason,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    writer.write(item)
    with progress_lock:
        if accepted:
            progress["accepted"] += 1
        else:
            progress["rejected"] += 1


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    api_key = os.getenv(args.openai_api_key_env, "")
    if not api_key:
        raise RuntimeError(f"env var {args.openai_api_key_env} is not set")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"distilled_cache_{args.output_suffix}.jsonl"
    if args.force_refresh and cache_path.exists():
        cache_path.rename(out_dir / f"distilled_cache_{args.output_suffix}.bak")
        cache = {}
    else:
        cache = load_cache(cache_path)
    writer = SafeWriter(cache_path)
    print(f"[cache] {len(cache)} pre-existing entries at {cache_path}")

    rng = random.Random(args.seed)

    progress = {"accepted": 0, "rejected": 0, "cached": 0}
    progress_lock = threading.Lock()
    progress_total = 0

    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for path in args.dataset_paths:
            rows = load_jsonl(path)
            dataset_name = Path(path).stem
            n_take = min(args.samples_per_dataset, len(rows))
            targets = rows[:n_take]
            pool = rows[:]    # full source pool (for seed candidates)
            print(f"[load] {dataset_name}: {len(targets)} prompts (pool={len(pool)})")
            for target in targets:
                if args.limit and progress_total >= args.limit:
                    break
                progress_total += 1
                fut = ex.submit(
                    process_one,
                    target, pool, rng, args, api_key, cache, writer,
                    progress, progress_lock, dataset_name)
                futures.append(fut)

        last_print = 0
        for done_count, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as e:
                print(f"[error] worker raised: {e}")
            if done_count - last_print >= 50 or done_count == len(futures):
                last_print = done_count
                with progress_lock:
                    p = dict(progress)
                print(f"[progress] {done_count}/{len(futures)} "
                      f"accepted={p['accepted']} "
                      f"rejected={p['rejected']} "
                      f"cached={p['cached']}", flush=True)

    writer.close()

    # Build HF dataset from cache
    final_cache = load_cache(cache_path)
    accepted_rows = [r for r in final_cache.values() if r.get("accepted")]
    print(f"\n[hf-dataset] writing {len(accepted_rows)} accepted rows")
    ds = Dataset.from_list(accepted_rows)
    hf_dir = out_dir / f"hf_distilled_icl_dataset_{args.output_suffix}"
    ds.save_to_disk(str(hf_dir))
    print(f"[saved] {hf_dir}")

    # Stats by dataset
    by_ds = {}
    for r in final_cache.values():
        ds_name = r["dataset_name"]
        by_ds.setdefault(ds_name, {"acc": 0, "rej": 0})
        by_ds[ds_name]["acc" if r.get("accepted") else "rej"] += 1
    print("\n[stats by dataset]")
    for k, v in by_ds.items():
        total = v["acc"] + v["rej"]
        rate = 100 * v["acc"] / total if total else 0
        print(f"  {k:20s} acc={v['acc']:5d} rej={v['rej']:5d} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
