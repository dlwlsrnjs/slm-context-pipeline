#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import json
import os
import random
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reusable distilled ICL dataset using OpenAI teacher")
    parser.add_argument("--dataset-paths", nargs="+", required=True, help="One or more local dataset paths")
    parser.add_argument("--samples-per-dataset", type=int, default=400)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--seed-pool-size", type=int, default=12)
    parser.add_argument("--teacher-model", default="gpt-4o-mini")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--output-dir", default="icl_distill_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def infer_dataset_name(dataset_path: str) -> str:
    return Path(dataset_path).name


def parse_mc_prediction(text: str) -> int:
    lowered = text.strip().lower()
    num_match = re.search(r"\b([1-5])\b", lowered)
    if num_match:
        return int(num_match.group(1)) - 1
    option_match = re.search(r"\boption\s*([1-5])\b", lowered)
    if option_match:
        return int(option_match.group(1)) - 1
    letter_match = re.search(r"\b([abcde])\b", lowered)
    if letter_match:
        return ord(letter_match.group(1)) - ord("a")
    return 0


def to_label(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1", "option1", "a"}:
        return 1
    if text in {"false", "no", "0", "option2", "b"}:
        return 0
    return parse_mc_prediction(text)


def format_seed_examples(rows: List[dict], shots: int, dataset_name: str) -> str:
    lines = []
    for row in rows[:shots]:
        if dataset_name == "BoolQ":
            question = str(row["question"]).strip()
            passage = str(row["passage"]).strip()
            answer = bool(row["answer"])
            lines.append(
                "Q: " + question + "\n"
                "Passage: " + passage + "\n"
                "A: " + ("True" if answer else "False")
            )
        elif dataset_name == "PIQA":
            goal = str(row["goal"]).strip()
            sol1 = str(row["sol1"]).strip()
            sol2 = str(row["sol2"]).strip()
            label = int(row["label"])
            lines.append(
                "Q: " + goal + "\n"
                "Option 1: " + sol1 + "\n"
                "Option 2: " + sol2 + "\n"
                "A: " + str(label + 1)
            )
        elif dataset_name == "WinoGrande":
            sentence = str(row["sentence"]).strip()
            option1 = str(row["option1"]).strip()
            option2 = str(row["option2"]).strip()
            answer = str(row["answer"]).strip()
            label = to_label(answer)
            lines.append(
                "Q: " + sentence + "\n"
                "Option 1: " + option1 + "\n"
                "Option 2: " + option2 + "\n"
                "A: " + str(label + 1)
            )
        elif dataset_name == "Hellaswag":
            ctx = str(row["ctx"]).strip()
            endings = [str(x).strip() for x in row["endings"]]
            label = int(row["label"])
            lines.append(
                "Q: " + ctx + "\n"
                "Option 1: " + endings[0] + "\n"
                "Option 2: " + endings[1] + "\n"
                "Option 3: " + endings[2] + "\n"
                "Option 4: " + endings[3] + "\n"
                "A: " + str(label + 1)
            )
        elif dataset_name in {"e2e_nlg", "viggo"}:
            mr = str(row["meaning_representation"]).strip()
            target = str(row["target"]).strip()
            lines.append(
                "MR: " + mr + "\n"
                "A: " + target
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    return "\n\n".join(lines)


def build_student_prompt(seed_examples: str, shots: int, dataset_name: str) -> str:
    if dataset_name == "BoolQ":
        format_rules = "Q: <question>\\nPassage: <passage>\\nA: True/False"
    elif dataset_name in {"PIQA", "WinoGrande"}:
        format_rules = "Q: <question>\\nOption 1: <text>\\nOption 2: <text>\\nA: 1 or 2"
    elif dataset_name == "Hellaswag":
        format_rules = "Q: <context>\\nOption 1: <text>\\nOption 2: <text>\\nOption 3: <text>\\nOption 4: <text>\\nA: 1/2/3/4"
    elif dataset_name in {"e2e_nlg", "viggo"}:
        format_rules = "MR: <meaning_representation>\\nA: <generated sentence>"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return (
        f"You are generating in-context demonstrations for {dataset_name}.\n"
        f"Generate exactly {shots} demonstrations with this format:\n"
        f"{format_rules}\n\n"
        "Keep each demo concise and high quality.\n"
        "Do not add explanations.\n"
        "Candidate examples:\n"
        f"{seed_examples}\n\n"
        "Now output demonstrations only:\n"
    )


def call_openai_chat_completion(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert ICL teacher. Output only high-quality demonstrations."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError):
        return ""


def quality_check(teacher_output: str, dataset_name: str, shots: int) -> Tuple[bool, str, Dict[str, int]]:
    counts = {"q": 0, "a": 0, "o1": 0, "o2": 0, "a1": 0, "a2": 0}
    if not teacher_output.strip():
        return False, "empty", counts

    if dataset_name in {"PIQA", "WinoGrande"}:
        counts["q"] = len(re.findall(r"(?m)^Q:\s+", teacher_output))
        counts["o1"] = len(re.findall(r"(?m)^Option\s*1:\s+", teacher_output))
        counts["o2"] = len(re.findall(r"(?m)^Option\s*2:\s+", teacher_output))
        answers = re.findall(r"(?m)^A:\s*([^\n]+)", teacher_output)
        counts["a"] = len(answers)
        counts["a1"] = sum(1 for item in answers if re.search(r"\b(1|option\s*1|a)\b", item.strip().lower()))
        counts["a2"] = sum(1 for item in answers if re.search(r"\b(2|option\s*2|b)\b", item.strip().lower()))

        if counts["q"] < shots or counts["a"] < shots or counts["o1"] < shots or counts["o2"] < shots:
            return False, "format", counts
        if dataset_name == "PIQA" and (counts["a1"] == 0 or counts["a2"] == 0):
            return False, "answer_imbalance", counts
        return True, "", counts

    if dataset_name == "BoolQ":
        counts["q"] = len(re.findall(r"(?m)^Q:\s+", teacher_output))
        counts["a"] = len(re.findall(r"(?m)^A:\s+", teacher_output))
        if counts["q"] < shots or counts["a"] < shots:
            return False, "format", counts
        return True, "", counts

    if dataset_name == "Hellaswag":
        counts["q"] = len(re.findall(r"(?m)^Q:\s+", teacher_output))
        counts["a"] = len(re.findall(r"(?m)^A:\s+", teacher_output))
        if counts["q"] < shots or counts["a"] < shots:
            return False, "format", counts
        return True, "", counts

    if dataset_name in {"e2e_nlg", "viggo"}:
        counts["a"] = len(re.findall(r"(?m)^A:\s+", teacher_output))
        if counts["a"] < shots:
            return False, "format", counts
        return True, "", counts

    return True, "", counts


def sample_rows(dataset_split, count: int) -> List[dict]:
    indices = random.sample(range(len(dataset_split)), k=min(count, len(dataset_split)))
    return [dataset_split[index] for index in indices]


def load_cache(cache_file: Path) -> Dict[str, dict]:
    if not cache_file.exists():
        return {}
    cache = {}
    with cache_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cache[item["prompt_hash"]] = item
    return cache


def append_jsonl(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    api_key = os.getenv(args.openai_api_key_env, "")
    if not api_key:
        raise RuntimeError(f"{args.openai_api_key_env} is not set")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / "distilled_icl_cache.jsonl"
    summary_file = output_dir / "distilled_icl_summary.json"
    hf_dataset_dir = output_dir / "hf_distilled_icl_dataset"

    cache = load_cache(cache_file)

    accepted_items: List[dict] = []
    stats = {
        "requested": 0,
        "cache_hit": 0,
        "teacher_calls": 0,
        "accepted": 0,
        "rejected": 0,
        "reject_reasons": {},
    }

    for dataset_path in args.dataset_paths:
        dataset_name = infer_dataset_name(dataset_path)
        data = load_from_disk(dataset_path)
        train_split = data["train"]

        for sample_idx in range(args.samples_per_dataset):
            stats["requested"] += 1

            seed_rows = sample_rows(train_split, args.seed_pool_size)
            seed_examples = format_seed_examples(seed_rows, shots=args.shots, dataset_name=dataset_name)
            student_prompt = build_student_prompt(seed_examples, shots=args.shots, dataset_name=dataset_name)

            prompt_hash = hashlib.sha256((dataset_name + "\n" + student_prompt).encode("utf-8")).hexdigest()

            if (not args.force_refresh) and prompt_hash in cache:
                accepted_items.append(cache[prompt_hash])
                stats["cache_hit"] += 1
                continue

            teacher_output = ""
            is_valid = False
            reject_reason = ""
            quality = {}

            for _ in range(max(1, args.retries)):
                stats["teacher_calls"] += 1
                teacher_output = call_openai_chat_completion(
                    api_key=api_key,
                    model=args.teacher_model,
                    prompt=student_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                is_valid, reject_reason, quality = quality_check(teacher_output, dataset_name, args.shots)
                if is_valid:
                    break

            if not is_valid:
                stats["rejected"] += 1
                stats["reject_reasons"][reject_reason] = stats["reject_reasons"].get(reject_reason, 0) + 1
                continue

            now = datetime.datetime.utcnow().isoformat()
            item = {
                "prompt_hash": prompt_hash,
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "student_prompt": student_prompt,
                "teacher_output": teacher_output,
                "teacher_model": args.teacher_model,
                "quality": quality,
                "created_at": now,
                "sample_index": sample_idx,
            }
            cache[prompt_hash] = item
            accepted_items.append(item)
            append_jsonl(cache_file, item)
            stats["accepted"] += 1

    merged = list(cache.values())
    merged.sort(key=lambda item: (item["dataset_name"], item["created_at"], item["prompt_hash"]))

    hf_data = {
        "dataset_name": [item["dataset_name"] for item in merged],
        "dataset_path": [item["dataset_path"] for item in merged],
        "prompt_hash": [item["prompt_hash"] for item in merged],
        "prompt": [item["student_prompt"] for item in merged],
        "target": [item["teacher_output"] for item in merged],
        "teacher_model": [item["teacher_model"] for item in merged],
        "created_at": [item["created_at"] for item in merged],
    }

    dataset = Dataset.from_dict(hf_data)
    dataset.save_to_disk(str(hf_dataset_dir))

    summary = {
        "total_cached": len(merged),
        "last_run": stats,
        "output_dataset": str(hf_dataset_dir),
        "cache_file": str(cache_file),
    }
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[DONE] Distilled ICL dataset build complete")
    print(f"Cached samples: {len(merged)}")
    print(f"Last run accepted/rejected: {stats['accepted']}/{stats['rejected']}")
    print(f"HF dataset: {hf_dataset_dir}")


if __name__ == "__main__":
    main()
