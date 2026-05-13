#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REQUIRED_KEYS = [
    "need_context",
    "question_type",
    "entities",
    "constraints",
    "subquestions",
    "useful_facts",
    "missing_info",
    "answer_hint",
]

LIST_KEYS = ["entities", "constraints", "subquestions", "useful_facts", "missing_info"]
SYSTEM_PROMPT = "You are a context generator for small language models. Generate minimal sufficient context."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CLI generation with schema-forced normalization")
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="name=path pairs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def parse_model_pairs(items: List[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --models item: {item}. Expected name=path")
        name, path = item.split("=", 1)
        parsed[name.strip()] = path.strip()
    return parsed


def normalize_instruction(instruction: str) -> str:
    text = str(instruction)
    if "Output JSON with:" in text and "subquestions" not in text:
        text = text.replace(
            "Output JSON with: need_context, question_type, entities, constraints, useful_facts, missing_info, answer_hint.",
            "Output JSON with: need_context, question_type, entities, constraints, subquestions, useful_facts, missing_info, answer_hint.",
        )
    return text


def make_prompt(instruction: str, input_text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{normalize_instruction(instruction)}\n\n{input_text}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_json_candidate(text: str) -> Optional[dict]:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def to_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def normalize_schema(raw: Optional[dict]) -> dict:
    raw = raw or {}
    norm = {
        "need_context": to_bool(raw.get("need_context", False)),
        "question_type": str(raw.get("question_type", "")).strip(),
        "entities": to_str_list(raw.get("entities", [])),
        "constraints": to_str_list(raw.get("constraints", [])),
        "subquestions": to_str_list(raw.get("subquestions", [])),
        "useful_facts": to_str_list(raw.get("useful_facts", [])),
        "missing_info": to_str_list(raw.get("missing_info", [])),
        "answer_hint": str(raw.get("answer_hint", "")).strip(),
    }
    return norm


def list_f1(pred_list: List[str], gold_list: List[str]) -> float:
    pred_set = set(item.strip().lower() for item in pred_list if item.strip())
    gold_set = set(item.strip().lower() for item in gold_list if item.strip())
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    return 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9가-힣]+", text.lower()))


def useful_facts_relaxed_f1(pred_list: List[str], gold_list: List[str]) -> float:
    if not pred_list and not gold_list:
        return 1.0
    if not pred_list or not gold_list:
        return 0.0

    pred_text = " ".join(pred_list[:1])
    pred_tokens = _tokenize(pred_text)
    if not pred_tokens:
        return 0.0

    best = 0.0
    for gold in gold_list:
        gold_tokens = _tokenize(gold)
        if not gold_tokens:
            continue
        inter = len(pred_tokens & gold_tokens)
        if inter == 0:
            continue
        precision = inter / len(pred_tokens)
        recall = inter / len(gold_tokens)
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
        if f1 > best:
            best = f1
    return best


def safe_target(record: dict) -> dict:
    try:
        parsed = json.loads(record["output"])
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def evaluate_model(model_name: str, model_path: str, records: List[dict], max_new_tokens: int) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()

    latencies: List[float] = []
    raw_valid = 0
    raw_full_schema = 0
    forced_full_schema = 0
    raw_key_presence_sum = 0.0
    raw_field_exact_sum = 0.0
    forced_field_exact_sum = 0.0

    need_context_acc = 0.0
    question_type_acc = 0.0
    entities_f1_sum = 0.0
    useful_facts_f1_sum = 0.0
    useful_facts_relaxed_sum = 0.0

    with torch.no_grad():
        for record in records:
            prompt = make_prompt(record["instruction"], record["input"])
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

            new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True)

            raw_obj = extract_json_candidate(generated)
            target = normalize_schema(safe_target(record))

            if isinstance(raw_obj, dict):
                raw_valid += 1
                present = sum(1 for key in REQUIRED_KEYS if key in raw_obj)
                raw_key_presence_sum += present / len(REQUIRED_KEYS)
                if present == len(REQUIRED_KEYS):
                    raw_full_schema += 1

                raw_exact = 0
                for key in REQUIRED_KEYS:
                    if key in raw_obj and raw_obj.get(key) == target.get(key):
                        raw_exact += 1
                raw_field_exact_sum += raw_exact / len(REQUIRED_KEYS)

            forced = normalize_schema(raw_obj)
            if all(key in forced for key in REQUIRED_KEYS):
                forced_full_schema += 1

            forced_exact = sum(1 for key in REQUIRED_KEYS if forced.get(key) == target.get(key))
            forced_field_exact_sum += forced_exact / len(REQUIRED_KEYS)

            need_context_acc += 1.0 if forced["need_context"] == target["need_context"] else 0.0
            question_type_acc += 1.0 if forced["question_type"] == target["question_type"] else 0.0
            entities_f1_sum += list_f1(forced["entities"], target["entities"])
            useful_facts_f1_sum += list_f1(forced["useful_facts"], target["useful_facts"])
            useful_facts_relaxed_sum += useful_facts_relaxed_f1(forced["useful_facts"], target["useful_facts"])

    n = len(records)
    result = {
        "model_name": model_name,
        "num_samples": n,
        "avg_latency_sec": round(sum(latencies) / n, 4),
        "p50_latency_sec": round(sorted(latencies)[n // 2], 4),
        "raw_json_valid_rate": round(raw_valid / n, 4),
        "raw_full_schema_rate": round(raw_full_schema / n, 4),
        "forced_full_schema_rate": round(forced_full_schema / n, 4),
        "raw_avg_key_presence": round(raw_key_presence_sum / n, 4),
        "raw_avg_field_exact_match": round(raw_field_exact_sum / n, 4),
        "forced_avg_field_exact_match": round(forced_field_exact_sum / n, 4),
        "forced_need_context_acc": round(need_context_acc / n, 4),
        "forced_question_type_acc": round(question_type_acc / n, 4),
        "forced_entities_f1": round(entities_f1_sum / n, 4),
        "forced_useful_facts_f1": round(useful_facts_f1_sum / n, 4),
        "forced_useful_facts_relaxed_f1": round(useful_facts_relaxed_sum / n, 4),
    }

    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    model_map = parse_model_pairs(args.models)

    holdout_path = Path(args.holdout)
    records: List[dict] = []
    with holdout_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if args.limit > 0:
        records = records[: args.limit]

    if not records:
        raise RuntimeError("No evaluation records found")

    summary = {
        "holdout": str(holdout_path),
        "num_samples": len(records),
        "results": {},
    }

    for name, path in model_map.items():
        summary["results"][name] = evaluate_model(name, path, records, args.max_new_tokens)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
