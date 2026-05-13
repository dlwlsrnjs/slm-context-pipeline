#!/usr/bin/env python3
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset

SYSTEM_PROMPT = "You are a context generator for small language models. Generate minimal sufficient context."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C/D retraining with CLI-generation fidelity early stopping")
    parser.add_argument("--task-c", default="slm_context_pipeline/data/train_run_en_v2/task_c_compression.jsonl")
    parser.add_argument("--task-d", default="slm_context_pipeline/data/train_run_en_v2/task_d_full_generation.jsonl")
    parser.add_argument("--holdout", default="experiment_results/cli_compare/holdout_d_eval.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--work-dir", default="experiment_results/cli_compare/cd_content_refine")
    parser.add_argument("--max-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--eval-limit", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--tune-strategy", choices=["norms", "norms_lm_head", "full"], default="norms_lm_head")
    parser.add_argument("--easy-useful-facts", action="store_true", help="Lower useful_facts difficulty by shrinking targets")
    parser.add_argument("--useful-facts-max-items", type=int, default=1)
    parser.add_argument("--useful-facts-max-chars", type=int, default=140)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_question(input_text: str) -> str:
    match = re.search(r"Question:\s*(.*)", input_text)
    if not match:
        return ""
    first = match.group(1)
    return first.split("\n", 1)[0].strip()


def normalize_instruction(instruction: str) -> str:
    text = str(instruction)
    if "Output JSON with:" in text and "subquestions" not in text:
        text = text.replace(
            "Output JSON with: need_context, question_type, entities, constraints, useful_facts, missing_info, answer_hint.",
            "Output JSON with: need_context, question_type, entities, constraints, subquestions, useful_facts, missing_info, answer_hint.",
        )
    return text


def to_prompt(instruction: str, input_text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{normalize_instruction(instruction)}\n\n{input_text}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_output_json(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def simplify_fact_text(text: str, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).strip())
    if not cleaned:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", cleaned)[0].strip()
    if len(sentence) > max_chars:
        sentence = sentence[:max_chars].rsplit(" ", 1)[0].strip()
    return sentence


def maybe_simplify_output(row: dict, args: argparse.Namespace) -> dict:
    if not args.easy_useful_facts:
        return row

    parsed = parse_output_json(row.get("output", ""))
    if not parsed:
        return row

    facts = parsed.get("useful_facts", [])
    if isinstance(facts, list):
        simplified = []
        for item in facts[: max(1, args.useful_facts_max_items)]:
            text = simplify_fact_text(item, args.useful_facts_max_chars)
            if text:
                simplified.append(text)
        parsed["useful_facts"] = simplified

    new_row = dict(row)
    new_row["output"] = json.dumps(parsed, ensure_ascii=False)
    return new_row


def good_c_sample(row: dict) -> bool:
    data = parse_output_json(row.get("output", ""))
    facts = data.get("useful_facts", [])
    if not isinstance(facts, list) or len(facts) < 1:
        return False
    total_chars = sum(len(str(x).strip()) for x in facts)
    return total_chars >= 40


def good_d_sample(row: dict) -> bool:
    data = parse_output_json(row.get("output", ""))
    entities = data.get("entities", [])
    facts = data.get("useful_facts", [])
    hint = str(data.get("answer_hint", "")).strip()
    qtype = str(data.get("question_type", "")).strip()
    if not isinstance(entities, list) or len(entities) < 1:
        return False
    if not isinstance(facts, list) or len(facts) < 1:
        return False
    if sum(len(str(x).strip()) for x in facts) < 40:
        return False
    if not qtype or not hint:
        return False
    return True


def build_quality_dataset(args: argparse.Namespace, work_dir: Path) -> Dict[str, int]:
    holdout = read_jsonl(Path(args.holdout))
    holdout_questions = {extract_question(item.get("input", "")) for item in holdout}

    c_rows = read_jsonl(Path(args.task_c))
    d_rows = read_jsonl(Path(args.task_d))

    c_kept: List[dict] = []
    d_kept: List[dict] = []

    for row in c_rows:
        q = extract_question(row.get("input", ""))
        if q in holdout_questions:
            continue
        if good_c_sample(row):
            c_kept.append(maybe_simplify_output(row, args))

    for row in d_rows:
        q = extract_question(row.get("input", ""))
        if q in holdout_questions:
            continue
        if good_d_sample(row):
            d_kept.append(maybe_simplify_output(row, args))

    merged = c_kept + d_kept

    hf_data = {
        "prompt": [to_prompt(item.get("instruction", ""), item.get("input", "")) for item in merged],
        "target": [item.get("output", "") for item in merged],
    }
    ds = Dataset.from_dict(hf_data)

    dataset_dir = work_dir / "hf_cd_quality_train"
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(dataset_dir))

    stats = {
        "holdout_questions": len(holdout_questions),
        "raw_c": len(c_rows),
        "raw_d": len(d_rows),
        "kept_c": len(c_kept),
        "kept_d": len(d_kept),
        "train_total": len(merged),
    }
    (work_dir / "quality_dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def run_cmd(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def read_eval_score(eval_path: Path) -> Dict[str, float]:
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    res = data["results"]["current"]
    raw_json_valid = float(res.get("raw_json_valid_rate", 0.0))
    raw_key_presence = float(res.get("raw_avg_key_presence", 0.0))
    forced_field_exact = float(res.get("forced_avg_field_exact_match", 0.0))
    need_context_acc = float(res.get("forced_need_context_acc", 0.0))
    question_type_acc = float(res.get("forced_question_type_acc", 0.0))

    score = (
        0.35 * forced_field_exact
        + 0.25 * raw_json_valid
        + 0.15 * raw_key_presence
        + 0.15 * need_context_acc
        + 0.10 * question_type_acc
    )

    return {
        "cli_score": score,
        "raw_json_valid_rate": raw_json_valid,
        "raw_avg_key_presence": raw_key_presence,
        "forced_avg_field_exact_match": forced_field_exact,
        "forced_need_context_acc": need_context_acc,
        "forced_question_type_acc": question_type_acc,
        "forced_entities_f1": float(res.get("forced_entities_f1", 0.0)),
        "forced_useful_facts_f1": float(res.get("forced_useful_facts_f1", 0.0)),
        "forced_useful_facts_relaxed_f1": float(res.get("forced_useful_facts_relaxed_f1", 0.0)),
    }


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    runs_dir = work_dir / "runs"
    evals_dir = work_dir / "evals"
    runs_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)

    stats = build_quality_dataset(args, work_dir)
    if stats["train_total"] < 100:
        raise RuntimeError(f"Too few quality samples after filtering: {stats['train_total']}")

    current_model = args.base_model
    best_model = current_model
    best_score = -1.0
    no_improve = 0
    history: List[dict] = []

    for epoch in range(1, args.max_epochs + 1):
        out_dir = runs_dir / f"epoch_{epoch}"
        run_cmd([
            "python3",
            "scripts/finetune_qwen_icl_sft.py",
            "--model-name",
            current_model,
            "--dataset-dir",
            str(work_dir / "hf_cd_quality_train"),
            "--max-length",
            str(args.max_length),
            "--epochs",
            "1",
            "--batch-size",
            str(args.batch_size),
            "--grad-accum",
            str(args.grad_accum),
            "--learning-rate",
            str(args.learning_rate),
            "--tune-strategy",
            args.tune_strategy,
            "--output-dir",
            str(out_dir),
        ])

        current_model = str(out_dir / "final")
        eval_path = evals_dir / f"epoch_{epoch}.json"
        run_cmd([
            "python3",
            "scripts/eval_cli_structured_outputs.py",
            "--holdout",
            args.holdout,
            "--models",
            f"current={current_model}",
            "--output",
            str(eval_path),
            "--max-new-tokens",
            "384",
            "--limit",
            str(args.eval_limit),
        ])

        metrics = read_eval_score(eval_path)
        entry = {"epoch": epoch, "model": current_model, **metrics}
        history.append(entry)

        if metrics["cli_score"] > best_score + args.min_delta:
            best_score = metrics["cli_score"]
            best_model = current_model
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            break

    summary = {
        "config": vars(args),
        "dataset_stats": stats,
        "best_model": best_model,
        "best_cli_score": best_score,
        "history": history,
        "recommend_upsize": len(history) >= args.patience and history[-1]["cli_score"] <= best_score + args.min_delta,
    }

    best_model_path = Path(best_model).resolve()
    removed_run_dirs = []
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.glob("epoch_*")):
            run_final = (run_dir / "final").resolve()
            if run_final != best_model_path:
                shutil.rmtree(run_dir, ignore_errors=True)
                removed_run_dirs.append(str(run_dir))

    summary["removed_run_dirs"] = removed_run_dirs
    (work_dir / "cd_refine_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
