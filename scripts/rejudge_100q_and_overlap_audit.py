import json
import re
from collections import Counter, defaultdict
from pathlib import Path


RESULT_PATH = Path("experiment_results/cli_compare_incremental/context_ablation_100q_1b.json")
TASK_C_PATH = Path("slm_context_pipeline/data/train_run_en_v2/task_c_compression.jsonl")
TASK_D_PATH = Path("slm_context_pipeline/data/train_run_en_v2/task_d_full_generation.jsonl")
HOLDOUT_SMALL_PATH = Path("experiment_results/cli_compare_incremental/holdout_d_eval_small.jsonl")
OUT_PATH = Path("experiment_results/cli_compare_incremental/context_ablation_100q_1b_rejudge_and_overlap.json")


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def extract_question_from_input(input_text: str) -> str:
    m = re.search(r"Question:\s*(.*)", input_text, flags=re.S)
    return m.group(1).strip() if m else ""


def extract_first_number(text: str):
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return matches[0] if matches else None


def extract_last_number(text: str):
    matches = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return matches[-1] if matches else None


def extract_yes_no(text: str):
    t = norm(text)
    if re.search(r"\b(yes|true)\b", t):
        return "yes"
    if re.search(r"\b(no|false)\b", t):
        return "no"
    return None


def parse_piqa_options(question_text: str):
    qa = re.search(r"Option\s*A:\s*(.*?)\s*(?:\n|$)", question_text, flags=re.I | re.S)
    qb = re.search(r"Option\s*B:\s*(.*?)\s*(?:\n|$)", question_text, flags=re.I | re.S)
    a = qa.group(1).strip() if qa else ""
    b = qb.group(1).strip() if qb else ""
    return a, b


def infer_gold_piqa_label(gold: str, question: str):
    a_text, b_text = parse_piqa_options(question)
    a_n = norm(a_text)
    b_n = norm(b_text)
    g_n = norm(gold)
    if a_n and (a_n in g_n or g_n in a_n):
        return "A"
    if b_n and (b_n in g_n or g_n in b_n):
        return "B"
    return None


def detect_predicted_option_label(pred: str):
    t = norm(pred)
    if re.search(r"\b(option\s*a|a\b|option\s*1|1\b)", t):
        return "A"
    if re.search(r"\b(option\s*b|b\b|option\s*2|2\b)", t):
        return "B"
    return None


def piqa_correct(pred: str, gold: str, question: str):
    pred_n = norm(pred)
    gold_n = norm(gold)
    gold_label = infer_gold_piqa_label(gold, question)
    pred_label = detect_predicted_option_label(pred)

    if gold_label and pred_label:
        return gold_label == pred_label

    if gold_n and (gold_n in pred_n or pred_n in gold_n):
        return True

    a_text, b_text = parse_piqa_options(question)
    label = pred_label
    if not label:
        return False

    chosen = norm(a_text if label == "A" else b_text)
    return bool(chosen and gold_n and (chosen in gold_n or gold_n in chosen))


def gsm8k_correct(pred: str, gold: str):
    gold_num = extract_first_number(gold)
    if not gold_num:
        return False
    first_num = extract_first_number(pred)
    last_num = extract_last_number(pred)
    return first_num == gold_num or last_num == gold_num


def boolq_correct(pred: str, gold: str):
    pb = extract_yes_no(pred)
    gb = extract_yes_no(gold)
    return pb is not None and gb is not None and pb == gb


def judge(source: str, question: str, pred: str, gold: str):
    s = norm(source)
    if s == "boolq":
        return boolq_correct(pred, gold)
    if s == "piqa":
        return piqa_correct(pred, gold, question)
    if s == "gsm8k":
        return gsm8k_correct(pred, gold)
    return norm(gold) in norm(pred)


def summarize(items):
    case_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    case_source_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for it in items:
        src = norm(it.get("source", "unknown"))
        for case in ["no_context", "base_context", "plus3_context"]:
            ok = bool(it["rejudge"][case])
            case_stats[case]["total"] += 1
            case_stats[case]["correct"] += int(ok)
            case_source_stats[case][src]["total"] += 1
            case_source_stats[case][src]["correct"] += int(ok)

    overall = {}
    by_source = {}
    for case, st in case_stats.items():
        overall[case] = {
            "correct": st["correct"],
            "total": st["total"],
            "accuracy": round(st["correct"] / max(st["total"], 1), 4),
        }

    for case, src_map in case_source_stats.items():
        by_source[case] = {}
        for src, st in src_map.items():
            by_source[case][src] = {
                "correct": st["correct"],
                "total": st["total"],
                "accuracy": round(st["correct"] / max(st["total"], 1), 4),
            }

    return {"overall": overall, "by_source": by_source}


def build_overlap_audit(eval_items):
    q_eval = [it["question"].strip() for it in eval_items]
    q_eval_set = set(q_eval)

    c_rows = read_jsonl(TASK_C_PATH)
    d_rows = read_jsonl(TASK_D_PATH)
    train_questions = set()
    for row in c_rows + d_rows:
        q = extract_question_from_input(row.get("input", ""))
        if q:
            train_questions.add(q)

    holdout_rows = read_jsonl(HOLDOUT_SMALL_PATH)
    holdout_questions = set()
    for row in holdout_rows:
        q = extract_question_from_input(row.get("input", ""))
        if q:
            holdout_questions.add(q)

    filtered_train = train_questions - holdout_questions

    source_counter = Counter(norm(it.get("source", "unknown")) for it in eval_items)
    per_source = {}
    for src in source_counter.keys():
        src_q = {it["question"].strip() for it in eval_items if norm(it.get("source", "unknown")) == src}
        per_source[src] = {
            "eval_count": len(src_q),
            "overlap_with_raw_train": len(src_q & train_questions),
            "overlap_with_filtered_train": len(src_q & filtered_train),
            "overlap_with_holdout_small": len(src_q & holdout_questions),
        }

    return {
        "eval_total": len(q_eval_set),
        "train_unique_questions": len(train_questions),
        "holdout_small_unique_questions": len(holdout_questions),
        "overlap_eval_with_raw_train": len(q_eval_set & train_questions),
        "overlap_eval_with_filtered_train": len(q_eval_set & filtered_train),
        "overlap_eval_with_holdout_small": len(q_eval_set & holdout_questions),
        "per_source": per_source,
    }


def main():
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    items = result["items"]

    judged_items = []
    changed_counts = {"no_context": 0, "base_context": 0, "plus3_context": 0}

    for it in items:
        src = it.get("source", "")
        q = it.get("question", "")
        gold = it.get("gold_answer", "")
        rejudge = {}
        for case in ["no_context", "base_context", "plus3_context"]:
            pred = it["results"][case]["prediction"]
            new_ok = judge(src, q, pred, gold)
            old_ok = bool(it["results"][case].get("correct", False))
            rejudge[case] = new_ok
            if new_ok != old_ok:
                changed_counts[case] += 1

        judged_items.append(
            {
                "id": it.get("id"),
                "source": src,
                "question": q,
                "gold_answer": gold,
                "results": it["results"],
                "rejudge": rejudge,
            }
        )

    rejudge_summary = summarize(judged_items)
    overlap_audit = build_overlap_audit(judged_items)

    report = {
        "input_result": str(RESULT_PATH),
        "rejudge_rules": {
            "boolq": "yes/no extraction",
            "piqa": "gold text match + option label parsing",
            "gsm8k": "first/last number exact match to gold number",
        },
        "changed_decision_counts": changed_counts,
        "rejudge_summary": rejudge_summary,
        "overlap_audit": overlap_audit,
        "items": judged_items,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["changed_decision_counts"], ensure_ascii=False, indent=2))
    print(json.dumps(report["rejudge_summary"]["overall"], ensure_ascii=False, indent=2))
    print(json.dumps(report["overlap_audit"], ensure_ascii=False, indent=2))
    print(f"[Saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
