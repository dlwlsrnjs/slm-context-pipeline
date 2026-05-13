import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen 1.5B using pre-generated context pack.")
    parser.add_argument(
        "--context-pack",
        default="experiment_results/cli_compare_incremental/context_pack_base_plus3_6000.jsonl",
    )
    parser.add_argument("--answer-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--output",
        default="experiment_results/cli_compare_incremental/eval_1b_on_context_pack_6000.json",
    )
    return parser.parse_args()


def read_jsonl(path: Path, limit: int):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def context_to_text(ctx_obj: dict) -> str:
    if not isinstance(ctx_obj, dict):
        return ""
    fields = [
        f"need_context: {ctx_obj.get('need_context')}",
        f"question_type: {ctx_obj.get('question_type', '')}",
        f"entities: {ctx_obj.get('entities', [])}",
        f"constraints: {ctx_obj.get('constraints', [])}",
        f"subquestions: {ctx_obj.get('subquestions', [])}",
        f"useful_facts: {ctx_obj.get('useful_facts', [])}",
        f"missing_info: {ctx_obj.get('missing_info', [])}",
        f"answer_hint: {ctx_obj.get('answer_hint', '')}",
    ]
    return "\n".join(fields)


def load_model_tok(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    return tokenizer, model


def answer_prompt(question: str, context_text: str):
    system = "You are a concise QA assistant. Give only the final answer."
    if context_text.strip():
        user = f"Question: {question}\n\nContext:\n{context_text}\n\nReturn only the final answer."
    else:
        user = f"Question: {question}\n\nReturn only the final answer."
    return (
        f"<|im_start|>system\n{system}\n<|im_end|>\n"
        f"<|im_start|>user\n{user}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def generate_text(tokenizer, model, prompt: str, max_new_tokens: int):
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def extract_num_first(text: str):
    m = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return m[0] if m else None


def extract_num_last(text: str):
    m = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return m[-1] if m else None


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


def detect_pred_label(pred: str):
    t = norm(pred)
    if re.search(r"\b(option\s*a|option\s*1)\b", t):
        return "A"
    if re.search(r"\b(option\s*b|option\s*2)\b", t):
        return "B"
    return None


def infer_gold_label(gold: str, question: str):
    a, b = parse_piqa_options(question)
    g = norm(gold)
    if a and (norm(a) in g or g in norm(a)):
        return "A"
    if b and (norm(b) in g or g in norm(b)):
        return "B"
    return None


def judge(source: str, question: str, pred: str, gold: str):
    s = norm(source)
    if s == "boolq":
        return extract_yes_no(pred) == extract_yes_no(gold) and extract_yes_no(gold) is not None
    if s == "gsm8k":
        g = extract_num_first(gold)
        return g is not None and (extract_num_first(pred) == g or extract_num_last(pred) == g)
    if s == "piqa":
        gold_label = infer_gold_label(gold, question)
        pred_label = detect_pred_label(pred)
        if gold_label and pred_label:
            return gold_label == pred_label
        return norm(gold) in norm(pred)
    return norm(gold) in norm(pred)


def summarize(items):
    case_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    source_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for it in items:
        src = norm(it.get("source", "unknown"))
        for case in ["no_context", "base_context", "plus3_context"]:
            ok = bool(it["results"][case]["correct"])
            case_stats[case]["total"] += 1
            case_stats[case]["correct"] += int(ok)
            source_stats[case][src]["total"] += 1
            source_stats[case][src]["correct"] += int(ok)

    overall = {}
    by_source = {}
    for case, st in case_stats.items():
        overall[case] = {
            "correct": st["correct"],
            "total": st["total"],
            "accuracy": round(st["correct"] / max(st["total"], 1), 4),
        }
    for case, src_map in source_stats.items():
        by_source[case] = {}
        for src, st in src_map.items():
            by_source[case][src] = {
                "correct": st["correct"],
                "total": st["total"],
                "accuracy": round(st["correct"] / max(st["total"], 1), 4),
            }
    return {"overall": overall, "by_source": by_source}


def main():
    args = parse_args()
    rows = read_jsonl(Path(args.context_pack), args.limit)
    if not rows:
        raise RuntimeError(f"No rows loaded from {args.context_pack}")

    tokenizer, model = load_model_tok(args.answer_model)
    items = []

    for idx, row in enumerate(rows, start=1):
        question = row.get("question", "")
        gold = str(row.get("gold_answer", ""))
        src = row.get("source", "")

        contexts = {
            "no_context": row.get("no_context", ""),
            "base_context": context_to_text(row.get("base_context")),
            "plus3_context": context_to_text(row.get("plus3_context")),
        }

        results = {}
        for case, context_text in contexts.items():
            pred = generate_text(tokenizer, model, answer_prompt(question, context_text), args.max_new_tokens)
            ok = judge(src, question, pred, gold)
            results[case] = {
                "prediction": pred,
                "correct": ok,
            }

        items.append(
            {
                "id": row.get("id", ""),
                "source": src,
                "question": question,
                "gold_answer": gold,
                "results": results,
            }
        )

        if idx % 100 == 0:
            print(f"[Eval] {idx}/{len(rows)}")

    summary = summarize(items)
    out = {
        "config": {
            "context_pack": args.context_pack,
            "limit": len(rows),
            "answer_model": args.answer_model,
            "max_new_tokens": args.max_new_tokens,
        },
        "summary": summary,
        "items": items,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
