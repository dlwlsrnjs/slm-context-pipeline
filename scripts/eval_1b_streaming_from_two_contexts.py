import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Streaming eval: no-context vs base-context vs plus3-context")
    parser.add_argument("--questions-file", default="slm_context_pipeline/data/questions_real_max.jsonl")
    parser.add_argument("--base-context-file", required=True)
    parser.add_argument("--plus3-context-file", required=True)
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--answer-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--save-every", type=int, default=20)
    return parser.parse_args()


def read_questions(path: Path, limit: int):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "id": obj.get("id", ""),
                    "source": obj.get("source", ""),
                    "question": obj.get("question", ""),
                    "gold_answer": str(obj.get("answer", "")),
                }
            )
            if len(rows) >= limit:
                break
    return rows


def read_jsonl_rows(path: Path, limit: int):
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


def context_to_text(ctx_obj):
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


def answer_prompt(source: str, question: str, context_text: str):
    system = "You are a concise QA assistant. Follow output format exactly."
    source_norm = norm(source)

    if source_norm == "boolq":
        format_rule = (
            "Output exactly one token: yes or no. "
            "Do not output any explanation, punctuation, or extra words."
        )
    elif source_norm == "piqa":
        format_rule = (
            "Output exactly one label: Option A or Option B. "
            "Do not output explanations or any additional text."
        )
    elif source_norm == "gsm8k":
        format_rule = (
            "Output only the final numeric answer (number only). "
            "Do not output equations, units, or explanations."
        )
    else:
        format_rule = "Return only the final answer with no explanation."

    if context_text.strip():
        user = (
            f"Question: {question}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Format rule: {format_rule}"
        )
    else:
        user = f"Question: {question}\n\nFormat rule: {format_rule}"
    return (
        f"<|im_start|>system\n{system}\n<|im_end|>\n"
        f"<|im_start|>user\n{user}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def norm(text: str):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def extract_num_first(text: str):
    values = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return values[0] if values else None


def extract_num_last(text: str):
    values = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return values[-1] if values else None


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
    return (qa.group(1).strip() if qa else ""), (qb.group(1).strip() if qb else "")


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
        pb = extract_yes_no(pred)
        gb = extract_yes_no(gold)
        return pb is not None and gb is not None and pb == gb
    if s == "gsm8k":
        g = extract_num_first(gold)
        return g is not None and (extract_num_first(pred) == g or extract_num_last(pred) == g)
    if s == "piqa":
        gl = infer_gold_label(gold, question)
        pl = detect_pred_label(pred)
        if gl and pl:
            return gl == pl
        return norm(gold) in norm(pred)
    return norm(gold) in norm(pred)


def load_model_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to("cuda").eval()
    return tokenizer, model


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


def generate_texts_batch(tokenizer, model, prompts, max_new_tokens: int):
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        decoded = []
        for idx in range(outputs.shape[0]):
            new_tokens = outputs[idx][prompt_len:]
            decoded.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return decoded


def main():
    args = parse_args()
    questions = read_questions(Path(args.questions_file), args.limit)
    base_rows = read_jsonl_rows(Path(args.base_context_file), args.limit)
    plus_rows = read_jsonl_rows(Path(args.plus3_context_file), args.limit)

    if len(base_rows) < len(questions) or len(plus_rows) < len(questions):
        raise RuntimeError(
            f"Context files are shorter than questions: questions={len(questions)}, base={len(base_rows)}, plus3={len(plus_rows)}"
        )

    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    if out_jsonl.exists():
        with out_jsonl.open("r", encoding="utf-8") as f:
            for _ in f:
                processed += 1

    tokenizer, model = load_model_tokenizer(args.answer_model)

    stats = {
        "no_context": {"correct": 0, "total": 0},
        "base_context": {"correct": 0, "total": 0},
        "plus3_context": {"correct": 0, "total": 0},
    }

    if processed > 0:
        with out_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                for key in stats.keys():
                    stats[key]["total"] += 1
                    stats[key]["correct"] += int(bool(obj["results"][key]["correct"]))

    mode = "a" if processed > 0 else "w"
    with out_jsonl.open(mode, encoding="utf-8") as out_file:
        for start in range(processed, len(questions), args.batch_size):
            end = min(start + args.batch_size, len(questions))
            q_batch = questions[start:end]
            b_batch = base_rows[start:end]
            p_batch = plus_rows[start:end]

            prompts_no = [answer_prompt(q["source"], q["question"], "") for q in q_batch]
            prompts_base = [
                answer_prompt(q["source"], q["question"], context_to_text(b.get("context")))
                for q, b in zip(q_batch, b_batch)
            ]
            prompts_plus3 = [
                answer_prompt(q["source"], q["question"], context_to_text(p.get("context")))
                for q, p in zip(q_batch, p_batch)
            ]

            preds_no = generate_texts_batch(tokenizer, model, prompts_no, args.max_new_tokens)
            preds_base = generate_texts_batch(tokenizer, model, prompts_base, args.max_new_tokens)
            preds_plus3 = generate_texts_batch(tokenizer, model, prompts_plus3, args.max_new_tokens)

            for i, q in enumerate(q_batch):
                pred_no = preds_no[i]
                pred_base = preds_base[i]
                pred_plus3 = preds_plus3[i]

                ok_no = judge(q["source"], q["question"], pred_no, q["gold_answer"])
                ok_base = judge(q["source"], q["question"], pred_base, q["gold_answer"])
                ok_plus3 = judge(q["source"], q["question"], pred_plus3, q["gold_answer"])

                stats["no_context"]["total"] += 1
                stats["no_context"]["correct"] += int(ok_no)
                stats["base_context"]["total"] += 1
                stats["base_context"]["correct"] += int(ok_base)
                stats["plus3_context"]["total"] += 1
                stats["plus3_context"]["correct"] += int(ok_plus3)

                row = {
                    "id": q["id"],
                    "source": q["source"],
                    "question": q["question"],
                    "gold_answer": q["gold_answer"],
                    "results": {
                        "no_context": {"prediction": pred_no, "correct": ok_no},
                        "base_context": {"prediction": pred_base, "correct": ok_base},
                        "plus3_context": {"prediction": pred_plus3, "correct": ok_plus3},
                    },
                }
                out_file.write(json.dumps(row, ensure_ascii=False) + "\n")

            done = end
            if done % args.save_every == 0 or done == len(questions):
                out_file.flush()
                os.fsync(out_file.fileno())
            if done % 50 == 0 or done == len(questions):
                print(f"[EvalProgress] {done}/{len(questions)}")

    overall = {}
    for key, st in stats.items():
        overall[key] = {
            "correct": st["correct"],
            "total": st["total"],
            "accuracy": round(st["correct"] / max(st["total"], 1), 4),
        }

    summary = {
        "config": {
            "questions_file": args.questions_file,
            "base_context_file": args.base_context_file,
            "plus3_context_file": args.plus3_context_file,
            "answer_model": args.answer_model,
            "limit": len(questions),
            "max_new_tokens": args.max_new_tokens,
        },
        "overall": overall,
    }

    Path(args.output_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"[Saved] {args.output_jsonl}")
    print(f"[Saved] {args.output_summary}")


if __name__ == "__main__":
    main()
