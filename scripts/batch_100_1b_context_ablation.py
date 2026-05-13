import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


QUESTION_FILE = Path("slm_context_pipeline/data/questions_real_max.jsonl")
NUM_QUESTIONS = 100

CONTEXT_INSTRUCTION = (
    "Generate minimal sufficient context for a small language model to answer this question.\n"
    "Output JSON with: need_context, question_type, entities, constraints, subquestions, useful_facts, missing_info, answer_hint.\n"
    "Do NOT include the answer directly."
)
CONTEXT_SYSTEM = "You are a context generator for small language models. Generate minimal sufficient context."

BASE_CONTEXT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PLUS3_CONTEXT_MODEL = "experiment_results/cli_compare_incremental/cd_refine_7b_cli_epochup_gpu2_ctn_plus3/final"
ANSWER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

OUT_PATH = Path("experiment_results/cli_compare_incremental/context_ablation_100q_1b.json")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_first_number(text: str):
    m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return m.group(0) if m else None


def canonical_bool(text: str):
    t = normalize_text(text)
    if re.search(r"\b(yes|true)\b", t):
        return "yes"
    if re.search(r"\b(no|false)\b", t):
        return "no"
    return None


def is_correct(pred: str, gold: str, source: str) -> bool:
    source = (source or "").strip().lower()
    pred_n = normalize_text(pred)
    gold_n = normalize_text(gold)

    if source == "gsm8k":
        pred_num = extract_first_number(pred)
        gold_num = extract_first_number(gold)
        return pred_num is not None and gold_num is not None and pred_num == gold_num

    if source == "boolq":
        pred_b = canonical_bool(pred)
        gold_b = canonical_bool(gold)
        return pred_b is not None and gold_b is not None and pred_b == gold_b

    if source == "piqa":
        return gold_n in pred_n or pred_n in gold_n

    return gold_n in pred_n or pred_n in gold_n


def make_ctx_prompt(q: str) -> str:
    return (
        f"<|im_start|>system\n{CONTEXT_SYSTEM}\n<|im_end|>\n"
        f"<|im_start|>user\n{CONTEXT_INSTRUCTION}\n\nQuestion: {q}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def extract_json_candidate(text: str):
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            if "subquestions" not in obj or not isinstance(obj.get("subquestions"), list):
                obj["subquestions"] = []
            return obj
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            if "subquestions" not in obj or not isinstance(obj.get("subquestions"), list):
                obj["subquestions"] = []
            return obj
    except Exception:
        return None
    return None


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


def generate_text(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
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


def answer_prompt(q: str, ctx_text: str | None) -> str:
    system = "You are a concise QA assistant. Give only the final answer."
    if ctx_text:
        user = f"Question: {q}\n\nContext:\n{ctx_text}\n\nReturn only the final answer."
    else:
        user = f"Question: {q}\n\nReturn only the final answer."
    return (
        f"<|im_start|>system\n{system}\n<|im_end|>\n"
        f"<|im_start|>user\n{user}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_questions(path: Path, k: int):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question" not in obj or "answer" not in obj:
                continue
            rows.append(
                {
                    "id": obj.get("id", ""),
                    "source": obj.get("source", ""),
                    "question": obj["question"],
                    "gold": str(obj["answer"]),
                }
            )
            if len(rows) >= k:
                break
    return rows


def summarize(items):
    total = len(items)
    by_case = defaultdict(lambda: {"correct": 0, "total": 0})
    by_case_source = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for item in items:
        source = (item.get("source") or "unknown").lower()
        for case in ["no_context", "base_context", "plus3_context"]:
            ok = bool(item["results"][case]["correct"])
            by_case[case]["total"] += 1
            by_case[case]["correct"] += int(ok)
            by_case_source[case][source]["total"] += 1
            by_case_source[case][source]["correct"] += int(ok)

    out = {"total_questions": total, "overall": {}, "by_source": {}}
    for case, stat in by_case.items():
        acc = stat["correct"] / max(stat["total"], 1)
        out["overall"][case] = {
            "correct": stat["correct"],
            "total": stat["total"],
            "accuracy": round(acc, 4),
        }

    for case, source_stats in by_case_source.items():
        out["by_source"][case] = {}
        for source, stat in source_stats.items():
            acc = stat["correct"] / max(stat["total"], 1)
            out["by_source"][case][source] = {
                "correct": stat["correct"],
                "total": stat["total"],
                "accuracy": round(acc, 4),
            }
    return out


def main():
    data = load_questions(QUESTION_FILE, NUM_QUESTIONS)
    if len(data) < NUM_QUESTIONS:
        raise RuntimeError(f"Not enough questions in {QUESTION_FILE}: got {len(data)}")

    print(f"[Info] Loaded {len(data)} questions from {QUESTION_FILE}")

    print(f"[Info] Loading base context model: {BASE_CONTEXT_MODEL}")
    base_tok, base_model = load_model_tok(BASE_CONTEXT_MODEL)
    for i, row in enumerate(data, start=1):
        raw = generate_text(base_tok, base_model, make_ctx_prompt(row["question"]), max_new_tokens=220)
        row["base_ctx"] = extract_json_candidate(raw)
        if i % 10 == 0:
            print(f"[BaseCtx] {i}/{len(data)}")
    del base_model
    torch.cuda.empty_cache()

    print(f"[Info] Loading plus3 context model: {PLUS3_CONTEXT_MODEL}")
    plus_tok, plus_model = load_model_tok(PLUS3_CONTEXT_MODEL)
    for i, row in enumerate(data, start=1):
        raw = generate_text(plus_tok, plus_model, make_ctx_prompt(row["question"]), max_new_tokens=220)
        row["plus3_ctx"] = extract_json_candidate(raw)
        if i % 10 == 0:
            print(f"[Plus3Ctx] {i}/{len(data)}")
    del plus_model
    torch.cuda.empty_cache()

    print(f"[Info] Loading answer model: {ANSWER_MODEL}")
    ans_tok, ans_model = load_model_tok(ANSWER_MODEL)

    items = []
    for i, row in enumerate(data, start=1):
        cases = {
            "no_context": None,
            "base_context": context_to_text(row.get("base_ctx")),
            "plus3_context": context_to_text(row.get("plus3_ctx")),
        }
        results = {}
        for name, ctx in cases.items():
            pred = generate_text(ans_tok, ans_model, answer_prompt(row["question"], ctx), max_new_tokens=64)
            ok = is_correct(pred, row["gold"], row["source"])
            results[name] = {
                "prediction": pred,
                "correct": ok,
            }

        items.append(
            {
                "id": row["id"],
                "source": row["source"],
                "question": row["question"],
                "gold_answer": row["gold"],
                "generated_contexts": {
                    "base": row.get("base_ctx"),
                    "plus3": row.get("plus3_ctx"),
                },
                "results": results,
            }
        )
        if i % 10 == 0:
            print(f"[AnswerEval] {i}/{len(data)}")

    summary = summarize(items)
    final = {
        "config": {
            "num_questions": NUM_QUESTIONS,
            "question_file": str(QUESTION_FILE),
            "answer_model": ANSWER_MODEL,
            "context_models": {
                "base": BASE_CONTEXT_MODEL,
                "plus3": PLUS3_CONTEXT_MODEL,
            },
        },
        "summary": summary,
        "items": items,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[Saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
