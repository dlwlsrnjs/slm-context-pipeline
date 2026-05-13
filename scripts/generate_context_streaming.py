import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


CONTEXT_INSTRUCTION = (
    "Generate minimal sufficient context for a small language model to answer this question.\n"
    "Output JSON with: need_context, question_type, entities, constraints, subquestions, useful_facts, missing_info, answer_hint.\n"
    "Do NOT include the answer directly."
)
CONTEXT_SYSTEM = "You are a context generator for small language models. Generate minimal sufficient context."


def parse_args():
    parser = argparse.ArgumentParser(description="Streaming context generation with batched inference and incremental writes")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--questions-file", default="slm_context_pipeline/data/questions_real_max.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--save-every", type=int, default=50)
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


def strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def first_balanced_json_object(text: str):
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    tail = text[start:]
    missing = tail.count("{") - tail.count("}")
    if missing > 0:
        return tail + ("}" * missing)
    return tail


def normalize_context_obj(obj: dict):
    if not isinstance(obj, dict):
        return None
    if "subquestions" not in obj or not isinstance(obj.get("subquestions"), list):
        obj["subquestions"] = []
    if "entities" in obj and not isinstance(obj.get("entities"), list):
        obj["entities"] = [obj["entities"]] if obj.get("entities") not in (None, "") else []
    if "constraints" in obj and not isinstance(obj.get("constraints"), (list, dict)):
        obj["constraints"] = [obj["constraints"]] if obj.get("constraints") not in (None, "") else []
    if "useful_facts" in obj and not isinstance(obj.get("useful_facts"), list):
        obj["useful_facts"] = [obj["useful_facts"]] if obj.get("useful_facts") not in (None, "") else []
    if "missing_info" in obj and not isinstance(obj.get("missing_info"), list):
        obj["missing_info"] = [obj["missing_info"]] if obj.get("missing_info") not in (None, "") else []
    return obj


def extract_json_candidate(text: str):
    text = strip_code_fence(text)

    try:
        obj = json.loads(text)
        return normalize_context_obj(obj)
    except Exception:
        pass

    candidate = first_balanced_json_object(text)
    if candidate:
        try:
            obj = json.loads(candidate)
            return normalize_context_obj(obj)
        except Exception:
            pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return normalize_context_obj(obj)
    except Exception:
        return None


def make_prompt(question: str):
    return (
        f"<|im_start|>system\n{CONTEXT_SYSTEM}\n<|im_end|>\n"
        f"<|im_start|>user\n{CONTEXT_INSTRUCTION}\n\nQuestion: {question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


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


def generate_batch(tokenizer, model, prompts, max_new_tokens: int):
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
        for i in range(outputs.shape[0]):
            new_tokens = outputs[i][int(prompt_len) :]
            decoded.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return decoded


def main():
    args = parse_args()
    questions = read_questions(Path(args.questions_file), args.limit)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for _ in f:
                processed += 1

    if processed >= len(questions):
        print(f"[DoneAlready] {processed}/{len(questions)} rows already exist in {out_path}")
        return

    print(f"[Info] resume_from={processed}, total={len(questions)}")
    tokenizer, model = load_model_tokenizer(args.model_name)

    to_process = questions[processed:]
    mode = "a" if processed > 0 else "w"
    with out_path.open(mode, encoding="utf-8") as out_file:
        done = processed
        for start in range(0, len(to_process), args.batch_size):
            batch_q = to_process[start : start + args.batch_size]
            prompts = [make_prompt(item["question"]) for item in batch_q]
            raws = generate_batch(tokenizer, model, prompts, args.max_new_tokens)

            for q_item, raw in zip(batch_q, raws):
                row = {
                    "id": q_item["id"],
                    "source": q_item["source"],
                    "question": q_item["question"],
                    "gold_answer": q_item["gold_answer"],
                    "no_context": "",
                    "context": extract_json_candidate(raw),
                }
                out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                done += 1

            if done % args.save_every == 0 or done == len(questions):
                out_file.flush()
                os.fsync(out_file.fileno())
            if done % 100 == 0 or done == len(questions):
                print(f"[Progress] {done}/{len(questions)}")

    del model
    torch.cuda.empty_cache()
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
