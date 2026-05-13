import argparse
import json
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
    parser = argparse.ArgumentParser(description="Generate base/plus3 contexts with identical settings.")
    parser.add_argument("--questions-file", default="slm_context_pipeline/data/questions_real_max.jsonl")
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--plus3-model",
        default="experiment_results/cli_compare_incremental/cd_refine_7b_cli_epochup_gpu2_ctn_plus3/final",
    )
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--output",
        default="experiment_results/cli_compare_incremental/context_pack_base_plus3_6000.jsonl",
    )
    return parser.parse_args()


def read_jsonl(path: Path, limit: int):
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


def make_ctx_prompt(question: str):
    return (
        f"<|im_start|>system\n{CONTEXT_SYSTEM}\n<|im_end|>\n"
        f"<|im_start|>user\n{CONTEXT_INSTRUCTION}\n\nQuestion: {question}\n<|im_end|>\n"
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


def generate_batch_texts(tokenizer, model, prompts, max_new_tokens: int):
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        decoded = []
        for idx, in_len in enumerate(input_lengths):
            new_tokens = outputs[idx][int(in_len) :]
            decoded.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return decoded


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


def save_rows_jsonl(rows, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_for_model(
    rows,
    model_name: str,
    field_name: str,
    max_new_tokens: int,
    batch_size: int,
    save_every: int,
    out_path: Path,
):
    print(f"[Info] Loading model for {field_name}: {model_name}")
    tokenizer, model = load_model_tok(model_name)

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prompts = [make_ctx_prompt(row["question"]) for row in batch_rows]
        raws = generate_batch_texts(tokenizer, model, prompts, max_new_tokens=max_new_tokens)
        for row, raw in zip(batch_rows, raws):
            row[field_name] = extract_json_candidate(raw)

        done = min(start + len(batch_rows), len(rows))
        if done % 100 == 0 or done == len(rows):
            print(f"[{field_name}] {done}/{len(rows)}")
        if done % save_every == 0 or done == len(rows):
            save_rows_jsonl(rows, out_path)

    del model
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    questions = read_jsonl(Path(args.questions_file), args.limit)
    if not questions:
        raise RuntimeError(f"No questions loaded from {args.questions_file}")

    for row in questions:
        row["no_context"] = ""
        row["base_context"] = None
        row["plus3_context"] = None

    out_path = Path(args.output)
    generate_for_model(
        questions,
        args.base_model,
        "base_context",
        args.max_new_tokens,
        args.batch_size,
        args.save_every,
        out_path,
    )
    generate_for_model(
        questions,
        args.plus3_model,
        "plus3_context",
        args.max_new_tokens,
        args.batch_size,
        args.save_every,
        out_path,
    )

    save_rows_jsonl(questions, out_path)

    print(f"[Saved] {out_path} ({len(questions)} rows)")


if __name__ == "__main__":
    main()
