import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

question = "how many minutes are in 3.5 hours"
gold_answer = "210"

context_instruction = (
    "Generate minimal sufficient context for a small language model to answer this question.\n"
    "Output JSON with: need_context, question_type, entities, constraints, subquestions, useful_facts, missing_info, answer_hint.\n"
    "Do NOT include the answer directly."
)
context_system = "You are a context generator for small language models. Generate minimal sufficient context."

base_context_model = "Qwen/Qwen2.5-7B-Instruct"
plus3_context_model = "experiment_results/cli_compare_incremental/cd_refine_7b_cli_epochup_gpu2_ctn_plus3/final"
answer_model = "Qwen/Qwen2.5-1.5B-Instruct"

out_path = Path("experiment_results/cli_compare_incremental/single_question_1b_context_ablation.json")
out_path.parent.mkdir(parents=True, exist_ok=True)


def make_ctx_prompt(q: str) -> str:
    return (
        f"<|im_start|>system\n{context_system}\n<|im_end|>\n"
        f"<|im_start|>user\n{context_instruction}\n\nQuestion: {q}\n<|im_end|>\n"
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


def is_correct_numeric(pred: str, gold: str) -> bool:
    m = re.search(r"-?\d+(?:\.\d+)?", pred.replace(",", ""))
    if not m:
        return False
    return m.group(0) == gold


base_tok, base_model = load_model_tok(base_context_model)
plus_tok, plus_model = load_model_tok(plus3_context_model)

base_ctx_raw = generate_text(base_tok, base_model, make_ctx_prompt(question), max_new_tokens=220)
plus_ctx_raw = generate_text(plus_tok, plus_model, make_ctx_prompt(question), max_new_tokens=220)
base_ctx_obj = extract_json_candidate(base_ctx_raw)
plus_ctx_obj = extract_json_candidate(plus_ctx_raw)

del base_model, plus_model
torch.cuda.empty_cache()

ans_tok, ans_model = load_model_tok(answer_model)
cases = {
    "no_context": None,
    "base_context": context_to_text(base_ctx_obj),
    "plus3_context": context_to_text(plus_ctx_obj),
}
results = {}
for name, ctx in cases.items():
    pred = generate_text(ans_tok, ans_model, answer_prompt(question, ctx), max_new_tokens=64)
    results[name] = {
        "prediction": pred,
        "correct": is_correct_numeric(pred, gold_answer),
    }

summary = {
    "question": question,
    "gold_answer": gold_answer,
    "answer_model": answer_model,
    "context_models": {
        "base": base_context_model,
        "plus3": plus3_context_model,
    },
    "generated_contexts": {
        "base": base_ctx_obj,
        "plus3": plus_ctx_obj,
    },
    "results": results,
}
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"[Saved] {out_path}")
