import json
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "experiment_results/cli_compare_incremental/cd_refine_7b_cli_epochup_gpu2_ctn_plus3/final"
out_path = Path("experiment_results/cli_compare_incremental/manual_probe_10_plus3.jsonl")
summary_path = Path("experiment_results/cli_compare_incremental/manual_probe_10_plus3_summary.json")
out_path.parent.mkdir(parents=True, exist_ok=True)

instruction = (
    "Generate minimal sufficient context for a small language model to answer this question.\n"
    "Output JSON with: need_context, question_type, entities, constraints, subquestions, useful_facts, missing_info, answer_hint.\n"
    "Do NOT include the answer directly."
)
system_prompt = "You are a context generator for small language models. Generate minimal sufficient context."

questions = [
    "is sodium chloride an ionic compound",
    "can you bring water through airport security",
    "Goal: clean a whiteboard\nOption A: use a dry tissue\nOption B: use a whiteboard eraser\nWhich option better achieves the goal?",
    "what is the capital city of australia",
    "does lightning always come with rain",
    "how many minutes are in 3.5 hours",
    "is it legal to turn right on red in california",
    "Goal: remove pencil marks from paper\nOption A: rub with eraser\nOption B: blow air on it\nWhich option better achieves the goal?",
    "do penguins live in the arctic",
    "if a shirt is 20% off from 50 dollars what is final price",
]

required_keys = [
    "need_context", "question_type", "entities", "constraints",
    "subquestions", "useful_facts", "missing_info", "answer_hint"
]

def build_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\n{instruction}\n\nQuestion: {question}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def extract_json(text: str):
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def ensure_schema_keys(obj):
    if not isinstance(obj, dict):
        return obj
    if "subquestions" not in obj or not isinstance(obj.get("subquestions"), list):
        obj["subquestions"] = []
    return obj

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()

rows = []
valid = 0
full_schema = 0

with torch.no_grad():
    for i, q in enumerate(questions, start=1):
        prompt = build_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed = ensure_schema_keys(extract_json(generated))
        is_valid = isinstance(parsed, dict)
        if is_valid:
            valid += 1
            present = sum(1 for k in required_keys if k in parsed)
            if present == len(required_keys):
                full_schema += 1
        else:
            present = 0
        rows.append({
            "id": i,
            "question": q,
            "parsed_json": parsed,
            "raw_output": generated,
            "is_json_valid": is_valid,
            "present_required_keys": present,
        })

with out_path.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

summary = {
    "model_path": model_path,
    "num_samples": len(rows),
    "json_valid_rate": round(valid / len(rows), 4),
    "full_schema_rate": round(full_schema / len(rows), 4),
    "output_file": str(out_path),
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
