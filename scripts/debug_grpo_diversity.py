#!/usr/bin/env python3
"""Diagnose why GRPO G-sample completions are identical."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

POLICY_PATH = "experiment_results/math_icl_sft/qwen1b_run2/final"
PROMPT = """You are generating in-context demonstrations for math word problems (gsm8k_5k).
Generate exactly 4 demonstrations with this format:
Q: <math word problem>
A: <step-by-step reasoning ending with 'The answer is [number].'>

Rules:
- Each demo must be self-contained.
- Show clear arithmetic steps.
- End every answer with 'The answer is [number].'

Candidate examples:
Q: John has 5 apples. He eats 2. How many are left?
A: John starts with 5 apples and eats 2, so he has 5 - 2 = 3 apples left.
The answer is 3.

Now output demonstrations only:
"""

tok = AutoTokenizer.from_pretrained(POLICY_PATH)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    POLICY_PATH, torch_dtype=torch.bfloat16, device_map="auto",
)
model.eval()

inputs = tok(PROMPT, return_tensors="pt").to(model.device)

print("=" * 60)
print("TEST 1: num_return_sequences=4, do_sample=True, temp=0.9")
print("=" * 60)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        num_return_sequences=4,
        pad_token_id=tok.pad_token_id,
    )
new_tokens = out[:, inputs["input_ids"].shape[1]:]
texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
for i, t in enumerate(texts):
    print(f"--- seq {i} ---")
    print(t[:200])
print(f"\nALL IDENTICAL: {len(set(texts)) == 1}")

print("\n" + "=" * 60)
print("TEST 2: generation_config.do_sample=False (should be greedy)")
print("=" * 60)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        num_return_sequences=1,
        pad_token_id=tok.pad_token_id,
    )
new_tokens = out[:, inputs["input_ids"].shape[1]:]
print(tok.decode(new_tokens[0], skip_special_tokens=True)[:200])

print("\n" + "=" * 60)
print("Model generation_config:")
print("=" * 60)
print(model.generation_config)
