"""Quick test for the LLM judge function.

Tests the JUDGE_PROMPT and llm_judge_batched on hand-crafted demo cases
to verify scoring works correctly before full training.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

# Load v4 script as module to access judge functions
spec = importlib.util.spec_from_file_location(
    "v4", os.path.join(os.path.dirname(__file__), "rl_train_icl_grpo_ra_v4.py"))
v4 = importlib.util.module_from_spec(spec)
# Run only top-level definitions (no main)
spec.loader.exec_module(v4)


# Test cases — varying demo quality
TEST_CASES = {
    "perfect_4": """Q: A baker made 100 cookies. He sold 60. How many left?
A: 100 - 60 = 40. The answer is 40.

Q: A car travels 60 mph for 3 hours. Distance?
A: 60 * 3 = 180. The answer is 180.

Q: Sarah has 18 marbles, gives 6.
A: 18 - 6 = 12. The answer is 12.

Q: Tom bought 4 books at $7 each.
A: 4 * 7 = 28. The answer is 28.""",

    "wrong_arithmetic": """Q: A baker made 100 cookies. He sold 60. How many left?
A: 100 - 60 = 30. The answer is 30.

Q: A car travels 60 mph for 3 hours. Distance?
A: 60 * 3 = 200. The answer is 200.

Q: Sarah has 18 marbles, gives 6.
A: 18 - 6 = 9. The answer is 9.

Q: Tom bought 4 books at $7 each.
A: 4 * 7 = 32. The answer is 32.""",

    "mixed_2_correct_2_wrong": """Q: A baker made 100 cookies. He sold 60. How many left?
A: 100 - 60 = 40. The answer is 40.

Q: A car travels 60 mph for 3 hours. Distance?
A: 60 * 3 = 200. The answer is 200.

Q: Sarah has 18 marbles, gives 6.
A: 18 - 6 = 12. The answer is 12.

Q: Tom bought 4 books at $7 each.
A: 4 * 7 = 30. The answer is 30.""",

    "answer_mismatch": """Q: A baker made 100 cookies. He sold 60. How many left?
A: 100 - 60 = 40. The answer is 50.

Q: A car travels 60 mph for 3 hours.
A: 60 * 3 = 180. The answer is 180.

Q: Sarah has 18 marbles, gives 6.
A: 18 - 6 = 12. The answer is 100.

Q: Tom bought 4 books at $7 each.
A: 4 * 7 = 28. The answer is 28.""",

    "only_one_demo": """Q: A baker made 100 cookies. He sold 60. How many left?
A: 100 - 60 = 40. The answer is 40.""",

    "empty": "",
}

print("=" * 70)
print("Loading LLM judge (Qwen2.5-7B-Instruct)…")
print("=" * 70)
v4.init_llm_judge("Qwen/Qwen2.5-7B-Instruct")

print("\n" + "=" * 70)
print("Running judge on test cases (batched)")
print("=" * 70)

names = list(TEST_CASES.keys())
demos_list = [TEST_CASES[n] for n in names]
scores = v4.llm_judge_batched(demos_list, max_new_tokens=200)

print("\n=== Results ===")
print(f"{'case':30s} | {'score':>6s} | expected")
print("-" * 70)
expected = {
    "perfect_4": "1.0 (4/4 correct)",
    "wrong_arithmetic": "0.0 (4/4 wrong)",
    "mixed_2_correct_2_wrong": "0.5 (2/4 correct)",
    "answer_mismatch": "0.5 (2/4 mismatched)",
    "only_one_demo": "varies (only 1 demo)",
    "empty": "0.0 (no demos)",
}
for n, s in zip(names, scores):
    print(f"{n:30s} | {s:6.2f} | {expected[n]}")
