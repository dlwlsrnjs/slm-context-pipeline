"""Test the v5 holistic LLM judge (covers tags + structure + count + math)."""
import os, sys, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

spec = importlib.util.spec_from_file_location(
    "v5", os.path.join(os.path.dirname(__file__), "rl_train_icl_grpo_ra_v5.py"))
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)


# Test cases — different format/quality combinations
TEST_CASES = {
    "perfect_4_with_structure": """<think>
The seed examples are basic arithmetic. I'll generate similar.
</think>
<answer>
Q: A baker made 100 cookies. He sold 60. How many left?
A: 100 - 60 = 40. The answer is 40.

Q: A car travels 60 mph for 3 hours. Distance?
A: 60 * 3 = 180. The answer is 180.

Q: Sarah has 18 marbles, gives 6.
A: 18 - 6 = 12. The answer is 12.

Q: Tom bought 4 books at $7 each.
A: 4 * 7 = 28. The answer is 28.
</answer>""",  # All 4 criteria pass

    "missing_think": """<answer>
Q: A baker made 100 cookies. He sold 60.
A: 100 - 60 = 40. The answer is 40.

Q: A car travels 60 mph for 3 hours.
A: 60 * 3 = 180. The answer is 180.

Q: Sarah has 18 marbles, gives 6.
A: 18 - 6 = 12. The answer is 12.

Q: Tom bought 4 books at $7 each.
A: 4 * 7 = 28. The answer is 28.
</answer>""",  # Missing <think> — A:N

    "wrong_order": """<answer>
4 valid demos here...
</answer>
<think>thoughts</think>""",  # Reversed order — B:N

    "only_one_demo": """<think>thinking</think><answer>
Q: A baker made 100 cookies. He sold 60.
A: 100 - 60 = 40. The answer is 40.
</answer>""",  # Only 1 demo — C:N

    "wrong_math": """<think>thinking</think><answer>
Q: 100 - 60 = ?
A: 100 - 60 = 30. The answer is 30.

Q: 60 * 3 = ?
A: 60 * 3 = 200. The answer is 200.

Q: 18 - 6 = ?
A: 18 - 6 = 9. The answer is 9.

Q: 4 * 7 = ?
A: 4 * 7 = 32. The answer is 32.
</answer>""",  # All math wrong — D:N

    "empty": "",
}

print("Loading v5 LLM judge…")
v5.init_llm_judge("Qwen/Qwen2.5-7B-Instruct")

print("\nTesting holistic judge…")
names = list(TEST_CASES.keys())
completions = [TEST_CASES[n] for n in names]
results = v5.llm_judge_holistic_batched(completions, max_new_tokens=300)

print("\n=== Results ===")
print(f"{'case':30s} | {'A':3s} {'B':3s} {'C':3s} {'D':3s} | score | expected")
print("-" * 80)
expected = {
    "perfect_4_with_structure": "Y Y Y Y / 1.00",
    "missing_think": "N ? ? ? / ~0.00-0.50",
    "wrong_order": "Y/Y N ? ? / ~0.50",
    "only_one_demo": "Y/Y Y N N / ~0.75",
    "wrong_math": "Y Y Y N / ~0.75",
    "empty": "N N N N / 0.00",
}
for n, (s, v) in zip(names, results):
    a, b, c, d = v
    print(f"{n:30s} | {'Y' if a else 'N':3s} {'Y' if b else 'N':3s} "
          f"{'Y' if c else 'N':3s} {'Y' if d else 'N':3s} | {s:5.2f} | {expected[n]}")
