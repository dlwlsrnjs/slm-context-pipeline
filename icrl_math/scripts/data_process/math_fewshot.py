"""Preprocess math datasets (MATH / GSM8K) to parquet format with few-shot examples for code-tool learning.

Mirrors ICRL's nq_search_fewshot.py pattern, but:
  - data_source='math' (routes to math_fewshot reward in our patched main_ppo_fewshot)
  - tool is a Python interpreter (<code>...</code> -> <output>...</output>) instead of web search
  - few-shot examples come from icrl_math/example/math_examples.txt by default
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Optional, Union

import datasets


# example/ next to this file's parent's parent (icrl_math/example/)
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "example"
DEFAULT_EXAMPLES_NAME = "math_examples.txt"
DEFAULT_EXAMPLES_FILE = EXAMPLES_DIR / DEFAULT_EXAMPLES_NAME


SYSTEM_PROMPT_FEWSHOT = """Solve the following math problem step by step. You must conduct reasoning inside <think> and </think> every time before you decide what to do next.

Whenever you need to carry out arithmetic, algebra, calculus, combinatorics, or any other computation, you may invoke a sandboxed Python interpreter by writing <search> python source </search>. The interpreter runs the snippet in a fresh namespace (state is NOT preserved between calls, so always re-import / re-declare variables) and returns its standard output back to you wrapped in <information> stdout </information>. You can invoke the interpreter as many times as you want. Common libraries available: math, sympy, fractions, itertools, collections, numpy.

Each reasoning step should be wrapped with <think> your thought here </think>.

When you call the Python interpreter, the code should be placed inside <search> python source here </search>.

The interpreter's output will be wrapped with <information> stdout here </information>.

The last part of your response must be in the following format: <answer> \\boxed{final answer} </answer>. Do NOT include any explanation inside the <answer> tag — only the boxed final answer.

Here are some existing math examples that you can refer to:
"""

SYSTEM_PROMPT_ZEROSHOT = """Solve the following math problem step by step. You must conduct reasoning inside <think> and </think> every time before you decide what to do next.

Whenever you need to carry out arithmetic, algebra, calculus, combinatorics, or any other computation, you may invoke a sandboxed Python interpreter by writing <search> python source </search>. The interpreter runs the snippet in a fresh namespace (state is NOT preserved between calls, so always re-import / re-declare variables) and returns its standard output back to you wrapped in <information> stdout </information>. You can invoke the interpreter as many times as you want. Common libraries available: math, sympy, fractions, itertools, collections, numpy.

Each reasoning step should be wrapped with <think> your thought here </think>.

When you call the Python interpreter, the code should be placed inside <search> python source here </search>.

The interpreter's output will be wrapped with <information> stdout here </information>.

The last part of your response must be in the following format: <answer> \\boxed{final answer} </answer>. Do NOT include any explanation inside the <answer> tag — only the boxed final answer.
"""

PROBLEM_TEXT = "Now solve the following problem with the ability you just learned from the examples. Do NOT reuse any problem from the examples."


def list_available_example_files() -> List[str]:
    if not EXAMPLES_DIR.exists():
        return []
    return sorted(p.name for p in EXAMPLES_DIR.glob("*.txt") if p.is_file())


def resolve_examples_file(examples_name: Optional[str] = None, examples_file: Optional[str] = None) -> Path:
    if examples_file is not None:
        return Path(examples_file)
    available = list_available_example_files()
    selected = examples_name or DEFAULT_EXAMPLES_NAME
    if available and selected not in available:
        raise ValueError(f"Unknown examples file '{selected}'. Available in {EXAMPLES_DIR}: {', '.join(available)}")
    return EXAMPLES_DIR / selected


def load_fewshot_examples(examples_file: Optional[Union[str, Path]] = None, num_examples: Optional[int] = None) -> str:
    """Load N math few-shot examples from file. Same parsing rule as ICRL's nq_search_fewshot."""
    if examples_file is None:
        examples_file = DEFAULT_EXAMPLES_FILE
    else:
        examples_file = Path(examples_file)
    if not examples_file.exists():
        print(f"Warning: Few-shot examples file not found at {examples_file}")
        return ""

    full_text = examples_file.read_text(encoding="utf-8").strip()
    if num_examples is None:
        return full_text

    example_pattern = r'(===============Example \d+===============.*?)(?================Example \d+===============|$)'
    examples = re.findall(example_pattern, full_text, re.DOTALL)
    if not examples:
        print("Warning: could not parse examples, returning full text")
        return full_text
    return "\n".join(ex.strip() for ex in examples[:num_examples])


def build_fewshot_prompt(problem: str, fewshot_examples: str) -> str:
    parts = [SYSTEM_PROMPT_FEWSHOT]
    if fewshot_examples:
        parts.append(fewshot_examples)
    parts.append(PROBLEM_TEXT)
    parts.append(f"Actual Problem: {problem}")
    return "\n\n".join(parts)


def build_zeroshot_prompt(problem: str) -> str:
    return "\n\n".join([SYSTEM_PROMPT_ZEROSHOT, f"Problem: {problem}"])


def extract_gsm8k_answer(answer_field: str) -> str:
    """GSM8K answers end with '#### <number>'."""
    m = re.search(r"####\s*(\-?[0-9\.,]+)", answer_field)
    if m is None:
        return answer_field.strip()
    return m.group(1).replace(",", "").strip()


def extract_math_answer(solution_field: str) -> str:
    """MATH solutions end with \\boxed{ans}."""
    idx = solution_field.rfind("\\boxed")
    if idx < 0:
        return solution_field.strip()
    if "\\boxed " in solution_field:
        return solution_field.split("\\boxed ")[-1].split("$")[0].strip()
    i = solution_field.find("{", idx)
    if i < 0:
        return solution_field[idx:].strip()
    depth = 0
    for j in range(i, len(solution_field)):
        if solution_field[j] == "{":
            depth += 1
        elif solution_field[j] == "}":
            depth -= 1
            if depth == 0:
                return solution_field[i + 1: j].strip()
    return solution_field[i + 1:].strip()


def load_math_dataset(name: str):
    """name: 'gsm8k' | 'math' | 'mixed'. Returns (train, test) HF datasets with normalized fields:
       - problem: str
       - golden_answer: str (just the final answer, no \\boxed)
    """
    if name == "gsm8k":
        d = datasets.load_dataset("openai/gsm8k", "main")
        def norm(ex):
            return {"problem": ex["question"].strip(), "golden_answer": extract_gsm8k_answer(ex["answer"])}
        return d["train"].map(norm, remove_columns=d["train"].column_names), \
               d["test"].map(norm, remove_columns=d["test"].column_names)

    if name == "math":
        # Hendrycks MATH; lighteval/MATH-Hard mirrors the original
        d = datasets.load_dataset("lighteval/MATH", "all")
        def norm(ex):
            return {"problem": ex["problem"].strip(), "golden_answer": extract_math_answer(ex["solution"])}
        return d["train"].map(norm, remove_columns=d["train"].column_names), \
               d["test"].map(norm, remove_columns=d["test"].column_names)

    if name == "mixed":
        gsm_tr, gsm_te = load_math_dataset("gsm8k")
        math_tr, math_te = load_math_dataset("math")
        train = datasets.concatenate_datasets([gsm_tr, math_tr]).shuffle(seed=42)
        test = datasets.concatenate_datasets([gsm_te, math_te]).shuffle(seed=42)
        return train, test

    raise ValueError(f"unknown dataset: {name}")


def main():
    available = list_available_example_files()
    available_help = ", ".join(available) if available else "no .txt found"

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/math_fewshot")
    parser.add_argument("--dataset", choices=["gsm8k", "math", "mixed"], default="math")
    parser.add_argument("--template_type", choices=["fewshot", "zeroshot"], default="fewshot")
    parser.add_argument("--examples_name", default=DEFAULT_EXAMPLES_NAME, help=f"available: {available_help}")
    parser.add_argument("--examples_file", default=None)
    parser.add_argument("--num_examples", type=int, default=None, help="None=all, or N (e.g. 3, 2, 0)")
    parser.add_argument("--train_data_num", type=int, default=None)
    parser.add_argument("--val_data_num", type=int, default=None)
    args = parser.parse_args()

    data_source = "math"

    fewshot_examples = ""
    examples_path = None
    if args.template_type == "fewshot":
        examples_path = resolve_examples_file(args.examples_name, args.examples_file)
        fewshot_examples = load_fewshot_examples(examples_path, args.num_examples)
        num_ex_str = f"{args.num_examples} example(s)" if args.num_examples else "all examples"
        print(f"Loaded {num_ex_str} from {examples_path} ({len(fewshot_examples)} chars)")

    print(f"Loading dataset: {args.dataset}")
    train_ds, test_ds = load_math_dataset(args.dataset)
    print(f"  train={len(train_ds)}, test={len(test_ds)}")

    if args.train_data_num:
        train_ds = train_ds.select(range(min(args.train_data_num, len(train_ds))))
    if args.val_data_num:
        test_ds = test_ds.select(range(min(args.val_data_num, len(test_ds))))

    def make_map_fn(split):
        def fn(example, idx):
            problem = example["problem"]
            if args.template_type == "fewshot":
                prompt_text = build_fewshot_prompt(problem, fewshot_examples)
            else:
                prompt_text = build_zeroshot_prompt(problem)
            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": "math-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {"target": example["golden_answer"]},
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "template_type": args.template_type,
                    "num_examples": args.num_examples if args.template_type == "fewshot" else 0,
                    "dataset": args.dataset,
                },
            }
        return fn

    train_ds = train_ds.map(make_map_fn("train"), with_indices=True)
    test_ds = test_ds.map(make_map_fn("test"), with_indices=True)

    os.makedirs(args.local_dir, exist_ok=True)
    train_path = os.path.join(args.local_dir, "train.parquet")
    test_path = os.path.join(args.local_dir, "test.parquet")
    train_ds.to_parquet(train_path)
    test_ds.to_parquet(test_path)
    print(f"saved {len(train_ds)} train -> {train_path}")
    print(f"saved {len(test_ds)} test  -> {test_path}")


if __name__ == "__main__":
    main()
