"""Convert existing GPT-4o-mini distillation data to PRL-style format.

Original target:
    Q: ... A: ...
    Q: ... A: ...
    ...

New PRL target:
    <think>
    Brief reasoning about pattern/difficulty matching candidates.
    </think>
    <answer>
    Q: ... A: ...
    Q: ... A: ...
    ...
    </answer>

Also rewrites prompt to request PRL format.
"""
import json
import random
from pathlib import Path
from datasets import load_from_disk, Dataset

REPO = Path(__file__).resolve().parent.parent
SRC_DIR = REPO / "icl_distill_math/hf_distilled_icl_dataset"
DST_DIR = REPO / "icl_distill_math/hf_distilled_icl_dataset_prl"

THINK_TEMPLATES = [
    "The candidate examples show {style}. I'll match the difficulty and style.",
    "Looking at the seed examples, they involve {style}. I'll create similar problems.",
    "The examples are {style} with clear arithmetic. I'll generate four matching demos.",
    "Seed candidates use {style}. I'll mirror this pattern across four new problems.",
    "Pattern check: {style} word problems. Producing four self-contained demonstrations.",
    "These look like {style} problems with simple operations. I'll keep the same structure.",
]

STYLE_HINTS = {
    "gsm8k_5k": ["grade-school arithmetic", "step-by-step word problems",
                 "real-life scenarios with simple math", "multi-step calculation problems"],
    "orcamath_5k": ["GSM-style word problems", "arithmetic reasoning",
                    "step-by-step word problems", "multi-step arithmetic"],
    "metamath_5k": ["GSM-derived word problems", "rephrased arithmetic problems",
                    "answer-augmented problems", "variable-substitution problems"],
}

PRL_PROMPT_HEADER = """You are generating in-context demonstrations for math word problems ({dataset}).

First, briefly think inside <think>...</think> about which patterns and difficulty are useful.
Then output exactly 4 self-contained demonstrations inside <answer>...</answer>.

Inside <answer>, each demonstration must follow:
Q: <math word problem>
A: <step-by-step reasoning ending with 'The answer is [number].'

Rules:
- Each demo must be self-contained (no external tables or formulas needed).
- Show clear arithmetic steps.
- End every answer with 'The answer is [number].'
- Do NOT add explanations outside the <think>/<answer> blocks.

Candidate examples:
"""


def make_prl_target(orig_target: str, dataset: str, seed: int) -> str:
    rng = random.Random(seed)
    style = rng.choice(STYLE_HINTS.get(dataset, ["arithmetic word problems"]))
    think = rng.choice(THINK_TEMPLATES).format(style=style)
    return f"<think>\n{think}\n</think>\n<answer>\n{orig_target.strip()}\n</answer>"


def make_prl_prompt(orig_prompt: str, dataset: str) -> str:
    # Replace the original instruction header but keep the candidate examples block.
    # Original prompt format (from build_math_icl_distill_dataset.py):
    #   "You are generating in-context demonstrations ...
    #    Generate exactly 4 ...
    #    ...
    #    Candidate examples:
    #    {seed_examples}
    #    Now output demonstrations only:"
    # We swap in PRL_PROMPT_HEADER but keep candidate examples.
    cand_idx = orig_prompt.find("Candidate examples:")
    if cand_idx < 0:
        # Fallback — just wrap original
        return PRL_PROMPT_HEADER.format(dataset=dataset) + "\n" + orig_prompt
    candidates = orig_prompt[cand_idx + len("Candidate examples:"):].strip()
    # Drop any trailing "Now output ..." line
    end_idx = candidates.rfind("Now output")
    if end_idx > 0:
        candidates = candidates[:end_idx].strip()
    new = PRL_PROMPT_HEADER.format(dataset=dataset) + candidates + "\n\nNow output:"
    return new


def main():
    print(f"Loading source dataset from {SRC_DIR}")
    src = load_from_disk(str(SRC_DIR))
    print(f"  rows: {len(src)}")

    new_rows = []
    for i, row in enumerate(src):
        ds_name = row["dataset_name"]
        new_target = make_prl_target(row["target"], ds_name, seed=i)
        new_prompt = make_prl_prompt(row["prompt"], ds_name)
        new_rows.append({
            "dataset_name": ds_name,
            "source_id": row["source_id"],
            "prompt": new_prompt,
            "target": new_target,
            "teacher_model": row["teacher_model"],
            "created_at": row["created_at"],
            "format_version": "prl_v1",
        })
        if i < 2:
            print(f"\n=== sample {i} ({ds_name}) ===")
            print(f"--- new target (first 500 chars) ---")
            print(new_target[:500])
            print(f"--- new prompt (first 400 chars) ---")
            print(new_prompt[:400])

    new_ds = Dataset.from_list(new_rows)
    DST_DIR.mkdir(parents=True, exist_ok=True)
    new_ds.save_to_disk(str(DST_DIR))
    print(f"\nSaved {len(new_ds)} rows to {DST_DIR}")


if __name__ == "__main__":
    main()
