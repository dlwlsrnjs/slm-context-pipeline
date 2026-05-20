"""Standalone math evaluation for ICRL-Math checkpoints.

Runs an inference loop that mirrors the training-time rollout:
    <think> ... </think> <search> code </search> <information> stdout </information>
    ... <answer> \\boxed{...} </answer>

Calls our Python sandbox at --sandbox-url for each <search>...</search> block,
caps the number of tool turns at --max-turns, then scores the final boxed answer
with our math_fewshot reward.

Supports the standard math evaluation suite:
    - gsm8k            (openai/gsm8k, split='test')
    - math500          (HuggingFaceH4/MATH-500)
    - aime2024         (Maxwell-Jia/AIME_2024)
    - aime2025         (yentinglin/aime_2025  /  Maxwell-Jia/AIME_2025  — first that loads)
    - minerva_math     (math-ai/Minerva-Math, if available)

Usage:
    python eval_math.py \\
        --model-path checkpoints/icrl-math-stage3-0shot-qwen2.5-3b/.../hf \\
        --datasets gsm8k math500 aime2024 aime2025 \\
        --sandbox-url http://127.0.0.1:8000/retrieve \\
        --num-shots 0 \\
        --out-dir eval_results/

Use --num-shots N to evaluate with N few-shot demos in the prompt (for
verifying the curriculum at intermediate stages).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# Ensure verl_patches is importable for reward scoring.
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "verl_patches"))
sys.path.insert(0, str(HERE / "scripts" / "data_process"))

from math_fewshot import (  # type: ignore  (our reward module)
    compute_accuracy,
    compute_format_score,
    extract_final_answer,
)
from math_fewshot import (  # data process module shares filename — pick this side first
    build_fewshot_prompt as _maybe_build,
)


# ---------------------------------------------------------------------------
# Prompt builders (re-import from data_process to avoid drift).
# ---------------------------------------------------------------------------

from importlib.util import spec_from_file_location, module_from_spec
_dp_spec = spec_from_file_location("dp_math_fewshot", str(HERE / "scripts" / "data_process" / "math_fewshot.py"))
_dp = module_from_spec(_dp_spec); _dp_spec.loader.exec_module(_dp)
build_fewshot_prompt = _dp.build_fewshot_prompt
build_zeroshot_prompt = _dp.build_zeroshot_prompt
load_fewshot_examples = _dp.load_fewshot_examples
resolve_examples_file = _dp.resolve_examples_file


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _gsm8k_loader(limit: Optional[int]) -> List[Dict]:
    from datasets import load_dataset
    d = load_dataset("openai/gsm8k", "main", split="test")
    if limit:
        d = d.select(range(min(limit, len(d))))
    out = []
    for ex in d:
        ans = ex["answer"]
        m = re.search(r"####\s*(\-?[0-9\.,]+)", ans)
        gold = m.group(1).replace(",", "").strip() if m else ans.strip()
        out.append({"problem": ex["question"].strip(), "gold": gold})
    return out


def _math500_loader(limit: Optional[int]) -> List[Dict]:
    from datasets import load_dataset
    d = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if limit:
        d = d.select(range(min(limit, len(d))))
    return [{"problem": ex["problem"].strip(), "gold": str(ex["answer"]).strip()} for ex in d]


def _aime_loader(year: int, limit: Optional[int]) -> List[Dict]:
    """Try a couple of mirrors for AIME 20{24,25}."""
    from datasets import load_dataset
    candidates = [
        (f"Maxwell-Jia/AIME_{year}", None, "train"),
        (f"yentinglin/aime_{year}", None, "train"),
        (f"HuggingFaceH4/aime_{year}", None, "test"),
    ]
    last_err = None
    d = None
    for repo, conf, split in candidates:
        try:
            d = load_dataset(repo, conf, split=split) if conf else load_dataset(repo, split=split)
            print(f"  [aime{year}] loaded {repo} (split={split})")
            break
        except Exception as e:
            last_err = e
            continue
    if d is None:
        raise RuntimeError(f"failed to load AIME{year}: {last_err}")
    if limit:
        d = d.select(range(min(limit, len(d))))
    out = []
    for ex in d:
        prob_key = "Problem" if "Problem" in ex else ("problem" if "problem" in ex else "question")
        ans_key = "Answer" if "Answer" in ex else ("answer" if "answer" in ex else "Solution")
        out.append({"problem": str(ex[prob_key]).strip(), "gold": str(ex[ans_key]).strip()})
    return out


def _minerva_loader(limit: Optional[int]) -> List[Dict]:
    from datasets import load_dataset
    try:
        d = load_dataset("math-ai/Minerva-Math", split="test")
    except Exception:
        return []
    if limit:
        d = d.select(range(min(limit, len(d))))
    return [{"problem": str(ex["problem"]).strip(), "gold": str(ex["answer"]).strip()} for ex in d]


_LOADERS = {
    "gsm8k": _gsm8k_loader,
    "math500": _math500_loader,
    "aime2024": lambda limit: _aime_loader(2024, limit),
    "aime2025": lambda limit: _aime_loader(2025, limit),
    "minerva_math": _minerva_loader,
}


# ---------------------------------------------------------------------------
# Inference loop with sandbox tool use
# ---------------------------------------------------------------------------


_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
_INFO_OPEN = "<information>"
_INFO_CLOSE = "</information>"
_ANSWER_OPEN_RE = re.compile(r"<answer>", re.IGNORECASE)


def _call_sandbox(url: str, code: str, timeout: float = 30.0) -> str:
    try:
        r = requests.post(url, json={"queries": [code], "topk": 1, "return_scores": False}, timeout=timeout)
        r.raise_for_status()
        result = r.json()["result"][0][0]["document"]["contents"]
        # contents = "stdout\nbody..." — strip title line for cleaner output.
        body = result.split("\n", 1)[1] if "\n" in result else result
        return body.strip()
    except Exception as e:
        return f"[sandbox error: {e}]"


def _generate_with_tool_loop(llm, sampling_params, prompt: str, sandbox_url: str,
                              max_turns: int, max_obs_chars: int = 500) -> str:
    """Run a multi-turn generation: stop at </search>, call sandbox, append, repeat."""
    from vllm import SamplingParams

    # Per-turn stop tokens.
    turn_sp = SamplingParams(
        n=1,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
        max_tokens=sampling_params.max_tokens,
        stop=["</search>", "</answer>"],
        include_stop_str_in_output=True,
    )

    response_so_far = ""
    for turn in range(max_turns + 1):
        full = prompt + response_so_far
        out = llm.generate([full], turn_sp, use_tqdm=False)[0]
        gen = out.outputs[0].text
        response_so_far += gen

        # Check stop reason: did we end at </search> or </answer>?
        if "</answer>" in gen:
            break
        if "</search>" in gen:
            # Extract the last <search>...</search> block from response_so_far,
            # call sandbox, append <information>.
            matches = list(_SEARCH_RE.finditer(response_so_far))
            if not matches:
                break
            code = matches[-1].group(1).strip()
            obs = _call_sandbox(sandbox_url, code)
            if len(obs) > max_obs_chars:
                obs = obs[:max_obs_chars] + "...[truncated]"
            response_so_far += f"\n{_INFO_OPEN}{obs}{_INFO_CLOSE}\n"
            continue
        # Hit max_tokens without stop → bail.
        break
    return response_so_far


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate(args):
    from vllm import LLM, SamplingParams

    # Build prompt prefix (fewshot or zeroshot).
    if args.num_shots and args.num_shots > 0:
        examples_path = resolve_examples_file(args.examples_name, None)
        fewshot_examples = load_fewshot_examples(examples_path, args.num_shots)
        prompt_builder = lambda q: build_fewshot_prompt(q, fewshot_examples)
        prompt_mode = f"{args.num_shots}-shot"
    else:
        prompt_builder = build_zeroshot_prompt
        prompt_mode = "0-shot"

    print(f"loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        n=1, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new_tokens,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}

    for ds_name in args.datasets:
        if ds_name not in _LOADERS:
            print(f"skip unknown dataset: {ds_name}")
            continue
        print(f"\n=== {ds_name} ({prompt_mode}) ===")
        examples = _LOADERS[ds_name](args.limit_per_dataset)
        print(f"  {len(examples)} problems")

        records = []
        n_correct = 0
        t0 = time.time()
        for i, ex in enumerate(examples):
            prompt = prompt_builder(ex["problem"])
            response = _generate_with_tool_loop(
                llm, sampling, prompt, args.sandbox_url, args.max_turns, args.max_obs_chars
            )
            # Score with our reward (use full text = prompt + response so format
            # extraction handles <answer> tag from the model's portion).
            acc = compute_accuracy(response, {"target": ex["gold"]})
            fmt, stats = compute_format_score(response, return_stats=True)
            pred = extract_final_answer(response)
            n_correct += int(acc >= 0.5)
            records.append({
                "idx": i,
                "problem": ex["problem"],
                "gold": ex["gold"],
                "pred": pred,
                "acc": acc,
                "fmt": fmt,
                "stats": stats,
                "response": response,
            })
            if (i + 1) % max(1, len(examples) // 20) == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(examples)}] running EM={n_correct/(i+1):.3f}  ({elapsed:.0f}s)")

        em = n_correct / max(1, len(examples))
        summary[ds_name] = {"em": em, "n": len(examples), "prompt": prompt_mode}
        print(f"  {ds_name}: EM = {em:.3f} ({n_correct}/{len(examples)})")

        out_path = Path(args.out_dir) / f"{ds_name}_{prompt_mode}.jsonl"
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  records -> {out_path}")

    summary_path = Path(args.out_dir) / f"summary_{prompt_mode}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {summary_path}")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "aime2024", "aime2025"],
                   choices=list(_LOADERS.keys()))
    p.add_argument("--sandbox-url", default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--num-shots", type=int, default=0,
                   help="0 for zero-shot eval (matches final-stage policy); use 3 to evaluate intermediate")
    p.add_argument("--examples-name", default="math_examples.txt")
    p.add_argument("--out-dir", default="./eval_results")
    p.add_argument("--limit-per-dataset", type=int, default=None,
                   help="cap problems per dataset for quick smoke tests")
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-obs-chars", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--tp-size", type=int, default=1)
    args = p.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
