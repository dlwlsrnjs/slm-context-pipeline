#!/usr/bin/env python3
"""GRPO fine-tune ICL generator with PRL-style reasoning step.

Key differences vs rl_train_icl_grpo.py:
  1) Prompt instructs generator to first reason in <think>...</think>,
     then output demos in <answer>...</answer>.
  2) Reward function adds:
       - token reward (each of 4 tags appears exactly once)
       - structure reward (overall <think>...</think><answer>...</answer> match)
       - format reward (4 valid demos inside <answer>)
       - alignment reward (downstream SLM correctness)
  3) Demos are extracted from inside <answer>...</answer> only.

Usage example:
    CUDA_VISIBLE_DEVICES=1 python scripts/rl_train_icl_grpo_prl.py \
        --policy-path experiment_results/math_icl_sft/qwen1b_run2/final \
        --reward-model Qwen/Qwen2.5-1.5B-Instruct \
        --output-dir experiment_results/math_icl_rl/qwen1b_grpo_prl
"""
import argparse
import json
import random
import re
from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent
MATH_5K_DIR = REPO_ROOT / "slm_context_pipeline/data/math_5k"

_SLM_TOK = None
_SLM_MODEL = None
_LLM_JUDGE_TOK = None
_LLM_JUDGE_MODEL = None
_validation_pools = {}      # short_name → list of {id, question, answer}
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
DEMO_PATTERN = re.compile(r"Q:\s*(.+?)\nA:\s*(.+?)(?=\nQ:\s|\Z)", re.DOTALL)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
TAGS = ["<think>", "</think>", "<answer>", "</answer>"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-path", required=True)
    p.add_argument("--reward-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--sft-dataset",
                   default="icl_distill_math/hf_distilled_icl_dataset")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--per-device-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-completion-tokens", type=int, default=900,
                   help="larger than no-reasoning (extra <think> budget)")
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--ans-max-tokens", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=8e-6)
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--lora-r", type=int, default=16)
    # PRL-style component reward weights (each scaled to ~1)
    p.add_argument("--r-token", type=float, default=0.25)      # 4 tags total
    p.add_argument("--r-structure", type=float, default=0.5)
    p.add_argument("--r-format", type=float, default=0.5)
    p.add_argument("--r-alignment", type=float, default=1.0)
    p.add_argument("--r-repetition", type=float, default=0.5,
                   help="Trigram repetition penalty weight (positive number; reward = -weight*ratio)")
    p.add_argument("--alignment-n", type=int, default=10,
                   help="PRL-style: number of held-out questions to average alignment reward over")
    p.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="LLM judge for demo quality (R_format)")
    p.add_argument("--r-judge", type=float, default=1.0,
                   help="Weight for LLM judge demo-quality reward (max value)")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cluster-dir", default="")
    return p.parse_args()


def load_id_index():
    idx = {}
    for fname in ["gsm8k_5k.jsonl", "orcamath_5k.jsonl", "metamath_5k.jsonl"]:
        with (MATH_5K_DIR / fname).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    idx[row["id"]] = row
    return idx


def format_seed_examples(rows, shots=4):
    lines = []
    for row in rows[:shots]:
        q = str(row["question"]).strip()
        sol = str(row.get("solution", row.get("answer", ""))).strip()
        sol = re.sub(r"<<[^>]*>>", "", sol)
        sol = re.sub(r"####\s*[\d,]+", "", sol).strip()
        final = str(row.get("answer", "")).strip()
        if final and not re.search(r"[Tt]he answer is", sol):
            sol = sol.rstrip(".") + f"\nThe answer is {final}."
        lines.append(f"Q: {q}\nA: {sol}")
    return "\n\n".join(lines)


def build_prl_prompt(seed_examples: str, dataset_name: str, shots: int = 4) -> str:
    """PRL-style: <think>reasoning</think><answer>demos</answer>."""
    return (
        f"You are generating in-context demonstrations for math word problems "
        f"({dataset_name}).\n"
        f"First, think briefly inside <think>...</think> about which patterns "
        f"and difficulty level are useful. Then output exactly {shots} "
        f"self-contained demonstrations inside <answer>...</answer>.\n\n"
        "Demonstration format inside <answer>:\n"
        "Q: <math word problem>\n"
        "A: <step-by-step reasoning ending with 'The answer is [number].'>\n\n"
        "Rules:\n"
        "- Each demo must be self-contained.\n"
        "- Show clear arithmetic steps.\n"
        "- End every answer with 'The answer is [number].'\n"
        "- Use exactly the tags <think>, </think>, <answer>, </answer> "
        "(each appears once).\n\n"
        "Candidate examples (for reference, do NOT copy verbatim):\n"
        f"{seed_examples}\n\n"
        "Now output:\n"
    )


def load_cluster_reps(cluster_dir: str, k: int = 8):
    cdir = REPO_ROOT / cluster_dir
    out = {}
    for short, full in [("gsm8k", "gsm8k_5k"),
                        ("orcamath", "orcamath_5k"),
                        ("metamath", "metamath_5k")]:
        path = cdir / f"{full}_clusters_k{k}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        out[short] = data["representatives"]
    print(f"[clusters] loaded reps from {cdir} (k={k})")
    return out


def build_rl_dataset(sft_path: str, cluster_dir: str = "",
                     n_shots: int = 4, seed: int = 42):
    ds = load_from_disk(str(REPO_ROOT / sft_path))
    idx = load_id_index()

    if cluster_dir:
        cluster_reps = load_cluster_reps(cluster_dir, k=8)
    else:
        cluster_reps = None

    def get_short(row):
        return row["source_id"].split("_")[0]

    def transform(row):
        info = idx.get(row["source_id"])
        if info is None:
            return {"prompt": None, "question": None, "gold": None}
        short = get_short(row)
        if cluster_reps is not None:
            reps = cluster_reps.get(short, [])
            if len(reps) < n_shots:
                return {"prompt": None, "question": None, "gold": None}
            candidates = [r for r in reps if r["rep_id"] != row["source_id"]]
            local_rng = random.Random(hash(row["source_id"]) & 0xFFFFFFFF)
            picks = local_rng.sample(candidates, n_shots)
        else:
            # use original SFT prompt's seed examples — re-extract isn't easy,
            # so we just rebuild from math_5k random seeds for consistency
            pool = idx
            local_rng = random.Random(hash(row["source_id"]) & 0xFFFFFFFF)
            same_source = [v for k, v in idx.items()
                           if k.startswith(short + "_") and k != row["source_id"]]
            picks = local_rng.sample(same_source, n_shots)
        seed_text = format_seed_examples(picks, shots=n_shots)
        new_prompt = build_prl_prompt(seed_text, f"{short}_5k", shots=n_shots)
        return {
            "prompt": new_prompt,
            "question": str(info["question"]).strip(),
            "gold": str(info["answer"]).replace(",", "").strip(),
        }

    ds = ds.map(transform, load_from_cache_file=False)
    ds = ds.filter(lambda r: r["question"] is not None and r["gold"])
    src = "AUTO-COT cluster" if cluster_dir else "random per-id"
    print(f"[data] total={len(ds)} (PRL-style prompts, seeds={src})")
    return ds


def extract_demos_from_answer(text: str, n: int = 4) -> str:
    """Extract first n demos from inside <answer>...</answer>.
    If no <answer> tag, fall back to whole text."""
    m = ANSWER_PATTERN.search(text)
    payload = m.group(1) if m else text
    clean = []
    for q, a in DEMO_PATTERN.findall(payload):
        q, a = q.strip(), a.strip()
        m2 = re.search(r"[Tt]he answer is[^.\n]*\.?", a)
        if m2:
            a = a[: m2.end()].strip()
        if q and a:
            clean.append(f"Q: {q}\nA: {a}")
        if len(clean) >= n:
            break
    return "\n\n".join(clean)


def extract_final_number(text: str):
    m = re.search(r"[Tt]he answer is[:\s]+([^\n]+)", text)
    if m:
        nums = NUMBER_RE.findall(m.group(1))
        if nums:
            return nums[0].replace(",", "")
    nums = NUMBER_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def numeric_match(pred, gold):
    if pred is None or not gold:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred.strip() == gold.strip()


def build_answer_prompt(question: str, demos: str) -> str:
    header = ("Solve the following math word problem. "
              "Show your reasoning and end with 'The answer is [number].'\n\n")
    if demos:
        return header + demos + "\n\n" + f"Q: {question}\nA:"
    return header + f"Q: {question}\nA:"


def init_slm(model_name: str):
    global _SLM_TOK, _SLM_MODEL
    _SLM_TOK = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if _SLM_TOK.pad_token is None:
        _SLM_TOK.pad_token = _SLM_TOK.eos_token
    _SLM_TOK.padding_side = "left"
    _SLM_MODEL = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    _SLM_MODEL.eval()
    print(f"[reward] SLM loaded: {model_name}")


def slm_answer_batched(prompts, max_new_tokens: int):
    enc = _SLM_TOK(prompts, return_tensors="pt", padding=True,
                   truncation=True, max_length=3072).to(_SLM_MODEL.device)
    with torch.no_grad():
        gen = _SLM_MODEL.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_SLM_TOK.pad_token_id,
            eos_token_id=_SLM_TOK.eos_token_id,
        )
    new_tokens = gen[:, enc["input_ids"].shape[1]:]
    return _SLM_TOK.batch_decode(new_tokens, skip_special_tokens=True)


# ==================== Comprehensive LLM Judge (v5) ====================
# Single comprehensive judge call covers what was R_token + R_structure +
# (math correctness). The auto-regex R_token, R_structure are dropped.
# Returns 4 binary verdicts (A/B/C/D) → score = (Y count)/4.
JUDGE_PROMPT_HOLISTIC = """You are a strict format auditor. Verify ONE response against EXACT rules.

RESPONSE TO AUDIT (between triple backticks):
```
{completion}
```

You MUST audit FOUR criteria using EXACT counting/checking. No leniency.

A) TAG_USAGE — count occurrences of EACH tag literal as substring:
   - "<think>" must appear exactly 1 time
   - "</think>" must appear exactly 1 time
   - "<answer>" must appear exactly 1 time
   - "</answer>" must appear exactly 1 time
   Y only if ALL FOUR counts equal 1. N otherwise.

B) STRUCTURE — only valid if A:Y AND closing of </think> appears at a position before opening of <answer>. N otherwise.

C) DEMO_COUNT — count "Q:" prefix occurrences inside the <answer>...</answer> block.
   Y only if exactly 4. N otherwise.

D) MATH_OK — for each Q/A pair inside <answer>, recompute the arithmetic from scratch.
   Y only if ALL 4 demos have correct arithmetic AND "The answer is X" matches.
   N if there are not 4 demos, or if any one demo has wrong math.

Reason step by step (count each tag, verify order, count Q:, recompute math). Then on the FINAL line output EXACTLY:
A:? B:? C:? D:?

where each ? is Y or N. The final line must contain ONLY this.

Audit:"""


JUDGE_PARSE_RE = re.compile(r"A\s*:\s*([YN]).*?B\s*:\s*([YN]).*?C\s*:\s*([YN]).*?D\s*:\s*([YN])",
                            re.IGNORECASE | re.DOTALL)


def _parse_judge_verdict(text: str):
    """Parse judge output into (a, b, c, d) tuple of bools."""
    m = JUDGE_PARSE_RE.search(text or "")
    if not m:
        return (False, False, False, False)
    return tuple(g.upper() == "Y" for g in m.groups())


def init_llm_judge(model_name: str):
    global _LLM_JUDGE_TOK, _LLM_JUDGE_MODEL
    _LLM_JUDGE_TOK = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if _LLM_JUDGE_TOK.pad_token is None:
        _LLM_JUDGE_TOK.pad_token = _LLM_JUDGE_TOK.eos_token
    _LLM_JUDGE_TOK.padding_side = "left"
    _LLM_JUDGE_MODEL = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    _LLM_JUDGE_MODEL.eval()
    print(f"[reward] LLM judge loaded: {model_name}")


def llm_judge_holistic_batched(completions, max_new_tokens: int = 300):
    """Holistic judge: one call per completion, evaluates 4 criteria.

    Returns: list of (score, verdict_tuple)
      - score: float in [0, 1] = (Y count) / 4
      - verdict_tuple: (a, b, c, d) booleans for logging
    """
    if not completions:
        return []

    # Truncate completion if too long for prompt budget
    truncated = [c[:6000] if c else "(empty)" for c in completions]
    prompts = [JUDGE_PROMPT_HOLISTIC.format(completion=c) for c in truncated]
    messages = [[{"role": "user", "content": p}] for p in prompts]
    chat_prompts = [
        _LLM_JUDGE_TOK.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    enc = _LLM_JUDGE_TOK(chat_prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=4096).to(_LLM_JUDGE_MODEL.device)
    with torch.no_grad():
        gen = _LLM_JUDGE_MODEL.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_LLM_JUDGE_TOK.pad_token_id,
            eos_token_id=_LLM_JUDGE_TOK.eos_token_id,
        )
    new_tokens = gen[:, enc["input_ids"].shape[1]:]
    decoded = _LLM_JUDGE_TOK.batch_decode(new_tokens, skip_special_tokens=True)

    results = []
    for text in decoded:
        verdicts = _parse_judge_verdict(text)
        score = sum(verdicts) / 4.0
        results.append((score, verdicts))
    return results


def make_reward_fn(args):

    def reward_fn(completions, **kwargs):
        questions = kwargs["question"]
        golds = kwargs["gold"]

        # Extract demos for downstream SLM call (and repetition check)
        demos_list = []
        repetition_rewards = []
        for c in completions:
            demos = extract_demos_from_answer(c, 4)
            demos_list.append(demos)
            # repetition penalty: trigram diversity
            words = demos.lower().split() if demos else []
            if len(words) >= 3:
                grams = list(zip(*[words[i:] for i in range(3)]))
                rep_ratio = 1 - len(set(grams)) / len(grams)
                r_rep = -args.r_repetition * rep_ratio
            else:
                r_rep = 0.0
            repetition_rewards.append(r_rep)

        # PRL-style alignment: each completion's demos applied to N held-out questions
        # → reward = average correctness over N samples (dense signal)
        N = args.alignment_n
        dataset_names = kwargs.get("dataset_name", [None] * len(completions))

        # Build N batched prompts per completion
        all_prompts = []                    # flat list of len(completions)*N
        all_golds = []                      # flat list of len(completions)*N
        for i, (c, q_orig, g_orig, demos, ds) in enumerate(zip(
                completions, questions, golds, demos_list, dataset_names)):
            # First sample = original target question (kept for compatibility)
            picks = [(q_orig, g_orig)]
            # Remaining N-1 samples from same dataset's pool (excluding original)
            short = (ds or "").split("_")[0] if ds else None
            pool = _validation_pools.get(short, [])
            if pool and len(pool) >= N:
                rng = random.Random(hash(q_orig) & 0xFFFFFFFF)
                extras = rng.sample(
                    [p for p in pool if p["question"] != q_orig], N - 1)
                picks.extend([(p["question"], p["answer"]) for p in extras])
            else:
                picks.extend([(q_orig, g_orig)] * (N - 1))    # fallback
            for q, g in picks:
                all_prompts.append(build_answer_prompt(q, demos))
                all_golds.append(g)

        # Single batched SLM inference for all len(completions)*N prompts
        all_answers = slm_answer_batched(all_prompts,
                                         max_new_tokens=args.ans_max_tokens)

        # ── R_judge_holistic: 7B LLM evaluates 4 criteria per completion ──
        # (replaces R_token + R_structure + R_format + math correctness)
        judge_results = llm_judge_holistic_batched(completions, max_new_tokens=300)

        # Compute final rewards
        rewards = []
        log_first = getattr(make_reward_fn, "_log_count", 0)
        for i, (r_rep, (j_score, j_verdicts)) in enumerate(zip(
                repetition_rewards, judge_results)):
            slc = slice(i * N, (i + 1) * N)
            correct = sum(
                1 for a, g in zip(all_answers[slc], all_golds[slc])
                if numeric_match(extract_final_number(a), g))
            r_align = args.r_alignment * correct / N
            r_judge = args.r_judge * j_score
            total = r_judge + r_align + r_rep
            rewards.append(total)
            if log_first < 4 and i < 4:
                a, b, c, d = j_verdicts
                print(f"[reward] c{i}: judge={r_judge:.3f} "
                      f"[A:{'Y' if a else 'N'} B:{'Y' if b else 'N'} "
                      f"C:{'Y' if c else 'N'} D:{'Y' if d else 'N'}] "
                      f"align={r_align:.3f} ({correct}/{N}) "
                      f"rep={r_rep:.3f} → total={total:.3f}", flush=True)
                log_first += 1
        make_reward_fn._log_count = log_first
        return rewards

    return reward_fn


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_slm(args.reward_model)
    init_llm_judge(args.judge_model)

    # Load validation pools for PRL-style N-sample alignment reward
    global _validation_pools
    for short, fname in [("gsm8k", "gsm8k_5k.jsonl"),
                         ("orcamath", "orcamath_5k.jsonl"),
                         ("metamath", "metamath_5k.jsonl")]:
        with (MATH_5K_DIR / fname).open(encoding="utf-8") as f:
            _validation_pools[short] = [
                {"id": r["id"], "question": str(r["question"]).strip(),
                 "answer": str(r["answer"]).replace(",", "").strip()}
                for r in (json.loads(line) for line in f if line.strip())
            ]
    print(f"[validation_pools] {{ {', '.join(f'{k}: {len(v)}' for k, v in _validation_pools.items())} }}")

    dataset = build_rl_dataset(args.sft_dataset,
                               cluster_dir=args.cluster_dir,
                               n_shots=4, seed=args.seed)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    grpo_cfg = GRPOConfig(
        output_dir=str(out_dir),
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        max_prompt_length=args.max_prompt_tokens,
        max_completion_length=args.max_completion_tokens,
        beta=args.beta,
        temperature=0.9,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
    )

    policy_tok = AutoTokenizer.from_pretrained(args.policy_path,
                                               trust_remote_code=True)
    if policy_tok.pad_token is None:
        policy_tok.pad_token = policy_tok.eos_token
    print(f"[policy tokenizer] pad_token={policy_tok.pad_token}")

    trainer = GRPOTrainer(
        model=args.policy_path,
        reward_funcs=make_reward_fn(args),
        args=grpo_cfg,
        train_dataset=dataset,
        peft_config=lora_cfg,
        processing_class=policy_tok,
    )

    print("[START] PRL-style GRPO training")
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    print(f"[DONE] saved → {out_dir / 'final'}")


if __name__ == "__main__":
    main()
