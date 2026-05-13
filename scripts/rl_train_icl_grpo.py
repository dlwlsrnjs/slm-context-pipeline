#!/usr/bin/env python3
"""GRPO fine-tune the ICL generator using SLM answer correctness as reward.

Policy: SFT'd ICL generator (TRAIN-1B)
Reward: frozen Qwen2.5-1.5B-Instruct answers the target question given generated demos.
        reward = 1.0 if answer matches gold, else 0.0, minus 0.3 format penalty if Q!=4.

Dataset: same HF dataset used for SFT (`icl_distill_math/hf_distilled_icl_dataset`),
enriched with target question + gold answer by looking up source_id in math_5k JSONL.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/rl_train_icl_grpo.py \
        --policy-path experiment_results/math_icl_sft/qwen1b_run2/final \
        --reward-model Qwen/Qwen2.5-1.5B-Instruct \
        --output-dir experiment_results/math_icl_rl/qwen1b_grpo
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
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
DEMO_PATTERN = re.compile(r"Q:\s*(.+?)\nA:\s*(.+?)(?=\nQ:\s|\Z)", re.DOTALL)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-path", required=True)
    p.add_argument("--reward-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--sft-dataset",
                   default="icl_distill_math/hf_distilled_icl_dataset")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--per-device-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-completion-tokens", type=int, default=600)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--ans-max-tokens", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="If >0, stop after N optimizer steps (for validation runs)")
    p.add_argument("--cluster-dir", default="",
                   help="Path to math_5k_clusters dir. If set, RL prompts are rebuilt "
                        "with cluster representatives as seeds (Auto-CoT style).")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--format-penalty", type=float, default=0.3)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_id_index():
    idx = {}
    for fname in ["gsm8k_5k.jsonl", "orcamath_5k.jsonl", "metamath_5k.jsonl"]:
        with (MATH_5K_DIR / fname).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                idx[row["id"]] = row
    return idx


def format_seed_examples(rows, shots=4):
    """Same logic as build_math_icl_distill_dataset.py."""
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


def build_student_prompt(seed_examples: str, dataset_name: str, shots: int = 4) -> str:
    format_rule = (
        "Q: <math word problem>\n"
        "A: <step-by-step reasoning ending with 'The answer is [number].'>"
    )
    return (
        f"You are generating in-context demonstrations for math word problems "
        f"({dataset_name}).\n"
        f"Generate exactly {shots} demonstrations with this format:\n"
        f"{format_rule}\n\n"
        "Rules:\n"
        "- Each demo must be self-contained (no external tables or formulas needed).\n"
        "- Show clear arithmetic steps.\n"
        "- End every answer with 'The answer is [number].'\n"
        "- Do NOT add explanations outside the Q/A format.\n\n"
        "Candidate examples:\n"
        f"{seed_examples}\n\n"
        "Now output demonstrations only:\n"
    )


def load_cluster_reps(cluster_dir: str, k: int = 8):
    """Load cluster representatives per dataset.
    Returns: {dataset_name: list of representative dicts (8 each)}.
    dataset_name keys: 'gsm8k', 'orcamath', 'metamath' (matching source field).
    """
    cdir = REPO_ROOT / cluster_dir
    out = {}
    for short, full in [("gsm8k", "gsm8k_5k"),
                        ("orcamath", "orcamath_5k"),
                        ("metamath", "metamath_5k")]:
        path = cdir / f"{full}_clusters_k{k}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        out[short] = data["representatives"]
    print(f"[clusters] loaded reps from {cdir} (k={k}): "
          f"{ {k_: len(v) for k_, v in out.items()} }")
    return out


def build_rl_dataset(sft_path: str, cluster_dir: str = "",
                     n_shots: int = 4, seed: int = 42):
    """Build RL dataset.

    If cluster_dir empty: use original SFT prompts (random seeds embedded at SFT time).
    If cluster_dir set: REBUILD prompts using cluster representatives as seeds.
    """
    ds = load_from_disk(str(REPO_ROOT / sft_path))
    idx = load_id_index()

    if not cluster_dir:
        def add_target(row):
            info = idx.get(row["source_id"])
            if info is None:
                return {"question": None, "gold": None}
            return {"question": str(info["question"]).strip(),
                    "gold": str(info["answer"]).replace(",", "").strip()}
        ds = ds.map(add_target)
        ds = ds.filter(lambda r: r["question"] is not None and r["gold"])
        print(f"[data] total={len(ds)} (RL dataset, original SFT prompts)")
        return ds

    # Cluster-based: rebuild prompts
    cluster_reps = load_cluster_reps(cluster_dir, k=8)
    rng = random.Random(seed)

    def get_source_short(row):
        # source_id like 'gsm8k_00123', 'orcamath_034237', 'metamath_012345'
        return row["source_id"].split("_")[0]

    def rebuild(row):
        info = idx.get(row["source_id"])
        if info is None:
            return {"prompt": None, "question": None, "gold": None}
        short = get_source_short(row)
        reps = cluster_reps.get(short, [])
        if len(reps) < n_shots:
            return {"prompt": None, "question": None, "gold": None}
        # Exclude rep that matches this sample (rare but possible)
        candidates = [r for r in reps if r["rep_id"] != row["source_id"]]
        # Deterministic shuffle by source_id hash for reproducibility
        local_rng = random.Random(hash(row["source_id"]) & 0xFFFFFFFF)
        picks = local_rng.sample(candidates, n_shots)
        seed_text = format_seed_examples(picks, shots=n_shots)
        new_prompt = build_student_prompt(seed_text, f"{short}_5k", shots=n_shots)
        return {
            "prompt": new_prompt,
            "question": str(info["question"]).strip(),
            "gold": str(info["answer"]).replace(",", "").strip(),
        }

    ds = ds.map(rebuild, load_from_cache_file=False)
    ds = ds.filter(lambda r: r["question"] is not None and r["gold"])
    print(f"[data] total={len(ds)} (RL dataset, AUTO-COT cluster-based prompts)")
    return ds


def extract_first_n_demos(text: str, n: int = 4) -> str:
    clean = []
    for q, a in DEMO_PATTERN.findall(text):
        q, a = q.strip(), a.strip()
        m = re.search(r"[Tt]he answer is[^.\n]*\.?", a)
        if m:
            a = a[: m.end()].strip()
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


def make_reward_fn(format_penalty: float, ans_max_tokens: int):
    # graded format penalty: smoother gradient than the previous cliff
    FORMAT_TABLE = {0: -0.50, 1: -0.30, 2: -0.15, 3: -0.05, 4: 0.0}

    def reward_fn(completions, **kwargs):
        questions = kwargs["question"]
        golds = kwargs["gold"]

        demos_list = [extract_first_n_demos(c, 4) for c in completions]
        ans_prompts = [build_answer_prompt(q, d)
                       for q, d in zip(questions, demos_list)]

        answers = slm_answer_batched(ans_prompts, max_new_tokens=ans_max_tokens)

        rewards = []
        for comp, demos, ans, gold in zip(completions, demos_list, answers, golds):
            pred = extract_final_number(ans)
            correct = numeric_match(pred, gold)
            n_q = min(demos.count("Q:"), 4)
            fmt_reward = FORMAT_TABLE[n_q]
            rewards.append(float(correct) + fmt_reward)
        return rewards

    return reward_fn


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_slm(args.reward_model)

    dataset = build_rl_dataset(args.sft_dataset,
                               cluster_dir=args.cluster_dir,
                               n_shots=4, seed=args.seed)

    # TRL GRPOTrainer expects "prompt" column; rest passed as kwargs to reward_fn
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

    # Pre-load policy tokenizer to ensure pad_token is set (fixes Llama / SmolLM)
    policy_tok = AutoTokenizer.from_pretrained(args.policy_path, trust_remote_code=True)
    if policy_tok.pad_token is None:
        policy_tok.pad_token = policy_tok.eos_token
    print(f"[policy tokenizer] pad_token={policy_tok.pad_token} eos={policy_tok.eos_token}")

    trainer = GRPOTrainer(
        model=args.policy_path,
        reward_funcs=make_reward_fn(args.format_penalty, args.ans_max_tokens),
        args=grpo_cfg,
        train_dataset=dataset,
        peft_config=lora_cfg,
        processing_class=policy_tok,
    )

    print("[START] GRPO training")
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    print(f"[DONE] saved → {out_dir / 'final'}")


if __name__ == "__main__":
    main()
