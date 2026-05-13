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
_question_to_cluster = {}   # short_name → {question_text: cluster_id}
_cluster_to_questions = {}  # short_name → {cluster_id: [items]}
_question_embeddings = {}   # short_name → {"questions": [...], "embeddings": ndarray}
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
    p.add_argument("--r-target", type=float, default=4.0,
                   help="v5 direct reward: SLM solves target_q with demos correctly?")
    p.add_argument("--r-alignment", type=float, default=0.0,
                   help="(legacy v4 — unused in v5)")
    p.add_argument("--r-repetition", type=float, default=0.5,
                   help="Trigram repetition penalty weight (positive number; reward = -weight*ratio)")
    p.add_argument("--alignment-n", type=int, default=10,
                   help="PRL-style: number of held-out questions to average alignment reward over")
    p.add_argument("--selection-strategy", default="random",
                   choices=["random", "cluster", "embedding"],
                   help="How to pick N-1 extra questions: random / cluster / embedding")
    p.add_argument("--cluster-reps-dir", default="slm_context_pipeline/data/math_5k_clusters",
                   help="Used when selection-strategy=cluster (loads kmeans labels)")
    p.add_argument("--embedder-name", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Used when selection-strategy=embedding")
    # v5 — judge removed (was self-bias). Reward is direct SLM accuracy on target.
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


# ==================== LLM Judge (R_format quality) ====================
# Per-demo judging: ask the judge to first solve, then compare.
# Demos that "lie" (wrong arithmetic but matching answer) get caught here.
JUDGE_PROMPT = """Verify ONE math demonstration. Output your VERDICT first, then brief reasoning.

Demo:
{demo}

Solve the question yourself and compare to the demo's "The answer is X."
- If both arithmetic steps AND final answer are correct → CORRECT
- If any arithmetic is wrong OR final answer is wrong → WRONG

Output format (FIRST line is verdict, then reasoning):
VERDICT: CORRECT
or
VERDICT: WRONG

Then briefly explain.

Output:"""

DEMO_SPLIT_RE = re.compile(r"(?=Q:\s)")


def _split_demos(demos_text: str, max_demos: int = 4):
    """Split a `Q: ... A: ... \\nQ: ... A: ...` string into individual demos."""
    if not demos_text or not demos_text.strip():
        return []
    parts = [p.strip() for p in DEMO_SPLIT_RE.split(demos_text) if p.strip().startswith("Q:")]
    return parts[:max_demos]


def _init_cluster_selection(cluster_dir: str):
    """Build mapping question_text → cluster_id and cluster_id → list of items.

    Uses kmeans labels saved alongside math_5k_clusters. The cluster files contain
    only 8 representative ids per dataset; for full N-sampling we need all 5000
    items partitioned by cluster — so we re-run a quick label assignment using
    sentence-transformers (same model as build_autocot_clusters.py).
    """
    print(f"[cluster-init] re-clustering {sum(len(v) for v in _validation_pools.values())} questions")
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    for short, items in _validation_pools.items():
        questions = [it["question"] for it in items]
        embs = encoder.encode(questions, batch_size=128, show_progress_bar=False)
        km = KMeans(n_clusters=8, random_state=192, n_init=10)
        labels = km.fit_predict(embs)
        q2c = {q: int(c) for q, c in zip(questions, labels)}
        c2items = {c: [] for c in range(8)}
        for it, lab in zip(items, labels):
            c2items[int(lab)].append(it)
        _question_to_cluster[short] = q2c
        _cluster_to_questions[short] = c2items
        sizes = [len(c2items[c]) for c in range(8)]
        print(f"  [cluster-init] {short}: cluster sizes {sizes}")


def _init_embedding_selection(model_name: str):
    """Pre-compute embeddings for all 5000 questions per dataset."""
    print(f"[embedding-init] using {model_name}")
    import numpy as np
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(model_name)
    for short, items in _validation_pools.items():
        questions = [it["question"] for it in items]
        embs = encoder.encode(questions, batch_size=128, show_progress_bar=False,
                              normalize_embeddings=True)
        _question_embeddings[short] = {
            "questions": questions,
            "embeddings": np.asarray(embs, dtype=np.float32),
            "items": items,
        }
        print(f"  [embedding-init] {short}: {len(questions)} embeddings, dim={embs.shape[1]}")


def _select_n_extras(strategy: str, short: str, q_orig: str, n_extra: int,
                     rng: random.Random):
    """Pick n_extra questions from the dataset's pool, using the chosen strategy."""
    if strategy == "cluster":
        cluster_id = _question_to_cluster.get(short, {}).get(q_orig)
        if cluster_id is None:
            # fall back to random
            pool = _validation_pools[short]
            return rng.sample([p for p in pool if p["question"] != q_orig], n_extra)
        cluster_items = [it for it in _cluster_to_questions[short][cluster_id]
                         if it["question"] != q_orig]
        if len(cluster_items) < n_extra:
            # cluster too small — pad from random
            pool = _validation_pools[short]
            extras = list(cluster_items)
            remaining = n_extra - len(extras)
            other = [p for p in pool if p["question"] != q_orig
                     and p not in cluster_items]
            extras.extend(rng.sample(other, min(remaining, len(other))))
            return extras
        return rng.sample(cluster_items, n_extra)

    if strategy == "embedding":
        import numpy as np
        cache = _question_embeddings.get(short)
        if cache is None:
            pool = _validation_pools[short]
            return rng.sample([p for p in pool if p["question"] != q_orig], n_extra)
        questions = cache["questions"]
        embs = cache["embeddings"]
        items = cache["items"]
        try:
            idx_orig = questions.index(q_orig)
            target_emb = embs[idx_orig]
        except ValueError:
            # q_orig not in pool exactly — embed on the fly
            from sentence_transformers import SentenceTransformer
            # rather than re-loading encoder, fall back to random
            pool = _validation_pools[short]
            return rng.sample([p for p in pool if p["question"] != q_orig], n_extra)
        sims = embs @ target_emb     # cosine since normalized
        # exclude self (idx_orig); take top-(n_extra+5) candidates and rng-sample n_extra
        topk = max(n_extra * 3, n_extra + 5)
        order = np.argsort(-sims)
        # drop self at first occurrence
        cand_idx = [i for i in order[:topk + 1] if i != idx_orig][:topk]
        sample_idx = rng.sample(cand_idx, n_extra)
        return [items[i] for i in sample_idx]

    # default: random
    pool = _validation_pools[short]
    return rng.sample([p for p in pool if p["question"] != q_orig], n_extra)


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


def llm_judge_batched(demos_list, max_new_tokens: int = 500):
    """Score each completion's demos by asking judge per-demo whether it's correct.

    Returns: list of floats in [0, 1] — fraction of valid demos (out of 4 expected).
    """
    if not demos_list:
        return []
    # Flatten: per (completion_idx, demo_idx) → judge once
    flat_prompts = []
    flat_owner = []   # which completion this demo belongs to
    # VERBOSE: log every step's first completion demos
    step = getattr(llm_judge_batched, "_step", 0)
    print(f"\n══════════ step {step} ══════════", flush=True)
    for ci, demos_text in enumerate(demos_list):
        demos = _split_demos(demos_text, max_demos=4)
        if ci == 0:    # only log first completion to keep logs readable
            print(f"[demos c0] split into {len(demos)} demos", flush=True)
            for k, d in enumerate(demos):
                print(f"--- demo[{k}] ({len(d)} chars) ---", flush=True)
                print(d, flush=True)
                print(f"--- end demo[{k}] ---", flush=True)
        for d in demos:
            flat_prompts.append(JUDGE_PROMPT.format(demo=d))
            flat_owner.append(ci)

    if not flat_prompts:
        return [0.0] * len(demos_list)

    messages = [[{"role": "user", "content": p}] for p in flat_prompts]
    chat_prompts = [
        _LLM_JUDGE_TOK.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    enc = _LLM_JUDGE_TOK(chat_prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=1024).to(_LLM_JUDGE_MODEL.device)
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

    # Tally per completion + log first completion's verdicts
    correct_per_comp = [0] * len(demos_list)
    first_verdicts = []
    for text, owner in zip(decoded, flat_owner):
        first_line = text.strip().split("\n")[0].strip().upper() if text.strip() else ""
        is_correct = "CORRECT" in first_line and "WRONG" not in first_line
        if is_correct:
            correct_per_comp[owner] += 1
        if owner == 0:
            first_verdicts.append((first_line[:60], "CORRECT" if is_correct else "WRONG"))

    print(f"[judge c0 verdicts]: {first_verdicts}", flush=True)
    llm_judge_batched._step = step + 1
    # Normalize by expected demo count (4)
    return [c / 4.0 for c in correct_per_comp]


def make_reward_fn(args):

    def reward_fn(completions, **kwargs):
        questions = kwargs["question"]
        golds = kwargs["gold"]

        # Pre-compute structural rewards (no LLM call)
        token_rewards = []
        structure_rewards = []
        format_rewards = []
        demos_list = []
        for c in completions:
            # 1) token: each tag appears exactly once
            token_hits = sum(1 for t in TAGS if c.count(t) == 1)
            r_tok = args.r_token * (token_hits / len(TAGS))
            token_rewards.append(r_tok)
            # 2) structure: matches <think>...</think><answer>...</answer>
            think_m = THINK_PATTERN.search(c)
            ans_m = ANSWER_PATTERN.search(c)
            structure_ok = (think_m is not None and ans_m is not None
                            and think_m.end() <= ans_m.start())
            structure_rewards.append(args.r_structure if structure_ok else 0.0)
            # 3) format: BINARY — Q=4 정확히 ✓ , 아니면 strict penalty
            #    v1의 graded scaling이 Q=1로 collapse 유발 → binary로 강제
            demos = extract_demos_from_answer(c, 4)
            n_q = demos.count("Q:")
            if n_q == 4:
                r_fmt = args.r_format
            else:
                r_fmt = -args.r_format * 0.5  # 4개 못 만들면 negative (strong signal)
            format_rewards.append(r_fmt)
            demos_list.append(demos)

            # 3b) repetition penalty: trigram 중복 비율
            words = demos.lower().split() if demos else []
            if len(words) >= 3:
                grams = list(zip(*[words[i:] for i in range(3)]))
                rep_ratio = 1 - len(set(grams)) / len(grams)
                r_rep = -args.r_repetition * rep_ratio
            else:
                r_rep = 0.0
            # store separately
            if not hasattr(make_reward_fn, "_rep_rewards"):
                pass
            # 간단히 token_rewards에 repetition penalty 추가
            token_rewards[-1] += r_rep

        # v5 — Direct target accuracy: SLM solves the *target* question with demos
        # No proxy, no judge, no N>1. Pure RL signal: did this demo set actually help?
        all_prompts = [build_answer_prompt(q, d)
                       for q, d in zip(questions, demos_list)]
        all_answers = slm_answer_batched(all_prompts,
                                         max_new_tokens=args.ans_max_tokens)

        rewards = []
        for i, (r_tok, r_str, r_fmt) in enumerate(zip(
                token_rewards, structure_rewards, format_rewards)):
            correct = numeric_match(extract_final_number(all_answers[i]), golds[i])
            r_target = args.r_target * (1.0 if correct else 0.0)
            total = r_tok + r_str + r_fmt + r_target
            rewards.append(total)
            print(f"[reward c{i}] tok={r_tok:+.3f} str={r_str:+.3f} fmt={r_fmt:+.3f} "
                  f"target={r_target:+.3f}({'OK' if correct else 'X'}) → total={total:+.3f}",
                  flush=True)
        return rewards

    return reward_fn


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    init_slm(args.reward_model)
    # v5 — judge removed (was self-bias)

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

    # Build selection-strategy auxiliary structures
    if args.selection_strategy == "cluster":
        _init_cluster_selection(args.cluster_reps_dir)
    elif args.selection_strategy == "embedding":
        _init_embedding_selection(args.embedder_name)
    print(f"[selection-strategy] {args.selection_strategy}")

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
