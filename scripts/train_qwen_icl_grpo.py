#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import random
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen ICL generator with GRPO-style policy updates")
    parser.add_argument("--policy-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--target-model", default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--dataset-path", default="local_datasets/BoolQ")
    parser.add_argument("--dataset-name", default="", help="Optional explicit dataset name (e.g., BoolQ, PIQA, WinoGrande)")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--off-policy-ratio", type=float, default=0.5)
    parser.add_argument("--n-demo", type=int, default=1)
    parser.add_argument("--demo-match-strategy", default="reward", choices=["reward", "random"])
    parser.add_argument("--cache-min-reward", type=float, default=0.5)
    parser.add_argument("--hinted-ratio-denom", type=float, default=0.1)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--seed-pool-size", type=int, default=12)
    parser.add_argument("--reward-batch-size", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--step-eval-episodes", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument("--adv-clip", type=float, default=2.0)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-clip-range", type=float, default=0.2)
    parser.add_argument("--hinted-loss-scale", type=float, default=1.0)
    parser.add_argument("--no-adv-normalize", action="store_true")
    parser.add_argument("--allow-regression", action="store_true", help="Do not fallback to initial weights when post reward is lower")
    parser.add_argument("--hint-provider", choices=["none", "openai"], default="none")
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--openai-max-tokens", type=int, default=220)
    parser.add_argument("--openai-hint-retries", type=int, default=2)
    parser.add_argument("--output-dir", default="experiment_results/grpo_icl")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_bool_prediction(text: str) -> int:
    lowered = text.strip().lower()
    token = re.search(r"\b(true|false|yes|no)\b", lowered)
    if token:
        value = token.group(1)
        return 1 if value in {"true", "yes"} else 0
    return 0


def infer_dataset_name(dataset_path: str, explicit_name: str) -> str:
    if explicit_name:
        return explicit_name
    return Path(dataset_path).name


def dataset_task(dataset_name: str) -> str:
    if dataset_name in {"BoolQ", "PIQA", "WinoGrande", "Hellaswag"}:
        return "classification"
    if dataset_name in {"e2e_nlg", "viggo"}:
        return "generation"
    raise ValueError(f"Unsupported dataset for GRPO trainer: {dataset_name}")


def parse_mc_prediction(text: str) -> int:
    lowered = text.strip().lower()
    num_match = re.search(r"\b([1-5])\b", lowered)
    if num_match:
        return int(num_match.group(1)) - 1
    option_match = re.search(r"\boption\s*([1-5])\b", lowered)
    if option_match:
        return int(option_match.group(1)) - 1
    letter_match = re.search(r"\b([abcde])\b", lowered)
    if letter_match:
        return ord(letter_match.group(1)) - ord("a")
    return 0


def tokenize_simple(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def unigram_f1(pred_text: str, gold_text: str) -> float:
    pred_tokens = tokenize_simple(pred_text)
    gold_tokens = tokenize_simple(gold_text)
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    gold_counts = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    overlap = 0
    for token, count in pred_counts.items():
        if token in gold_counts:
            overlap += min(count, gold_counts[token])
    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(gold_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def to_label(value) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        return 1 if lowered in {"true", "yes", "1"} else 0
    return int(value)


def format_seed_examples(rows: List[dict], shots: int, dataset_name: str) -> str:
    lines = []
    for item in rows[:shots]:
        if dataset_name == "BoolQ":
            question = str(item["question"]).strip()
            passage = str(item["passage"]).strip()
            label = "True" if to_label(item["answer"]) == 1 else "False"
            lines.append(
                "Q: " + question + "\n"
                "Passage: " + passage + "\n"
                "A: " + label
            )
        elif dataset_name == "PIQA":
            goal = str(item["goal"]).strip()
            sol1 = str(item["sol1"]).strip()
            sol2 = str(item["sol2"]).strip()
            label = to_label(item["label"]) + 1
            lines.append(
                "Q: " + goal + "\n"
                "Option 1: " + sol1 + "\n"
                "Option 2: " + sol2 + "\n"
                "A: " + str(label)
            )
        elif dataset_name == "WinoGrande":
            sentence = str(item["sentence"]).strip()
            opt1 = str(item["option1"]).strip()
            opt2 = str(item["option2"]).strip()
            label = str(item["answer"]).strip()
            lines.append(
                "Q: " + sentence + "\n"
                "Option 1: " + opt1 + "\n"
                "Option 2: " + opt2 + "\n"
                "A: " + label
            )
        elif dataset_name == "Hellaswag":
            ctx = str(item["ctx"]).strip()
            endings = [str(x).strip() for x in item["endings"]]
            label_idx = int(item["label"])
            lines.append(
                "Q: " + ctx + "\n"
                "Option 1: " + endings[0] + "\n"
                "Option 2: " + endings[1] + "\n"
                "Option 3: " + endings[2] + "\n"
                "Option 4: " + endings[3] + "\n"
                "A: " + str(label_idx + 1)
            )
        elif dataset_name in {"e2e_nlg", "viggo"}:
            mr = str(item["meaning_representation"]).strip()
            target = str(item["target"]).strip()
            lines.append(
                "MR: " + mr + "\n"
                "A: " + target
            )
        else:
            raise ValueError(f"Unsupported dataset for GRPO trainer: {dataset_name}")
    return "\n\n".join(lines)


def build_policy_prompt(seed_examples: str, shots: int, dataset_name: str) -> str:
    if dataset_name == "BoolQ":
        format_rules = "Q: <question>\\nPassage: <passage>\\nA: True/False"
    elif dataset_name in {"PIQA", "WinoGrande"}:
        format_rules = "Q: <question>\\nOption 1: <text>\\nOption 2: <text>\\nA: 1 or 2"
    elif dataset_name == "Hellaswag":
        format_rules = "Q: <context>\\nOption 1: <text>\\nOption 2: <text>\\nOption 3: <text>\\nOption 4: <text>\\nA: 1/2/3/4"
    elif dataset_name in {"e2e_nlg", "viggo"}:
        format_rules = "MR: <meaning_representation>\\nA: <generated sentence>"
    else:
        raise ValueError(f"Unsupported dataset for GRPO trainer: {dataset_name}")

    return (
        f"You are generating in-context demonstrations for {dataset_name}.\n"
        f"Generate exactly {shots} demonstrations with this format:\n"
        f"{format_rules}\n\n"
        "Keep each demo concise. Do not add explanations.\n"
        "Candidate examples:\n"
        f"{seed_examples}\n\n"
        "Now output demonstrations only:\n"
    )


def build_off_policy_prompt(base_prompt: str, retrieved_demos: List[str]) -> str:
    if not retrieved_demos:
        return base_prompt
    demo_blob = "\n\n".join(retrieved_demos)
    return (
        "High-reward demonstration blocks from previous rollouts:\n"
        f"{demo_blob}\n\n"
        f"{base_prompt}"
    )


def call_openai_chat_completion(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Generate concise in-context demonstrations only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError):
        return ""


def build_openai_off_policy_prompt(
    on_prompt: str,
    seed_examples: str,
    retrieved_demo_blocks: List[str],
    dataset_name: str,
    args: argparse.Namespace,
    openai_api_key: str,
) -> Tuple[str, dict]:
    info = {
        "openai_calls": 0,
        "openai_success": 0,
        "hint_valid": 0,
        "hint_used": 0,
        "fallback": 0,
        "reject_reason": "",
        "a1_count": 0,
        "a2_count": 0,
    }

    if args.hint_provider != "openai" or not openai_api_key:
        return build_off_policy_prompt(on_prompt, retrieved_demo_blocks), info

    cache_blob = "\n\n".join(retrieved_demo_blocks) if retrieved_demo_blocks else "(none)"
    user_prompt = (
        f"Dataset: {dataset_name}\n"
        "Create a short demonstration block to improve reasoning and answer quality.\n"
        "Use the same answer format as the examples.\n"
        "For multiple-choice tasks, include both answer labels across the block when possible.\n"
        "Do not explain meta-reasoning. Output demos only.\n\n"
        "Current high-reward demo cache:\n"
        f"{cache_blob}\n\n"
        "Current sampled examples:\n"
        f"{seed_examples}\n"
    )

    def validate_hint_block(block_text: str) -> Tuple[bool, str, int, int]:
        if not block_text.strip():
            return False, "empty", 0, 0
        if dataset_name != "PIQA":
            return True, "", 0, 0

        q_count = len(re.findall(r"(?m)^Q:\s+", block_text))
        o1_count = len(re.findall(r"(?m)^Option\s*1:\s+", block_text))
        o2_count = len(re.findall(r"(?m)^Option\s*2:\s+", block_text))
        a_vals = re.findall(r"(?m)^A:\s*([^\n]+)", block_text)
        a1_count = sum(1 for x in a_vals if re.search(r"\b(1|option\s*1|a)\b", x.strip().lower()))
        a2_count = sum(1 for x in a_vals if re.search(r"\b(2|option\s*2|b)\b", x.strip().lower()))

        min_count = max(2, args.shots)
        if q_count < min_count or o1_count < min_count or o2_count < min_count or len(a_vals) < min_count:
            return False, "format", a1_count, a2_count
        if a1_count == 0 or a2_count == 0:
            return False, "piqa_answer_imbalance", a1_count, a2_count
        return True, "", a1_count, a2_count

    best_hint = ""
    retries = max(1, args.openai_hint_retries)
    for _ in range(retries):
        info["openai_calls"] += 1
        hint_block = call_openai_chat_completion(
            api_key=openai_api_key,
            model=args.openai_model,
            prompt=user_prompt,
            max_tokens=args.openai_max_tokens,
            temperature=max(0.2, args.temperature),
        )
        if hint_block:
            info["openai_success"] += 1
        is_valid, reason, a1_count, a2_count = validate_hint_block(hint_block)
        info["a1_count"] = a1_count
        info["a2_count"] = a2_count
        if is_valid:
            info["hint_valid"] = 1
            best_hint = hint_block
            break
        if reason:
            info["reject_reason"] = reason

    if best_hint:
        info["hint_used"] = 1
        return build_off_policy_prompt(on_prompt, [best_hint] + retrieved_demo_blocks), info

    info["fallback"] = 1
    return build_off_policy_prompt(on_prompt, retrieved_demo_blocks), info


def get_demo_blocks(cache_entries: List[dict], n_demo: int, strategy: str) -> List[str]:
    if not cache_entries or n_demo <= 0:
        return []
    if strategy == "reward":
        sorted_entries = sorted(cache_entries, key=lambda item: item["reward"], reverse=True)
        return [item["text"] for item in sorted_entries[:n_demo]]
    pick_count = min(n_demo, len(cache_entries))
    return [item["text"] for item in random.sample(cache_entries, k=pick_count)]


def build_eval_prompt(icl_text: str, row: dict, dataset_name: str) -> str:
    if dataset_name == "BoolQ":
        question = str(row["question"]).strip()
        passage = str(row["passage"]).strip()
        body = f"Q: {question}\\nPassage: {passage}\\nA:"
    elif dataset_name == "PIQA":
        goal = str(row["goal"]).strip()
        sol1 = str(row["sol1"]).strip()
        sol2 = str(row["sol2"]).strip()
        body = f"Q: {goal}\\nOption 1: {sol1}\\nOption 2: {sol2}\\nA:"
    elif dataset_name == "WinoGrande":
        sentence = str(row["sentence"]).strip()
        opt1 = str(row["option1"]).strip()
        opt2 = str(row["option2"]).strip()
        body = f"Q: {sentence}\\nOption 1: {opt1}\\nOption 2: {opt2}\\nA:"
    elif dataset_name == "Hellaswag":
        ctx = str(row["ctx"]).strip()
        endings = [str(x).strip() for x in row["endings"]]
        body = (
            f"Q: {ctx}\\n"
            f"Option 1: {endings[0]}\\n"
            f"Option 2: {endings[1]}\\n"
            f"Option 3: {endings[2]}\\n"
            f"Option 4: {endings[3]}\\n"
            "A:"
        )
    elif dataset_name in {"e2e_nlg", "viggo"}:
        mr = str(row["meaning_representation"]).strip()
        body = f"MR: {mr}\\nA:"
    else:
        raise ValueError(f"Unsupported dataset for GRPO trainer: {dataset_name}")

    return f"{icl_text.strip()}\\n\\n{body}"


def reward_candidate(
    icl_text: str,
    eval_rows: List[dict],
    target_model,
    target_tokenizer,
    device: torch.device,
    dataset_name: str,
) -> float:
    task_type = dataset_task(dataset_name)
    correct = 0
    reward_sum = 0.0
    for row in eval_rows:
        prompt = build_eval_prompt(icl_text, row, dataset_name)
        enc = target_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
        enc = {k: v.to(device) for k, v in enc.items()}
        max_new = 64 if task_type == "generation" else 4
        with torch.no_grad():
            out = target_model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=target_tokenizer.pad_token_id,
                eos_token_id=target_tokenizer.eos_token_id,
            )
        gen_text = target_tokenizer.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)
        if dataset_name == "BoolQ":
            pred = parse_bool_prediction(gen_text)
            label = to_label(row["answer"])
        elif dataset_name == "PIQA":
            pred = parse_mc_prediction(gen_text)
            label = to_label(row["label"])
        elif dataset_name == "WinoGrande":
            pred = parse_mc_prediction(gen_text)
            label = int(str(row["answer"]).strip()) - 1
        elif dataset_name == "Hellaswag":
            pred = parse_mc_prediction(gen_text)
            label = int(row["label"])
        elif dataset_name in {"e2e_nlg", "viggo"}:
            label_text = str(row["target"]).strip()
            reward_sum += unigram_f1(gen_text, label_text)
            continue
        else:
            raise ValueError(f"Unsupported dataset for GRPO trainer: {dataset_name}")
        if pred == label:
            correct += 1
    if task_type == "generation":
        return reward_sum / max(1, len(eval_rows))
    return correct / max(1, len(eval_rows))


def capture_trainable_state(policy_model) -> dict:
    state = {}
    for name, param in policy_model.named_parameters():
        if param.requires_grad:
            state[name] = param.detach().cpu().clone()
    return state


def restore_trainable_state(policy_model, state: dict, device: torch.device) -> None:
    for name, param in policy_model.named_parameters():
        if not param.requires_grad:
            continue
        if name in state:
            param.data.copy_(state[name].to(device))


def sequence_logprob(policy_model, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    outputs = policy_model(input_ids=full_ids)
    logits = outputs.logits[:, :-1, :]
    targets = full_ids[:, 1:]

    start = max(0, prompt_len - 1)
    logits = logits[:, start:, :]
    targets = targets[:, start:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.mean()


def freeze_for_lightweight_update(policy_model) -> int:
    for param in policy_model.parameters():
        param.requires_grad = False

    trainable = 0
    for name, param in policy_model.named_parameters():
        if "norm" in name and param.ndim == 1:
            param.requires_grad = True
            trainable += param.numel()

    return trainable


def sample_rows(dataset_split, count: int) -> List[dict]:
    indices = random.sample(range(len(dataset_split)), k=min(count, len(dataset_split)))
    return [dataset_split[i] for i in indices]


def build_fixed_eval_pool(dataset_split, total_count: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    indices = list(range(len(dataset_split)))
    rng.shuffle(indices)
    selected = indices[: min(total_count, len(indices))]
    return [dataset_split[i] for i in selected]


def take_eval_slice(pool: List[dict], start: int, size: int) -> List[dict]:
    if not pool:
        return []
    end = start + size
    if end <= len(pool):
        return pool[start:end]
    wrap = end - len(pool)
    return pool[start:] + pool[:wrap]


def evaluate_policy(
    train_split,
    val_split,
    policy_model,
    policy_tokenizer,
    target_model,
    target_tokenizer,
    device: torch.device,
    episodes: int,
    shots: int,
    seed_pool_size: int,
    reward_batch_size: int,
    max_new_tokens: int,
    temperature: float,
    dataset_name: str,
    eval_pool: List[dict] | None = None,
) -> float:
    rewards = []
    policy_model.eval()

    for _ in range(episodes):
        seed_rows = sample_rows(train_split, seed_pool_size)
        seed_examples = format_seed_examples(seed_rows, shots=shots, dataset_name=dataset_name)
        prompt = build_policy_prompt(seed_examples, shots=shots, dataset_name=dataset_name)

        enc = policy_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            generated = policy_model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=policy_tokenizer.pad_token_id,
                eos_token_id=policy_tokenizer.eos_token_id,
            )

        text = policy_tokenizer.decode(generated[0], skip_special_tokens=True)
        candidate = text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()

        if eval_pool:
            start = (_ * reward_batch_size) % max(1, len(eval_pool))
            eval_rows = take_eval_slice(eval_pool, start, reward_batch_size)
        else:
            eval_rows = sample_rows(val_split, reward_batch_size)
        reward = reward_candidate(candidate, eval_rows, target_model, target_tokenizer, device, dataset_name)
        rewards.append(reward)

    return float(sum(rewards) / max(1, len(rewards)))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    openai_api_key = os.getenv(args.openai_api_key_env, "")
    if args.hint_provider == "openai" and not openai_api_key:
        print(f"[WARN] {args.openai_api_key_env} is not set; using cached offline hints only.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for GRPO training")
    device = torch.device("cuda:0")

    dataset = load_from_disk(args.dataset_path)
    dataset_name = infer_dataset_name(args.dataset_path, args.dataset_name)
    train_split = dataset["train"]
    val_split = dataset["validation"] if "validation" in dataset else dataset["test"]

    policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_model, trust_remote_code=True)
    if policy_tokenizer.pad_token is None:
        policy_tokenizer.pad_token = policy_tokenizer.eos_token

    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)

    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)
    target_model.eval()

    trainable_params = freeze_for_lightweight_update(policy_model)
    optimizer = torch.optim.AdamW(
        [param for param in policy_model.parameters() if param.requires_grad],
        lr=args.lr,
        eps=1e-6,
    )

    fixed_eval_pool = build_fixed_eval_pool(
        val_split,
        total_count=max(1, args.eval_episodes * args.reward_batch_size),
        seed=args.seed + 2026,
    )

    pre_reward = evaluate_policy(
        train_split,
        val_split,
        policy_model,
        policy_tokenizer,
        target_model,
        target_tokenizer,
        device,
        episodes=args.eval_episodes,
        shots=args.shots,
        seed_pool_size=args.seed_pool_size,
        reward_batch_size=args.reward_batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        dataset_name=dataset_name,
        eval_pool=fixed_eval_pool,
    )
    initial_state = capture_trainable_state(policy_model)

    step_logs = []
    best_step_eval_reward = float("-inf")
    best_step = 0
    best_state = capture_trainable_state(policy_model)
    demo_cache_entries: List[dict] = []
    openai_stats = {
        "calls": 0,
        "success": 0,
        "valid": 0,
        "used": 0,
        "fallback": 0,
        "reject_reasons": {},
        "piqa_a1_total": 0,
        "piqa_a2_total": 0,
    }

    for step in range(1, args.steps + 1):
        policy_model.eval()

        seed_rows = sample_rows(train_split, args.seed_pool_size)
        seed_examples = format_seed_examples(seed_rows, shots=args.shots, dataset_name=dataset_name)
        on_prompt = build_policy_prompt(seed_examples, shots=args.shots, dataset_name=dataset_name)

        retrieved_demo_blocks = get_demo_blocks(
            demo_cache_entries,
            n_demo=args.n_demo,
            strategy=args.demo_match_strategy,
        )
        off_prompt, hint_info = build_openai_off_policy_prompt(
            on_prompt=on_prompt,
            seed_examples=seed_examples,
            retrieved_demo_blocks=retrieved_demo_blocks,
            dataset_name=dataset_name,
            args=args,
            openai_api_key=openai_api_key,
        )
        openai_stats["calls"] += hint_info["openai_calls"]
        openai_stats["success"] += hint_info["openai_success"]
        openai_stats["valid"] += hint_info["hint_valid"]
        openai_stats["used"] += hint_info["hint_used"]
        openai_stats["fallback"] += hint_info["fallback"]
        openai_stats["piqa_a1_total"] += hint_info["a1_count"]
        openai_stats["piqa_a2_total"] += hint_info["a2_count"]
        if hint_info["reject_reason"]:
            reason = hint_info["reject_reason"]
            openai_stats["reject_reasons"][reason] = openai_stats["reject_reasons"].get(reason, 0) + 1

        off_count = int(round(args.group_size * args.off_policy_ratio))
        off_count = max(0, min(args.group_size, off_count))
        on_count = max(0, args.group_size - off_count)

        candidate_specs = ([{"prompt": on_prompt, "hinted": 0}] * on_count) + ([{"prompt": off_prompt, "hinted": 1}] * off_count)
        if not candidate_specs:
            candidate_specs = [{"prompt": on_prompt, "hinted": 0}]

        candidates: List[Tuple[str, torch.Tensor, float, int, int, float]] = []
        eval_rows = sample_rows(val_split, args.reward_batch_size)

        for idx, spec in enumerate(candidate_specs):
            prompt = spec["prompt"]
            hinted = int(spec["hinted"])
            enc = policy_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            enc = {k: v.to(device) for k, v in enc.items()}
            prompt_len = enc["input_ids"].shape[1]

            with torch.no_grad():
                generated = policy_model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=max(0.2, args.temperature + (0.03 * idx)),
                    top_p=0.9,
                    pad_token_id=policy_tokenizer.pad_token_id,
                    eos_token_id=policy_tokenizer.eos_token_id,
                )

            text = policy_tokenizer.decode(generated[0], skip_special_tokens=True)
            candidate = text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
            reward = reward_candidate(candidate, eval_rows, target_model, target_tokenizer, device, dataset_name)
            with torch.no_grad():
                old_logp = sequence_logprob(policy_model, generated, prompt_len).detach().item()
            candidates.append((candidate, generated[0].detach(), reward, hinted, prompt_len, old_logp))

            if reward >= args.cache_min_reward:
                demo_cache_entries.append({"text": candidate, "reward": float(reward)})
                if len(demo_cache_entries) > 4000:
                    if args.demo_match_strategy == "reward":
                        demo_cache_entries = sorted(demo_cache_entries, key=lambda item: item["reward"], reverse=True)[:2000]
                    else:
                        demo_cache_entries = random.sample(demo_cache_entries, k=2000)

        rewards = torch.tensor([item[2] for item in candidates], dtype=torch.float32, device=device)
        advantages = rewards - rewards.mean()
        if not args.no_adv_normalize:
            advantages = advantages / (rewards.std() + 1e-6)
        advantages = advantages.clamp(-args.adv_clip, args.adv_clip)

        policy_model.train()
        total_loss_value = 0.0
        ratio_mean_value = 1.0
        ratio_clip_fraction = 0.0

        for _ in range(max(1, args.ppo_epochs)):
            optimizer.zero_grad()

            losses = []
            ratio_values = []
            ratio_clipped_flags = []

            for idx, (_, full_ids, _, hinted, prompt_len, old_logp_value) in enumerate(candidates):
                full_ids = full_ids.unsqueeze(0).to(device)
                new_logp = sequence_logprob(policy_model, full_ids, prompt_len)
                old_logp = torch.tensor(old_logp_value, dtype=torch.float32, device=device)

                ratio = torch.exp(torch.clamp(new_logp - old_logp, min=-20.0, max=20.0))
                clipped_ratio = torch.clamp(ratio, 1.0 - args.ppo_clip_range, 1.0 + args.ppo_clip_range)
                surrogate = torch.minimum(ratio * advantages[idx], clipped_ratio * advantages[idx])

                if hinted == 1:
                    hinted_gate = ratio / (ratio + max(1e-6, args.hinted_ratio_denom))
                    surrogate = surrogate * hinted_gate * args.hinted_loss_scale

                loss = -surrogate
                if args.kl_beta > 0:
                    loss = loss + (args.kl_beta * ((new_logp - old_logp) ** 2))

                losses.append(loss)
                ratio_values.append(ratio.detach())
                ratio_clipped_flags.append((torch.abs(ratio - clipped_ratio) > 1e-6).float().detach())

            total_loss = torch.stack(losses).mean()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [param for param in policy_model.parameters() if param.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()

            total_loss_value = float(total_loss.item())
            ratio_mean_value = float(torch.stack(ratio_values).mean().item())
            ratio_clip_fraction = float(torch.stack(ratio_clipped_flags).mean().item())

        step_eval_reward = evaluate_policy(
            train_split,
            val_split,
            policy_model,
            policy_tokenizer,
            target_model,
            target_tokenizer,
            device,
            episodes=args.step_eval_episodes,
            shots=args.shots,
            seed_pool_size=args.seed_pool_size,
            reward_batch_size=args.reward_batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            dataset_name=dataset_name,
            eval_pool=fixed_eval_pool,
        )

        if step_eval_reward > best_step_eval_reward:
            best_step_eval_reward = step_eval_reward
            best_step = step
            best_state = capture_trainable_state(policy_model)

        step_log = {
            "step": step,
            "mean_reward": float(rewards.mean().item()),
            "best_reward": float(rewards.max().item()),
            "loss": total_loss_value,
            "ratio_mean": ratio_mean_value,
            "ratio_clip_fraction": ratio_clip_fraction,
            "step_eval_reward": float(step_eval_reward),
            "demo_cache_size": len(demo_cache_entries),
            "n_on_policy": on_count,
            "n_off_policy": off_count,
            "openai_hint_used": hint_info["hint_used"],
            "openai_hint_valid": hint_info["hint_valid"],
            "openai_fallback": hint_info["fallback"],
            "openai_reject_reason": hint_info["reject_reason"],
        }
        step_logs.append(step_log)
        print(
            f"[STEP {step}] mean_reward={step_log['mean_reward']:.4f} "
            f"best_reward={step_log['best_reward']:.4f} "
            f"step_eval={step_log['step_eval_reward']:.4f} "
            f"loss={step_log['loss']:.4f} "
            f"ratio={step_log['ratio_mean']:.4f} "
            f"clipfrac={step_log['ratio_clip_fraction']:.4f}"
        )

    restore_trainable_state(policy_model, best_state, device)

    post_reward = evaluate_policy(
        train_split,
        val_split,
        policy_model,
        policy_tokenizer,
        target_model,
        target_tokenizer,
        device,
        episodes=args.eval_episodes,
        shots=args.shots,
        seed_pool_size=args.seed_pool_size,
        reward_batch_size=args.reward_batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        dataset_name=dataset_name,
        eval_pool=fixed_eval_pool,
    )

    raw_post_reward = post_reward
    raw_delta_reward = raw_post_reward - pre_reward

    fallback_to_initial = False
    if (not args.allow_regression) and post_reward < pre_reward:
        restore_trainable_state(policy_model, initial_state, device)
        post_reward = pre_reward
        fallback_to_initial = True

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name = dataset_name.lower().replace("-", "_")
    report_path = output_dir / f"grpo_{safe_name}_report_{timestamp}.json"

    report = {
        "policy_model": args.policy_model,
        "target_model": args.target_model,
        "dataset_name": dataset_name,
        "dataset_path": args.dataset_path,
        "steps": args.steps,
        "group_size": args.group_size,
        "ppo_epochs": args.ppo_epochs,
        "ppo_clip_range": args.ppo_clip_range,
        "hinted_loss_scale": args.hinted_loss_scale,
        "adv_normalize": not args.no_adv_normalize,
        "hint_provider": args.hint_provider,
        "openai_model": args.openai_model if args.hint_provider == "openai" else "",
        "openai_api_key_set": bool(openai_api_key),
        "openai_hint_stats": openai_stats,
        "trainable_params": trainable_params,
        "best_step": best_step,
        "best_step_eval_reward": best_step_eval_reward,
        "pre_reward": pre_reward,
        "raw_post_reward": raw_post_reward,
        "raw_delta_reward": raw_delta_reward,
        "post_reward": post_reward,
        "delta_reward": post_reward - pre_reward,
        "fallback_to_initial": fallback_to_initial,
        "step_logs": step_logs,
    }

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("[DONE] GRPO-style training finished")
    print(f"Report: {report_path}")
    print(f"Pre reward:  {pre_reward:.4f}")
    print(f"Post reward: {post_reward:.4f}")
    print(f"Delta:       {post_reward - pre_reward:+.4f}")


if __name__ == "__main__":
    main()
