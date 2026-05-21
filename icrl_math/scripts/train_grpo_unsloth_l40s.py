"""Unsloth + TRL GRPO trainer, tuned for a single L40S 40GB.

Why this exists alongside `train_curriculum_math.sh`:
  - The verl-based curriculum trainer assumes 4×A100/A6000 with FSDP, vLLM
    rollout, and multi-turn tool use. It does NOT fit on one L40S 40GB.
  - This script is a memory-efficient, single-GPU GRPO baseline using:
      * 4-bit (bnb) weights via Unsloth `FastLanguageModel`
      * vLLM fast inference for rollouts (`fast_inference=True`)
      * LoRA rank 8 on q_proj/v_proj only (cheapest target set)
      * num_generations=2  (smallest non-trivial GRPO group)
      * gradient_accumulation_steps=4
      * paged_adamw_8bit, bf16
  - It is a single-stage GRPO (no curriculum, no tool use, no sandbox call).
    Use this to first prove the GRPO loop runs on our hardware, then scale up.

Phase plan (run in order, only move forward when previous phase converges):
  Phase A — this script as-is (3B, 4bit, GSM8K, XML <answer>, no tool)
  Phase B — same script with our 5 math demos prepended to system prompt
  Phase C — multi-stage curriculum (3→2→0 shot) on top of B
  Phase D — once stable, port to verl/FSDP on 4+ GPUs for tool-use rollouts

Defaults below match the user's L40S baseline. Tighten if OOM in this order:
    num_generations: 2 -> 2 (already min)
    max_completion_length: 512 -> 384 -> 256
    max_seq_length: 1024 -> 768 -> 640
    gpu_memory_utilization: 0.55 -> 0.50 -> 0.45
    target_modules: q,v only (already min)
"""

import os
# Pin to GPU 1 (user requested). Must be set BEFORE torch / unsloth import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
# Reduce CUDA fragmentation under variable-length sequences.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Unsloth-recommended toggle for vLLM standby memory in RL.
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

import argparse
import re
from pathlib import Path

import torch
from datasets import load_dataset

# Unsloth must be imported before transformers/trl so it can monkey-patch them.
from unsloth import FastLanguageModel  # noqa: E402

from trl import GRPOConfig, GRPOTrainer  # noqa: E402


# ---------------------------------------------------------------------------
# Prompt & answer format (single-turn, no tool — matches Phase A)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_BASE = """You are a careful step-by-step math reasoner.

Respond in EXACTLY this format and nothing else:

<reasoning>
your step-by-step reasoning here
</reasoning>
<answer>
the final numeric answer here, e.g. 42
</answer>
"""


def build_system_prompt(demos_file=None):
    """Optionally prepend few-shot demos (ICRL-style) to the system prompt."""
    if not demos_file:
        return SYSTEM_PROMPT_BASE
    from pathlib import Path
    text = Path(demos_file).read_text(encoding="utf-8").strip()
    return (
        SYSTEM_PROMPT_BASE
        + "\nHere are some worked examples you can refer to:\n\n"
        + text
        + "\n\nNow solve the new problem using the same format. Do NOT reuse problems from the examples."
    )

_HASH_RE = re.compile(r"####\s*(\-?[0-9\.,]+)")
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_REASONING_RE = re.compile(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", re.DOTALL)
_STRICT_RE = re.compile(r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n?$", re.DOTALL)


def _extract_hash_answer(text):
    m = _HASH_RE.search(text or "")
    return m.group(1).replace(",", "").strip() if m else None


def _extract_xml_answer(text):
    m = _ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else ""


def _completion_text(c):
    """TRL passes completions as either a list[{role,content}] or a plain string."""
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return c[0].get("content", "")
    return str(c)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


_CORRECTNESS_W = 2.0
_INT_W = 0.5
_SOFT_FMT_W = 0.5
_STRICT_FMT_W = 0.5


def set_reward_weights(correctness, integer, soft_fmt, strict_fmt):
    """Override the (otherwise hardcoded) reward weights at runtime."""
    global _CORRECTNESS_W, _INT_W, _SOFT_FMT_W, _STRICT_FMT_W
    _CORRECTNESS_W = correctness
    _INT_W = integer
    _SOFT_FMT_W = soft_fmt
    _STRICT_FMT_W = strict_fmt


def correctness_reward(prompts, completions, answer, **kw):
    """correctness weight if predicted answer string-matches gold (after light normalization)."""
    out = []
    for c, gold in zip(completions, answer):
        pred = _extract_xml_answer(_completion_text(c))
        ok = False
        if pred and gold is not None:
            p, g = pred.replace(",", "").strip(), str(gold).replace(",", "").strip()
            if p == g:
                ok = True
            else:
                try:
                    ok = abs(float(p) - float(g)) < 1e-6
                except ValueError:
                    pass
        out.append(_CORRECTNESS_W if ok else 0.0)
    return out


def int_reward(completions, **kw):
    """int_weight if extracted answer parses as an integer (encourages clean numeric output)."""
    return [_INT_W if _extract_xml_answer(_completion_text(c)).replace("-", "").isdigit() else 0.0
            for c in completions]


def soft_format_reward(completions, **kw):
    """soft_fmt weight if response contains <reasoning>...</reasoning><answer>...</answer> somewhere."""
    return [_SOFT_FMT_W if _REASONING_RE.search(_completion_text(c)) else 0.0 for c in completions]


def strict_format_reward(completions, **kw):
    """strict_fmt weight if response matches the strict newline-delimited template exactly."""
    return [_STRICT_FMT_W if _STRICT_RE.match(_completion_text(c)) else 0.0 for c in completions]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_dataset(split, system_prompt):
    ds = load_dataset("openai/gsm8k", "main")[split]

    def convert(x):
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "answer": _extract_hash_answer(x["answer"]),
        }
    return ds.map(convert)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
                   help="HF model id. Fallbacks: 'Qwen/Qwen2.5-3B-Instruct' (no 4bit), "
                        "'unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit', etc.")
    p.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "grpo_l40s"))

    # Sequence budgets (most-likely OOM knobs)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--max-prompt-length", type=int, default=256)
    p.add_argument("--max-completion-length", type=int, default=512)

    # LoRA
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-targets", default="q_proj,v_proj",
                   help="comma list; 'all' = qkvo+gate+up+down")

    # GRPO
    p.add_argument("--num-generations", type=int, default=2)
    p.add_argument("--per-device-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--gpu-mem-util", type=float, default=0.55)
    p.add_argument("--seed", type=int, default=3407)

    # Smoke
    p.add_argument("--smoke", action="store_true",
                   help="set max_steps=2, save_steps=2, train_data_num=8 for a fast OOM check")

    # ICRL-style few-shot demos in the system prompt
    p.add_argument("--demos-file", default=None,
                   help="path to a math_examples.txt; if set, demos are prepended to system prompt")

    # Resume LoRA from a previously saved adapter (for curriculum stages)
    p.add_argument("--resume-from-lora", default=None,
                   help="path to a previously saved LoRA adapter directory; if set, train continues "
                        "from those weights instead of fresh random LoRA init")

    # Reward weights (rebalanced in Phase B to lean on correctness)
    p.add_argument("--correctness-weight", type=float, default=2.0)
    p.add_argument("--int-weight", type=float, default=0.5)
    p.add_argument("--soft-format-weight", type=float, default=0.5)
    p.add_argument("--strict-format-weight", type=float, default=0.5)

    args = p.parse_args()

    if args.smoke:
        args.max_steps = 2
        args.save_steps = 2

    if args.lora_targets == "all":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = [t.strip() for t in args.lora_targets.split(",") if t.strip()]

    print(f"[boot] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}  "
          f"device_count={torch.cuda.device_count()}  "
          f"current={torch.cuda.current_device() if torch.cuda.is_available() else '-'}")
    print(f"[boot] model={args.model}")
    print(f"[boot] seq_len={args.max_seq_length} prompt={args.max_prompt_length} comp={args.max_completion_length}")
    print(f"[boot] LoRA rank={args.lora_rank} targets={target_modules}")
    print(f"[boot] GRPO num_gen={args.num_generations} batch={args.per_device_batch} accum={args.grad_accum} "
          f"steps={args.max_steps}")
    print(f"[boot] demos_file={args.demos_file}")
    print(f"[boot] reward weights: correctness={args.correctness_weight} int={args.int_weight} "
          f"soft_fmt={args.soft_format_weight} strict_fmt={args.strict_format_weight}")

    # Install reward weights and (optionally) demos into the system prompt.
    set_reward_weights(
        args.correctness_weight,
        args.int_weight,
        args.soft_format_weight,
        args.strict_format_weight,
    )
    system_prompt = build_system_prompt(args.demos_file)
    print(f"[boot] system_prompt len = {len(system_prompt)} chars")

    # 1) load 4bit model + vLLM rollout
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_mem_util,
    )

    # 2) LoRA — fresh or resumed
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=target_modules,
        lora_alpha=args.lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    if args.resume_from_lora:
        from safetensors.torch import load_file
        from pathlib import Path as _P
        adapter_path = _P(args.resume_from_lora) / "adapter_model.safetensors"
        if not adapter_path.exists():
            raise FileNotFoundError(f"adapter weights not found: {adapter_path}")
        print(f"[boot] resuming LoRA weights from {adapter_path}")
        prev = load_file(str(adapter_path))
        # Map saved keys (peft format: "base_model.model.<...>.lora_A.weight") onto our
        # fresh LoRA's parameters. Unsloth's get_peft_model wraps params the same way.
        own = {k: v for k, v in model.named_parameters() if "lora_" in k}
        n_match = 0
        for k, v in prev.items():
            # PEFT saves "...lora_A.weight" but the runtime LoRA names include the
            # adapter id: "...lora_A.default.weight". Try both, then the bare key.
            candidates = [k.replace(".weight", ".default.weight"), k]
            for ck in candidates:
                if ck in own:
                    own[ck].data.copy_(v.to(own[ck].device, dtype=own[ck].dtype))
                    n_match += 1
                    break
        print(f"[boot] copied {n_match}/{len(prev)} adapter tensors into the fresh LoRA")
        if n_match == 0:
            print("[boot] WARNING: 0 matches. saved keys sample:", list(prev.keys())[:3])
            print("[boot] own keys sample:", list(own.keys())[:3])
            raise RuntimeError("LoRA resume failed: no key overlap")

    # 3) data
    dataset = build_dataset("train", system_prompt)
    if args.smoke:
        dataset = dataset.select(range(min(8, len(dataset))))

    # 4) GRPO config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=True,
        fp16=False,
        logging_steps=1,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        max_grad_norm=0.1,
        report_to="none",
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            soft_format_reward,
            strict_format_reward,
            int_reward,
            correctness_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    lora_save = os.path.join(args.output_dir, "lora_final")
    model.save_lora(lora_save)
    print(f"[done] LoRA adapters saved -> {lora_save}")


if __name__ == "__main__":
    main()
