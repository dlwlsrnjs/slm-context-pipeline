#!/usr/bin/env python3
"""
SFT training for math ICL generator (7B).

Key settings:
- Model : Qwen2.5-7B-Instruct (default)
- Epochs: 3
- Best-only checkpoint: save_strategy=epoch + load_best_model_at_end + save_total_limit=1
- Speed : gradient_checkpointing, bf16, dataloader workers, larger effective batch

Usage:
    python scripts/finetune_math_icl_sft.py \
        --dataset-dir icl_distill_math/hf_distilled_icl_dataset \
        --output-dir experiment_results/math_icl_sft/qwen7b_run1
"""
import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dataset-dir",
                   default="icl_distill_math/hf_distilled_icl_dataset")
    p.add_argument("--output-dir",
                   default="experiment_results/math_icl_sft/qwen7b_run1")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Per-device train batch size")
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Gradient accumulation → effective batch = batch * grad_accum * n_gpu")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--eval-split", type=float, default=0.05,
                   help="Fraction held out for eval (best-model selection)")
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--tune-strategy",
                   choices=["norms_lm_head", "full"], default="norms_lm_head",
                   help="norms_lm_head=light, full=full fine-tune")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def apply_tune_strategy(model, strategy: str) -> int:
    if strategy == "full":
        for param in model.parameters():
            param.requires_grad = True
        return sum(p.numel() for p in model.parameters())

    for param in model.parameters():
        param.requires_grad = False

    has_untied_lm_head = any("lm_head" in n for n, _ in model.named_parameters())
    output_key = "lm_head" if has_untied_lm_head else "embed_tokens"

    trainable = 0
    for name, param in model.named_parameters():
        if ("norm" in name and param.ndim == 1) or output_key in name:
            param.requires_grad = True
            trainable += param.numel()
    return trainable


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(args.dataset_dir)
    print(f"Dataset size: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    # Gradient checkpointing for memory efficiency on 7B
    if torch.cuda.is_available():
        model.gradient_checkpointing_enable()

    trainable = apply_tune_strategy(model, args.tune_strategy)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    def preprocess(example):
        prompt = example["prompt"].strip()
        target = example["target"].strip()

        prompt_ids = tokenizer(
            prompt + "\n", add_special_tokens=False).input_ids
        target_ids = tokenizer(
            target, add_special_tokens=False).input_ids

        input_ids = (prompt_ids + target_ids + [tokenizer.eos_token_id])[
            : args.max_length]

        n_prompt = min(len(prompt_ids), len(input_ids))
        labels = [-100] * n_prompt + \
            (target_ids + [tokenizer.eos_token_id])[: len(input_ids) - n_prompt]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    processed = dataset.map(
        preprocess,
        remove_columns=dataset.column_names,
        num_proc=4,
    )
    before = len(processed)
    processed = processed.filter(
        lambda ex: any(l != -100 for l in ex["labels"]),
        num_proc=4,
    )
    dropped = before - len(processed)
    print(f"Dropped {dropped} samples with all-masked labels "
          f"(prompt exceeded max_length); kept {len(processed)}")
    n_eval = max(1, int(len(processed) * args.eval_split))
    splits = processed.train_test_split(test_size=n_eval, seed=args.seed)
    print(f"Train: {len(splits['train'])}  Eval: {len(splits['test'])}")

    def collate_fn(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [tokenizer.pad_token_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=torch.cuda.is_available(),
        dataloader_num_workers=4,
        # ── Best model only ──────────────────────────────
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,          # 최고 성능 1개만 보관
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # ─────────────────────────────────────────────────
        logging_steps=args.logging_steps,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        data_collator=collate_fn,
    )

    print("[START] Training...")
    trainer.train()

    # Best model already loaded by load_best_model_at_end
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print(f"\n[DONE] Best model saved → {final_dir}")


if __name__ == "__main__":
    main()
