#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen ICL generator with distilled SFT data")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset-dir", default="icl_distill_data/hf_distilled_icl_dataset")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--train-split-ratio", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-strategy", choices=["no", "steps", "epoch"], default="no")
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune-strategy", choices=["norms", "norms_lm_head", "full"], default="norms_lm_head")
    parser.add_argument("--output-dir", default="experiment_results/qwen_icl_sft")
    return parser.parse_args()


def apply_tune_strategy(model, strategy: str) -> int:
    if strategy == "full":
        total = 0
        for param in model.parameters():
            param.requires_grad = True
            total += param.numel()
        return total

    for param in model.parameters():
        param.requires_grad = False

    trainable = 0
    for name, param in model.named_parameters():
        if "norm" in name and param.ndim == 1:
            param.requires_grad = True
            trainable += param.numel()

    if strategy == "norms_lm_head":
        for name, param in model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = True
                trainable += param.numel()

    return trainable


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(args.dataset_dir)
    if len(dataset) < 10:
        raise RuntimeError(f"Dataset too small: {len(dataset)} samples")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    trainable_params = apply_tune_strategy(model, args.tune_strategy)

    def preprocess(example):
        prompt = example["prompt"].strip()
        target = example["target"].strip()

        prompt_ids = tokenizer(prompt + "\n", add_special_tokens=False).input_ids
        target_ids = tokenizer(target, add_special_tokens=False).input_ids

        input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        input_ids = input_ids[: args.max_length]

        labels = [-100] * min(len(prompt_ids), len(input_ids))
        remaining = len(input_ids) - len(labels)
        labels += (target_ids + [tokenizer.eos_token_id])[:remaining]

        attn = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
        }

    processed = dataset.map(preprocess, remove_columns=dataset.column_names)
    split = processed.train_test_split(test_size=max(1, int(len(processed) * (1.0 - args.train_split_ratio))), seed=args.seed)

    def collate_fn(features):
        max_len = max(len(item["input_ids"]) for item in features)

        input_ids = []
        attention_mask = []
        labels = []

        for item in features:
            cur_len = len(item["input_ids"])
            pad_len = max_len - cur_len

            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        bf16=torch.cuda.is_available(),
        fp16=False,
        dataloader_num_workers=2,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        data_collator=collate_fn,
    )

    print(f"[INFO] Trainable params: {trainable_params}")
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    print("[DONE] Qwen ICL SFT completed")
    print(f"Model saved to: {output_dir / 'final'}")


if __name__ == "__main__":
    main()
