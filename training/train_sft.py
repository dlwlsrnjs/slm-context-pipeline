"""
SFT Training Script for SLM Context Generator
Structured context generation을 위한 supervised fine-tuning
"""
import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from utils.file_utils import load_yaml, iter_jsonl, ensure_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required packages are installed"""
    required = ["torch", "transformers", "datasets", "peft", "trl"]
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    return True


class SFTTrainer:
    """Supervised Fine-Tuning for Context Generation"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_config = config.get("student", {})
        self.train_config = config.get("training", {}).get("sft", {})
        
    def load_data(self, data_path: Path):
        """Load training data"""
        from datasets import Dataset
        
        records = list(iter_jsonl(data_path))
        logger.info(f"Loaded {len(records)} training examples")
        
        # Format for training
        formatted = []
        for record in records:
            instruction = record.get("instruction", "")
            input_text = record.get("input", "")
            output = record.get("output", "")
            
            # Create chat format
            text = f"""<|im_start|>system
You are a context generator for small language models. Generate minimal sufficient context.
<|im_end|>
<|im_start|>user
{instruction}

{input_text}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>"""
            
            formatted.append({"text": text})
        
        return Dataset.from_list(formatted)
    
    def train(self, train_data_path: Path, output_dir: Path):
        """Run SFT training"""
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        # Load model and tokenizer
        model_name = self.model_config.get("model_name", "Qwen/Qwen2.5-1.5B")
        tokenizer_name = self.model_config.get("tokenizer_name", model_name)
        
        logger.info(f"Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.train_config.get("fp16", True) else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Load and tokenize data
        dataset = self.load_data(train_data_path)
        
        def tokenize(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.model_config.get("max_length", 1024),
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text"]
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.train_config.get("epochs", 3),
            per_device_train_batch_size=self.train_config.get("batch_size", 8),
            gradient_accumulation_steps=self.train_config.get("gradient_accumulation_steps", 4),
            learning_rate=self.train_config.get("learning_rate", 2e-5),
            warmup_ratio=self.train_config.get("warmup_ratio", 0.1),
            logging_steps=10,
            save_steps=self.train_config.get("save_steps", 500),
            save_total_limit=3,
            fp16=self.train_config.get("fp16", True),
            gradient_checkpointing=self.train_config.get("gradient_checkpointing", True),
            report_to="none",
            seed=self.config.get("training", {}).get("seed", 42)
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        trainer.save_model(str(output_dir / "final"))
        tokenizer.save_pretrained(str(output_dir / "final"))
        
        logger.info(f"Model saved to {output_dir / 'final'}")
        
        return str(output_dir / "final")


def main():
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    parser.add_argument("--data", type=str, help="Training data path (overrides config)")
    parser.add_argument("--output", type=str, help="Output directory (overrides config)")
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Load config
    config = load_yaml(args.config) if Path(args.config).exists() else {}
    
    # Paths
    train_data = Path(args.data or config.get("data", {}).get("train_dir", "data/train") + "/sft_combined.jsonl")
    output_dir = Path(args.output or config.get("training", {}).get("output_dir", "outputs/models/sft"))
    
    ensure_dir(output_dir)
    
    # Train
    trainer = SFTTrainer(config)
    model_path = trainer.train(train_data, output_dir)
    
    print(f"\nSFT training complete!")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
