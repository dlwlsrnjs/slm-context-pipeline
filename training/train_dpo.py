"""
DPO Training Script for SLM Context Generator
Preference learning: 좋은 컨텍스트 vs 나쁜 컨텍스트
"""
import argparse
import logging
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


class DPOTrainer:
    """Direct Preference Optimization for Context Quality"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_config = config.get("student", {})
        self.train_config = config.get("training", {}).get("dpo", {})
    
    def load_data(self, data_path: Path):
        """Load DPO preference data"""
        from datasets import Dataset
        
        records = list(iter_jsonl(data_path))
        logger.info(f"Loaded {len(records)} preference pairs")
        
        # Format for DPO
        formatted = []
        for record in records:
            instruction = record.get("instruction", "")
            input_text = record.get("input", "")
            chosen = record.get("chosen", "")
            rejected = record.get("rejected", "")
            
            # Create prompt
            prompt = f"""<|im_start|>system
You are a context generator for small language models. Generate minimal sufficient context.
<|im_end|>
<|im_start|>user
{instruction}

{input_text}
<|im_end|>
<|im_start|>assistant
"""
            
            formatted.append({
                "prompt": prompt,
                "chosen": chosen + "<|im_end|>",
                "rejected": rejected + "<|im_end|>"
            })
        
        return Dataset.from_list(formatted)
    
    def train(
        self,
        train_data_path: Path,
        base_model_path: Path,
        output_dir: Path
    ):
        """Run DPO training"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from trl import DPOConfig, DPOTrainer as TRLDPOTrainer
        
        # Load tokenizer
        tokenizer_path = base_model_path if (base_model_path / "tokenizer_config.json").exists() else self.model_config.get("tokenizer_name", "Qwen/Qwen2.5-1.5B")
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model (SFT checkpoint)
        logger.info(f"Loading SFT model from: {base_model_path}")
        
        base_model_name = self.model_config.get("model_name", "Qwen/Qwen2.5-1.5B")
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights if exists
        if (base_model_path / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, str(base_model_path))
            model = model.merge_and_unload()
        
        # Reference model (same as base for DPO)
        ref_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load data
        dataset = self.load_data(train_data_path)
        
        # DPO config
        dpo_config = DPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=self.train_config.get("epochs", 1),
            per_device_train_batch_size=self.train_config.get("batch_size", 4),
            learning_rate=self.train_config.get("learning_rate", 1e-6),
            beta=self.train_config.get("beta", 0.1),
            max_length=self.model_config.get("max_length", 1024),
            max_prompt_length=512,
            logging_steps=10,
            save_steps=200,
            gradient_checkpointing=True,
            fp16=True,
            report_to="none",
            seed=self.config.get("training", {}).get("seed", 42),
            remove_unused_columns=False
        )
        
        # Create DPO trainer
        trainer = TRLDPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=dataset,
            processing_class=tokenizer
        )
        
        # Train
        logger.info("Starting DPO training...")
        trainer.train()
        
        # Save
        trainer.save_model(str(output_dir / "final"))
        tokenizer.save_pretrained(str(output_dir / "final"))
        
        logger.info(f"DPO model saved to {output_dir / 'final'}")
        
        return str(output_dir / "final")


def main():
    parser = argparse.ArgumentParser(description="DPO Training")
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    parser.add_argument("--data", type=str, help="DPO preference data path")
    parser.add_argument("--base-model", type=str, help="SFT model path")
    parser.add_argument("--output", type=str, help="Output directory")
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Load config
    config = load_yaml(args.config) if Path(args.config).exists() else {}
    
    # Paths
    train_data = Path(args.data or config.get("data", {}).get("train_dir", "data/train") + "/dpo_preferences.jsonl")
    base_model = Path(args.base_model or config.get("training", {}).get("output_dir", "outputs/models/sft") + "/final")
    output_dir = Path(args.output or config.get("training", {}).get("output_dir", "outputs/models") + "/dpo")
    
    ensure_dir(output_dir)
    
    # Check if base model exists
    if not base_model.exists():
        logger.error(f"Base model not found at {base_model}")
        logger.info("Run SFT training first: python -m training.train_sft")
        return
    
    # Check if DPO data exists
    if not train_data.exists():
        logger.error(f"DPO training data not found at {train_data}")
        logger.info("Run evaluation first to generate preference labels")
        return
    
    # Train
    trainer = DPOTrainer(config)
    model_path = trainer.train(train_data, base_model, output_dir)
    
    print(f"\nDPO training complete!")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
