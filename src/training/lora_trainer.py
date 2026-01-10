"""LoRA/QLoRA trainer for fine-tuning student models."""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from loguru import logger


class LoRATrainer:
    """Trainer for fine-tuning models with LoRA/QLoRA."""

    def __init__(self, config_path: str):
        """
        Initialize LoRA trainer.

        Args:
            config_path: Path to training configuration YAML
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.model = None
        self.tokenizer = None
        self.trainer = None

        logger.info(f"LoRA Trainer initialized with config: {config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def setup_model(self) -> None:
        """Setup base model with quantization and LoRA."""
        model_config = self.config["model"]
        quant_config = self.config["quantization"]
        lora_config = self.config["lora"]

        logger.info(f"Loading base model: {model_config['base_model']}")

        # Configure quantization
        if quant_config.get("enabled", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=quant_config.get("load_in_4bit", True),
                bnb_4bit_compute_dtype=getattr(
                    torch, quant_config.get("bnb_4bit_compute_dtype", "float16")
                ),
                bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=quant_config.get(
                    "bnb_4bit_use_double_quant", True
                ),
            )
        else:
            bnb_config = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config["base_model"],
            quantization_config=bnb_config,
            device_map=model_config.get("device_map", "auto"),
            trust_remote_code=model_config.get("trust_remote_code", False),
            torch_dtype=getattr(torch, model_config.get("torch_dtype", "float16")),
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config["base_model"],
            trust_remote_code=model_config.get("trust_remote_code", False),
        )

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # Prepare model for k-bit training
        if quant_config.get("enabled", False):
            self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            task_type=lora_config["task_type"],
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        logger.info("Model setup complete")

    def prepare_dataset(self, dataset_path: str) -> tuple:
        """
        Prepare dataset for training.

        Args:
            dataset_path: Path to JSONL dataset file

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        logger.info(f"Loading dataset from {dataset_path}")

        # Load dataset
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        data_config = self.config["data"]
        max_seq_length = data_config.get("max_seq_length", 2048)

        # Tokenize function
        def tokenize_function(examples):
            # Use the 'text' field which contains the formatted prompt
            texts = examples["text"]

            # Tokenize
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )

            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()

            return tokenized

        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )

        # Split into train/val
        dataset_params = self.config.get("dataset", {})
        train_split = dataset_params.get("train_split", 0.9)

        split_dataset = tokenized_dataset.train_test_split(
            test_size=1 - train_split,
            seed=self.config["training"].get("seed", 42),
        )

        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

        logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

        return train_dataset, eval_dataset

    def train(
        self,
        train_dataset,
        eval_dataset,
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Output directory (overrides config)
        """
        training_config = self.config["training"]

        # Override output dir if provided
        if output_dir:
            training_config["output_dir"] = output_dir

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=training_config["output_dir"],
            num_train_epochs=training_config["num_train_epochs"],
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
            gradient_checkpointing=training_config.get("gradient_checkpointing", True),
            optim=training_config["optim"],
            learning_rate=training_config["learning_rate"],
            weight_decay=training_config["weight_decay"],
            lr_scheduler_type=training_config["lr_scheduler_type"],
            warmup_ratio=training_config.get("warmup_ratio", 0.03),
            logging_steps=training_config["logging_steps"],
            logging_dir=training_config.get("logging_dir", "./logs"),
            eval_strategy=training_config.get("evaluation_strategy", "steps"),  # Changed from evaluation_strategy
            eval_steps=training_config.get("eval_steps", 100),
            save_strategy=training_config["save_strategy"],
            save_steps=training_config["save_steps"],
            save_total_limit=training_config.get("save_total_limit", 3),
            load_best_model_at_end=training_config.get("load_best_model_at_end", True),
            metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
            fp16=training_config.get("fp16", False),
            bf16=training_config.get("bf16", False),
            tf32=training_config.get("tf32", True),
            max_grad_norm=training_config.get("max_grad_norm", 1.0),
            seed=training_config.get("seed", 42),
            data_seed=training_config.get("data_seed", 42),
            group_by_length=training_config.get("group_by_length", True),
            dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
            report_to=self._get_report_to(),
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # Start training
        logger.info("Starting training...")
        self.trainer.train()

        logger.info("Training complete!")

    def save_model(self, output_dir: Optional[str] = None) -> None:
        """
        Save the trained model.

        Args:
            output_dir: Output directory (uses config default if not provided)
        """
        if output_dir is None:
            output_dir = self.config["training"]["output_dir"]

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        logger.info(f"Model saved to {output_path}")

    def _get_report_to(self) -> list:
        """Get reporting destinations based on config."""
        monitoring_config = self.config.get("monitoring", {})
        report_to = []

        if monitoring_config.get("use_wandb", False):
            report_to.append("wandb")

        if monitoring_config.get("use_tensorboard", True):
            report_to.append("tensorboard")

        if not report_to:
            report_to.append("none")

        return report_to


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Train model with LoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Training config path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to training dataset (JSONL)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config)",
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = LoRATrainer(args.config)

    # Setup model
    trainer.setup_model()

    # Prepare dataset
    train_ds, eval_ds = trainer.prepare_dataset(args.dataset)

    # Train
    trainer.train(train_ds, eval_ds, output_dir=args.output_dir)

    # Save
    trainer.save_model(args.output_dir)

    logger.info("Training pipeline complete!")
