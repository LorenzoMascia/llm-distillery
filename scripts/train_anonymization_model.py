#!/usr/bin/env python3
"""
Train the anonymization model using LoRA fine-tuning.
Uses the training_config_1b_anonymization.yaml configuration.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.lora_trainer import LoRATrainer
from src.utils.logger import setup_logger, get_logger


def main():
    parser = argparse.ArgumentParser(
        description="Train anonymization model with LoRA"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/anonymization_training_data.jsonl",
        help="Path to training dataset (JSONL format)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config_1b_anonymization.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/anonymization_1b",
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logger
    setup_logger(level="DEBUG" if args.verbose else "INFO")
    logger = get_logger("anonymization_trainer")

    logger.info("Starting anonymization model training")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")

    # Check if dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {args.dataset}")
        logger.info("Please generate the dataset first using:")
        logger.info("  python scripts/generate_anonymization_dataset.py")
        sys.exit(1)

    # Initialize trainer
    try:
        trainer = LoRATrainer(
            config_path=args.config,
            dataset_path=args.dataset,
            output_dir=args.output_dir
        )
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        sys.exit(1)

    # Train model
    try:
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        logger.info(f"Training completed! Model saved to: {args.output_dir}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
