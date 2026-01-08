#!/usr/bin/env python3
"""
Generate synthetic dataset for anonymization training using OpenAI API.
Uses the prompts_anonymization.yaml configuration to generate diverse examples.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.dataset_generator import DatasetGenerator
from src.data_generation.openai_client import OpenAIClient
from src.data_generation.prompt_manager import PromptManager
from src.utils.logger import setup_logger, get_logger
import os


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic anonymization dataset using teacher model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/prompts_anonymization.yaml",
        help="Path to prompts configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/anonymization_training_data.jsonl",
        help="Output path for generated dataset"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of training samples to generate"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (or set OPENAI_API_KEY env variable)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4-turbo-preview",
        help="OpenAI model to use as teacher"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logger
    setup_logger(level="DEBUG" if args.verbose else "INFO")
    logger = get_logger("anonymization_dataset_generator")

    logger.info("Starting anonymization dataset generation")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Target samples: {args.num_samples}")
    logger.info(f"Teacher model: {args.model}")

    # Get API key from args or environment
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key not provided. Set OPENAI_API_KEY env variable or use --api-key")
        sys.exit(1)

    # Initialize components
    try:
        logger.info("Initializing OpenAI client...")
        openai_client = OpenAIClient(api_key=api_key, model=args.model)

        logger.info(f"Loading prompts from {args.config}...")
        prompt_manager = PromptManager(args.config)

        logger.info("Creating dataset generator...")
        generator = DatasetGenerator(
            openai_client=openai_client,
            prompt_manager=prompt_manager
        )
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Generate dataset
    try:
        logger.info("Starting dataset generation...")
        examples = generator.generate_dataset(
            task_names=None,  # Generate for all tasks
            num_examples_per_task=args.num_samples,
            output_path=args.output
        )
        logger.info(f"Dataset generated successfully: {args.output}")
        logger.info(f"Total examples: {len(examples)}")

    except KeyboardInterrupt:
        logger.warning("Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during dataset generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
