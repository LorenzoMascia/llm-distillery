"""Quick start script for running the complete pipeline."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.openai_client import OpenAIClient
from src.data_generation.prompt_manager import PromptManager
from src.data_generation.dataset_generator import DatasetGenerator
from src.training.lora_trainer import LoRATrainer
from src.utils.logger import setup_logger


def main():
    """Run the complete pipeline."""
    # Setup logging
    setup_logger(log_file="logs/quickstart.log", level="INFO")
    logger.info("Starting LLM Knowledge Distillation Pipeline")

    # Load environment variables
    load_dotenv()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found in environment variables!")
        logger.error("Please create a .env file with your OpenAI API key")
        return

    # Configuration
    PROMPTS_CONFIG = "config/prompts.yaml"
    TRAINING_CONFIG = "config/training_config.yaml"
    DATASET_PATH = "data/processed/training_data.jsonl"
    MODEL_OUTPUT_DIR = "models/student_model"
    NUM_SAMPLES = 100  # Small number for quick start

    # Step 1: Generate Dataset
    logger.info("Step 1: Generating synthetic dataset")
    logger.info(f"Target: {NUM_SAMPLES} examples")

    try:
        client = OpenAIClient()
        prompt_manager = PromptManager(PROMPTS_CONFIG)
        generator = DatasetGenerator(client, prompt_manager)

        # Validate API key
        if not client.validate_api_key():
            logger.error("API key validation failed!")
            return

        # Generate dataset
        generator.generate_dataset(
            num_examples_per_task=NUM_SAMPLES,
            output_path=DATASET_PATH,
        )

        logger.info(f"Dataset generated: {DATASET_PATH}")

    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        return

    # Step 2: Train Model
    logger.info("Step 2: Training student model with LoRA")

    try:
        trainer = LoRATrainer(TRAINING_CONFIG)

        logger.info("Setting up model...")
        trainer.setup_model()

        logger.info("Preparing dataset...")
        train_ds, eval_ds = trainer.prepare_dataset(DATASET_PATH)

        logger.info("Starting training...")
        trainer.train(train_ds, eval_ds, output_dir=MODEL_OUTPUT_DIR)

        logger.info("Saving model...")
        trainer.save_model(MODEL_OUTPUT_DIR)

        logger.info(f"Model trained and saved: {MODEL_OUTPUT_DIR}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

    # Step 3: Test Inference
    logger.info("Step 3: Testing inference")

    try:
        from src.training.inference import InferenceEngine

        engine = InferenceEngine(
            model_path=MODEL_OUTPUT_DIR,
            load_in_4bit=True,
        )

        # Test with a simple prompt
        test_prompt = engine.format_prompt(
            instruction="Parse the following CLI output",
            input_text="GigabitEthernet0/0/0 is up, line protocol is up",
        )

        response = engine.generate(test_prompt, max_new_tokens=256)

        logger.info("Test inference successful!")
        logger.info(f"Response: {response[:200]}...")

    except Exception as e:
        logger.error(f"Inference test failed: {e}")
        return

    logger.info("Pipeline complete! ✓")
    logger.info(f"Model available at: {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
