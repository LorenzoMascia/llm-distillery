"""Dataset generator for creating synthetic training data."""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import jsonlines
from loguru import logger

from .openai_client import OpenAIClient
from .prompt_manager import PromptManager
from .input_generator import InputGenerator
from ..utils.parser import ResponseParser


class DatasetGenerator:
    """Generator for creating synthetic datasets using teacher models."""

    def __init__(
        self,
        openai_client: OpenAIClient,
        prompt_manager: PromptManager,
        parser: Optional[ResponseParser] = None,
        input_generator: Optional[InputGenerator] = None,
    ):
        """
        Initialize dataset generator.

        Args:
            openai_client: OpenAI client instance
            prompt_manager: Prompt manager instance
            parser: Response parser (creates new one if not provided)
            input_generator: Input generator (creates new one if not provided)
        """
        self.client = openai_client
        self.prompt_manager = prompt_manager
        self.parser = parser or ResponseParser(strict_mode=False)
        self.input_generator = input_generator or InputGenerator(openai_client)

        self.generated_examples = []
        self.failed_examples = []

        logger.info("Dataset generator initialized")

    def generate_dataset(
        self,
        task_names: Optional[List[str]] = None,
        num_examples_per_task: int = 1000,
        input_generator: Optional[callable] = None,
        output_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a complete dataset for specified tasks.

        Args:
            task_names: List of task names (None = all tasks)
            num_examples_per_task: Number of examples to generate per task
            input_generator: Function to generate input data for tasks
            output_path: Path to save the dataset (JSONL format)

        Returns:
            List of generated training examples
        """
        if task_names is None:
            tasks = self.prompt_manager.get_all_tasks()
            task_names = [task["name"] for task in tasks]

        logger.info(f"Generating dataset for tasks: {task_names}")
        logger.info(f"Target: {num_examples_per_task} examples per task")

        all_examples = []

        for task_name in task_names:
            logger.info(f"Processing task: {task_name}")

            examples = self._generate_task_examples(
                task_name=task_name,
                num_examples=num_examples_per_task,
                input_generator=input_generator,
            )

            all_examples.extend(examples)

            logger.info(
                f"Generated {len(examples)} valid examples for task '{task_name}'"
            )

        # Shuffle the dataset
        random.shuffle(all_examples)

        self.generated_examples = all_examples

        # Save if output path provided
        if output_path:
            self.save_dataset(all_examples, output_path)

        logger.info(f"Dataset generation complete: {len(all_examples)} total examples")
        logger.info(f"Failed examples: {len(self.failed_examples)}")

        return all_examples

    def _generate_task_examples(
        self,
        task_name: str,
        num_examples: int,
        input_generator: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """Generate examples for a single task."""
        examples = []
        task = self.prompt_manager.get_task(task_name)

        if not task:
            logger.error(f"Task '{task_name}' not found")
            return examples

        system_message = self.prompt_manager.get_system_message(task_name)
        output_format = self.prompt_manager.get_output_format(task_name)
        output_schema = self.prompt_manager.get_output_schema(task_name)

        # Use custom input generator if provided, otherwise use the built-in one
        if input_generator is None:
            # Use the InputGenerator instance
            def default_generator(task_name):
                return self.input_generator.generate_input(task_name, task)
            input_generator = default_generator

        # Generate examples with progress bar
        with tqdm(total=num_examples, desc=f"Task: {task_name}") as pbar:
            attempts = 0
            max_attempts = num_examples * 3  # Allow some failures

            while len(examples) < num_examples and attempts < max_attempts:
                attempts += 1

                try:
                    # Generate input
                    input_data = input_generator(task_name)

                    # Create prompt
                    prompt = self.prompt_manager.format_prompt(
                        task_name, input_data, include_examples=True
                    )

                    # Get response from teacher
                    response = self.client.generate(
                        prompt,
                        system_message=system_message,
                        temperature=self.prompt_manager.get_generation_params().get(
                            "temperature", 0.3
                        ),
                    )

                    # Parse response
                    if output_format == "json":
                        parsed_output = self.parser.parse_json_response(response)

                        # Validate against schema if available
                        if output_schema and parsed_output:
                            if not self.parser.validate_schema(
                                parsed_output, output_schema
                            ):
                                logger.warning("Schema validation failed")
                                self.failed_examples.append(
                                    {
                                        "task": task_name,
                                        "input": input_data,
                                        "response": response,
                                        "reason": "schema_validation_failed",
                                    }
                                )
                                continue
                    else:
                        parsed_output = self.parser.parse_text_response(response)

                    if parsed_output is None:
                        self.failed_examples.append(
                            {
                                "task": task_name,
                                "input": input_data,
                                "response": response,
                                "reason": "parsing_failed",
                            }
                        )
                        continue

                    # Create training example
                    training_example = self.parser.create_training_example(
                        instruction=task["instruction"],
                        input_text=str(input_data),
                        output=parsed_output,
                    )

                    # Add metadata
                    training_example["task"] = task_name
                    training_example["metadata"] = {
                        "output_format": output_format,
                        "has_schema": output_schema is not None,
                    }

                    examples.append(training_example)
                    pbar.update(1)

                except Exception as e:
                    logger.error(f"Error generating example: {e}")
                    self.failed_examples.append(
                        {
                            "task": task_name,
                            "error": str(e),
                            "reason": "generation_error",
                        }
                    )

        return examples

    def set_custom_input_generator(
        self, task_name: str, generator_func: callable
    ) -> None:
        """
        Set a custom input generator for a specific task.

        Args:
            task_name: Name of the task
            generator_func: Function that takes task_config and returns input

        Example:
            def my_generator(task_config):
                return "custom input"

            dataset_gen.set_custom_input_generator("my_task", my_generator)
        """
        self.input_generator.register_generator(task_name, generator_func)
        logger.info(f"Registered custom generator for task: {task_name}")

    def save_dataset(self, examples: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save dataset to JSONL file.

        Args:
            examples: List of training examples
            output_path: Path to output file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with jsonlines.open(output_file, mode="w") as writer:
            for example in examples:
                writer.write(example)

        logger.info(f"Dataset saved to {output_path}")

    def split_dataset(
        self,
        train_ratio: float = 0.9,
        shuffle: bool = True,
        seed: int = 42,
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Split dataset into train and validation sets.

        Args:
            train_ratio: Ratio of training data
            shuffle: Whether to shuffle before splitting
            seed: Random seed

        Returns:
            Tuple of (train_examples, val_examples)
        """
        examples = self.generated_examples.copy()

        if shuffle:
            random.seed(seed)
            random.shuffle(examples)

        split_idx = int(len(examples) * train_ratio)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]

        logger.info(f"Dataset split: {len(train_examples)} train, {len(val_examples)} val")

        return train_examples, val_examples

    def export_failed_examples(self, output_path: str) -> None:
        """Export failed examples for analysis."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.failed_examples, f, indent=2)

        logger.info(f"Failed examples exported to {output_path}")


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate synthetic dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config/prompts.yaml",
        help="Path to prompts config",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/training_data.jsonl",
        help="Output path",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000, help="Number of samples per task"
    )
    parser.add_argument("--tasks", nargs="+", help="Specific tasks to generate")

    args = parser.parse_args()

    # Initialize components
    client = OpenAIClient()
    prompt_mgr = PromptManager(args.config)
    generator = DatasetGenerator(client, prompt_mgr)

    # Generate dataset
    generator.generate_dataset(
        task_names=args.tasks,
        num_examples_per_task=args.num_samples,
        output_path=args.output,
    )

    # Export failed examples
    generator.export_failed_examples("data/processed/failed_examples.json")
