"""Prompt management and template handling."""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger


class PromptManager:
    """Manager for loading and formatting prompts from configuration."""

    def __init__(self, config_path: str):
        """
        Initialize prompt manager.

        Args:
            config_path: Path to prompts configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.tasks = self.config.get("tasks", [])
        self.generation_params = self.config.get("generation", {})
        self.dataset_params = self.config.get("dataset", {})

        logger.info(f"Loaded {len(self.tasks)} tasks from {config_path}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config

    def get_task(self, task_name: str) -> Optional[Dict[str, Any]]:
        """
        Get task configuration by name.

        Args:
            task_name: Name of the task

        Returns:
            Task configuration dictionary or None if not found
        """
        for task in self.tasks:
            if task.get("name") == task_name:
                return task

        logger.warning(f"Task '{task_name}' not found")
        return None

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all task configurations.

        Returns:
            List of task dictionaries
        """
        return self.tasks

    def format_prompt(
        self,
        task_name: str,
        input_data: Any,
        include_examples: bool = False,
    ) -> str:
        """
        Format a prompt for a specific task.

        Args:
            task_name: Name of the task
            input_data: Input data for the task
            include_examples: Whether to include examples in the prompt

        Returns:
            Formatted prompt string
        """
        task = self.get_task(task_name)
        if not task:
            raise ValueError(f"Task '{task_name}' not found")

        instruction = task.get("instruction", "")

        # Format input data
        if isinstance(input_data, dict):
            input_str = yaml.dump(input_data, default_flow_style=False)
        else:
            input_str = str(input_data)

        prompt_parts = [instruction]

        # Add examples if requested
        if include_examples and "examples" in task:
            prompt_parts.append("\n\nExamples:")
            for i, example in enumerate(task["examples"][:3], 1):  # Limit to 3 examples
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"Input: {example.get('input', '')}")
                prompt_parts.append(f"Output: {example.get('output', '')}")

        # Add the actual input
        prompt_parts.append(f"\n\nInput:\n{input_str}")

        # Add output format hint if schema exists
        if "output_schema" in task:
            prompt_parts.append("\n\nProvide the output as valid JSON.")

        return "\n".join(prompt_parts)

    def get_system_message(self, task_name: str) -> Optional[str]:
        """
        Get system message for a task.

        Args:
            task_name: Name of the task

        Returns:
            System message string or None
        """
        task = self.get_task(task_name)
        if task:
            return task.get("system_message")
        return None

    def get_output_schema(self, task_name: str) -> Optional[Dict[str, Any]]:
        """
        Get output schema for a task.

        Args:
            task_name: Name of the task

        Returns:
            Output schema dictionary or None
        """
        task = self.get_task(task_name)
        if task:
            return task.get("output_schema")
        return None

    def get_output_format(self, task_name: str) -> str:
        """
        Get expected output format for a task.

        Args:
            task_name: Name of the task

        Returns:
            Output format ('json' or 'text')
        """
        task = self.get_task(task_name)
        if task:
            if "output_schema" in task:
                return "json"
            return task.get("output_format", "text")
        return "text"

    def get_generation_params(self) -> Dict[str, Any]:
        """
        Get generation parameters from config.

        Returns:
            Dictionary of generation parameters
        """
        return self.generation_params

    def get_dataset_params(self) -> Dict[str, Any]:
        """
        Get dataset parameters from config.

        Returns:
            Dictionary of dataset parameters
        """
        return self.dataset_params

    def generate_variations(
        self,
        task_name: str,
        base_input: Any,
        num_variations: int = 5,
    ) -> List[str]:
        """
        Generate prompt variations for data augmentation.

        Args:
            task_name: Name of the task
            base_input: Base input to create variations from
            num_variations: Number of variations to generate

        Returns:
            List of prompt variations
        """
        task = self.get_task(task_name)
        if not task:
            raise ValueError(f"Task '{task_name}' not found")

        variations = []

        # Simple variation strategies
        rephrasings = [
            "Please process the following:",
            "Analyze this:",
            "Handle the following input:",
            "Process this data:",
            "Perform the task on:",
        ]

        for i in range(min(num_variations, len(rephrasings))):
            variation_instruction = task["instruction"]
            if i > 0 and i <= len(rephrasings):
                variation_instruction = rephrasings[i - 1] + " " + task["instruction"]

            task_copy = task.copy()
            task_copy["instruction"] = variation_instruction

            # Temporarily replace task to use modified instruction
            original_task = self.get_task(task_name)
            idx = self.tasks.index(original_task)
            self.tasks[idx] = task_copy

            prompt = self.format_prompt(task_name, base_input)
            variations.append(prompt)

            # Restore original task
            self.tasks[idx] = original_task

        return variations
