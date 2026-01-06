"""Script for evaluating trained models."""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from loguru import logger
import jsonlines

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.inference import InferenceEngine
from src.utils.parser import ResponseParser
from src.utils.logger import setup_logger


class ModelEvaluator:
    """Evaluator for trained models."""

    def __init__(
        self,
        model_path: str,
        test_dataset_path: str,
        output_path: str = "evaluation_results.json",
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model
            test_dataset_path: Path to test dataset
            output_path: Path to save results
        """
        self.model_path = model_path
        self.test_dataset_path = test_dataset_path
        self.output_path = output_path

        logger.info(f"Loading model from {model_path}")
        self.engine = InferenceEngine(model_path, load_in_4bit=True)

        self.parser = ResponseParser(strict_mode=False)
        self.results = []

    def load_test_data(self) -> List[Dict[str, Any]]:
        """Load test dataset."""
        logger.info(f"Loading test data from {self.test_dataset_path}")

        examples = []
        with jsonlines.open(self.test_dataset_path) as reader:
            for obj in reader:
                examples.append(obj)

        logger.info(f"Loaded {len(examples)} test examples")
        return examples

    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation on test set."""
        test_examples = self.load_test_data()

        correct = 0
        total = 0
        errors = []

        logger.info("Starting evaluation...")

        for example in tqdm(test_examples, desc="Evaluating"):
            try:
                # Create prompt
                prompt = self.engine.format_prompt(
                    instruction=example["instruction"],
                    input_text=example["input"],
                )

                # Generate prediction
                prediction = self.engine.generate(prompt, max_new_tokens=512)

                # Compare with expected output
                expected = example["output"]

                # Simple string match for now (can be made more sophisticated)
                is_correct = self._compare_outputs(prediction, expected)

                if is_correct:
                    correct += 1

                total += 1

                # Store result
                self.results.append(
                    {
                        "input": example["input"],
                        "expected": expected,
                        "predicted": prediction,
                        "correct": is_correct,
                        "task": example.get("task", "unknown"),
                    }
                )

            except Exception as e:
                logger.error(f"Error evaluating example: {e}")
                errors.append({"example": example, "error": str(e)})

        # Calculate metrics
        accuracy = correct / total if total > 0 else 0

        results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "errors": len(errors),
            "per_example_results": self.results,
            "error_details": errors,
        }

        # Save results
        self.save_results(results)

        logger.info(f"Evaluation complete!")
        logger.info(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
        logger.info(f"Errors: {len(errors)}")

        return results

    def _compare_outputs(self, predicted: str, expected: str) -> bool:
        """
        Compare predicted and expected outputs.

        Args:
            predicted: Model prediction
            expected: Expected output

        Returns:
            True if outputs match
        """
        # Try JSON comparison if both are JSON
        try:
            pred_json = self.parser.parse_json_response(predicted)
            exp_json = json.loads(expected) if isinstance(expected, str) else expected

            if pred_json and exp_json:
                return pred_json == exp_json
        except:
            pass

        # Fallback to string similarity
        pred_clean = predicted.strip().lower()
        exp_clean = str(expected).strip().lower()

        return pred_clean == exp_clean

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results."""
        with open(self.output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {self.output_path}")

    def generate_report(self) -> str:
        """Generate evaluation report."""
        if not self.results:
            return "No evaluation results available"

        correct = sum(1 for r in self.results if r["correct"])
        total = len(self.results)
        accuracy = correct / total if total > 0 else 0

        # Per-task accuracy
        task_stats = {}
        for result in self.results:
            task = result.get("task", "unknown")
            if task not in task_stats:
                task_stats[task] = {"correct": 0, "total": 0}

            task_stats[task]["total"] += 1
            if result["correct"]:
                task_stats[task]["correct"] += 1

        report = f"""
Evaluation Report
================

Overall Metrics:
- Accuracy: {accuracy:.2%} ({correct}/{total})
- Total Examples: {total}
- Correct Predictions: {correct}
- Incorrect Predictions: {total - correct}

Per-Task Performance:
"""

        for task, stats in task_stats.items():
            task_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            report += f"\n- {task}: {task_acc:.2%} ({stats['correct']}/{stats['total']})"

        return report


if __name__ == "__main__":
    import argparse

    setup_logger(level="INFO")

    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="Path to test dataset (JSONL)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output path for results",
    )

    args = parser.parse_args()

    evaluator = ModelEvaluator(args.model_path, args.test_data, args.output)
    results = evaluator.evaluate()

    print("\n" + evaluator.generate_report())
