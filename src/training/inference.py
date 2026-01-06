"""Inference engine for trained student models."""

import torch
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from loguru import logger


class InferenceEngine:
    """Engine for running inference with trained LoRA models."""

    def __init__(
        self,
        model_path: str,
        base_model: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to the trained LoRA adapter
            base_model: Base model name (if different from adapter config)
            device: Device to run inference on
            load_in_8bit: Load model in 8-bit precision
            load_in_4bit: Load model in 4-bit precision
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.tokenizer = None

        logger.info(f"Loading model from {model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Determine base model
        if base_model is None:
            # Try to read from adapter config
            try:
                from peft import PeftConfig

                peft_config = PeftConfig.from_pretrained(model_path)
                base_model = peft_config.base_model_name_or_path
                logger.info(f"Using base model from config: {base_model}")
            except Exception as e:
                logger.error(f"Could not determine base model: {e}")
                raise ValueError(
                    "Base model not specified and could not be determined from config"
                )

        # Load base model
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": False,
        }

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        else:
            model_kwargs["torch_dtype"] = torch.float16

        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model, **model_kwargs
        )

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model_obj, model_path)

        # Set to evaluation mode
        self.model.eval()

        logger.info("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        num_beams: int = 1,
    ) -> str:
        """
        Generate response for a prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search

        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                num_beams=num_beams,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the input prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :].strip()

        return generated_text

    def generate_batch(
        self, prompts: list[str], batch_size: int = 4, **kwargs
    ) -> list[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts
            batch_size: Batch size for processing
            **kwargs: Additional arguments for generate()

        Returns:
            List of generated texts
        """
        results = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]

            for prompt in batch:
                result = self.generate(prompt, **kwargs)
                results.append(result)

        return results

    def format_prompt(
        self, instruction: str, input_text: str, template: Optional[str] = None
    ) -> str:
        """
        Format a prompt using the training template.

        Args:
            instruction: Task instruction
            input_text: Input data
            template: Custom template (uses default if not provided)

        Returns:
            Formatted prompt
        """
        if template is None:
            template = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

        return template.format(instruction=instruction, input=input_text)

    def interactive_mode(self) -> None:
        """Run interactive inference mode."""
        logger.info("Entering interactive mode. Type 'exit' to quit.")

        while True:
            try:
                user_input = input("\n> ")

                if user_input.lower() in ["exit", "quit", "q"]:
                    break

                response = self.generate(user_input)
                print(f"\nModel: {response}")

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error during generation: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model/adapter",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        help="Base model name (optional)",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input text for inference",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit precision",
    )

    args = parser.parse_args()

    # Initialize engine
    engine = InferenceEngine(
        model_path=args.model_path,
        base_model=args.base_model,
        load_in_4bit=args.load_in_4bit,
    )

    # Run inference
    if args.interactive:
        engine.interactive_mode()
    elif args.input:
        response = engine.generate(args.input)
        print(f"\nResponse:\n{response}")
    else:
        print("Please provide --input or use --interactive mode")
