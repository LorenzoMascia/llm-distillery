#!/usr/bin/env python3
"""
Test script for the anonymization model.
Tests the fine-tuned model on sample texts and displays results.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_model(model_path: str, base_model: str = None):
    """
    Load the fine-tuned model and tokenizer.

    Args:
        model_path: Path to the fine-tuned model (LoRA adapters)
        base_model: Base model name (if not specified, reads from adapter config)

    Returns:
        model, tokenizer
    """
    print(f"Loading model from {model_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model and adapters
    if base_model:
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=False
        )
        print(f"Loading LoRA adapters from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
    else:
        # Try to load directly (works if model is merged)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=False
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please specify the base model with --base-model flag")
            sys.exit(1)

    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer


def create_prompt(text: str) -> str:
    """
    Create the prompt for anonymization.

    Args:
        text: Input text to anonymize

    Returns:
        Formatted prompt
    """
    prompt = f"""### Instruction:
Analyze the following text and anonymize all personally identifiable information (PII). Return a JSON object with the anonymized text and all replaced tokens.

### Input:
{text}

### Response:
"""
    return prompt


def anonymize_text(model, tokenizer, text: str, max_new_tokens: int = 1024) -> dict:
    """
    Anonymize the input text using the model.

    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        text: Input text to anonymize
        max_new_tokens: Maximum number of tokens to generate

    Returns:
        Dictionary with anonymized_text and replaced_tokens
    """
    prompt = create_prompt(text)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the response part (after "### Response:")
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    # Try to parse as JSON
    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        # If not valid JSON, return raw response
        return {
            "error": "Failed to parse JSON response",
            "raw_response": response
        }


def main():
    parser = argparse.ArgumentParser(description="Test the anonymization model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/anonymization_1b",
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name (optional, if not merged)"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to anonymize"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File containing text to anonymize"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate"
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model(args.model_path, args.base_model)

    if args.interactive:
        # Interactive mode
        print("\n=== Interactive Anonymization Mode ===")
        print("Enter text to anonymize (or 'quit' to exit):\n")

        while True:
            try:
                text = input("Input: ").strip()
                if text.lower() in ['quit', 'exit', 'q']:
                    break

                if not text:
                    continue

                print("\nAnonymizing...")
                result = anonymize_text(model, tokenizer, text, args.max_new_tokens)

                print("\n=== Result ===")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                print()

            except KeyboardInterrupt:
                print("\nExiting...")
                break

    elif args.file:
        # Read from file
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        print(f"Anonymizing text from file: {args.file}")
        result = anonymize_text(model, tokenizer, text, args.max_new_tokens)

        print("\n=== Result ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.text:
        # Single text input
        print(f"Anonymizing text: {args.text}")
        result = anonymize_text(model, tokenizer, args.text, args.max_new_tokens)

        print("\n=== Result ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    else:
        # Run test examples
        print("\n=== Running Test Examples ===\n")

        test_cases = [
            "My name is Mario Rossi and my email is mario.rossi@gmail.com. You can call me at +39 333 1234567.",
            "Il dottor Giovanni Bianchi, nato il 15/03/1980, risiede in Via Roma 42, 20100 Milano.",
            "Contact: John Smith (john.smith@company.com), Phone: +1-555-123-4567.",
        ]

        for i, text in enumerate(test_cases, 1):
            print(f"Test Case {i}:")
            print(f"Input: {text}")
            print()

            result = anonymize_text(model, tokenizer, text, args.max_new_tokens)

            print("Output:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
