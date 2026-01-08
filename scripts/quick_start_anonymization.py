#!/usr/bin/env python3
"""
Quick start script for anonymization project.
Guides the user through the complete workflow.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def print_banner():
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║     LLM Distillery - Anonymization Quick Start           ║
    ║     Fine-tune a 1B model for PII anonymization           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_env_file():
    """Check if .env file exists and has API key"""
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found!")
        print("📝 Please create .env file and add your OpenAI API key:")
        print("   OPENAI_API_KEY=sk-your-key-here")
        return False

    with open(env_path) as f:
        content = f.read()
        if "your_openai_api_key_here" in content or "OPENAI_API_KEY=sk-" not in content:
            print("⚠️  OpenAI API key not configured!")
            print("📝 Please edit .env file and add your actual API key")
            return False

    print("✅ Environment configured")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    try:
        import torch
        import transformers
        import peft
        print("✅ Dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("📦 Please install: pip install -r requirements.txt")
        return False


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed")
        return False

    print(f"✅ {description} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="Quick start for anonymization project")
    parser.add_argument(
        "--skip-dataset-generation",
        action="store_true",
        help="Skip dataset generation (use existing dataset)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (use existing model)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100 for quick test)"
    )
    parser.add_argument(
        "--fast-test",
        action="store_true",
        help="Fast test mode (100 samples, 2 epochs)"
    )

    args = parser.parse_args()

    print_banner()

    # Check prerequisites
    print("\n📋 Checking prerequisites...")
    if not check_dependencies():
        sys.exit(1)

    if not check_env_file():
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Workflow steps
    print("\n\n📍 Starting workflow...")

    # Step 1: Generate dataset
    if not args.skip_dataset_generation:
        print("\n" + "="*60)
        print("STEP 1: Generate Training Dataset")
        print("="*60)
        print(f"\nThis will generate {args.num_samples} examples using OpenAI GPT-4")
        if args.num_samples >= 1000:
            print(f"⚠️  This may take 1-2 hours and cost ~${args.num_samples * 0.01:.2f}")
        else:
            print(f"⏱️  This should take ~{args.num_samples // 10} minutes")

        response = input("\nProceed with dataset generation? (y/n): ")
        if response.lower() == 'y':
            cmd = [
                sys.executable,
                "scripts/generate_anonymization_dataset.py",
                "--config", "config/prompts_anonymization.yaml",
                "--output", "data/processed/anonymization_training_data.jsonl",
                "--num-samples", str(args.num_samples),
                "--model", "gpt-4-turbo-preview"
            ]
            if not run_command(cmd, "Dataset Generation"):
                sys.exit(1)
        else:
            print("⏭️  Skipping dataset generation")
    else:
        print("\n⏭️  Skipping dataset generation (using existing dataset)")

    # Check if dataset exists
    dataset_path = Path("data/processed/anonymization_training_data.jsonl")
    if not dataset_path.exists():
        print("\n❌ Dataset not found! Please generate it first.")
        sys.exit(1)

    # Step 2: Train model
    if not args.skip_training:
        print("\n\n" + "="*60)
        print("STEP 2: Train Anonymization Model")
        print("="*60)
        print("\nThis will fine-tune TinyLlama 1.1B with LoRA")
        print("Requirements:")
        print("  - GPU with 8GB+ VRAM")
        print("  - ~2 hours training time (for 5000 samples)")

        response = input("\nProceed with training? (y/n): ")
        if response.lower() == 'y':
            cmd = [
                sys.executable,
                "scripts/train_anonymization_model.py",
                "--dataset", "data/processed/anonymization_training_data.jsonl",
                "--config", "config/training_config_1b_anonymization.yaml",
                "--output-dir", "models/anonymization_1b"
            ]
            if not run_command(cmd, "Model Training"):
                sys.exit(1)
        else:
            print("⏭️  Skipping training")
    else:
        print("\n⏭️  Skipping training (using existing model)")

    # Check if model exists
    model_path = Path("models/anonymization_1b")
    if not model_path.exists():
        print("\n❌ Model not found! Please train it first.")
        sys.exit(1)

    # Step 3: Test model
    print("\n\n" + "="*60)
    print("STEP 3: Test the Model")
    print("="*60)
    print("\nRunning test examples...")

    cmd = [
        sys.executable,
        "scripts/test_anonymization.py",
        "--model-path", "models/anonymization_1b",
        "--base-model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ]
    run_command(cmd, "Model Testing")

    # Final message
    print("\n\n" + "="*60)
    print("✅ Quick Start Completed!")
    print("="*60)
    print("\nYour anonymization model is ready to use!")
    print("\nNext steps:")
    print("  1. Test interactively:")
    print("     python scripts/test_anonymization.py --interactive")
    print("\n  2. Test with your own text:")
    print("     python scripts/test_anonymization.py --text 'Your text here'")
    print("\n  3. Test with a file:")
    print("     python scripts/test_anonymization.py --file input.txt")
    print("\n  4. Read the full guide:")
    print("     README_ANONYMIZATION.md")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
