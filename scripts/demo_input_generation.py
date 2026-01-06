"""Demo script showing input generation capabilities."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.input_generator import InputGenerator
from src.utils.logger import setup_logger
import json


def main():
    """Demonstrate input generation for different tasks."""
    setup_logger(level="INFO")

    print("=" * 80)
    print("INPUT GENERATION DEMO")
    print("=" * 80)
    print()

    # Initialize generator (no API client needed for programmatic generation)
    generator = InputGenerator()

    # Demo 1: CLI Parsing
    print("\n📡 DEMO 1: CLI Parsing Input Generation")
    print("-" * 80)
    print("Generating 5 diverse CLI outputs...\n")

    for i in range(5):
        cli_output = generator.generate_input("cli_parsing", {})
        print(f"Example {i+1}:")
        print(cli_output)
        print()

    # Demo 2: Config Generation
    print("\n⚙️  DEMO 2: Configuration Parameters Generation")
    print("-" * 80)
    print("Generating 3 diverse configuration sets...\n")

    for i in range(3):
        config_params = generator.generate_input("config_generation", {})
        print(f"Example {i+1}:")
        print(json.dumps(config_params, indent=2))
        print()

    # Demo 3: Troubleshooting
    print("\n🔧 DEMO 3: Troubleshooting Scenarios Generation")
    print("-" * 80)
    print("Generating 5 diverse troubleshooting scenarios...\n")

    for i in range(5):
        scenario = generator.generate_input("troubleshooting", {})
        print(f"Scenario {i+1}: {scenario}")
        print()

    # Demo 4: YANG Conversion
    print("\n📝 DEMO 4: YANG Conversion Input Generation")
    print("-" * 80)
    print("Generating 3 diverse CLI configs...\n")

    for i in range(3):
        yang_input = generator.generate_input("yang_conversion", {})
        print(f"Config {i+1}:")
        print(yang_input)
        print()

    # Demo 5: Batch Generation
    print("\n🚀 DEMO 5: Batch Generation")
    print("-" * 80)
    print("Generating 100 inputs in batch...\n")

    inputs = generator.batch_generate("cli_parsing", {}, count=100)
    print(f"✓ Generated {len(inputs)} unique inputs")
    print(f"✓ First input length: {len(inputs[0])} chars")
    print(f"✓ Last input length: {len(inputs[-1])} chars")
    print(f"✓ All inputs are unique: {len(set(inputs)) == len(inputs)}")

    # Show diversity
    print("\n📊 Diversity Check:")
    print(f"   - Unique inputs: {len(set(inputs))}/{len(inputs)}")
    print(f"   - Average length: {sum(len(i) for i in inputs) / len(inputs):.0f} chars")

    print("\n" + "=" * 80)
    print("✓ Demo complete!")
    print("=" * 80)
    print("\nKey Points:")
    print("  1. Each input is UNIQUE and REALISTIC")
    print("  2. No API calls needed (FREE and FAST)")
    print("  3. Can generate unlimited examples")
    print("  4. Perfect for creating large training datasets")
    print()
    print("Next Steps:")
    print("  - Run: python -m src.data_generation.dataset_generator --num-samples 10000")
    print("  - This will generate 10,000 diverse examples for training!")
    print()


if __name__ == "__main__":
    main()
