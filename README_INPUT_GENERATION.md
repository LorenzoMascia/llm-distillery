# 🎯 Input Generation - Quick Guide

## The Problem Solved

**Before**: You had to manually write thousands of examples in the `prompts.yaml` file

**Now**: Just 2-3 examples are enough! The system automatically generates thousands of diverse inputs.

## How It Works

### 1. Define Few Examples (prompts.yaml)

```yaml
tasks:
  - name: "cli_parsing"
    instruction: "Parse the CLI output"
    examples:
      - input: "GigabitEthernet0/0/0 is up..."  # Just 1 example!
        output: {...}
```

### 2. Generate Thousands of Variations

```python
from src.data_generation import OpenAIClient, PromptManager, DatasetGenerator

client = OpenAIClient()
prompt_mgr = PromptManager("config/prompts.yaml")
dataset_gen = DatasetGenerator(client, prompt_mgr)

# Generate 10,000 DIVERSE examples automatically!
dataset_gen.generate_dataset(
    task_names=["cli_parsing"],
    num_examples_per_task=10000,
    output_path="data/processed/dataset.jsonl"
)
```

### 3. The System Generates Varied Inputs

For each example, the `InputGenerator` automatically creates:

- ✅ Different interfaces (GigabitEthernet, FastEthernet, TenGigE, etc.)
- ✅ Random but valid IP addresses (10.x.x.x, 192.168.x.x, etc.)
- ✅ Random MAC addresses
- ✅ Different states (up/down, errors, etc.)
- ✅ Varied parameters (MTU, bandwidth, counters, etc.)

**Result**: 10,000 UNIQUE inputs instead of reusing the same 3!

## Quick Demo

```bash
# See the generator in action
cd code
python scripts/demo_input_generation.py
```

Output:
```
Example 1:
GigabitEthernet2/15/3 is up, line protocol is up
  Hardware is GigabitEthernet, address is a4:5e:60:b2:3f:91
  Internet address is 10.142.78.45/24

Example 2:
TenGigabitEthernet0/8/1 is administratively down, line protocol is down
  Hardware is TenGigabitEthernet, address is 2c:f4:32:8a:1d:5b
  Internet address is 192.168.23.101/28

Example 3:
FastEthernet1/22/0 is up, line protocol is down
  45 input errors, 12 CRC, 5 frame
  23 output errors, 3 collisions

... (each example is DIFFERENT!)
```

## Pre-configured Generators

| Task | What It Generates | Examples |
|------|-------------------|----------|
| **cli_parsing** | Varied CLI outputs | 10,000 different interfaces, IPs, MACs, states |
| **config_generation** | Config parameters | Router names, multiple interfaces, services |
| **troubleshooting** | Troubleshooting scenarios | Layer 1/2/3 issues, protocol problems |
| **yang_conversion** | CLI configs | Interface configs, VLANs, Loopbacks |

## Custom Generators

For your specific domain:

```python
from src.data_generation import InputGenerator

generator = InputGenerator()

# Define a custom generator
def my_sql_generator(task_config):
    return f"SELECT * FROM table_{random.randint(1, 100)} WHERE id = {random.randint(1, 1000)}"

# Register the generator
generator.register_generator("sql_parsing", my_sql_generator)

# Use in dataset generator
dataset_gen.set_custom_input_generator("sql_parsing", my_sql_generator)
```

## Advantages

| Method | YAML Inputs | Generated Outputs | Cost | Time |
|--------|-------------|-------------------|------|------|
| **Manual** | 10,000 | 10,000 | High (labor) | Days |
| **AI-only** | 3 | 10,000 | High (API) | Hours |
| **InputGenerator** | 3 | 10,000+ | **FREE** | **Minutes** |

## Quick Start

```bash
# 1. See the demo
python scripts/demo_input_generation.py

# 2. Generate dataset
python -m src.data_generation.dataset_generator \
    --config config/prompts.yaml \
    --output data/processed/training_data.jsonl \
    --num-samples 10000

# 3. Train!
python -m src.training.lora_trainer \
    --config config/training_config.yaml \
    --dataset data/processed/training_data.jsonl
```

## Full Documentation

See [docs/INPUT_GENERATION_GUIDE.md](docs/INPUT_GENERATION_GUIDE.md) for:
- Technical details
- Custom generators
- Best practices
- Advanced examples

## Short Answer to Your Question

> "Should it be a file capable of generating many examples for the dataset from each prompt? Does it already work like this?"

**Answer**: Yes! ✅

- **Before**: The YAML file had fixed examples that were reused
- **Now**: Just 2-3 examples in YAML, the system automatically generates thousands of different variations
- **How**: `InputGenerator` creates random but realistic inputs (IPs, MACs, interfaces, states, etc.)
- **Result**: From 3 examples → 10,000+ unique inputs for training

🚀 **Ready to use!** No extra configuration needed.
