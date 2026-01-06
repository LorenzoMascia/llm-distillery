# Input Generation Guide

## How Diversified Input Generation Works

### Original Problem

In the `config/prompts.yaml` file, you could only define **a few examples** per task:

```yaml
tasks:
  - name: "cli_parsing"
    examples:
      - input: "GigabitEthernet0/0/0 is up..."
      - input: "FastEthernet0/1 is down..."
      - input: "TenGigE0/2/0 is up..."
```

❌ **Problem**: With only 3 examples, the system would reuse them repeatedly, generating a poorly diversified dataset.

### Solution: InputGenerator

The new `InputGenerator` solves this problem by automatically generating **synthetic and varied inputs**.

## How It Works

### 1. Pre-configured Generators

For common tasks, there are generators that create random but realistic inputs:

```python
from src.data_generation import InputGenerator

generator = InputGenerator()

# Generate 10 different inputs for CLI parsing
for i in range(10):
    cli_output = generator.generate_input("cli_parsing", task_config)
    print(cli_output)
```

**Example output**:
```
GigabitEthernet2/15/3 is up, line protocol is up
  Hardware is GigabitEthernet, address is a4:5e:60:b2:3f:91
  Internet address is 10.142.78.45/24
  MTU 1500 bytes, BW 1000000 Kbit
  ...

TenGigabitEthernet0/8/1 is administratively down, line protocol is down
  Hardware is TenGigabitEthernet, address is 2c:f4:32:8a:1d:5b
  Internet address is 192.168.23.101/28
  ...

FastEthernet1/22/0 is up, line protocol is down
  45 input errors, 12 CRC, 5 frame
  23 output errors, 3 collisions
```

### 2. Parametric Generation

The generator creates variations using random parameters:

- **Interface types**: GigabitEthernet, FastEthernet, TenGigE, Loopback, etc.
- **IP addresses**: Random private IPs (10.x.x.x, 192.168.x.x, 172.16-31.x.x)
- **MAC addresses**: Random valid MACs
- **States**: up/down, various combinations
- **Errors**: Random counters for errors

### 3. Task-Specific Generators

#### CLI Parsing
```python
# Generate varied CLI output
cli_output = generator.generate_input("cli_parsing", task_config)
# Result: Cisco IOS/IOS-XR output with random parameters
```

#### Config Generation
```python
# Generate configuration parameters
config_params = generator.generate_input("config_generation", task_config)
# Result:
# {
#   "device_type": "cisco_ios",
#   "hostname": "router-042",
#   "interfaces": [
#     {"name": "GigabitEthernet0/0", "ip": "10.45.67.1", "mask": "255.255.255.0"},
#     {"name": "GigabitEthernet0/1", "ip": "192.168.1.1", "mask": "255.255.255.252"}
#   ],
#   "ntp_servers": ["10.10.10.1", "10.10.10.2"]
# }
```

#### Troubleshooting
```python
# Generate troubleshooting scenarios
scenario = generator.generate_input("troubleshooting", task_config)
# Result: "Interface GigabitEthernet0/15 shows up/down status.
#          High CRC errors detected. Cable is connected."
```

#### YANG Conversion
```python
# Generate CLI configurations to convert
cli_config = generator.generate_input("yang_conversion", task_config)
# Result: Formatted CLI configuration
```

## Integration with Dataset Generator

The `DatasetGenerator` automatically uses `InputGenerator`:

```python
from src.data_generation import OpenAIClient, PromptManager, DatasetGenerator

client = OpenAIClient()
prompt_mgr = PromptManager("config/prompts.yaml")
dataset_gen = DatasetGenerator(client, prompt_mgr)

# Generate 10,000 examples with DIVERSIFIED inputs
dataset_gen.generate_dataset(
    task_names=["cli_parsing"],
    num_examples_per_task=10000,  # Each example will have a DIFFERENT input!
    output_path="data/processed/training_data.jsonl"
)
```

## Generation Modes

### Mode 1: Programmatic Generators (Default)

Uses templates and randomization to generate inputs:

```python
# Automatic for pre-configured tasks
generator = InputGenerator()
input_data = generator.generate_input("cli_parsing", task_config)
```

**Advantages**:
- ✅ Fast (no API calls)
- ✅ Free (no cost)
- ✅ Deterministic (you can control variation)
- ✅ High diversity

### Mode 2: AI-Powered Generation

Uses the teacher model to also generate inputs:

```python
generator = InputGenerator(openai_client=client)

# For custom tasks not pre-configured
input_data = generator.generate_input("custom_task", task_config)
# The system uses GPT to generate varied inputs
```

**Advantages**:
- ✅ Works for any domain
- ✅ Very realistic inputs
- ✅ No programming required

**Disadvantages**:
- ❌ Slower (API calls)
- ❌ More expensive (token usage)

## Custom Generators

You can create custom generators for your domain:

```python
from src.data_generation import DatasetGenerator, InputGenerator
import random

# Define a custom generator
def my_custom_generator(task_config):
    """Generate inputs for my specific domain."""
    # Your logic here
    return {
        "query": f"SELECT * FROM table_{random.randint(1, 100)}",
        "parameters": [random.randint(1, 1000)]
    }

# Register the generator
generator = InputGenerator()
generator.register_generator("sql_parsing", my_custom_generator)

# Use in dataset generator
dataset_gen = DatasetGenerator(client, prompt_mgr)
dataset_gen.set_custom_input_generator("sql_parsing", my_custom_generator)

# Generate dataset with your generator
dataset_gen.generate_dataset(
    task_names=["sql_parsing"],
    num_examples_per_task=5000
)
```

## Practical Examples

### Example 1: CLI Parsing Dataset (10K examples)

```python
from src.data_generation import OpenAIClient, PromptManager, DatasetGenerator

client = OpenAIClient()
prompt_mgr = PromptManager("config/prompts.yaml")
dataset_gen = DatasetGenerator(client, prompt_mgr)

# Generate 10,000 DIFFERENT examples
dataset_gen.generate_dataset(
    task_names=["cli_parsing"],
    num_examples_per_task=10000,
    output_path="data/processed/cli_dataset.jsonl"
)
```

**Result**: 10,000 examples with:
- 10,000 different interfaces
- 10,000 different IPs
- 10,000 different MACs
- Varied states (up/down, errors, etc.)

### Example 2: Mixed Tasks (50K total examples)

```python
dataset_gen.generate_dataset(
    task_names=["cli_parsing", "config_generation", "troubleshooting", "yang_conversion"],
    num_examples_per_task=12500,  # 12.5K per task = 50K total
    output_path="data/processed/full_dataset.jsonl"
)
```

### Example 3: AI Generation for Custom Tasks

```python
# For non-pre-configured tasks, use AI
dataset_gen = DatasetGenerator(client, prompt_mgr)

# The system will use GPT to generate varied inputs
dataset_gen.generate_dataset(
    task_names=["my_custom_task"],
    num_examples_per_task=1000
)
```

## Configuration File: prompts.yaml

The YAML file **no longer needs** to contain thousands of examples. Just a few examples as reference is enough:

```yaml
tasks:
  - name: "cli_parsing"
    description: "Parse CLI output and extract structured information"
    instruction: "Parse the following network device CLI output..."
    system_message: "You are an expert network engineer..."

    # Just 2-3 examples as reference!
    examples:
      - input: "GigabitEthernet0/0/0 is up, line protocol is up..."
        output: {"interface": "GigabitEthernet0/0/0", "admin_state": "up", ...}

      - input: "FastEthernet0/1 is down, line protocol is down..."
        output: {"interface": "FastEthernet0/1", "admin_state": "down", ...}

    # Schema for validation
    output_schema:
      type: "object"
      required: ["interface", "admin_state", "oper_state"]
      ...
```

## Complete Workflow

```
1. config/prompts.yaml
   ↓ (defines few examples as reference)

2. InputGenerator
   ↓ (generates thousands of varied inputs)

3. Teacher Model (OpenAI API)
   ↓ (processes each input and generates output)

4. Dataset (JSONL)
   ↓ (thousands of different examples)

5. Student Model (LoRA)
   ↓ (learns from diversified data)

6. Production
```

## Advantages

✅ **High Diversity**: Each example is unique
✅ **Scalable**: Generate 10K+ examples easily
✅ **Economical**: Programmatic generators are free
✅ **Flexible**: Supports custom generators
✅ **Realistic**: Outputs look like real data
✅ **Controllable**: You can configure variation

## Comparison

| Approach | Examples in YAML | Generated Examples | Diversity | Cost |
|----------|------------------|-------------------|-----------|------|
| **Old** | 10,000 | 10,000 | Low (reuse) | High (manual writing) |
| **New (programmatic)** | 3 | 10,000+ | High | Free |
| **New (AI)** | 3 | 10,000+ | Very High | API tokens |

## FAQ

**Q: Do I still need to put examples in prompts.yaml?**
A: Yes, but only 2-3 as reference. The system uses them to understand the format, not to replicate them.

**Q: How many examples can I generate?**
A: Unlimited! The programmatic generator is free and fast.

**Q: How do I control the quality of generated inputs?**
A: Inputs follow realistic templates. You can customize them by creating custom generators.

**Q: Can I use only AI to generate inputs?**
A: Yes, but it costs more. Programmatic generators are recommended for common tasks.

**Q: How do I add a new type of input?**
A: Create a custom generator and register it with `register_generator()`.

## Advanced Features

### Batch Generation

Generate multiple inputs at once:

```python
inputs = generator.batch_generate("cli_parsing", task_config, count=1000)
# Returns list of 1000 unique inputs
```

### Diversity Control

Control how varied the outputs are:

```python
# High diversity (default)
generator.generate_input("cli_parsing", task_config)

# For AI-powered generation, adjust temperature
generator_with_ai = InputGenerator(openai_client=client)
# Temperature is set to 0.9 for high diversity
```

### Domain-Specific Customization

Create generators for specific vendors or device types:

```python
def cisco_ios_generator(task_config):
    """Generate Cisco IOS-specific outputs."""
    # Your Cisco-specific logic
    pass

def juniper_generator(task_config):
    """Generate Juniper-specific outputs."""
    # Your Juniper-specific logic
    pass

generator.register_generator("cisco_cli_parsing", cisco_ios_generator)
generator.register_generator("juniper_cli_parsing", juniper_generator)
```

## Best Practices

1. **Start with 2-3 high-quality examples** in prompts.yaml
2. **Use programmatic generators** for speed and cost efficiency
3. **Reserve AI generation** for complex or custom domains
4. **Test your generators** with `demo_input_generation.py`
5. **Validate outputs** to ensure quality
6. **Iterate and refine** based on model performance

## Troubleshooting

**Issue**: Generated inputs are too similar
- **Solution**: Increase randomization ranges or add more variation patterns

**Issue**: Inputs don't match domain requirements
- **Solution**: Create a custom generator with domain-specific logic

**Issue**: AI generation is too slow/expensive
- **Solution**: Switch to programmatic generators for common tasks

**Issue**: Need more control over generation
- **Solution**: Implement custom generators with your specific rules

## Performance

- **Programmatic generation**: ~10,000 inputs/second
- **AI-powered generation**: ~10-100 inputs/minute (depends on API)
- **Memory usage**: Minimal (generators don't store data)
- **Disk usage**: Only the final JSONL dataset

## Next Steps

1. Run the demo: `python scripts/demo_input_generation.py`
2. Generate a small dataset (100 examples) to test
3. Validate the quality of inputs and outputs
4. Scale up to 10,000+ examples
5. Train your student model
6. Evaluate and iterate
