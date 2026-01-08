# LLM Distillery - Data Anonymization

A production-ready framework for distilling knowledge from large teacher models (GPT-4, Claude) into smaller, efficient student models using synthetic data generation and parameter-efficient fine-tuning (LoRA/QLoRA).

**Current Use Case: PII (Personally Identifiable Information) Anonymization**

Fine-tune a 1B parameter model to anonymize sensitive data in text, producing structured JSON output with replaced tokens.

Complete implementation of the knowledge distillation approach described in the HOWAI article ["Fine-Tuning LLMs with a Bigger Teacher"](https://www.howai.cloud/fine-tuning-knowledge-distillation-llm.html).

## Overview

This project implements an end-to-end pipeline for distilling knowledge from a large teacher model (GPT-4, Claude, etc.) into a smaller, more efficient student model using synthetic data generation and parameter-efficient fine-tuning (LoRA/QLoRA).

**Current Implementation:** Data anonymization model that takes text input and returns JSON with anonymized text and token mappings.

## Use Case: Data Anonymization

Given an input text, the model produces:

```json
{
  "anonymized_text": "My name is [NAME_1] and my email is [EMAIL_1].",
  "replaced_tokens": [
    {"replaced_value": "[NAME_1]", "original_value": "Mario Rossi"},
    {"replaced_value": "[EMAIL_1]", "original_value": "mario.rossi@gmail.com"}
  ]
}
```

### Supported PII Types

-  Personal Names
-  Email Addresses
-  Phone Numbers (international formats)
-  Physical Addresses
-  Dates (birth dates, appointments)
-  Tax IDs / Social Security Numbers
-  Credit Card Numbers
-  Bank Account Numbers (IBAN)
-  VAT Numbers
-  Patient/Insurance IDs
-  Employee IDs

## Pipeline Architecture

```
┌──────────────┐
│ Teacher LLM  │  (GPT-4 via OpenAI API)
│              │  Generates synthetic anonymization examples
└──────┬───────┘
       │ creates training dataset
┌──────▼────────┐
│   Dataset     │  (5,000-10,000 examples)
│   JSONL       │  Input texts → Anonymized JSON
└──────┬────────┘
       │ fine-tuning with LoRA
┌──────▼────────┐
│ Student LLM   │  (TinyLlama 1.1B)
│   1B params   │  Specialized for anonymization
└──────┬────────┘
       │
┌──────▼────────┐
│  Inference    │  (fast, accurate, structured output)
└───────────────┘
```

## Features

- **Synthetic Data Generation**: Automated dataset creation using GPT-4 as teacher model
- **Flexible Prompt Management**: YAML-based configuration for anonymization tasks
- **Structured Output**: JSON format with anonymized text and token mappings
- **LoRA Fine-Tuning**: Efficient training with LoRA on 1B parameter model (TinyLlama)
- **Multiple Test Modes**: Interactive, file-based, and single-text testing
- **Production Ready**: Easy integration via Python API or REST API

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/LorenzoMascia/llm-distillery.git
cd llm-distillery

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure OpenAI API key
cp .env.example .env
# Edit .env and add your API key
```

### One-Command Setup

```bash
# Run complete pipeline (generates 5000 examples, trains model, tests)
python scripts/quick_start_anonymization.py --num-samples 5000
```

Or using Make:

```bash
make anonymization-all
```

### Step-by-Step Workflow

#### 1. Generate Training Dataset

```bash
python scripts/generate_anonymization_dataset.py \
    --config config/prompts_anonymization.yaml \
    --output data/processed/anonymization_training_data.jsonl \
    --num-samples 5000
```

Or: `make anonymization-generate`

#### 2. Train Model

```bash
python scripts/train_anonymization_model.py \
    --dataset data/processed/anonymization_training_data.jsonl \
    --config config/training_config_1b_anonymization.yaml \
    --output-dir models/anonymization_1b
```

Or: `make anonymization-train`

#### 3. Test Model

```bash
# Interactive mode
python scripts/test_anonymization.py --interactive

# Test with file
python scripts/test_anonymization.py --file data/examples/test_input.txt

# Test single text
python scripts/test_anonymization.py --text "Mario Rossi, email: m.rossi@example.com"
```

Or: `make anonymization-test` / `make anonymization-interactive`

## Project Structure

```
llm-distillery/
├── config/                                      # Configuration files
│   ├── prompts_anonymization.yaml              # Prompts for data generation
│   └── training_config_1b_anonymization.yaml   # Training configuration (1B model)
├── src/
│   ├── data_generation/                        # Synthetic data creation
│   │   ├── openai_client.py                   # OpenAI API client
│   │   ├── prompt_manager.py                  # Prompt template management
│   │   ├── input_generator.py                 # Input text generation
│   │   └── dataset_generator.py               # Complete dataset generation
│   ├── training/                              # Model training and inference
│   │   ├── lora_trainer.py                    # LoRA fine-tuning
│   │   └── inference.py                       # Model inference
│   └── utils/                                 # Helper functions
│       ├── parser.py                          # Response parsing
│       └── logger.py                          # Logging utilities
├── scripts/                                    # Executable scripts
│   ├── generate_anonymization_dataset.py      # Dataset generation
│   ├── train_anonymization_model.py           # Model training
│   ├── test_anonymization.py                  # Model testing
│   └── quick_start_anonymization.py           # One-command setup
├── data/
│   ├── examples/                              # Example data
│   │   ├── anonymization_samples.jsonl       # Sample training data
│   │   └── test_input.txt                    # Test input file
│   └── processed/                             # Generated datasets
├── models/                                     # Trained model checkpoints
├── tests/                                      # Unit tests
└── README_ANONYMIZATION.md                     # Detailed anonymization guide
```

## Configuration

### Prompt Configuration

[config/prompts_anonymization.yaml](config/prompts_anonymization.yaml) defines:
- Task description and instructions
- System message for the model
- Output JSON schema
- Example input/output pairs
- Generation parameters

### Training Configuration

[config/training_config_1b_anonymization.yaml](config/training_config_1b_anonymization.yaml) defines:
- Base model (TinyLlama 1.1B)
- LoRA parameters (rank, alpha, target modules)
- Training hyperparameters
- Evaluation strategy
- Hardware optimization

## Requirements

- Python 3.8+
- GPU with 8GB+ VRAM (for training)
- OpenAI API key (for dataset generation)

### Hardware Recommendations

**For Training:**
- GPU: RTX 3060/4060 or better (8GB+ VRAM)
- RAM: 16GB system memory
- Storage: 10GB free space

**For Inference:**
- GPU: Any CUDA-capable GPU (4GB+ VRAM)
- CPU: Modern x86_64 processor
- RAM: 8GB system memory

## Performance

With a well-trained model (5000+ examples, 5 epochs):

- **Accuracy**: ~95% PII recognition for common types
- **Inference Speed**: ~50 tokens/second (RTX 3090)
- **Latency**: 1-3 seconds for average text (100-200 words)
- **Model Size**: ~2.5GB (including LoRA adapters)
- **Training Time**: 1-2 hours (5000 samples, RTX 3090)

## Example Usage

### Python Integration

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, "models/anonymization_1b")
tokenizer = AutoTokenizer.from_pretrained("models/anonymization_1b")

def anonymize(text: str) -> dict:
    prompt = f"""### Instruction:
Analyze the following text and anonymize all personally identifiable information (PII). Return a JSON object with the anonymized text and all replaced tokens.

### Input:
{text}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:")[-1].strip()
    return json.loads(response)

# Use the function
result = anonymize("Mario Rossi lives at Via Roma 1, Milano. Email: m.rossi@email.com")
print(result["anonymized_text"])
print(result["replaced_tokens"])
```

## Documentation

- **[README_ANONYMIZATION.md](README_ANONYMIZATION.md)** - Complete anonymization guide
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Getting started tutorial
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed project structure
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

## Available Commands (Make)

```bash
make help                        # Show all available commands
make anonymization-generate      # Generate anonymization dataset
make anonymization-train         # Train anonymization model
make anonymization-test          # Test model with examples
make anonymization-interactive   # Interactive testing mode
make anonymization-all          # Complete pipeline
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{llm-distillery,
  title={LLM Distillery: A Framework for Knowledge Distillation in Large Language Models},
  author={Mascia, Lorenzo},
  year={2025},
  publisher={GitHub},
  url={https://github.com/LorenzoMascia/llm-distillery}
}
```

## Acknowledgments

This project is based on the knowledge distillation approach described at [HOWAI](https://www.howai.cloud/fine-tuning-knowledge-distillation-llm.html).

## Support

-  Full Documentation: See [README_ANONYMIZATION.md](README_ANONYMIZATION.md)
-  Issues: Report bugs on [GitHub Issues](https://github.com/LorenzoMascia/llm-distillery/issues)
-  Discussions: Join [GitHub Discussions](https://github.com/LorenzoMascia/llm-distillery/discussions)

---

**Note:** This framework can be easily adapted for other use cases by modifying the prompt configuration and training data. The anonymization use case serves as a comprehensive example of the distillation pipeline.
