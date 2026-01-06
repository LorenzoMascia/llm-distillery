# LLM Distillery

A production-ready framework for distilling knowledge from large teacher models (GPT-4, Claude) into smaller, efficient student models using synthetic data generation and parameter-efficient fine-tuning (LoRA/QLoRA).

Complete implementation of the knowledge distillation approach described in the HOWAI article ["Fine-Tuning LLMs with a Bigger Teacher"](https://www.howai.cloud/fine-tuning-knowledge-distillation-llm.html).

## Overview

This project implements an end-to-end pipeline for distilling knowledge from a large teacher model (GPT-4, Claude, etc.) into a smaller, more efficient student model using synthetic data generation and parameter-efficient fine-tuning (LoRA/QLoRA).

## Pipeline Architecture

```
┌──────────────┐
│ Teacher LLM  │  (OpenAI API)
└──────┬───────┘
       │ generates synthetic data
┌──────▼───────┐
│   Dataset    │  (domain-specific)
└──────┬───────┘
       │ fine-tuning with QLoRA
┌──────▼───────┐
│ Student LLM  │  (7-8B parameters)
└──────┬───────┘
       │
┌──────▼───────┐
│  Inference   │  (fast, specialized)
└──────────────┘
```

## Features

- **Synthetic Data Generation**: Automated dataset creation using teacher models via OpenAI API
- **Flexible Prompt Management**: YAML-based configuration for different domain tasks
- **Response Parsing**: Intelligent parsing and validation of teacher outputs
- **LoRA Fine-Tuning**: Efficient training with QLoRA on 7-8B parameter models
- **Evaluation Suite**: Comprehensive testing and validation tools

## Project Structure

```
code/
├── config/                 # Configuration files
│   ├── prompts.yaml       # Prompt templates for data generation
│   └── training_config.yaml  # Training hyperparameters
├── src/
│   ├── data_generation/   # Synthetic data creation
│   │   ├── openai_client.py
│   │   ├── prompt_manager.py
│   │   └── dataset_generator.py
│   ├── training/          # Model training and inference
│   │   ├── lora_trainer.py
│   │   └── inference.py
│   └── utils/             # Helper functions
│       ├── parser.py
│       └── logger.py
├── tests/                 # Unit tests
├── data/                  # Generated datasets
└── models/                # Trained model checkpoints
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/LorenzoMascia/llm-distillery.git
cd llm-distillery
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

### 1. Generate Synthetic Dataset

```bash
python -m src.data_generation.dataset_generator \
    --config config/prompts.yaml \
    --output data/processed/training_data.jsonl \
    --num-samples 10000
```

### 2. Train Student Model with LoRA

```bash
python -m src.training.lora_trainer \
    --dataset data/processed/training_data.jsonl \
    --config config/training_config.yaml \
    --output-dir models/student_model
```

### 3. Run Inference

```bash
python -m src.training.inference \
    --model-path models/student_model \
    --input "Your input text here"
```

## Configuration

### Prompts Configuration (`config/prompts.yaml`)

Define your domain-specific prompts and tasks:

```yaml
tasks:
  - name: "cli_parsing"
    instruction: "Parse the following CLI output..."
    examples: [...]
```

### Training Configuration (`config/training_config.yaml`)

Customize training hyperparameters:

```yaml
model:
  base_model: "mistralai/Mistral-7B-v0.1"
  lora_r: 16
  lora_alpha: 32
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (8-12GB VRAM recommended)
- OpenAI API key

## Performance

- **Dataset Generation**: ~1000 examples per hour (depends on API limits)
- **Training Time**: 2-4 hours on a single RTX 3090
- **Inference Speed**: ~50 tokens/second on quantized 7B model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

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

Based on the knowledge distillation approach described at [HOWAI](https://www.howai.cloud/fine-tuning-knowledge-distillation-llm.html).
