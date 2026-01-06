# Getting Started Guide

Welcome to LLM Distillery! This guide will help you get up and running quickly.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (8-12GB VRAM recommended for training)
- OpenAI API key
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/LorenzoMascia/llm-distillery.git
cd llm-distillery
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install the package and dependencies
pip install -e .

# Or use make (if available)
make install
```

### 4. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-api-key-here
```

### 5. Setup Project Structure

```bash
# Create necessary directories
make setup
```

## Quick Start

### Option 1: Run Complete Pipeline (Recommended for First Time)

```bash
python scripts/quick_start.py
```

This will:
1. Validate your OpenAI API key
2. Generate a small synthetic dataset (100 examples)
3. Train a student model with LoRA
4. Test inference

**Note**: This is a minimal run for demonstration. For production, use larger datasets.

### Option 2: Step-by-Step Execution

#### Step 1: Generate Synthetic Dataset

```bash
python -m src.data_generation.dataset_generator \
    --config config/prompts.yaml \
    --output data/processed/training_data.jsonl \
    --num-samples 10000
```

**Parameters**:
- `--config`: Path to prompts configuration
- `--output`: Where to save the dataset
- `--num-samples`: Number of examples per task
- `--tasks`: (Optional) Specific tasks to generate

**Expected output**:
- `data/processed/training_data.jsonl`: Training dataset
- `data/processed/failed_examples.json`: Failed generations (for debugging)

#### Step 2: Train Model with LoRA

```bash
python -m src.training.lora_trainer \
    --config config/training_config.yaml \
    --dataset data/processed/training_data.jsonl \
    --output-dir models/student_model
```

**Parameters**:
- `--config`: Training configuration file
- `--dataset`: Path to training dataset
- `--output-dir`: Where to save the trained model

**Expected output**:
- `models/student_model/`: Trained LoRA adapter and tokenizer
- Training logs in `logs/` directory

**Training time**: 2-4 hours on RTX 3090 (depends on dataset size)

#### Step 3: Run Inference

```bash
# Interactive mode
python -m src.training.inference \
    --model-path models/student_model \
    --interactive

# Single query
python -m src.training.inference \
    --model-path models/student_model \
    --input "Your input text here"
```

## Configuration

### Customize Prompts (`config/prompts.yaml`)

Add new tasks or modify existing ones:

```yaml
tasks:
  - name: "your_custom_task"
    description: "Description of the task"
    instruction: "Instruction for the model"
    system_message: "System message (optional)"
    output_schema:  # Optional JSON schema
      type: "object"
      required: ["field1", "field2"]
      properties:
        field1:
          type: "string"
```

### Customize Training (`config/training_config.yaml`)

Adjust training parameters:

```yaml
model:
  base_model: "mistralai/Mistral-7B-v0.1"  # Change base model

lora:
  r: 16                # LoRA rank (higher = more parameters)
  lora_alpha: 32       # LoRA scaling factor

training:
  num_train_epochs: 3  # Number of training epochs
  learning_rate: 2.0e-4  # Learning rate
```

## Common Use Cases

### 1. Generate Dataset Only

```bash
python -m src.data_generation.dataset_generator \
    --config config/prompts.yaml \
    --output data/processed/my_dataset.jsonl \
    --num-samples 5000 \
    --tasks cli_parsing config_generation
```

### 2. Train on Custom Dataset

```bash
python -m src.training.lora_trainer \
    --config config/training_config.yaml \
    --dataset path/to/your/dataset.jsonl \
    --output-dir models/custom_model
```

### 3. Evaluate Trained Model

```bash
python scripts/evaluate_model.py \
    --model-path models/student_model \
    --test-data data/processed/test_data.jsonl \
    --output evaluation_results.json
```

## Monitoring Training

### Using TensorBoard

```bash
# Training logs are saved to ./runs by default
tensorboard --logdir=./runs
```

### Using Weights & Biases (Optional)

1. Enable in `config/training_config.yaml`:
```yaml
monitoring:
  use_wandb: true
  wandb_project: "my-project"
```

2. Login to W&B:
```bash
wandb login
```

## Troubleshooting

### OpenAI API Rate Limits

If you hit rate limits:
- Reduce batch size in dataset generation
- Add delays between requests
- Use a higher tier API key

### CUDA Out of Memory

If training fails with OOM:
- Reduce `per_device_train_batch_size` in training config
- Enable gradient checkpointing (already enabled by default)
- Use smaller model (e.g., 7B instead of 13B)
- Ensure 4-bit quantization is enabled

### Low Quality Outputs

If the model generates poor results:
- Increase dataset size (10,000+ examples recommended)
- Improve prompt quality in `config/prompts.yaml`
- Add more examples to each task
- Increase LoRA rank (16 → 32)
- Train for more epochs

### Import Errors

If you see import errors:
```bash
# Reinstall in development mode
pip install -e .
```

## Next Steps

1. **Customize for Your Domain**: Modify `config/prompts.yaml` with your domain-specific tasks
2. **Generate Larger Dataset**: Aim for 10,000-50,000 examples for production
3. **Experiment with Models**: Try different base models (LLaMA, Qwen, etc.)
4. **Fine-tune Hyperparameters**: Adjust LoRA rank, learning rate, etc.
5. **Evaluate Performance**: Use `scripts/evaluate_model.py` to measure quality

## Resources

- **Documentation**: See [README.md](README.md) and [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **Examples**: Check `config/prompts.yaml` for task examples
- **Article**: Read the full guide at [HOWAI](https://howai.com/fine-tuning-knowledge-distillation-llm.html)

## Getting Help

- Check existing [Issues](https://github.com/LorenzoMascia/llm-distillery/issues)
- Read [CONTRIBUTING.md](CONTRIBUTING.md)
- Open a new issue with detailed information

Happy distilling! 🚀
