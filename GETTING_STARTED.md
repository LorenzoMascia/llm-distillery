# Getting Started Guide - Data Anonymization

Welcome to LLM Distillery! This guide will help you get up and running quickly with the data anonymization use case.

## What You'll Build

A specialized 1B parameter model that anonymizes Personally Identifiable Information (PII) in text and returns structured JSON output.

**Input**: "My name is Mario Rossi, email: mario.rossi@gmail.com"
**Output**:
```json
{
  "anonymized_text": "My name is [NAME_1], email: [EMAIL_1]",
  "replaced_tokens": [
    {"replaced_value": "[NAME_1]", "original_value": "Mario Rossi"},
    {"replaced_value": "[EMAIL_1]", "original_value": "mario.rossi@gmail.com"}
  ]
}
```

## Prerequisites

- Python 3.8 or higher
- GPU with 8GB+ VRAM (for training) - RTX 3060/4060 or better
- OpenAI API key (for generating training data)
- Git
- 10GB free disk space

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
# Install all required packages
pip install -r requirements.txt

# Or use the setup script
pip install -e .
```

### 4. Configure OpenAI API Key

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# Open .env in your editor and replace:
# OPENAI_API_KEY=your_openai_api_key_here
# with your actual key:
# OPENAI_API_KEY=sk-your-actual-key-here
```

### 5. Verify Installation

```bash
# Test that imports work
python -c "import torch; import transformers; import peft; print('All dependencies installed!')"
```

## Quick Start (Recommended)

### One-Command Pipeline

Run the entire workflow with a single command:

```bash
python scripts/quick_start_anonymization.py --num-samples 5000
```

Or using Make:

```bash
make anonymization-all
```

This will:
1. ✅ Verify prerequisites and API key
2. 📊 Generate 5,000 training examples using GPT-4
3. 🎓 Train TinyLlama 1.1B with LoRA
4. 🧪 Test the model with examples

**Estimated time**: 3-5 hours total
- Dataset generation: 2-3 hours
- Training: 1-2 hours

**Estimated cost**: $50-100 in OpenAI API credits

### Fast Test Mode (Recommended for First Try)

Test the pipeline with a small dataset:

```bash
python scripts/quick_start_anonymization.py --num-samples 100 --fast-test
```

This creates a quick prototype in ~30 minutes.

## Step-by-Step Workflow

### Step 1: Generate Training Dataset

Generate synthetic anonymization examples using GPT-4 as teacher:

```bash
python scripts/generate_anonymization_dataset.py \
    --config config/prompts_anonymization.yaml \
    --output data/processed/anonymization_training_data.jsonl \
    --num-samples 5000 \
    --model gpt-4-turbo-preview
```

**Or using Make:**
```bash
make anonymization-generate
```

**Parameters**:
- `--config`: Prompt configuration file (defines PII types and examples)
- `--output`: Output path for training dataset
- `--num-samples`: Number of examples to generate (5000-10000 recommended)
- `--model`: OpenAI model to use as teacher
- `--verbose`: Enable detailed logging

**What happens**:
- GPT-4 generates diverse text examples containing PII
- Each example includes the anonymized version and token mappings
- Progress is saved (you can stop and resume)

**Expected output**:
- `data/processed/anonymization_training_data.jsonl`: Training dataset in JSONL format
- Each line contains a formatted prompt with input and expected JSON output

**Tips**:
- Start with 100 samples for testing (`--num-samples 100`)
- For production, use 5,000-10,000 samples
- Generation speed: ~10-20 examples per minute

### Step 2: Train the Model

Fine-tune TinyLlama 1.1B on the generated dataset:

```bash
python scripts/train_anonymization_model.py \
    --dataset data/processed/anonymization_training_data.jsonl \
    --config config/training_config_1b_anonymization.yaml \
    --output-dir models/anonymization_1b
```

**Or using Make:**
```bash
make anonymization-train
```

**Parameters**:
- `--dataset`: Path to training dataset (JSONL)
- `--config`: Training configuration file
- `--output-dir`: Directory to save model checkpoints
- `--resume-from-checkpoint`: Resume from a checkpoint (optional)
- `--verbose`: Enable detailed logging

**What happens**:
- Downloads TinyLlama 1.1B base model (~2.5GB)
- Trains LoRA adapters on the anonymization task
- Saves checkpoints every 50 steps
- Evaluates on validation set

**Expected output**:
- `models/anonymization_1b/`: Trained model directory
  - `adapter_model.bin`: LoRA adapters
  - `adapter_config.json`: LoRA configuration
  - `tokenizer/`: Tokenizer files
- `logs/anonymization_1b/`: Training logs
- `runs/anonymization_1b/`: TensorBoard logs

**Training time**:
- 100 samples: ~5 minutes
- 5,000 samples: ~1-2 hours (RTX 3090)
- 10,000 samples: ~3-4 hours

**Hardware requirements**:
- GPU: 8GB+ VRAM
- RAM: 16GB system memory
- Storage: 10GB free space

### Step 3: Test the Model

Test your trained model:

#### 3a. Test with Predefined Examples

```bash
python scripts/test_anonymization.py \
    --model-path models/anonymization_1b \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

**Or using Make:**
```bash
make anonymization-test
```

#### 3b. Interactive Mode

```bash
python scripts/test_anonymization.py \
    --model-path models/anonymization_1b \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --interactive
```

**Or using Make:**
```bash
make anonymization-interactive
```

Enter text interactively and see instant anonymization results.

#### 3c. Test with File

```bash
python scripts/test_anonymization.py \
    --model-path models/anonymization_1b \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --file data/examples/test_input.txt
```

#### 3d. Test Single Text

```bash
python scripts/test_anonymization.py \
    --model-path models/anonymization_1b \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --text "Mario Rossi lives in Via Roma 1, Milano. Email: m.rossi@example.com"
```

## Configuration

### Customize PII Types (`config/prompts_anonymization.yaml`)

Add or modify PII types to detect:

```yaml
tasks:
  - name: "data_anonymization"
    examples:
      - input: "Your custom input with PII"
        output:
          anonymized_text: "Your [PLACEHOLDER] text"
          replaced_tokens:
            - replaced_value: "[PLACEHOLDER]"
              original_value: "original value"
```

### Adjust Training Parameters (`config/training_config_1b_anonymization.yaml`)

```yaml
model:
  base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Or try other 1B models

lora:
  r: 8                     # LoRA rank (higher = more parameters, better quality)
  lora_alpha: 16           # LoRA scaling

training:
  num_train_epochs: 5      # More epochs = better training
  per_device_train_batch_size: 8  # Adjust based on GPU memory
  learning_rate: 3.0e-4    # Learning rate
```

## Monitoring Training

### TensorBoard

View training progress in real-time:

```bash
# Start TensorBoard
tensorboard --logdir=runs/anonymization_1b

# Open browser to http://localhost:6006
```

### Weights & Biases (Optional)

Enable W&B in `config/training_config_1b_anonymization.yaml`:

```yaml
monitoring:
  use_wandb: true
  wandb_project: "anonymization-1b"
```

Then login:

```bash
wandb login
```

## Troubleshooting

### OpenAI API Rate Limits

**Symptom**: "Rate limit exceeded" errors during dataset generation

**Solutions**:
- Use a higher tier API key
- Reduce `--num-samples` and run multiple times
- Add delays between requests (modify `generate_anonymization_dataset.py`)

### CUDA Out of Memory

**Symptom**: "CUDA out of memory" during training

**Solutions**:
```yaml
# In config/training_config_1b_anonymization.yaml:
training:
  per_device_train_batch_size: 4  # Reduce from 8
  gradient_accumulation_steps: 4   # Increase from 2
```

### Model Not Generating Valid JSON

**Symptom**: Model outputs invalid JSON or doesn't follow format

**Solutions**:
- Increase training samples (5,000 → 10,000)
- Train for more epochs (5 → 8)
- Check that training completed without errors
- Verify examples in `prompts_anonymization.yaml` are correct

### Low PII Detection Accuracy

**Symptom**: Model misses some PII or makes incorrect replacements

**Solutions**:
- Add more diverse examples to `prompts_anonymization.yaml`
- Increase dataset size (10,000+ examples)
- Train for more epochs
- Fine-tune temperature during inference (default: 0.1)

### Import Errors

**Symptom**: "ModuleNotFoundError" when running scripts

**Solutions**:
```bash
# Reinstall in development mode
pip install -e .

# Or add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%cd%  # Windows
```

## Next Steps

### 1. Test on Your Data

```bash
python scripts/test_anonymization.py \
    --model-path models/anonymization_1b \
    --file your_data.txt
```

### 2. Integrate into Your Application

See [README_ANONYMIZATION.md](README_ANONYMIZATION.md) for Python integration examples.

### 3. Fine-Tune for Your Domain

- Add domain-specific PII types in `prompts_anonymization.yaml`
- Generate more training data focused on your domain
- Retrain the model

### 4. Deploy to Production

See production integration examples in [README_ANONYMIZATION.md](README_ANONYMIZATION.md):
- Python API integration
- FastAPI REST endpoint
- Batch processing

## Resources

- **[README.md](README.md)** - Main project overview
- **[README_ANONYMIZATION.md](README_ANONYMIZATION.md)** - Detailed anonymization guide
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Project structure details
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[HOWAI Article](https://www.howai.cloud/fine-tuning-knowledge-distillation-llm.html)** - Knowledge distillation approach

## Available Commands

```bash
make help                        # Show all commands
make anonymization-generate      # Generate dataset
make anonymization-train         # Train model
make anonymization-test          # Test with examples
make anonymization-interactive   # Interactive mode
make anonymization-all          # Complete pipeline
```

## Getting Help

- 📖 Check [README_ANONYMIZATION.md](README_ANONYMIZATION.md) for detailed documentation
- 🐛 Report issues on [GitHub Issues](https://github.com/LorenzoMascia/llm-distillery/issues)
- 💬 Ask questions in [GitHub Discussions](https://github.com/LorenzoMascia/llm-distillery/discussions)

Happy anonymizing! 🔒🚀
