# Quick Start Guide

Get up and running with LLM Distillery data anonymization in minutes.

## Prerequisites

- Python 3.8+
- GPU with 8GB+ VRAM (RTX 3060/4060 or better)
- OpenAI API key
- 10GB free disk space

## Installation

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/LorenzoMascia/llm-distillery.git
cd llm-distillery

# Create and activate virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Edit the `.env` file and add your OpenAI API key:

```bash
OPENAI_API_KEY=sk-your-actual-api-key-here
```

## Launch Options

### Option 1: One-Command Quick Start (Recommended)

Run the complete pipeline with a single command:

```bash
# Fast test mode (100 samples, ~30 minutes)
python scripts/quick_start_anonymization.py --num-samples 100

# Production mode (5000 samples, ~3-5 hours)
python scripts/quick_start_anonymization.py --num-samples 5000
```

This automatically:
1.  Verifies prerequisites
2.  Generates training dataset
3.  Trains the model
4.  Tests the results

### Option 2: Using Make

```bash
# Complete pipeline (5000 samples)
make anonymization-all

# Or individual steps:
make anonymization-generate    # Generate dataset
make anonymization-train       # Train model
make anonymization-test        # Test model
make anonymization-interactive # Interactive mode
```

### Option 3: Manual Step-by-Step

#### Step 1: Generate Training Data

```bash
# Quick test (100 samples, ~10 minutes)
python scripts/generate_anonymization_dataset.py \
    --config config/prompts_anonymization.yaml \
    --output data/processed/anonymization_training_data.jsonl \
    --num-samples 100

# Production (5000 samples, ~2-3 hours, ~$50-100 in API costs)
python scripts/generate_anonymization_dataset.py \
    --config config/prompts_anonymization.yaml \
    --output data/processed/anonymization_training_data.jsonl \
    --num-samples 5000
```

#### Step 2: Train Model

```bash
python scripts/train_anonymization_model.py \
    --dataset data/processed/anonymization_training_data.jsonl \
    --config config/training_config_1b_anonymization.yaml \
    --output-dir models/anonymization_1b
```

**Training time:**
- 100 samples: ~5 minutes
- 5000 samples: ~1-2 hours (RTX 3090)

#### Step 3: Test Model

```bash
# Test with examples
python scripts/test_anonymization.py \
    --model-path models/anonymization_1b \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Interactive mode
python scripts/test_anonymization.py \
    --model-path models/anonymization_1b \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --interactive

# Test with file
python scripts/test_anonymization.py \
    --model-path models/anonymization_1b \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --file data/examples/test_input.txt

# Test single text
python scripts/test_anonymization.py \
    --model-path models/anonymization_1b \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --text "Mario Rossi, email: m.rossi@gmail.com, phone: +39 333 1234567"
```

## Expected Output

The model produces structured JSON:

```json
{
  "anonymized_text": "My name is [NAME_1] and my email is [EMAIL_1].",
  "replaced_tokens": [
    {
      "replaced_value": "[NAME_1]",
      "original_value": "Mario Rossi"
    },
    {
      "replaced_value": "[EMAIL_1]",
      "original_value": "m.rossi@gmail.com"
    }
  ]
}
```

## Estimated Times & Costs

### Fast Test (100 samples)
- **Dataset generation**: ~10 minutes
- **Training**: ~5 minutes
- **Total time**: ~15-20 minutes
- **API cost**: ~$1-2

### Production (5000 samples)
- **Dataset generation**: ~2-3 hours
- **Training**: ~1-2 hours (RTX 3090)
- **Total time**: ~3-5 hours
- **API cost**: ~$50-100

## Troubleshooting

### No GPU Available

If you don't have a GPU, you can still run on CPU (slower):

Edit `config/training_config_1b_anonymization.yaml`:
```yaml
hardware:
  cuda_visible_devices: ""  # Empty = CPU only
```

### OpenAI API Rate Limits

If you hit rate limits:
- Start with fewer samples (100 instead of 5000)
- Use a higher tier API key
- The script will retry automatically with backoff

### Import Errors

```bash
# Reinstall package
pip install -e .
```

### CUDA Out of Memory

Reduce batch size in `config/training_config_1b_anonymization.yaml`:
```yaml
training:
  per_device_train_batch_size: 4  # Reduce from 8
  gradient_accumulation_steps: 4   # Increase from 2
```

## Next Steps

1. **Test on your data**: Use `--file your_data.txt`
2. **Integrate into your app**: See [README_ANONYMIZATION.md](README_ANONYMIZATION.md) for Python integration
3. **Customize PII types**: Edit `config/prompts_anonymization.yaml`
4. **Deploy to production**: See integration examples in documentation

## Verify Installation

Check that everything is installed correctly:

```bash
# Test imports
python -c "import torch; import transformers; import peft; print(' All dependencies installed!')"

# Check GPU (if available)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Commands Reference

```bash
# Setup
make setup                          # Create directories
make install                        # Install package

# Full pipeline
make anonymization-all              # Complete workflow

# Individual steps
make anonymization-generate         # Generate dataset
make anonymization-train            # Train model
make anonymization-test             # Test model
make anonymization-interactive      # Interactive mode

# Development
make test                           # Run tests
make clean                          # Clean generated files
make help                           # Show all commands
```

## Documentation

- **[README.md](README.md)** - Project overview
- **[README_ANONYMIZATION.md](README_ANONYMIZATION.md)** - Detailed anonymization guide
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Complete tutorial
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Project structure details

## Support

-  Check documentation for detailed guides
-  [Report issues](https://github.com/LorenzoMascia/llm-distillery/issues)
-  [Ask questions](https://github.com/LorenzoMascia/llm-distillery/discussions)

---

**Ready to start?** Run:
```bash
python scripts/quick_start_anonymization.py --num-samples 100
```
