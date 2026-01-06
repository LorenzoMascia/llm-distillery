# Project Structure

```
llm-knowledge-distillation/
│
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── CONTRIBUTING.md                    # Contribution guidelines
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup script
├── Makefile                          # Build and automation commands
├── .env.example                      # Environment variables template
├── .gitignore                        # Git ignore rules
│
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI/CD
│
├── config/
│   ├── prompts.yaml                  # Prompt templates and tasks
│   └── training_config.yaml          # LoRA training configuration
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_generation/              # Synthetic data generation
│   │   ├── __init__.py
│   │   ├── openai_client.py          # OpenAI API client
│   │   ├── prompt_manager.py         # Prompt configuration manager
│   │   └── dataset_generator.py      # Dataset generation pipeline
│   │
│   ├── training/                     # Model training and inference
│   │   ├── __init__.py
│   │   ├── lora_trainer.py           # LoRA/QLoRA training
│   │   └── inference.py              # Inference engine
│   │
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── parser.py                 # Response parser and validator
│       └── logger.py                 # Logging configuration
│
├── tests/                            # Test suite
│   ├── __init__.py
│   ├── test_parser.py                # Parser tests
│   ├── test_data_generation.py       # Data generation tests
│   └── test_training.py              # Training tests
│
├── scripts/                          # Utility scripts
│   ├── __init__.py
│   ├── quick_start.py                # End-to-end pipeline script
│   └── evaluate_model.py             # Model evaluation script
│
├── data/                             # Data directory
│   ├── raw/                          # Raw data (if any)
│   │   └── .gitkeep
│   └── processed/                    # Generated datasets
│       └── .gitkeep
│
├── models/                           # Trained models
│   └── .gitkeep
│
└── logs/                             # Application logs
    └── .gitkeep
```

## Module Descriptions

### Data Generation (`src/data_generation/`)

- **openai_client.py**: Client for interacting with OpenAI API
  - API key management
  - Request handling with retry logic
  - Token counting and cost estimation
  - Batch generation support

- **prompt_manager.py**: Manages prompt templates and configurations
  - Loads tasks from YAML configuration
  - Formats prompts with examples
  - Handles multiple task types
  - Schema management

- **dataset_generator.py**: Orchestrates synthetic data generation
  - Generates training examples using teacher model
  - Validates responses against schemas
  - Saves datasets in JSONL format
  - Handles train/validation splits

### Training (`src/training/`)

- **lora_trainer.py**: LoRA/QLoRA fine-tuning implementation
  - Quantized model loading (4-bit)
  - LoRA adapter configuration
  - Training loop with HuggingFace Transformers
  - Model checkpointing and evaluation

- **inference.py**: Inference engine for trained models
  - Loads LoRA adapters
  - Generation with customizable parameters
  - Batch inference support
  - Interactive mode

### Utilities (`src/utils/`)

- **parser.py**: Response parsing and validation
  - JSON extraction from markdown
  - Schema validation
  - Training example formatting
  - Batch processing

- **logger.py**: Centralized logging configuration
  - Console and file logging
  - Configurable log levels
  - Rotation and retention

### Scripts (`scripts/`)

- **quick_start.py**: Complete pipeline execution
  - Generates dataset
  - Trains model
  - Tests inference
  - All-in-one script for quick experimentation

- **evaluate_model.py**: Model evaluation suite
  - Runs inference on test set
  - Calculates accuracy metrics
  - Per-task performance analysis
  - Generates evaluation reports

## Configuration Files

### `config/prompts.yaml`

Defines:
- Task definitions (CLI parsing, config generation, etc.)
- System messages for each task
- Output schemas and validation rules
- Example inputs and outputs
- Generation parameters

### `config/training_config.yaml`

Defines:
- Base model configuration
- Quantization settings (QLoRA)
- LoRA parameters (rank, alpha, target modules)
- Training hyperparameters
- Monitoring and logging options

## Data Flow

```
1. Teacher Model (OpenAI API)
   ↓ (generates synthetic examples)
2. Dataset (JSONL)
   ↓ (tokenized and formatted)
3. Student Model Training (LoRA)
   ↓ (fine-tuned adapter)
4. Trained Model (Student + Adapter)
   ↓ (inference)
5. Production Deployment
```

## Key Features

✓ Modular architecture
✓ Comprehensive test suite
✓ CI/CD pipeline (GitHub Actions)
✓ Configurable via YAML
✓ CLI tools and scripts
✓ Extensive documentation
✓ Production-ready code structure
✓ Cost estimation and monitoring
✓ Error handling and logging
✓ Type hints and docstrings

## Quick Commands

```bash
# Setup
make setup                 # Create directories and .env
make install               # Install package

# Development
make test                  # Run tests
make lint                  # Check code quality
make format                # Format code

# Pipeline
make generate              # Generate dataset
make train                 # Train model
make infer                 # Run inference

# All-in-one
make quickstart            # Complete pipeline
```
