# Makefile for LLM Knowledge Distillation Pipeline

.PHONY: help install install-dev setup clean test lint format train generate infer

help:
	@echo "Available commands:"
	@echo "  make install      - Install package and dependencies"
	@echo "  make install-dev  - Install with development dependencies"
	@echo "  make setup        - Setup environment and create directories"
	@echo "  make clean        - Remove generated files and caches"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make generate     - Generate synthetic dataset"
	@echo "  make train        - Train model with LoRA"
	@echo "  make infer        - Run inference"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,monitoring]"

setup:
	@echo "Setting up project structure..."
	mkdir -p data/raw data/processed models logs
	touch data/raw/.gitkeep data/processed/.gitkeep models/.gitkeep logs/.gitkeep
	@echo "Copying .env.example to .env..."
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@echo "Setup complete! Don't forget to add your OPENAI_API_KEY to .env"

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist
	@echo "Clean complete!"

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/ --max-line-length=100

format:
	black src/ tests/ --line-length=100
	isort src/ tests/

# Data generation
generate:
	python -m src.data_generation.dataset_generator \
		--config config/prompts.yaml \
		--output data/processed/training_data.jsonl \
		--num-samples 1000

# Training
train:
	python -m src.training.lora_trainer \
		--config config/training_config.yaml \
		--dataset data/processed/training_data.jsonl \
		--output-dir models/student_model

# Inference
infer:
	python -m src.training.inference \
		--model-path models/student_model \
		--interactive

# Quick start - full pipeline
quickstart: setup install generate train
	@echo "Pipeline complete! Model saved in models/student_model"
