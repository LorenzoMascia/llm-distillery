"""Setup script for LLM Knowledge Distillation Pipeline."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="llm-distillery",
    version="1.0.0",
    author="Lorenzo Mascia",
    author_email="your.email@example.com",
    description="A production-ready framework for distilling knowledge from large teacher models into smaller, efficient student models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LorenzoMascia/llm-distillery",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.25.0",
        "datasets>=2.16.0",
        "openai>=1.10.0",
        "tiktoken>=0.5.2",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "loguru>=0.7.2",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "jsonlines>=4.0.0",
        "scikit-learn>=1.3.0",
        "evaluate>=0.4.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "monitoring": [
            "wandb>=0.16.0",
            "tensorboard>=2.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "distill-generate=src.data_generation.dataset_generator:main",
            "distill-train=src.training.lora_trainer:main",
            "distill-infer=src.training.inference:main",
        ],
    },
)
