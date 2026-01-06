"""Tests for training module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestLoRATrainer:
    """Test cases for LoRATrainer."""

    def test_load_config(self, tmp_path):
        """Test loading training configuration."""
        from src.training.lora_trainer import LoRATrainer

        config_content = """
model:
  base_model: "test/model"
  torch_dtype: "float16"
quantization:
  enabled: true
  load_in_4bit: true
lora:
  r: 8
  lora_alpha: 16
  target_modules: ["q_proj", "v_proj"]
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
training:
  output_dir: "./output"
  num_train_epochs: 1
  per_device_train_batch_size: 1
  learning_rate: 0.0001
"""
        config_file = tmp_path / "train_config.yaml"
        config_file.write_text(config_content)

        trainer = LoRATrainer(str(config_file))

        assert trainer.config["model"]["base_model"] == "test/model"
        assert trainer.config["lora"]["r"] == 8
        assert trainer.config["training"]["num_train_epochs"] == 1


class TestInferenceEngine:
    """Test cases for InferenceEngine."""

    @patch("src.training.inference.AutoModelForCausalLM")
    @patch("src.training.inference.AutoTokenizer")
    @patch("src.training.inference.PeftModel")
    @patch("src.training.inference.PeftConfig")
    def test_initialization(
        self, mock_peft_config, mock_peft_model, mock_tokenizer, mock_model
    ):
        """Test inference engine initialization."""
        from src.training.inference import InferenceEngine

        # Mock the config
        mock_config = Mock()
        mock_config.base_model_name_or_path = "test/base-model"
        mock_peft_config.from_pretrained.return_value = mock_config

        # Mock model and tokenizer
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        mock_peft_model.from_pretrained.return_value = Mock()

        engine = InferenceEngine(model_path="test/model")

        assert engine.model is not None
        assert engine.tokenizer is not None

    def test_format_prompt(self):
        """Test prompt formatting."""
        from src.training.inference import InferenceEngine

        # Create a minimal mock engine
        with patch("src.training.inference.AutoModelForCausalLM"), patch(
            "src.training.inference.AutoTokenizer"
        ), patch("src.training.inference.PeftModel"), patch(
            "src.training.inference.PeftConfig"
        ) as mock_config:

            mock_cfg = Mock()
            mock_cfg.base_model_name_or_path = "test/model"
            mock_config.from_pretrained.return_value = mock_cfg

            engine = InferenceEngine(model_path="test/model")

            prompt = engine.format_prompt(
                instruction="Parse the data", input_text="test input"
            )

            assert "Parse the data" in prompt
            assert "test input" in prompt
            assert "### Instruction:" in prompt
            assert "### Input:" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
