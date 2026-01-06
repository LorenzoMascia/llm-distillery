"""Tests for data generation module."""

import pytest
from unittest.mock import Mock, patch
from src.data_generation.prompt_manager import PromptManager
from src.data_generation.openai_client import OpenAIClient


class TestPromptManager:
    """Test cases for PromptManager."""

    def test_load_config(self, tmp_path):
        """Test loading configuration from file."""
        # Create temporary config file
        config_content = """
tasks:
  - name: "test_task"
    instruction: "Do something"
    system_message: "You are helpful"
generation:
  temperature: 0.5
dataset:
  train_split: 0.8
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        # Load config
        manager = PromptManager(str(config_file))

        assert len(manager.tasks) == 1
        assert manager.tasks[0]["name"] == "test_task"
        assert manager.generation_params["temperature"] == 0.5

    def test_get_task(self, tmp_path):
        """Test retrieving task by name."""
        config_content = """
tasks:
  - name: "task1"
    instruction: "First task"
  - name: "task2"
    instruction: "Second task"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        manager = PromptManager(str(config_file))

        task = manager.get_task("task1")
        assert task is not None
        assert task["instruction"] == "First task"

        # Test non-existent task
        assert manager.get_task("task999") is None

    def test_get_system_message(self, tmp_path):
        """Test retrieving system message."""
        config_content = """
tasks:
  - name: "test"
    system_message: "Test message"
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        manager = PromptManager(str(config_file))

        assert manager.get_system_message("test") == "Test message"


class TestOpenAIClient:
    """Test cases for OpenAIClient."""

    @patch("src.data_generation.openai_client.OpenAI")
    def test_initialization(self, mock_openai):
        """Test client initialization."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient(model="gpt-4")

            assert client.model == "gpt-4"
            assert client.api_key == "test-key"

    def test_count_tokens(self):
        """Test token counting."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient()

            text = "Hello, world!"
            count = client.count_tokens(text)

            assert count > 0
            assert isinstance(count, int)

    @patch("src.data_generation.openai_client.OpenAI")
    def test_generate_with_retry(self, mock_openai):
        """Test generation with retry logic."""
        # Mock API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated response"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client_instance

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient()
            result = client.generate("Test prompt")

            assert result == "Generated response"

    def test_estimate_cost(self):
        """Test cost estimation."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            client = OpenAIClient(model="gpt-4-turbo-preview")

            cost = client.estimate_cost(
                prompt_tokens=1000, completion_tokens=500, model="gpt-4-turbo-preview"
            )

            assert cost > 0
            assert isinstance(cost, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
