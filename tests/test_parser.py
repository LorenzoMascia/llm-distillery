"""Tests for response parser."""

import pytest
from src.utils.parser import ResponseParser


class TestResponseParser:
    """Test cases for ResponseParser."""

    def setup_method(self):
        """Setup test fixtures."""
        self.parser = ResponseParser(strict_mode=False)

    def test_parse_json_response(self):
        """Test parsing JSON from response."""
        response = """```json
{
    "interface": "GigabitEthernet0/0/0",
    "admin_state": "up",
    "oper_state": "up"
}
```"""

        result = self.parser.parse_json_response(response)

        assert result is not None
        assert result["interface"] == "GigabitEthernet0/0/0"
        assert result["admin_state"] == "up"

    def test_parse_json_without_markdown(self):
        """Test parsing plain JSON."""
        response = '{"status": "success", "value": 42}'

        result = self.parser.parse_json_response(response)

        assert result is not None
        assert result["status"] == "success"
        assert result["value"] == 42

    def test_parse_invalid_json(self):
        """Test handling of invalid JSON."""
        response = "This is not JSON"

        result = self.parser.parse_json_response(response)

        # Should return None in non-strict mode
        assert result is None

    def test_validate_schema_success(self):
        """Test schema validation with valid data."""
        schema = {
            "required": ["name", "value"],
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "integer"},
            },
        }

        data = {"name": "test", "value": 123}

        assert self.parser.validate_schema(data, schema) is True

    def test_validate_schema_missing_field(self):
        """Test schema validation with missing required field."""
        schema = {
            "required": ["name", "value"],
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "integer"},
            },
        }

        data = {"name": "test"}  # missing 'value'

        assert self.parser.validate_schema(data, schema) is False

    def test_validate_schema_wrong_type(self):
        """Test schema validation with wrong type."""
        schema = {
            "required": ["value"],
            "properties": {
                "value": {"type": "integer"},
            },
        }

        data = {"value": "not an integer"}

        assert self.parser.validate_schema(data, schema) is False

    def test_validate_enum(self):
        """Test enum validation."""
        schema = {
            "required": ["status"],
            "properties": {
                "status": {"type": "string", "enum": ["up", "down"]},
            },
        }

        valid_data = {"status": "up"}
        invalid_data = {"status": "unknown"}

        assert self.parser.validate_schema(valid_data, schema) is True
        assert self.parser.validate_schema(invalid_data, schema) is False

    def test_create_training_example(self):
        """Test creating training example."""
        result = self.parser.create_training_example(
            instruction="Parse the output",
            input_text="interface eth0",
            output={"interface": "eth0", "state": "up"},
        )

        assert "instruction" in result
        assert "input" in result
        assert "output" in result
        assert "text" in result
        assert "Parse the output" in result["text"]

    def test_remove_markdown_blocks(self):
        """Test markdown block removal."""
        text = "Here is some code:\n```python\nprint('hello')\n```\nDone."

        cleaned = self.parser._remove_markdown_blocks(text)

        assert "```" not in cleaned
        assert "print('hello')" in cleaned

    def test_batch_parse_json(self):
        """Test batch parsing."""
        responses = [
            '{"value": 1}',
            '{"value": 2}',
            '{"value": 3}',
        ]

        results = self.parser.batch_parse(responses, output_format="json")

        assert len(results) == 3
        assert all(r is not None for r in results)
        assert results[0]["value"] == 1
        assert results[2]["value"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
