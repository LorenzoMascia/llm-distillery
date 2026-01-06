"""Response parser for teacher model outputs."""

import json
import re
from typing import Dict, Any, Optional, List
from loguru import logger


class ResponseParser:
    """Parser for teacher model responses with validation and cleaning."""

    def __init__(self, strict_mode: bool = True):
        """
        Initialize the response parser.

        Args:
            strict_mode: If True, raise exceptions on parsing errors.
                        If False, return None on errors.
        """
        self.strict_mode = strict_mode

    def parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON response from teacher model.

        Args:
            response: Raw response string from the model

        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        # Remove markdown code blocks if present
        cleaned = self._remove_markdown_blocks(response)

        # Try to extract JSON object
        json_str = self._extract_json(cleaned)

        if not json_str:
            logger.warning("No JSON found in response")
            if self.strict_mode:
                raise ValueError("Failed to extract JSON from response")
            return None

        try:
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            if self.strict_mode:
                raise
            return None

    def parse_text_response(self, response: str) -> str:
        """
        Parse plain text response from teacher model.

        Args:
            response: Raw response string

        Returns:
            Cleaned text response
        """
        # Remove markdown code blocks
        cleaned = self._remove_markdown_blocks(response)

        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()

        return cleaned

    def validate_schema(
        self, data: Dict[str, Any], schema: Dict[str, Any]
    ) -> bool:
        """
        Validate data against a simple schema.

        Args:
            data: Data to validate
            schema: Schema definition (simplified JSON Schema)

        Returns:
            True if valid, False otherwise
        """
        if "required" in schema:
            for field in schema["required"]:
                if field not in data:
                    logger.error(f"Missing required field: {field}")
                    return False

        if "properties" in schema:
            for field, field_schema in schema["properties"].items():
                if field in data:
                    if not self._validate_field(data[field], field_schema):
                        return False

        return True

    def _validate_field(self, value: Any, field_schema: Dict[str, Any]) -> bool:
        """Validate a single field against its schema."""
        expected_type = field_schema.get("type")

        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        if expected_type in type_mapping:
            if not isinstance(value, type_mapping[expected_type]):
                logger.error(
                    f"Type mismatch: expected {expected_type}, got {type(value).__name__}"
                )
                return False

        # Validate enum values
        if "enum" in field_schema:
            if value not in field_schema["enum"]:
                logger.error(f"Value '{value}' not in allowed enum: {field_schema['enum']}")
                return False

        return True

    def _remove_markdown_blocks(self, text: str) -> str:
        """Remove markdown code blocks from text."""
        # Remove ```json ... ``` blocks
        text = re.sub(r"```json\s*\n(.*?)\n```", r"\1", text, flags=re.DOTALL)
        # Remove generic ``` ... ``` blocks
        text = re.sub(r"```\s*\n(.*?)\n```", r"\1", text, flags=re.DOTALL)
        # Remove inline code
        text = re.sub(r"`([^`]*)`", r"\1", text)
        return text

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object or array from text."""
        # Try to find JSON object
        obj_match = re.search(r"\{.*\}", text, re.DOTALL)
        if obj_match:
            return obj_match.group(0)

        # Try to find JSON array
        arr_match = re.search(r"\[.*\]", text, re.DOTALL)
        if arr_match:
            return arr_match.group(0)

        return None

    def create_training_example(
        self,
        instruction: str,
        input_text: str,
        output: Any,
        template: str = None,
    ) -> Dict[str, str]:
        """
        Create a formatted training example for fine-tuning.

        Args:
            instruction: The task instruction
            input_text: The input for the task
            output: The expected output (will be converted to string if needed)
            template: Optional custom template

        Returns:
            Dictionary with formatted training example
        """
        # Convert output to string if it's a dict or other type
        if isinstance(output, (dict, list)):
            output_str = json.dumps(output, indent=2)
        else:
            output_str = str(output)

        if template is None:
            template = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

        formatted_text = template.format(
            instruction=instruction, input=input_text, output=output_str
        )

        return {
            "instruction": instruction,
            "input": input_text,
            "output": output_str,
            "text": formatted_text,
        }

    def batch_parse(
        self, responses: List[str], output_format: str = "json"
    ) -> List[Optional[Any]]:
        """
        Parse multiple responses in batch.

        Args:
            responses: List of response strings
            output_format: Format type ('json' or 'text')

        Returns:
            List of parsed responses
        """
        results = []

        for i, response in enumerate(responses):
            try:
                if output_format == "json":
                    parsed = self.parse_json_response(response)
                else:
                    parsed = self.parse_text_response(response)

                results.append(parsed)
            except Exception as e:
                logger.error(f"Error parsing response {i}: {e}")
                if self.strict_mode:
                    raise
                results.append(None)

        return results
