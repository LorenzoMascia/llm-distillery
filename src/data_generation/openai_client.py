"""OpenAI API client for teacher model interactions."""

import os
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from loguru import logger
import tiktoken


class OpenAIClient:
    """Client for interacting with OpenAI API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        max_tokens: int = 2000,
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-2)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize tokenizer for counting
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Model {model} not found, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")

        logger.info(f"OpenAI client initialized with model: {model}")

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from the teacher model.

        Args:
            prompt: User prompt
            system_message: Optional system message
            temperature: Override default temperature
            max_tokens: Override default max tokens

        Returns:
            Generated response text
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        # Log token usage
        prompt_tokens = self.count_tokens(prompt)
        if system_message:
            prompt_tokens += self.count_tokens(system_message)

        logger.debug(f"Prompt tokens: {prompt_tokens}")

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )

                content = response.choices[0].message.content

                # Log usage
                if hasattr(response, "usage"):
                    logger.debug(
                        f"API usage - Prompt: {response.usage.prompt_tokens}, "
                        f"Completion: {response.usage.completion_tokens}, "
                        f"Total: {response.usage.total_tokens}"
                    )

                return content

            except Exception as e:
                logger.error(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}")

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

        raise Exception("Max retries exceeded")

    def batch_generate(
        self,
        prompts: List[str],
        system_message: Optional[str] = None,
        delay: float = 0.5,
    ) -> List[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of prompts
            system_message: Optional system message for all prompts
            delay: Delay between requests to avoid rate limits

        Returns:
            List of generated responses
        """
        responses = []

        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i + 1}/{len(prompts)}")

            try:
                response = self.generate(prompt, system_message=system_message)
                responses.append(response)

                # Rate limiting
                if i < len(prompts) - 1:
                    time.sleep(delay)

            except Exception as e:
                logger.error(f"Failed to generate response for prompt {i}: {e}")
                responses.append(None)

        return responses

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int, model: Optional[str] = None
    ) -> float:
        """
        Estimate the cost of an API call.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model name (defaults to self.model)

        Returns:
            Estimated cost in USD
        """
        model = model or self.model

        # Pricing as of Jan 2025 (update as needed)
        pricing = {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},  # per 1K tokens
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        }

        if model not in pricing:
            logger.warning(f"Pricing not available for model {model}")
            return 0.0

        input_cost = (prompt_tokens / 1000) * pricing[model]["input"]
        output_cost = (completion_tokens / 1000) * pricing[model]["output"]

        return input_cost + output_cost

    def validate_api_key(self) -> bool:
        """
        Validate that the API key works.

        Returns:
            True if valid, False otherwise
        """
        try:
            self.generate("Test", max_tokens=5)
            logger.info("API key validated successfully")
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False
