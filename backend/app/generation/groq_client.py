"""
Groq Client for CorrectRAG.

Wraps the official Groq Python SDK for LLM text generation.
API key and model name are read from environment variables only —
no secrets are ever hardcoded.

Configuration:
    GROQ_API_KEY  (required) — Groq Cloud API key.
    GROQ_MODEL    (optional) — Model identifier; defaults to openai/gpt-oss-120b.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    from groq import Groq
except ImportError as exc:
    raise ImportError(
        "groq package is not installed. "
        "Run: pip install groq"
    ) from exc


class GroqAPIError(Exception):
    """Raised when the Groq API returns an error or is misconfigured."""


class GroqClient:
    """Wrapper around the official Groq Python SDK.

    Configuration:
        GROQ_API_KEY  (required) — Groq Cloud API key.
        GROQ_MODEL    (optional) — Model identifier; defaults to openai/gpt-oss-120b.

    Example:
        client = GroqClient()
        answer = client.generate("What is CRAG?")
    """

    DEFAULT_MODEL: str = "openai/gpt-oss-120b"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initialise the Groq client.

        Args:
            api_key: Override the GROQ_API_KEY environment variable.
            model:   Override the GROQ_MODEL environment variable.

        Raises:
            GroqAPIError: If no API key is available.
        """
        resolved_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not resolved_key:
            raise GroqAPIError(
                "GROQ_API_KEY is not set. "
                "Export the variable or pass api_key= explicitly."
            )

        self.model: str = (
            model
            or os.environ.get("GROQ_MODEL", "")
            or self.DEFAULT_MODEL
        )

        # Initialise the official SDK client
        self._client = Groq(api_key=resolved_key)

    def generate(self, prompt: str) -> str:
        """Generate a text response for the given prompt.

        Args:
            prompt: Full prompt string (question + context already formatted).

        Returns:
            Generated text from the model.

        Raises:
            TypeError:  If prompt is not a string.
            ValueError: If prompt is empty or whitespace-only.
            GroqAPIError: On API-level failures.
        """
        if not isinstance(prompt, str):
            raise TypeError(f"Prompt must be a string, got {type(prompt).__name__}")
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            if not response.choices:
                return ""
            content = response.choices[0].message.content or ""
            return content.strip()
        except GroqAPIError:
            raise
        except Exception as exc:
            raise GroqAPIError(
                f"Groq API call failed [{self.model}]: {exc}"
            ) from exc
