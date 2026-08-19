"""
Gemini Client for CorrectRAG.

Wraps the official Google GenAI Python SDK with clean error handling.
API key and model name are read from environment variables only —
no secrets are ever hardcoded.
"""

import os
from typing import Optional

try:
    from google import genai
except ImportError as exc:
    raise ImportError(
        "google-genai package is not installed. "
        "Run: pip install google-genai"
    ) from exc


class GeminiAPIError(Exception):
    """Raised when the Gemini API returns an error or is misconfigured."""


class GeminiClient:
    """Thin wrapper around the Google GenAI Python SDK.

    Configuration:
        GEMINI_API_KEY  (required) — Google AI Studio API key.
        GEMINI_MODEL    (optional) — Model identifier; defaults to gemini-3.6-flash.

    Example:
        client = GeminiClient()
        answer = client.generate("What is RAG?")
    """

    DEFAULT_MODEL: str = "gemini-3.6-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initialise the Gemini client.

        Args:
            api_key: Override the GEMINI_API_KEY environment variable.
            model:   Override the GEMINI_MODEL environment variable.

        Raises:
            GeminiAPIError: If no API key is available.
        """
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            raise GeminiAPIError(
                "GEMINI_API_KEY is not set. "
                "Export the variable or pass api_key= explicitly."
            )

        self.model: str = (
            model
            or os.environ.get("GEMINI_MODEL", "")
            or self.DEFAULT_MODEL
        )

        # Initialise the official SDK client
        self._client = genai.Client(api_key=resolved_key)

    def generate(self, prompt: str) -> str:
        """Generate a text response for the given prompt.

        Args:
            prompt: Full prompt string (question + context already formatted).

        Returns:
            Generated text from the model.

        Raises:
            GeminiAPIError: On API-level failures.
            ValueError: If prompt is empty.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            # The SDK stores the generated text in response.text
            text: str = response.text or ""
            return text.strip()
        except GeminiAPIError:
            raise
        except Exception as exc:
            raise GeminiAPIError(
                f"Gemini API call failed [{self.model}]: {exc}"
            ) from exc
