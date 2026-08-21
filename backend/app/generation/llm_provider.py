"""
LLM Provider Protocol for CorrectRAG.

Defines a minimal interface for LLM text generation:
    class LLMProvider(Protocol):
        def generate(self, prompt: str) -> str:
            ...

Supported providers:
    - GeminiClient (primary production & demo provider: Gemini 3.6 Flash)
    - GroqClient   (alternate evaluation provider: openai/gpt-oss-120b)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface for LLM generation providers."""

    def generate(self, prompt: str) -> str:
        """Generate a text response for the given prompt.

        Args:
            prompt: Full prompt string (question + context already formatted).

        Returns:
            Generated text response from the model.
        """
        ...
