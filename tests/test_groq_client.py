"""
Offline unit tests for GroqClient.

All tests in this module are offline and mock the Groq Python SDK.
No actual API calls to Groq or network requests occur during testing.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from app.generation.groq_client import GroqAPIError, GroqClient
from app.generation.llm_provider import LLMProvider


class TestGroqClientInit:
    """Test initialization, configuration, and API key handling."""

    def test_missing_api_key_raises_groq_api_error(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(GroqAPIError, match="GROQ_API_KEY is not set"):
            GroqClient()

    def test_explicit_api_key_accepted(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with patch("app.generation.groq_client.Groq") as mock_groq:
            client = GroqClient(api_key="explicit_gsk_test123")
            mock_groq.assert_called_once_with(api_key="explicit_gsk_test123")
            assert client.model == "openai/gpt-oss-120b"

    def test_env_api_key_accepted(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "env_gsk_test456")
        with patch("app.generation.groq_client.Groq") as mock_groq:
            client = GroqClient()
            mock_groq.assert_called_once_with(api_key="env_gsk_test456")
            assert client.model == "openai/gpt-oss-120b"

    def test_default_model_is_gpt_oss_120b(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        with patch("app.generation.groq_client.Groq"):
            client = GroqClient()
            assert client.model == "openai/gpt-oss-120b"

    def test_custom_model_parameter(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        with patch("app.generation.groq_client.Groq"):
            client = GroqClient(model="llama-3.3-70b-versatile")
            assert client.model == "llama-3.3-70b-versatile"

    def test_custom_model_env_var(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        monkeypatch.setenv("GROQ_MODEL", "mixtral-8x7b-32768")
        with patch("app.generation.groq_client.Groq"):
            client = GroqClient()
            assert client.model == "mixtral-8x7b-32768"


class TestGroqClientGeneration:
    """Test generate() validation, mocked execution, and error handling."""

    def test_empty_prompt_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        with patch("app.generation.groq_client.Groq"):
            client = GroqClient()
            with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
                client.generate("")
            with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
                client.generate("   ")

    def test_non_string_prompt_raises_type_error(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        with patch("app.generation.groq_client.Groq"):
            client = GroqClient()
            with pytest.raises(TypeError, match="Prompt must be a string"):
                client.generate(None)  # type: ignore
            with pytest.raises(TypeError, match="Prompt must be a string"):
                client.generate(123)  # type: ignore

    def test_successful_mocked_generation(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        with patch("app.generation.groq_client.Groq") as mock_groq_cls:
            mock_instance = MagicMock()
            mock_groq_cls.return_value = mock_instance

            # Mock chat.completions.create response
            mock_choice = MagicMock()
            mock_choice.message.content = "  This is a generated answer from Groq.  "
            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_instance.chat.completions.create.return_value = mock_response

            client = GroqClient()
            result = client.generate("What is CRAG?")

            assert result == "This is a generated answer from Groq."
            mock_instance.chat.completions.create.assert_called_once_with(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": "What is CRAG?"}],
            )

    def test_empty_choices_returns_empty_string(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        with patch("app.generation.groq_client.Groq") as mock_groq_cls:
            mock_instance = MagicMock()
            mock_groq_cls.return_value = mock_instance

            mock_response = MagicMock()
            mock_response.choices = []
            mock_instance.chat.completions.create.return_value = mock_response

            client = GroqClient()
            assert client.generate("Test prompt") == ""

    def test_provider_error_wrapped_in_groq_api_error(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        with patch("app.generation.groq_client.Groq") as mock_groq_cls:
            mock_instance = MagicMock()
            mock_groq_cls.return_value = mock_instance
            mock_instance.chat.completions.create.side_effect = RuntimeError("Rate limit exceeded")

            client = GroqClient()
            with pytest.raises(GroqAPIError, match=r"Groq API call failed \[openai/gpt-oss-120b\]: Rate limit exceeded"):
                client.generate("Test query")

    def test_implements_llm_provider_protocol(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test_key")
        with patch("app.generation.groq_client.Groq"):
            client = GroqClient()
            assert isinstance(client, LLMProvider)
