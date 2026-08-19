"""
Offline unit tests for CRAG Web Search Adapter.

All tests are fully offline using a mocked TavilyClient.
No external network calls or search API requests occur during testing.

Coverage:
  - Valid query execution and conversion to WebSearchResult objects
  - Preservation of title, url, content, and score metadata
  - Enforcement of max_results (default and per-search override)
  - Missing API key raises clear WebSearchError
  - Provider exceptions wrapped in WebSearchError
  - Empty provider responses handled safely (returns [])
  - Input validation (empty queries, invalid types, invalid bounds)
  - Provider called exactly once per search
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from app.external.web_search import (
    WebSearchClient,
    WebSearchResult,
    WebSearchError,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration & Initialization Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWebSearchClientInit:
    """Test API key resolution and parameter validation."""

    def test_missing_api_key_raises_web_search_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(WebSearchError, match="TAVILY_API_KEY is not set"):
                WebSearchClient()

    def test_explicit_api_key_accepted(self):
        with patch("app.external.web_search.TavilyClient") as mock_tavily:
            client = WebSearchClient(api_key="tvly-test-12345")
            assert client is not None
            mock_tavily.assert_called_once_with(api_key="tvly-test-12345")

    def test_env_api_key_accepted(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "tvly-env-67890"}):
            with patch("app.external.web_search.TavilyClient") as mock_tavily:
                client = WebSearchClient()
                assert client is not None
                mock_tavily.assert_called_once_with(api_key="tvly-env-67890")

    def test_default_max_results_is_five(self):
        with patch("app.external.web_search.TavilyClient"):
            client = WebSearchClient(api_key="mock-key")
            assert client.max_results == 5

    def test_custom_max_results_stored(self):
        with patch("app.external.web_search.TavilyClient"):
            client = WebSearchClient(api_key="mock-key", max_results=10)
            assert client.max_results == 10

    def test_invalid_max_results_raises_error(self):
        with patch("app.external.web_search.TavilyClient"):
            with pytest.raises(ValueError, match="max_results must be a positive integer"):
                WebSearchClient(api_key="mock-key", max_results=0)
            with pytest.raises(ValueError, match="max_results must be a positive integer"):
                WebSearchClient(api_key="mock-key", max_results=-3)
            with pytest.raises(TypeError, match="max_results must be an integer"):
                WebSearchClient(api_key="mock-key", max_results="5")  # type: ignore
            with pytest.raises(TypeError, match="max_results must be an integer"):
                WebSearchClient(api_key="mock-key", max_results=True)  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 2. Web Search Execution Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWebSearchExecution:
    """Test search execution, response parsing, and error wrapping."""

    @pytest.fixture
    def client(self) -> WebSearchClient:
        with patch("app.external.web_search.TavilyClient"):
            return WebSearchClient(api_key="tvly-mock-key", max_results=5)

    def test_valid_query_returns_structured_web_search_results(self, client: WebSearchClient):
        mock_response = {
            "results": [
                {
                    "title": "Henry Feilden Biography",
                    "url": "https://example.com/feilden",
                    "content": "Henry Feilden was a British army officer and naturalist.",
                    "score": 0.95,
                },
                {
                    "title": "Historical Records: Feilden",
                    "url": "https://example.com/records",
                    "content": "Served in the Royal Artillery and Arctic expeditions.",
                    "score": 0.88,
                },
            ]
        }
        client._client.search = MagicMock(return_value=mock_response)

        results = client.search("Henry Feilden, occupation")

        assert len(results) == 2
        assert isinstance(results[0], WebSearchResult)
        assert results[0].title == "Henry Feilden Biography"
        assert results[0].url == "https://example.com/feilden"
        assert results[0].content == "Henry Feilden was a British army officer and naturalist."
        assert results[0].score == 0.95

        assert results[1].title == "Historical Records: Feilden"
        assert results[1].url == "https://example.com/records"
        assert results[1].content == "Served in the Royal Artillery and Arctic expeditions."
        assert results[1].score == 0.88

        client._client.search.assert_called_once_with(
            query="Henry Feilden, occupation",
            max_results=5,
            search_depth="basic",
        )

    def test_per_search_max_results_override(self, client: WebSearchClient):
        mock_response = {
            "results": [
                {"title": f"Result {i}", "url": f"https://example.com/{i}", "content": f"Content {i}", "score": 0.5}
                for i in range(5)
            ]
        }
        client._client.search = MagicMock(return_value=mock_response)

        results = client.search("query terms", max_results=2)

        assert len(results) == 2
        client._client.search.assert_called_once_with(
            query="query terms",
            max_results=2,
            search_depth="basic",
        )

    def test_empty_query_raises_value_error(self, client: WebSearchClient):
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            client.search("")
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            client.search("   \n\t ")

    def test_non_string_query_raises_type_error(self, client: WebSearchClient):
        with pytest.raises(TypeError, match="query must be a string"):
            client.search(12345)  # type: ignore

    def test_invalid_per_search_max_results_raises_error(self, client: WebSearchClient):
        with pytest.raises(ValueError, match="max_results must be a positive integer"):
            client.search("valid query", max_results=0)
        with pytest.raises(TypeError, match="max_results must be an integer"):
            client.search("valid query", max_results="two")  # type: ignore

    def test_provider_exception_wrapped_in_web_search_error(self, client: WebSearchClient):
        client._client.search = MagicMock(side_effect=RuntimeError("Rate limit exceeded"))

        with pytest.raises(WebSearchError, match="Tavily search request failed"):
            client.search("Henry Feilden, occupation")

    def test_empty_provider_response_returns_empty_list(self, client: WebSearchClient):
        client._client.search = MagicMock(return_value={"results": []})

        results = client.search("unknown obscure query")
        assert results == []

    def test_provider_called_exactly_once(self, client: WebSearchClient):
        client._client.search = MagicMock(return_value={"results": []})

        client.search("test query")
        assert client._client.search.call_count == 1
