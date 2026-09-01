"""
Offline unit tests for CRAG Query Rewriter.

All tests are fully offline using a mocked GeminiClient.
No external network calls or model downloads occur during testing.

Coverage:
  - Valid question rewriting to at most 3 comma-separated terms
  - Empty or whitespace query validation
  - Output sanitization (prefix stripping, quote stripping, bullet stripping)
  - Limiting results to at most max_keywords terms
  - Handling multi-line / bulleted LLM output
  - Handling empty or malformed LLM responses
  - Determinism and single LLM invocation per rewrite
  - Constructor parameter validation
"""

import pytest
from unittest.mock import MagicMock

from app.external.query_rewriter import (
    QueryRewriter,
    sanitize_rewritten_query,
    REWRITE_PROMPT_TEMPLATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Output Sanitization Helper Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSanitizeRewrittenQuery:
    """Test response sanitization, prefix removal, and keyword count limiting."""

    def test_standard_comma_separated_response(self):
        raw = "Henry Feilden, occupation"
        sanitized = sanitize_rewritten_query(raw, max_keywords=3)
        assert sanitized == "Henry Feilden, occupation"

    def test_limits_to_at_most_three_keywords_when_llm_outputs_more(self):
        raw = "term one, term two, term three, term four, term five"
        sanitized = sanitize_rewritten_query(raw, max_keywords=3)
        assert sanitized == "term one, term two, term three"

    def test_removes_common_prefixes(self):
        prefixes = [
            "Search Query: Henry Feilden, occupation",
            "search query: Henry Feilden, occupation",
            "Query: Henry Feilden, occupation",
            "Keywords: Henry Feilden, occupation",
            "Search terms: Henry Feilden, occupation",
            "Output: Henry Feilden, occupation",
        ]
        for raw in prefixes:
            assert sanitize_rewritten_query(raw, max_keywords=3) == "Henry Feilden, occupation"

    def test_removes_surrounding_quotes_and_backticks(self):
        raw_double = '"Henry Feilden, occupation"'
        raw_single = "'Henry Feilden, occupation'"
        raw_ticks = "`Henry Feilden, occupation`"
        raw_per_term = '"Henry Feilden", "occupation"'

        assert sanitize_rewritten_query(raw_double, max_keywords=3) == "Henry Feilden, occupation"
        assert sanitize_rewritten_query(raw_single, max_keywords=3) == "Henry Feilden, occupation"
        assert sanitize_rewritten_query(raw_ticks, max_keywords=3) == "Henry Feilden, occupation"
        assert sanitize_rewritten_query(raw_per_term, max_keywords=3) == "Henry Feilden, occupation"

    def test_handles_bulleted_or_newline_separated_terms(self):
        raw = "- Henry Feilden\n- occupation\n- army officer"
        sanitized = sanitize_rewritten_query(raw, max_keywords=3)
        assert sanitized == "Henry Feilden, occupation, army officer"

    def test_handles_numbered_list_terms(self):
        raw = "1. Henry Feilden\n2. occupation"
        sanitized = sanitize_rewritten_query(raw, max_keywords=3)
        assert sanitized == "Henry Feilden, occupation"

    def test_cleans_extra_whitespace_between_terms(self):
        raw = "  Henry Feilden   ,    occupation  ,   army   "
        sanitized = sanitize_rewritten_query(raw, max_keywords=3)
        assert sanitized == "Henry Feilden, occupation, army"

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="empty response"):
            sanitize_rewritten_query("")
        with pytest.raises(ValueError, match="empty response"):
            sanitize_rewritten_query("   \n\t  ")

    def test_only_delimiters_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not extract search keywords"):
            sanitize_rewritten_query(", , , \"\" ")


# ─────────────────────────────────────────────────────────────────────────────
# 2. QueryRewriter Initialization & Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryRewriterInit:
    """Test constructor parameter validation."""

    def test_valid_initialization(self):
        mock_client = MagicMock()
        rewriter = QueryRewriter(llm_client=mock_client)
        assert rewriter.max_keywords == 3
        assert rewriter.llm_client == mock_client

    def test_valid_custom_max_keywords(self):
        mock_client = MagicMock()
        rewriter = QueryRewriter(llm_client=mock_client, max_keywords=2)
        assert rewriter.max_keywords == 2

    def test_none_llm_client_raises_type_error(self):
        with pytest.raises(TypeError, match="llm_client cannot be None"):
            QueryRewriter(llm_client=None)  # type: ignore

    def test_invalid_max_keywords_raises_error(self):
        mock_client = MagicMock()
        with pytest.raises(ValueError, match="max_keywords must be a positive integer"):
            QueryRewriter(llm_client=mock_client, max_keywords=0)
        with pytest.raises(ValueError, match="max_keywords must be a positive integer"):
            QueryRewriter(llm_client=mock_client, max_keywords=-2)
        with pytest.raises(TypeError, match="max_keywords must be an integer"):
            QueryRewriter(llm_client=mock_client, max_keywords=3.5)  # type: ignore
        with pytest.raises(TypeError, match="max_keywords must be an integer"):
            QueryRewriter(llm_client=mock_client, max_keywords=True)  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 3. QueryRewriter Rewrite Execution Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryRewriterExecution:
    """Test rewrite method with mocked LLM client."""

    def test_rewrite_returns_sanitized_string(self):
        mock_client = MagicMock()
        mock_client.generate.return_value = "Henry Feilden, occupation"
        rewriter = QueryRewriter(llm_client=mock_client)

        result = rewriter.rewrite("What is Henry Feilden's occupation?")
        assert isinstance(result, str)
        assert result == "Henry Feilden, occupation"

    def test_llm_is_called_exactly_once_with_prompt(self):
        mock_client = MagicMock()
        mock_client.generate.return_value = "Paris 2024, Olympic games, host"
        rewriter = QueryRewriter(llm_client=mock_client)

        question = "Where were the 2024 Olympic Games held?"
        rewriter.rewrite(question)

        mock_client.generate.assert_called_once()
        prompt_arg = mock_client.generate.call_args[0][0]
        assert question in prompt_arg
        assert "at most 3 concise web search keywords" in prompt_arg

    def test_empty_query_raises_value_error(self):
        mock_client = MagicMock()
        rewriter = QueryRewriter(llm_client=mock_client)

        with pytest.raises(ValueError, match="query must be a non-empty string"):
            rewriter.rewrite("")
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            rewriter.rewrite("   \n\t ")

    def test_non_string_query_raises_type_error(self):
        mock_client = MagicMock()
        rewriter = QueryRewriter(llm_client=mock_client)

        with pytest.raises(TypeError, match="query must be a string"):
            rewriter.rewrite(42)  # type: ignore

    def test_empty_llm_response_falls_back_to_original_query(self):
        mock_client = MagicMock()
        mock_client.generate.return_value = ""
        rewriter = QueryRewriter(llm_client=mock_client)

        result = rewriter.rewrite("What is ChatGPT and how does it relate to RAG systems?")
        assert result == "What is ChatGPT and how does it relate to RAG systems?"

    def test_whitespace_only_llm_response_falls_back_to_original_query(self):
        mock_client = MagicMock()
        mock_client.generate.return_value = "   \n\t  "
        rewriter = QueryRewriter(llm_client=mock_client)

        result = rewriter.rewrite("What is the capital of France?")
        assert result == "What is the capital of France?"

    def test_unusable_delimiters_llm_response_falls_back_to_original_query(self):
        mock_client = MagicMock()
        mock_client.generate.return_value = ", , , \"\" "
        rewriter = QueryRewriter(llm_client=mock_client)

        result = rewriter.rewrite("How does BM25 work?")
        assert result == "How does BM25 work?"

    def test_provider_exception_is_not_swallowed(self):
        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("Upstream provider connection timeout")
        rewriter = QueryRewriter(llm_client=mock_client)

        with pytest.raises(RuntimeError, match="Upstream provider connection timeout"):
            rewriter.rewrite("Any valid question?")

    def test_deterministic_behavior_across_calls(self):
        mock_client = MagicMock()
        mock_client.generate.return_value = "Einstein, Nobel Prize, 1921"
        rewriter = QueryRewriter(llm_client=mock_client)

        res1 = rewriter.rewrite("When did Einstein win the Nobel Prize?")
        res2 = rewriter.rewrite("When did Einstein win the Nobel Prize?")
        assert res1 == res2 == "Einstein, Nobel Prize, 1921"
        assert mock_client.generate.call_count == 2
