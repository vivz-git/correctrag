"""
Tests for Baseline RAG Generation — prompt, pipeline, client config.

All tests are fully offline. No real Gemini API calls are made.
The GeminiClient is mocked at the method level via unittest.mock.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from app.generation.prompt import (
    build_rag_prompt,
    format_context_block,
    SYSTEM_INSTRUCTION,
)
from app.generation.gemini_client import GeminiClient, GeminiAPIError
from app.generation.rag_pipeline import (
    BaselineRAG,
    RAGResult,
    SourceCitation,
    _NO_CONTEXT_ANSWER,
)
from app.retrieval.retriever import RetrievedChunk


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_chunk(
    chunk_id: str = "doc_p1_c001",
    text: str = "Sample text about RAG.",
    source: str = "test.pdf",
    page_number: int = 1,
    score: float = 0.85,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page_number=page_number,
        score=score,
        metadata={},
    )


def make_mock_llm(answer: str = "This is the generated answer.") -> MagicMock:
    mock = MagicMock()
    mock.generate.return_value = answer
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# prompt.py — format_context_block
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatContextBlock:

    def test_empty_list_returns_empty_string(self):
        assert format_context_block([]) == ""

    def test_single_chunk_contains_source_and_page(self):
        chunk = make_chunk(source="paper.pdf", page_number=3)
        result = format_context_block([chunk])
        assert "paper.pdf" in result
        assert "Page: 3" in result
        assert chunk.text.strip() in result

    def test_multiple_chunks_are_numbered_sequentially(self):
        chunks = [
            make_chunk(chunk_id=f"doc_p1_c00{i}", text=f"Text {i}")
            for i in range(1, 4)
        ]
        result = format_context_block(chunks)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_chunks_are_separated_by_blank_line(self):
        chunks = [make_chunk(text="A"), make_chunk(text="B")]
        result = format_context_block(chunks)
        # There must be at least one blank line between passages
        assert "\n\n" in result

    def test_source_and_page_appear_for_each_chunk(self):
        c1 = make_chunk(source="a.pdf", page_number=1, text="First")
        c2 = make_chunk(source="b.pdf", page_number=7, text="Second")
        result = format_context_block([c1, c2])
        assert "a.pdf" in result
        assert "b.pdf" in result
        assert "Page: 7" in result


# ─────────────────────────────────────────────────────────────────────────────
# prompt.py — build_rag_prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildRagPrompt:

    def test_prompt_contains_question(self):
        question = "What is corrective RAG?"
        prompt = build_rag_prompt(question, [])
        assert question in prompt

    def test_prompt_contains_system_instruction(self):
        prompt = build_rag_prompt("Any question?", [])
        assert "Answer ONLY using information" in prompt

    def test_prompt_with_chunks_contains_context(self):
        chunk = make_chunk(text="Corrective RAG evaluates retrieval quality.")
        prompt = build_rag_prompt("What does CRAG do?", [chunk])
        assert "Corrective RAG evaluates retrieval quality." in prompt

    def test_prompt_with_no_chunks_signals_no_context(self):
        prompt = build_rag_prompt("What is RAG?", [])
        assert "No relevant context was retrieved" in prompt

    def test_prompt_ends_with_answer_marker(self):
        prompt = build_rag_prompt("What is RAG?", [make_chunk()])
        assert prompt.rstrip().endswith("ANSWER:")

    def test_prompt_strips_whitespace_from_question(self):
        prompt = build_rag_prompt("  padded question  ", [])
        assert "padded question" in prompt
        assert "  padded question  " not in prompt


# ─────────────────────────────────────────────────────────────────────────────
# gemini_client.py — configuration & error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiClientConfig:

    def test_raises_error_when_no_api_key(self):
        """No key set anywhere — must raise GeminiAPIError."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove GEMINI_API_KEY if present
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(GeminiAPIError, match="GEMINI_API_KEY"):
                GeminiClient(api_key="")

    def test_accepts_explicit_api_key(self):
        """Explicit key bypasses env-var requirement."""
        with patch("app.generation.gemini_client.genai.Client") as mock_cls:
            mock_cls.return_value = MagicMock()
            client = GeminiClient(api_key="test-key-123")
            assert client.model == GeminiClient.DEFAULT_MODEL
            mock_cls.assert_called_once_with(api_key="test-key-123")

    def test_model_defaults_to_class_default(self):
        with patch("app.generation.gemini_client.genai.Client"):
            client = GeminiClient(api_key="key")
            assert client.model == GeminiClient.DEFAULT_MODEL

    def test_model_can_be_overridden_explicitly(self):
        with patch("app.generation.gemini_client.genai.Client"):
            client = GeminiClient(api_key="key", model="gemini-2.5-flash")
            assert client.model == "gemini-2.5-flash"

    def test_model_can_be_overridden_via_env(self):
        with patch("app.generation.gemini_client.genai.Client"):
            with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-custom"}):
                client = GeminiClient(api_key="key")
                assert client.model == "gemini-custom"

    def test_generate_raises_on_empty_prompt(self):
        with patch("app.generation.gemini_client.genai.Client"):
            client = GeminiClient(api_key="key")
            with pytest.raises(ValueError, match="non-empty"):
                client.generate("   ")

    def test_generate_wraps_sdk_exception_as_gemini_api_error(self):
        with patch("app.generation.gemini_client.genai.Client") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.models.generate_content.side_effect = RuntimeError("network error")
            mock_cls.return_value = mock_instance
            client = GeminiClient(api_key="key")
            with pytest.raises(GeminiAPIError, match="network error"):
                client.generate("Some question")


# ─────────────────────────────────────────────────────────────────────────────
# rag_pipeline.py — BaselineRAG
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselineRAG:

    def _make_rag(
        self,
        chunks: list[RetrievedChunk],
        llm_answer: str = "Test answer.",
        top_k: int = 5,
        skip_llm_on_empty: bool = True,
    ) -> BaselineRAG:
        """Helper: build a BaselineRAG with a mocked retriever and LLM."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = chunks
        mock_llm = make_mock_llm(llm_answer)
        return BaselineRAG(
            retriever=mock_retriever,
            llm_client=mock_llm,
            top_k=top_k,
            skip_llm_on_empty=skip_llm_on_empty,
        )

    def test_returns_rag_result_instance(self):
        rag = self._make_rag([make_chunk()])
        result = rag.run("What is RAG?")
        assert isinstance(result, RAGResult)

    def test_answer_is_propagated_from_llm(self):
        rag = self._make_rag([make_chunk()], llm_answer="42 is the answer.")
        result = rag.run("What is the answer?")
        assert result.answer == "42 is the answer."

    def test_sources_contain_chunk_metadata(self):
        chunk = make_chunk(chunk_id="doc_p3_c007", source="crag.pdf", page_number=3)
        rag = self._make_rag([chunk])
        result = rag.run("Some question")
        assert len(result.sources) == 1
        citation: SourceCitation = result.sources[0]
        assert citation.source == "crag.pdf"
        assert citation.page_number == 3
        assert citation.chunk_id == "doc_p3_c007"

    def test_retrieved_chunks_are_preserved(self):
        chunks = [make_chunk(chunk_id=f"doc_p1_c00{i}") for i in range(3)]
        rag = self._make_rag(chunks)
        result = rag.run("Question")
        assert len(result.retrieved_chunks) == 3

    def test_sources_are_deduplicated_by_chunk_id(self):
        # Two chunks with same chunk_id — should collapse to one citation
        chunk = make_chunk(chunk_id="dup_id")
        rag = self._make_rag([chunk, chunk])
        result = rag.run("Question")
        ids = [s.chunk_id for s in result.sources]
        assert ids.count("dup_id") == 1

    def test_empty_retrieval_does_not_call_llm_by_default(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_llm = make_mock_llm()
        rag = BaselineRAG(
            retriever=mock_retriever,
            llm_client=mock_llm,
            skip_llm_on_empty=True,
        )
        result = rag.run("Anything")
        mock_llm.generate.assert_not_called()
        assert result.answer == _NO_CONTEXT_ANSWER
        assert result.sources == []

    def test_empty_retrieval_calls_llm_when_skip_disabled(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        mock_llm = make_mock_llm("LLM fallback answer.")
        rag = BaselineRAG(
            retriever=mock_retriever,
            llm_client=mock_llm,
            skip_llm_on_empty=False,
        )
        result = rag.run("Anything")
        mock_llm.generate.assert_called_once()
        assert result.answer == "LLM fallback answer."

    def test_raises_on_empty_query(self):
        rag = self._make_rag([])
        with pytest.raises(ValueError, match="non-empty"):
            rag.run("   ")

    def test_retriever_called_with_correct_top_k(self):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [make_chunk()]
        mock_llm = make_mock_llm()
        rag = BaselineRAG(
            retriever=mock_retriever,
            llm_client=mock_llm,
            top_k=7,
        )
        rag.run("Test query")
        mock_retriever.retrieve.assert_called_once_with("Test query", top_k=7)
