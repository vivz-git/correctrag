"""
Offline unit tests for LLMProvider protocol compatibility.

Validates that:
  - GeminiClient implements LLMProvider
  - GroqClient implements LLMProvider
  - BaselineRAG accepts any LLMProvider
  - CRAGPipeline accepts any LLMProvider
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.generation.gemini_client import GeminiClient
from app.generation.groq_client import GroqClient
from app.generation.llm_provider import LLMProvider
from app.generation.rag_pipeline import BaselineRAG
from app.pipeline.crag_pipeline import CRAGPipeline
from app.retrieval.retriever import RetrievedChunk


class CustomDummyProvider:
    """A minimal custom provider implementing generate()."""

    def generate(self, prompt: str) -> str:
        return f"Custom response to: {prompt[:20]}"


class IncompleteProvider:
    """A class that does NOT implement generate()."""
    pass


class TestLLMProviderProtocol:
    """Test runtime checkable protocol behavior."""

    def test_gemini_client_implements_llm_provider(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
        with patch("app.generation.gemini_client.genai.Client"):
            client = GeminiClient()
            assert isinstance(client, LLMProvider)

    def test_groq_client_implements_llm_provider(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test_groq_key")
        with patch("app.generation.groq_client.Groq"):
            client = GroqClient()
            assert isinstance(client, LLMProvider)

    def test_custom_class_implements_llm_provider(self):
        provider = CustomDummyProvider()
        assert isinstance(provider, LLMProvider)

    def test_incomplete_class_does_not_implement_llm_provider(self):
        provider = IncompleteProvider()
        assert not isinstance(provider, LLMProvider)


class TestPipelinesWithGenericLLMProvider:
    """Test BaselineRAG and CRAGPipeline using generic LLMProvider instances."""

    def test_baseline_rag_with_generic_provider(self):
        mock_retriever = MagicMock()
        mock_chunk = RetrievedChunk(
            chunk_id="c1",
            text="Corrective RAG improves factual accuracy.",
            source="paper.pdf",
            page_number=1,
            score=0.92,
        )
        mock_retriever.retrieve.return_value = [mock_chunk]

        custom_provider = CustomDummyProvider()
        rag = BaselineRAG(retriever=mock_retriever, llm_client=custom_provider)

        result = rag.run("What is CRAG?")
        assert result.answer.startswith("Custom response to:")
        assert len(result.sources) == 1
        assert result.sources[0].source == "paper.pdf"

    def test_crag_pipeline_with_generic_provider_correct_branch(self):
        mock_retriever = MagicMock()
        mock_evaluator = MagicMock()
        mock_router = MagicMock()
        mock_refiner = MagicMock()
        mock_rewriter = MagicMock()
        mock_web_search = MagicMock()

        mock_chunk = RetrievedChunk(
            chunk_id="c1",
            text="Internal evidence text.",
            source="CRAG.pdf",
            page_number=2,
            score=0.95,
        )
        mock_retriever.retrieve.return_value = [mock_chunk]
        mock_evaluator.score_batch.return_value = [0.85]
        mock_router.route.return_value = "CORRECT"

        from app.evaluation.knowledge_refiner import KnowledgeStrip
        mock_refiner.refine.return_value = [
            KnowledgeStrip(
                strip_id="s1",
                text="Internal refined strip.",
                source="CRAG.pdf",
                page_number=2,
                parent_chunk_id="c1",
                score=0.88,
                position=0,
            )
        ]

        generic_provider = CustomDummyProvider()
        crag = CRAGPipeline(
            retriever=mock_retriever,
            evaluator=mock_evaluator,
            router=mock_router,
            refiner=mock_refiner,
            query_rewriter=mock_rewriter,
            web_search=mock_web_search,
            llm_client=generic_provider,
        )

        result = crag.run("What does CRAG do?")
        assert result.action == "CORRECT"
        assert result.answer.startswith("Custom response to:")
        assert result.trace.final_context_source == "internal"
