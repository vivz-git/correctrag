"""
Offline tests for the CRAGPipeline orchestrator.

All tests are fully offline. Every component is mocked — no model downloads,
no ChromaDB queries, no Gemini API calls, no Tavily API calls occur during pytest.

Test groups:
  - TestCRAGPipelineInit            — constructor validation
  - TestCRAGPipelineInputGuards     — query type / empty query guards
  - TestCRAGPipelineCorrect         — CORRECT branch (high relevance scores)
  - TestCRAGPipelineIncorrect       — INCORRECT branch (low relevance scores)
  - TestCRAGPipelineAmbiguous       — AMBIGUOUS branch (mid-range scores)
  - TestCRAGPipelineEmptyStore      — empty vector store adaptation
  - TestCRAGExternalRefinement      — external knowledge goes through KnowledgeRefiner
  - TestCRAGExecutionTrace          — ExecutionTrace populated correctly for all branches
  - TestCRAGResultModel             — CRAGResult and ExecutionTrace field contract
  - TestBuildCRAGPrompt             — _build_crag_prompt() helper
  - TestWebResultsToChunksAdapter   — _web_results_to_chunks() adapter
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, call

from app.pipeline.crag_pipeline import (
    CRAGPipeline,
    CRAGResult,
    ExecutionTrace,
    _build_crag_prompt,
    _web_results_to_chunks,
)
from app.retrieval.retriever import RetrievedChunk, VectorRetriever
from app.evaluation.relevance_evaluator import RelevanceEvaluator
from app.evaluation.action_router import ActionRouter
from app.evaluation.knowledge_refiner import KnowledgeRefiner, KnowledgeStrip
from app.external.query_rewriter import QueryRewriter
from app.external.web_search import WebSearchClient, WebSearchResult
from app.generation.gemini_client import GeminiClient


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixture helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_chunk(text: str = "test content", score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        text=text,
        source="doc.pdf",
        page_number=1,
        score=score,
        metadata={},
    )


def _make_strip(text: str = "refined strip", score: float = 0.7) -> KnowledgeStrip:
    return KnowledgeStrip(
        text=text,
        source="doc.pdf",
        page_number=1,
        parent_chunk_id="chunk-1",
        score=score,
        position=0,
    )


def _make_external_strip(
    url: str = "https://example.com",
    text: str = "External web content about the topic.",
) -> KnowledgeStrip:
    """KnowledgeStrip representing a refined external web knowledge strip."""
    return KnowledgeStrip(
        text=text,
        source=url,
        page_number=1,
        parent_chunk_id=f"web-0-{url[:30]}",
        score=0.9,
        position=0,
    )


def _make_web_result(
    title: str = "Web Article",
    url: str = "https://example.com",
) -> WebSearchResult:
    return WebSearchResult(
        title=title,
        url=url,
        content="External web content about the topic.",
        score=0.9,
    )


def _make_trace(
    action: str = "CORRECT",
    retrieved_count: int = 1,
    max_relevance_score: float | None = 0.85,
    web_search_used: bool = False,
    rewritten_query: str | None = None,
    internal_strip_count: int = 1,
    external_strip_count: int = 0,
    final_context_source: str = "internal",
) -> ExecutionTrace:
    return ExecutionTrace(
        retrieved_count=retrieved_count,
        action=action,
        max_relevance_score=max_relevance_score,
        web_search_used=web_search_used,
        rewritten_query=rewritten_query,
        internal_strip_count=internal_strip_count,
        external_strip_count=external_strip_count,
        final_context_source=final_context_source,
    )


def _mock_component_set(
    retriever_chunks: list[RetrievedChunk],
    evaluator_scores: list[float],
    router_action: str,
    refiner_strips: list[KnowledgeStrip] | None = None,
    rewritten_query: str = "rewritten, query",
    web_results: list[WebSearchResult] | None = None,
    generated_answer: str = "Generated answer.",
) -> dict:
    """Build a dictionary of fully-mocked pipeline components.

    refiner_strips is the return value for ALL calls to refiner.refine()
    (both internal and external calls use the same mock return value).
    """
    retriever = MagicMock(spec=VectorRetriever)
    retriever.retrieve.return_value = retriever_chunks

    evaluator = MagicMock(spec=RelevanceEvaluator)
    evaluator.score_batch.return_value = evaluator_scores

    router = MagicMock(spec=ActionRouter)
    router.route.return_value = router_action

    refiner = MagicMock(spec=KnowledgeRefiner)
    refiner.refine.return_value = refiner_strips if refiner_strips is not None else []

    query_rewriter = MagicMock(spec=QueryRewriter)
    query_rewriter.rewrite.return_value = rewritten_query

    web_search = MagicMock(spec=WebSearchClient)
    web_search.search.return_value = web_results if web_results is not None else []

    llm_client = MagicMock(spec=GeminiClient)
    llm_client.generate.return_value = generated_answer

    return {
        "retriever": retriever,
        "evaluator": evaluator,
        "router": router,
        "refiner": refiner,
        "query_rewriter": query_rewriter,
        "web_search": web_search,
        "llm_client": llm_client,
    }


def _build_pipeline(components: dict, top_k: int = 5) -> CRAGPipeline:
    return CRAGPipeline(
        retriever=components["retriever"],
        evaluator=components["evaluator"],
        router=components["router"],
        refiner=components["refiner"],
        query_rewriter=components["query_rewriter"],
        web_search=components["web_search"],
        llm_client=components["llm_client"],
        top_k=top_k,
    )


# ──────────────────────────────────────────────────────────────────────────────
# TestCRAGPipelineInit
# ──────────────────────────────────────────────────────────────────────────────

class TestCRAGPipelineInit:
    """Constructor validation tests."""

    def test_valid_construction_succeeds(self):
        comps = _mock_component_set([], [], "CORRECT")
        pipeline = _build_pipeline(comps)
        assert pipeline.top_k == 5

    def test_custom_top_k_is_stored(self):
        comps = _mock_component_set([], [], "CORRECT")
        pipeline = _build_pipeline(comps, top_k=10)
        assert pipeline.top_k == 10

    def test_none_retriever_raises_type_error(self):
        comps = _mock_component_set([], [], "CORRECT")
        with pytest.raises(TypeError, match="retriever cannot be None"):
            CRAGPipeline(
                retriever=None,
                evaluator=comps["evaluator"],
                router=comps["router"],
                refiner=comps["refiner"],
                query_rewriter=comps["query_rewriter"],
                web_search=comps["web_search"],
                llm_client=comps["llm_client"],
            )

    def test_none_evaluator_raises_type_error(self):
        comps = _mock_component_set([], [], "CORRECT")
        with pytest.raises(TypeError, match="evaluator cannot be None"):
            CRAGPipeline(
                retriever=comps["retriever"],
                evaluator=None,
                router=comps["router"],
                refiner=comps["refiner"],
                query_rewriter=comps["query_rewriter"],
                web_search=comps["web_search"],
                llm_client=comps["llm_client"],
            )

    def test_none_router_raises_type_error(self):
        comps = _mock_component_set([], [], "CORRECT")
        with pytest.raises(TypeError, match="router cannot be None"):
            CRAGPipeline(
                retriever=comps["retriever"],
                evaluator=comps["evaluator"],
                router=None,
                refiner=comps["refiner"],
                query_rewriter=comps["query_rewriter"],
                web_search=comps["web_search"],
                llm_client=comps["llm_client"],
            )

    def test_none_refiner_raises_type_error(self):
        comps = _mock_component_set([], [], "CORRECT")
        with pytest.raises(TypeError, match="refiner cannot be None"):
            CRAGPipeline(
                retriever=comps["retriever"],
                evaluator=comps["evaluator"],
                router=comps["router"],
                refiner=None,
                query_rewriter=comps["query_rewriter"],
                web_search=comps["web_search"],
                llm_client=comps["llm_client"],
            )

    def test_none_query_rewriter_raises_type_error(self):
        comps = _mock_component_set([], [], "CORRECT")
        with pytest.raises(TypeError, match="query_rewriter cannot be None"):
            CRAGPipeline(
                retriever=comps["retriever"],
                evaluator=comps["evaluator"],
                router=comps["router"],
                refiner=comps["refiner"],
                query_rewriter=None,
                web_search=comps["web_search"],
                llm_client=comps["llm_client"],
            )

    def test_none_web_search_raises_type_error(self):
        comps = _mock_component_set([], [], "CORRECT")
        with pytest.raises(TypeError, match="web_search cannot be None"):
            CRAGPipeline(
                retriever=comps["retriever"],
                evaluator=comps["evaluator"],
                router=comps["router"],
                refiner=comps["refiner"],
                query_rewriter=comps["query_rewriter"],
                web_search=None,
                llm_client=comps["llm_client"],
            )

    def test_none_llm_client_raises_type_error(self):
        comps = _mock_component_set([], [], "CORRECT")
        with pytest.raises(TypeError, match="llm_client cannot be None"):
            CRAGPipeline(
                retriever=comps["retriever"],
                evaluator=comps["evaluator"],
                router=comps["router"],
                refiner=comps["refiner"],
                query_rewriter=comps["query_rewriter"],
                web_search=comps["web_search"],
                llm_client=None,
            )

    def test_invalid_top_k_type_raises_type_error(self):
        comps = _mock_component_set([], [], "CORRECT")
        with pytest.raises(TypeError, match="top_k must be an integer"):
            _build_pipeline(comps, top_k="five")  # type: ignore

    def test_zero_top_k_raises_value_error(self):
        comps = _mock_component_set([], [], "CORRECT")
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            _build_pipeline(comps, top_k=0)

    def test_negative_top_k_raises_value_error(self):
        comps = _mock_component_set([], [], "CORRECT")
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            _build_pipeline(comps, top_k=-1)

    def test_bool_top_k_raises_type_error(self):
        comps = _mock_component_set([], [], "CORRECT")
        with pytest.raises(TypeError, match="top_k must be an integer"):
            _build_pipeline(comps, top_k=True)


# ──────────────────────────────────────────────────────────────────────────────
# TestCRAGPipelineInputGuards
# ──────────────────────────────────────────────────────────────────────────────

class TestCRAGPipelineInputGuards:
    """Input validation for run()."""

    def test_non_string_query_raises_type_error(self):
        comps = _mock_component_set([], [], "CORRECT")
        pipeline = _build_pipeline(comps)
        with pytest.raises(TypeError, match="query must be a string"):
            pipeline.run(42)  # type: ignore

    def test_empty_query_raises_value_error(self):
        comps = _mock_component_set([], [], "CORRECT")
        pipeline = _build_pipeline(comps)
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            pipeline.run("")

    def test_whitespace_only_query_raises_value_error(self):
        comps = _mock_component_set([], [], "CORRECT")
        pipeline = _build_pipeline(comps)
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            pipeline.run("   ")


# ──────────────────────────────────────────────────────────────────────────────
# TestCRAGPipelineCorrect
# ──────────────────────────────────────────────────────────────────────────────

class TestCRAGPipelineCorrect:
    """CORRECT branch: high relevance scores → refine internal docs → generate."""

    def _make_correct_pipeline(self):
        chunks = [_make_chunk("Highly relevant text.", score=0.95)]
        scores = [0.85]
        strips = [_make_strip("Highly relevant text.", score=0.85)]

        comps = _mock_component_set(
            retriever_chunks=chunks,
            evaluator_scores=scores,
            router_action="CORRECT",
            refiner_strips=strips,
            generated_answer="Answer from internal docs.",
        )
        return _build_pipeline(comps), comps

    def test_correct_action_returned(self):
        pipeline, _ = self._make_correct_pipeline()
        result = pipeline.run("What is CRAG?")
        assert result.action == "CORRECT"

    def test_correct_answer_is_generated(self):
        pipeline, _ = self._make_correct_pipeline()
        result = pipeline.run("What is CRAG?")
        assert result.answer == "Answer from internal docs."

    def test_correct_no_rewritten_query(self):
        pipeline, _ = self._make_correct_pipeline()
        result = pipeline.run("What is CRAG?")
        assert result.rewritten_query is None

    def test_correct_no_web_results(self):
        pipeline, _ = self._make_correct_pipeline()
        result = pipeline.run("What is CRAG?")
        assert result.web_results == []

    def test_correct_no_external_strips(self):
        pipeline, _ = self._make_correct_pipeline()
        result = pipeline.run("What is CRAG?")
        assert result.external_strips == []

    def test_correct_refined_strips_populated(self):
        pipeline, _ = self._make_correct_pipeline()
        result = pipeline.run("What is CRAG?")
        assert len(result.refined_strips) == 1
        assert result.refined_strips[0].text == "Highly relevant text."

    def test_correct_retrieved_chunks_populated(self):
        pipeline, _ = self._make_correct_pipeline()
        result = pipeline.run("What is CRAG?")
        assert len(result.retrieved_chunks) == 1

    def test_correct_scores_populated(self):
        pipeline, _ = self._make_correct_pipeline()
        result = pipeline.run("What is CRAG?")
        assert result.relevance_scores == [0.85]

    def test_correct_refiner_called_once_for_internal(self):
        """CORRECT branch: refiner called exactly once, for internal documents."""
        pipeline, comps = self._make_correct_pipeline()
        pipeline.run("What is CRAG?")
        comps["refiner"].refine.assert_called_once()
        call_args = comps["refiner"].refine.call_args
        # First positional arg is the query
        assert call_args[0][0] == "What is CRAG?" or call_args[1].get("query") == "What is CRAG?"

    def test_correct_web_search_not_called(self):
        pipeline, comps = self._make_correct_pipeline()
        pipeline.run("What is CRAG?")
        comps["web_search"].search.assert_not_called()

    def test_correct_query_rewriter_not_called(self):
        pipeline, comps = self._make_correct_pipeline()
        pipeline.run("What is CRAG?")
        comps["query_rewriter"].rewrite.assert_not_called()

    def test_correct_llm_called_once(self):
        pipeline, comps = self._make_correct_pipeline()
        pipeline.run("What is CRAG?")
        comps["llm_client"].generate.assert_called_once()

    def test_correct_query_preserved_in_result(self):
        pipeline, _ = self._make_correct_pipeline()
        result = pipeline.run("  What is CRAG?  ")
        assert result.query == "What is CRAG?"

    def test_correct_empty_strips_returns_no_context_answer_without_llm(self):
        """If refiner returns no strips, fallback answer returned without calling LLM."""
        chunks = [_make_chunk()]
        scores = [0.85]
        comps = _mock_component_set(
            retriever_chunks=chunks,
            evaluator_scores=scores,
            router_action="CORRECT",
            refiner_strips=[],
        )
        pipeline = _build_pipeline(comps)
        result = pipeline.run("What is CRAG?")
        assert "cannot answer" in result.answer.lower()
        comps["llm_client"].generate.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# TestCRAGPipelineIncorrect
# ──────────────────────────────────────────────────────────────────────────────

class TestCRAGPipelineIncorrect:
    """INCORRECT branch: low relevance → rewrite → web search → external refinement → generate."""

    def _make_incorrect_pipeline(self):
        chunks = [_make_chunk("Irrelevant text.", score=0.1)]
        scores = [-0.7]
        web_results = [_make_web_result()]
        # Refiner returns strips when called on external chunks
        external_strips = [_make_external_strip()]

        comps = _mock_component_set(
            retriever_chunks=chunks,
            evaluator_scores=scores,
            router_action="INCORRECT",
            refiner_strips=external_strips,
            web_results=web_results,
            rewritten_query="CRAG, corrective retrieval",
            generated_answer="Answer from web.",
        )
        return _build_pipeline(comps), comps

    def test_incorrect_action_returned(self):
        pipeline, _ = self._make_incorrect_pipeline()
        result = pipeline.run("What is CRAG?")
        assert result.action == "INCORRECT"

    def test_incorrect_answer_is_generated(self):
        pipeline, _ = self._make_incorrect_pipeline()
        result = pipeline.run("What is CRAG?")
        assert result.answer == "Answer from web."

    def test_incorrect_rewritten_query_populated(self):
        pipeline, _ = self._make_incorrect_pipeline()
        result = pipeline.run("What is CRAG?")
        assert result.rewritten_query == "CRAG, corrective retrieval"

    def test_incorrect_web_results_preserved_for_provenance(self):
        """Raw web results are preserved in CRAGResult for provenance."""
        pipeline, _ = self._make_incorrect_pipeline()
        result = pipeline.run("What is CRAG?")
        assert len(result.web_results) == 1

    def test_incorrect_no_internal_refined_strips(self):
        """INCORRECT branch does not produce internal refined strips."""
        pipeline, _ = self._make_incorrect_pipeline()
        result = pipeline.run("What is CRAG?")
        assert result.refined_strips == []

    def test_incorrect_external_strips_populated(self):
        """INCORRECT branch: external_strips contains KnowledgeRefiner output."""
        pipeline, _ = self._make_incorrect_pipeline()
        result = pipeline.run("What is CRAG?")
        assert len(result.external_strips) == 1

    def test_incorrect_refiner_called_once_for_external_knowledge(self):
        """INCORRECT branch: refiner called exactly once for converted web chunks."""
        pipeline, comps = self._make_incorrect_pipeline()
        pipeline.run("What is CRAG?")
        comps["refiner"].refine.assert_called_once()

    def test_incorrect_refiner_not_called_for_internal(self):
        """INCORRECT branch: internal documents discarded, refiner never called with them."""
        pipeline, comps = self._make_incorrect_pipeline()
        pipeline.run("What is CRAG?")
        # Refiner called exactly once means it was NOT called for internal docs
        assert comps["refiner"].refine.call_count == 1

    def test_incorrect_query_rewriter_called_once(self):
        pipeline, comps = self._make_incorrect_pipeline()
        pipeline.run("What is CRAG?")
        comps["query_rewriter"].rewrite.assert_called_once_with("What is CRAG?")

    def test_incorrect_web_search_called_with_rewritten_query(self):
        pipeline, comps = self._make_incorrect_pipeline()
        pipeline.run("What is CRAG?")
        comps["web_search"].search.assert_called_once_with("CRAG, corrective retrieval")

    def test_incorrect_llm_called_once(self):
        pipeline, comps = self._make_incorrect_pipeline()
        pipeline.run("What is CRAG?")
        comps["llm_client"].generate.assert_called_once()

    def test_incorrect_retrieved_chunks_preserved_for_provenance(self):
        """Internal chunks are preserved in CRAGResult for provenance even in INCORRECT."""
        pipeline, _ = self._make_incorrect_pipeline()
        result = pipeline.run("What is CRAG?")
        assert len(result.retrieved_chunks) == 1

    def test_incorrect_scores_preserved(self):
        pipeline, _ = self._make_incorrect_pipeline()
        result = pipeline.run("What is CRAG?")
        assert result.relevance_scores == [-0.7]

    def test_incorrect_empty_web_results_returns_no_context_answer(self):
        """When web search returns nothing, refiner is not called and fallback is returned."""
        chunks = [_make_chunk()]
        scores = [-0.7]
        comps = _mock_component_set(
            retriever_chunks=chunks,
            evaluator_scores=scores,
            router_action="INCORRECT",
            web_results=[],
        )
        pipeline = _build_pipeline(comps)
        result = pipeline.run("What is CRAG?")
        assert "cannot answer" in result.answer.lower()
        comps["llm_client"].generate.assert_not_called()
        # Refiner not called since there are no external chunks to refine
        comps["refiner"].refine.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# TestCRAGPipelineAmbiguous
# ──────────────────────────────────────────────────────────────────────────────

class TestCRAGPipelineAmbiguous:
    """AMBIGUOUS branch: internal refinement + external refinement → combined generation."""

    def _make_ambiguous_pipeline(self):
        chunks = [_make_chunk("Partially relevant text.", score=0.5)]
        scores = [0.1]
        # Both internal and external refiner calls return the same mock strips
        strips = [_make_strip("Partially relevant text.", score=0.1)]
        web_results = [_make_web_result()]

        comps = _mock_component_set(
            retriever_chunks=chunks,
            evaluator_scores=scores,
            router_action="AMBIGUOUS",
            refiner_strips=strips,
            web_results=web_results,
            rewritten_query="CRAG, ambiguous query",
            generated_answer="Combined answer.",
        )
        return _build_pipeline(comps), comps

    def test_ambiguous_action_returned(self):
        pipeline, _ = self._make_ambiguous_pipeline()
        result = pipeline.run("What is CRAG?")
        assert result.action == "AMBIGUOUS"

    def test_ambiguous_answer_is_generated(self):
        pipeline, _ = self._make_ambiguous_pipeline()
        result = pipeline.run("What is CRAG?")
        assert result.answer == "Combined answer."

    def test_ambiguous_rewritten_query_populated(self):
        pipeline, _ = self._make_ambiguous_pipeline()
        result = pipeline.run("What is CRAG?")
        assert result.rewritten_query == "CRAG, ambiguous query"

    def test_ambiguous_refined_strips_populated(self):
        """AMBIGUOUS: internal refined strips populated."""
        pipeline, _ = self._make_ambiguous_pipeline()
        result = pipeline.run("What is CRAG?")
        assert len(result.refined_strips) == 1

    def test_ambiguous_external_strips_populated(self):
        """AMBIGUOUS: external refined strips also populated."""
        pipeline, _ = self._make_ambiguous_pipeline()
        result = pipeline.run("What is CRAG?")
        assert len(result.external_strips) == 1

    def test_ambiguous_web_results_preserved_for_provenance(self):
        pipeline, _ = self._make_ambiguous_pipeline()
        result = pipeline.run("What is CRAG?")
        assert len(result.web_results) == 1

    def test_ambiguous_refiner_called_twice(self):
        """AMBIGUOUS: refiner called once for internal, once for external chunks."""
        pipeline, comps = self._make_ambiguous_pipeline()
        pipeline.run("What is CRAG?")
        assert comps["refiner"].refine.call_count == 2

    def test_ambiguous_query_rewriter_called_once(self):
        pipeline, comps = self._make_ambiguous_pipeline()
        pipeline.run("What is CRAG?")
        comps["query_rewriter"].rewrite.assert_called_once()

    def test_ambiguous_web_search_called(self):
        pipeline, comps = self._make_ambiguous_pipeline()
        pipeline.run("What is CRAG?")
        comps["web_search"].search.assert_called_once_with("CRAG, ambiguous query")

    def test_ambiguous_llm_called_once(self):
        pipeline, comps = self._make_ambiguous_pipeline()
        pipeline.run("What is CRAG?")
        comps["llm_client"].generate.assert_called_once()

    def test_ambiguous_empty_both_sources_returns_no_context_answer(self):
        """If both internal refiner and external refiner return nothing, return fallback."""
        chunks = [_make_chunk()]
        scores = [0.1]
        comps = _mock_component_set(
            retriever_chunks=chunks,
            evaluator_scores=scores,
            router_action="AMBIGUOUS",
            refiner_strips=[],  # both internal + external refiner calls return empty
            web_results=[],
        )
        pipeline = _build_pipeline(comps)
        result = pipeline.run("What is CRAG?")
        assert "cannot answer" in result.answer.lower()
        comps["llm_client"].generate.assert_not_called()

    def test_ambiguous_empty_web_results_skips_external_refiner(self):
        """When web search returns nothing, external refiner is not called."""
        chunks = [_make_chunk()]
        strips = [_make_strip()]
        comps = _mock_component_set(
            retriever_chunks=chunks,
            evaluator_scores=[0.1],
            router_action="AMBIGUOUS",
            refiner_strips=strips,
            web_results=[],  # no web results
        )
        pipeline = _build_pipeline(comps)
        pipeline.run("What is CRAG?")
        # Refiner called once (internal only), not twice
        assert comps["refiner"].refine.call_count == 1


# ──────────────────────────────────────────────────────────────────────────────
# TestCRAGPipelineEmptyStore
# ──────────────────────────────────────────────────────────────────────────────

class TestCRAGPipelineEmptyStore:
    """Edge case: VectorRetriever returns empty list (empty knowledge base).

    [OUR ADAPTATION] When the retriever returns no documents, there is no s_max
    to compute. The pipeline bypasses scoring and routing and directly follows
    the external correction path. This is explicitly NOT Algorithm 1 behavior.
    """

    def test_empty_store_bypasses_router_and_evaluator(self):
        web_results = [_make_web_result()]
        external_strips = [_make_external_strip()]
        comps = _mock_component_set(
            retriever_chunks=[],
            evaluator_scores=[],
            router_action="INCORRECT",
            refiner_strips=external_strips,
            web_results=web_results,
            rewritten_query="CRAG search",
            generated_answer="Web-based answer.",
        )
        pipeline = _build_pipeline(comps)
        pipeline.run("What is CRAG?")
        # Router and evaluator must NOT be called — no s_max to compute
        comps["router"].route.assert_not_called()
        comps["evaluator"].score_batch.assert_not_called()

    def test_empty_store_action_is_incorrect(self):
        web_results = [_make_web_result()]
        external_strips = [_make_external_strip()]
        comps = _mock_component_set(
            retriever_chunks=[],
            evaluator_scores=[],
            router_action="INCORRECT",
            refiner_strips=external_strips,
            web_results=web_results,
            rewritten_query="CRAG search",
            generated_answer="Web answer.",
        )
        pipeline = _build_pipeline(comps)
        result = pipeline.run("What is CRAG?")
        assert result.action == "INCORRECT"

    def test_empty_store_no_retrieved_chunks_or_scores_in_result(self):
        comps = _mock_component_set(
            retriever_chunks=[],
            evaluator_scores=[],
            router_action="INCORRECT",
            refiner_strips=[_make_external_strip()],
            web_results=[_make_web_result()],
            rewritten_query="search terms",
            generated_answer="Answer.",
        )
        pipeline = _build_pipeline(comps)
        result = pipeline.run("question?")
        assert result.retrieved_chunks == []
        assert result.relevance_scores == []

    def test_empty_store_no_internal_refined_strips(self):
        comps = _mock_component_set(
            retriever_chunks=[],
            evaluator_scores=[],
            router_action="INCORRECT",
            refiner_strips=[_make_external_strip()],
            web_results=[_make_web_result()],
            rewritten_query="search",
            generated_answer="Answer.",
        )
        pipeline = _build_pipeline(comps)
        result = pipeline.run("question?")
        assert result.refined_strips == []

    def test_empty_store_external_refinement_still_occurs(self):
        """Even with empty store, web results are still refined through KnowledgeRefiner."""
        external_strips = [_make_external_strip()]
        comps = _mock_component_set(
            retriever_chunks=[],
            evaluator_scores=[],
            router_action="INCORRECT",
            refiner_strips=external_strips,
            web_results=[_make_web_result()],
            rewritten_query="search",
            generated_answer="Answer.",
        )
        pipeline = _build_pipeline(comps)
        result = pipeline.run("question?")
        comps["refiner"].refine.assert_called_once()
        assert len(result.external_strips) == 1


# ──────────────────────────────────────────────────────────────────────────────
# TestCRAGExternalRefinement
# ──────────────────────────────────────────────────────────────────────────────

class TestCRAGExternalRefinement:
    """Verify that external web knowledge is routed through KnowledgeRefiner,
    not sent directly to the LLM as raw WebSearchResult objects."""

    def test_incorrect_branch_refiner_receives_converted_chunks(self):
        """Refiner is called with RetrievedChunk objects converted from web results."""
        web = _make_web_result(url="https://example.com")
        external_strips = [_make_external_strip()]
        comps = _mock_component_set(
            retriever_chunks=[_make_chunk()],
            evaluator_scores=[-0.7],
            router_action="INCORRECT",
            refiner_strips=external_strips,
            web_results=[web],
            rewritten_query="search terms",
            generated_answer="Answer.",
        )
        pipeline = _build_pipeline(comps)
        pipeline.run("Query?")

        comps["refiner"].refine.assert_called_once()
        # Second argument to refine() is the documents list
        docs_arg = comps["refiner"].refine.call_args[0][1]
        assert isinstance(docs_arg, list)
        assert len(docs_arg) == 1
        # Converted chunk must have the URL as its source
        assert docs_arg[0].source == "https://example.com"
        assert docs_arg[0].text == web.content

    def test_ambiguous_branch_refiner_receives_external_chunks_on_second_call(self):
        """In AMBIGUOUS, the second refiner call receives converted web chunks."""
        web = _make_web_result(url="https://example.org")
        strips = [_make_strip()]
        comps = _mock_component_set(
            retriever_chunks=[_make_chunk()],
            evaluator_scores=[0.1],
            router_action="AMBIGUOUS",
            refiner_strips=strips,
            web_results=[web],
            rewritten_query="search",
            generated_answer="Answer.",
        )
        pipeline = _build_pipeline(comps)
        pipeline.run("Query?")

        assert comps["refiner"].refine.call_count == 2
        second_call_docs = comps["refiner"].refine.call_args_list[1][0][1]
        assert second_call_docs[0].source == "https://example.org"

    def test_incorrect_generation_uses_external_strips_not_raw_web_results(self):
        """LLM prompt is built from refined external_strips, not raw WebSearchResult."""
        external_strips = [_make_external_strip(text="Refined external knowledge.")]
        comps = _mock_component_set(
            retriever_chunks=[_make_chunk()],
            evaluator_scores=[-0.7],
            router_action="INCORRECT",
            refiner_strips=external_strips,
            web_results=[_make_web_result()],
            rewritten_query="terms",
            generated_answer="Answer.",
        )
        pipeline = _build_pipeline(comps)
        pipeline.run("Query?")

        prompt_sent = comps["llm_client"].generate.call_args[0][0]
        assert "Refined external knowledge." in prompt_sent
        assert "(web)" in prompt_sent

    def test_ambiguous_generation_uses_both_refined_sources(self):
        """LLM prompt contains both internal and external refined strips."""
        internal_strips = [_make_strip("Internal knowledge.")]
        comps = _mock_component_set(
            retriever_chunks=[_make_chunk()],
            evaluator_scores=[0.1],
            router_action="AMBIGUOUS",
            refiner_strips=internal_strips,
            web_results=[_make_web_result()],
            rewritten_query="terms",
            generated_answer="Answer.",
        )
        pipeline = _build_pipeline(comps)
        pipeline.run("Query?")

        prompt_sent = comps["llm_client"].generate.call_args[0][0]
        # Internal strips labeled as (internal document)
        assert "(internal document)" in prompt_sent
        # External strips labeled as (web)
        assert "(web)" in prompt_sent

    def test_correct_branch_does_not_use_external_refinement(self):
        """CORRECT: refiner called once (internal only), no web search or external strips."""
        strips = [_make_strip()]
        comps = _mock_component_set(
            retriever_chunks=[_make_chunk()],
            evaluator_scores=[0.9],
            router_action="CORRECT",
            refiner_strips=strips,
        )
        pipeline = _build_pipeline(comps)
        result = pipeline.run("Query?")

        comps["refiner"].refine.assert_called_once()
        comps["web_search"].search.assert_not_called()
        assert result.external_strips == []


# ──────────────────────────────────────────────────────────────────────────────
# TestCRAGExecutionTrace
# ──────────────────────────────────────────────────────────────────────────────

class TestCRAGExecutionTrace:
    """ExecutionTrace is populated correctly for all three branches and edge cases."""

    def test_correct_branch_trace_fields(self):
        chunks = [_make_chunk(score=0.95)]
        scores = [0.85]
        strips = [_make_strip()]
        comps = _mock_component_set(
            retriever_chunks=chunks,
            evaluator_scores=scores,
            router_action="CORRECT",
            refiner_strips=strips,
        )
        pipeline = _build_pipeline(comps)
        result = pipeline.run("What is CRAG?")

        trace = result.trace
        assert trace.action == "CORRECT"
        assert trace.retrieved_count == 1
        assert trace.max_relevance_score == pytest.approx(0.85)
        assert trace.web_search_used is False
        assert trace.rewritten_query is None
        assert trace.internal_strip_count == 1
        assert trace.external_strip_count == 0
        assert trace.final_context_source == "internal"

    def test_correct_branch_trace_no_strips_gives_none_context_source(self):
        chunks = [_make_chunk()]
        comps = _mock_component_set(
            retriever_chunks=chunks,
            evaluator_scores=[0.85],
            router_action="CORRECT",
            refiner_strips=[],
        )
        pipeline = _build_pipeline(comps)
        result = pipeline.run("What is CRAG?")
        assert result.trace.final_context_source == "none"
        assert result.trace.internal_strip_count == 0

    def test_incorrect_branch_trace_fields(self):
        chunks = [_make_chunk(score=0.1)]
        scores = [-0.7]
        ext_strips = [_make_external_strip()]
        comps = _mock_component_set(
            retriever_chunks=chunks,
            evaluator_scores=scores,
            router_action="INCORRECT",
            refiner_strips=ext_strips,
            web_results=[_make_web_result()],
            rewritten_query="CRAG, retrieval",
        )
        pipeline = _build_pipeline(comps)
        result = pipeline.run("Query?")

        trace = result.trace
        assert trace.action == "INCORRECT"
        assert trace.retrieved_count == 1
        assert trace.max_relevance_score == pytest.approx(-0.7)
        assert trace.web_search_used is True
        assert trace.rewritten_query == "CRAG, retrieval"
        assert trace.internal_strip_count == 0
        assert trace.external_strip_count == 1
        assert trace.final_context_source == "external"

    def test_incorrect_branch_trace_no_external_strips_gives_none_context(self):
        comps = _mock_component_set(
            retriever_chunks=[_make_chunk()],
            evaluator_scores=[-0.7],
            router_action="INCORRECT",
            refiner_strips=[],
            web_results=[],
        )
        pipeline = _build_pipeline(comps)
        result = pipeline.run("Query?")
        assert result.trace.final_context_source == "none"
        assert result.trace.external_strip_count == 0

    def test_ambiguous_branch_trace_combined_source(self):
        chunks = [_make_chunk(score=0.5)]
        scores = [0.1]
        strips = [_make_strip()]
        comps = _mock_component_set(
            retriever_chunks=chunks,
            evaluator_scores=scores,
            router_action="AMBIGUOUS",
            refiner_strips=strips,
            web_results=[_make_web_result()],
            rewritten_query="CRAG, ambiguous",
        )
        pipeline = _build_pipeline(comps)
        result = pipeline.run("Query?")

        trace = result.trace
        assert trace.action == "AMBIGUOUS"
        assert trace.retrieved_count == 1
        assert trace.max_relevance_score == pytest.approx(0.1)
        assert trace.web_search_used is True
        assert trace.rewritten_query == "CRAG, ambiguous"
        assert trace.internal_strip_count == 1
        assert trace.external_strip_count == 1
        assert trace.final_context_source == "combined"

    def test_ambiguous_trace_internal_only_when_no_web(self):
        strips = [_make_strip()]
        comps = _mock_component_set(
            retriever_chunks=[_make_chunk()],
            evaluator_scores=[0.1],
            router_action="AMBIGUOUS",
            refiner_strips=strips,
            web_results=[],
        )
        pipeline = _build_pipeline(comps)
        result = pipeline.run("Query?")
        assert result.trace.final_context_source == "internal"
        assert result.trace.external_strip_count == 0

    def test_ambiguous_trace_external_only_when_internal_empty(self):
        """When internal refiner returns empty, external only contributes."""
        ext_strips = [_make_external_strip()]
        # Set up refiner to return empty first call (internal), non-empty second (external)
        refiner_mock = MagicMock(spec=KnowledgeRefiner)
        refiner_mock.refine.side_effect = [[], ext_strips]

        comps = _mock_component_set(
            retriever_chunks=[_make_chunk()],
            evaluator_scores=[0.1],
            router_action="AMBIGUOUS",
            web_results=[_make_web_result()],
            rewritten_query="terms",
            generated_answer="Answer.",
        )
        comps["refiner"] = refiner_mock
        pipeline = _build_pipeline(comps)
        result = pipeline.run("Query?")
        assert result.trace.final_context_source == "external"
        assert result.trace.internal_strip_count == 0
        assert result.trace.external_strip_count == 1

    def test_empty_store_trace_has_none_max_score(self):
        """Empty store: no scores computed, max_relevance_score must be None."""
        comps = _mock_component_set(
            retriever_chunks=[],
            evaluator_scores=[],
            router_action="INCORRECT",
            refiner_strips=[_make_external_strip()],
            web_results=[_make_web_result()],
            rewritten_query="search",
            generated_answer="Answer.",
        )
        pipeline = _build_pipeline(comps)
        result = pipeline.run("Query?")
        assert result.trace.max_relevance_score is None
        assert result.trace.retrieved_count == 0
        assert result.trace.action == "INCORRECT"

    def test_trace_is_included_in_result(self):
        """CRAGResult.trace is always an ExecutionTrace instance."""
        comps = _mock_component_set(
            retriever_chunks=[_make_chunk()],
            evaluator_scores=[0.8],
            router_action="CORRECT",
            refiner_strips=[_make_strip()],
        )
        pipeline = _build_pipeline(comps)
        result = pipeline.run("Query?")
        assert isinstance(result.trace, ExecutionTrace)


# ──────────────────────────────────────────────────────────────────────────────
# TestCRAGResultModel
# ──────────────────────────────────────────────────────────────────────────────

class TestCRAGResultModel:
    """CRAGResult and ExecutionTrace Pydantic model field contract."""

    def _make_trace(self, action: str = "CORRECT") -> ExecutionTrace:
        return ExecutionTrace(
            retrieved_count=1,
            action=action,
            max_relevance_score=0.85,
            web_search_used=False,
            rewritten_query=None,
            internal_strip_count=1,
            external_strip_count=0,
            final_context_source="internal",
        )

    def test_result_has_required_fields_with_defaults(self):
        trace = self._make_trace()
        result = CRAGResult(
            answer="Test answer.",
            action="CORRECT",
            query="Test query?",
            trace=trace,
        )
        assert result.answer == "Test answer."
        assert result.action == "CORRECT"
        assert result.query == "Test query?"
        assert result.rewritten_query is None
        assert result.retrieved_chunks == []
        assert result.relevance_scores == []
        assert result.refined_strips == []
        assert result.external_strips == []
        assert result.web_results == []
        assert isinstance(result.trace, ExecutionTrace)

    def test_result_with_all_fields(self):
        chunk = _make_chunk()
        strip = _make_strip()
        ext_strip = _make_external_strip()
        web = _make_web_result()
        trace = ExecutionTrace(
            retrieved_count=1,
            action="AMBIGUOUS",
            max_relevance_score=0.1,
            web_search_used=True,
            rewritten_query="search terms",
            internal_strip_count=1,
            external_strip_count=1,
            final_context_source="combined",
        )
        result = CRAGResult(
            answer="Full answer.",
            action="AMBIGUOUS",
            query="Full query?",
            rewritten_query="search terms",
            retrieved_chunks=[chunk],
            relevance_scores=[0.5],
            refined_strips=[strip],
            external_strips=[ext_strip],
            web_results=[web],
            trace=trace,
        )
        assert result.rewritten_query == "search terms"
        assert len(result.retrieved_chunks) == 1
        assert len(result.refined_strips) == 1
        assert len(result.external_strips) == 1
        assert len(result.web_results) == 1
        assert result.trace.final_context_source == "combined"

    def test_action_is_one_of_three_values(self):
        trace = self._make_trace()
        for action in ("CORRECT", "INCORRECT", "AMBIGUOUS"):
            trace = self._make_trace(action=action)
            result = CRAGResult(answer="a", action=action, query="q", trace=trace)
            assert result.action == action

    def test_execution_trace_model_fields(self):
        trace = ExecutionTrace(
            retrieved_count=3,
            action="AMBIGUOUS",
            max_relevance_score=0.2,
            web_search_used=True,
            rewritten_query="key terms",
            internal_strip_count=2,
            external_strip_count=4,
            final_context_source="combined",
        )
        assert trace.retrieved_count == 3
        assert trace.action == "AMBIGUOUS"
        assert trace.max_relevance_score == pytest.approx(0.2)
        assert trace.web_search_used is True
        assert trace.rewritten_query == "key terms"
        assert trace.internal_strip_count == 2
        assert trace.external_strip_count == 4
        assert trace.final_context_source == "combined"

    def test_execution_trace_optional_fields_default_to_none(self):
        trace = ExecutionTrace(
            retrieved_count=0,
            action="INCORRECT",
            web_search_used=True,
            internal_strip_count=0,
            external_strip_count=0,
            final_context_source="none",
        )
        assert trace.max_relevance_score is None
        assert trace.rewritten_query is None


# ──────────────────────────────────────────────────────────────────────────────
# TestBuildCRAGPrompt
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildCRAGPrompt:
    """Unit tests for the _build_crag_prompt() helper.

    After Issue 1 fix: third argument is external_strips: list[KnowledgeStrip],
    not list[WebSearchResult].
    """

    def test_internal_only_prompt_contains_internal_passage(self):
        strip = _make_strip("Internal knowledge.", score=0.8)
        prompt = _build_crag_prompt("Query?", [strip], [])
        assert "Internal knowledge." in prompt
        assert "(internal document)" in prompt

    def test_external_only_prompt_contains_web_snippet(self):
        ext_strip = _make_external_strip(
            url="https://example.com",
            text="External web content about the topic.",
        )
        prompt = _build_crag_prompt("Query?", [], [ext_strip])
        assert "External web content about the topic." in prompt
        assert "(web)" in prompt

    def test_external_only_prompt_contains_source_url(self):
        ext_strip = _make_external_strip(url="https://example.com")
        prompt = _build_crag_prompt("Query?", [], [ext_strip])
        assert "https://example.com" in prompt

    def test_combined_prompt_contains_both_sources(self):
        strip = _make_strip("Internal strip.")
        ext_strip = _make_external_strip(text="External web content.")
        prompt = _build_crag_prompt("Query?", [strip], [ext_strip])
        assert "Internal strip." in prompt
        assert "(internal document)" in prompt
        assert "External web content." in prompt
        assert "(web)" in prompt

    def test_combined_prompt_numbers_passages_sequentially(self):
        strip = _make_strip("Internal.")
        ext_strip = _make_external_strip(text="External.")
        prompt = _build_crag_prompt("Query?", [strip], [ext_strip])
        assert "[1]" in prompt
        assert "[2]" in prompt

    def test_empty_context_shows_no_context_message(self):
        prompt = _build_crag_prompt("Query?", [], [])
        assert "No relevant context was found." in prompt

    def test_prompt_contains_question(self):
        prompt = _build_crag_prompt("What is CRAG?", [], [])
        assert "What is CRAG?" in prompt

    def test_prompt_ends_with_answer_marker(self):
        prompt = _build_crag_prompt("Query?", [], [])
        assert prompt.strip().endswith("ANSWER:")


# ──────────────────────────────────────────────────────────────────────────────
# TestWebResultsToChunksAdapter
# ──────────────────────────────────────────────────────────────────────────────

class TestWebResultsToChunksAdapter:
    """Unit tests for the _web_results_to_chunks() adapter function."""

    def test_empty_input_returns_empty_list(self):
        assert _web_results_to_chunks([]) == []

    def test_single_result_produces_one_chunk(self):
        web = _make_web_result(url="https://example.com")
        chunks = _web_results_to_chunks([web])
        assert len(chunks) == 1

    def test_chunk_source_is_url(self):
        web = _make_web_result(url="https://example.com")
        chunks = _web_results_to_chunks([web])
        assert chunks[0].source == "https://example.com"

    def test_chunk_text_is_content(self):
        web = _make_web_result()
        chunks = _web_results_to_chunks([web])
        assert chunks[0].text == web.content

    def test_chunk_page_number_is_one(self):
        """Web pages have no page concept; page_number is set to 1."""
        web = _make_web_result()
        chunks = _web_results_to_chunks([web])
        assert chunks[0].page_number == 1

    def test_chunk_score_matches_result_score(self):
        web = _make_web_result()
        chunks = _web_results_to_chunks([web])
        assert chunks[0].score == pytest.approx(web.score)

    def test_metadata_contains_title_and_source_type(self):
        web = _make_web_result(title="My Article")
        chunks = _web_results_to_chunks([web])
        assert chunks[0].metadata["title"] == "My Article"
        assert chunks[0].metadata["source_type"] == "web"

    def test_multiple_results_produce_multiple_chunks(self):
        results = [
            _make_web_result(url="https://a.com"),
            _make_web_result(url="https://b.com"),
            _make_web_result(url="https://c.com"),
        ]
        chunks = _web_results_to_chunks(results)
        assert len(chunks) == 3

    def test_chunks_have_unique_chunk_ids(self):
        results = [
            _make_web_result(url="https://a.com"),
            _make_web_result(url="https://b.com"),
        ]
        chunks = _web_results_to_chunks(results)
        ids = [c.chunk_id for c in chunks]
        assert len(set(ids)) == 2

    def test_output_is_list_of_retrieved_chunks(self):
        web = _make_web_result()
        chunks = _web_results_to_chunks([web])
        assert all(isinstance(c, RetrievedChunk) for c in chunks)
