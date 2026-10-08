"""
Regression tests for the WHO World Health Statistics 2026 production bug.

ROOT CAUSE SUMMARY
==================
The LLM judge in ActionRouter was classifying valid PDF evidence as INCORRECT
because its prompt described CORRECT as requiring evidence that is "directly
relevant and sufficient" -- a criterion the LLM applied too strictly to
factual/numerical queries where the chunk wording was approximate (e.g. the
document says "73 years" while the query context implies 73.3/73.4).

The fix tightens the judge prompt:
  - CORRECT: approximate figures, equivalent statistics, or prose that clearly
    describes the same fact are sufficient.
  - INCORRECT: only genuinely unrelated or contradictory content.
  - AMBIGUOUS: tangential, incomplete, or requires significant inference.

Required regression scenarios (all offline, no real API calls):
  1. Clearly-supported internal factual query  -> CORRECT, no web fallback.
  2. Relevant-but-insufficient (partial) query -> AMBIGUOUS.
  3. Genuinely unsupported query               -> INCORRECT / web fallback.
  4. PDF filename + page provenance preserved.
  5. Judge prompt content and wiring validation.
  6. End-to-end pipeline integration for the WHO scenario.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.pipeline.crag_pipeline import CRAGPipeline, CRAGResult
from app.retrieval.retriever import RetrievedChunk, VectorRetriever
from app.evaluation.relevance_evaluator import RelevanceEvaluator
from app.evaluation.action_router import ActionRouter, RoutingDecision, JUDGE_PROMPT_TEMPLATE
from app.evaluation.knowledge_refiner import KnowledgeRefiner, KnowledgeStrip
from app.external.query_rewriter import QueryRewriter
from app.external.web_search import WebSearchClient, WebSearchResult
from app.generation.llm_provider import LLMProvider


WHO_QUERY = (
    "According to the WHO World Health Statistics 2026 report, "
    "what was global life expectancy at birth in 2019 and in 2023?"
)
WHO_CHUNK_TEXT = (
    "Global life expectancy at birth increased to 73 years in 2019, "
    "the highest ever recorded. During the COVID-19 pandemic it fell "
    "sharply, but by 2022 and 2023 it had recovered to 73 years."
)
WHO_PDF_SOURCE = "World Health Statistics 2026.pdf"
WHO_PDF_PAGE = 42


def _make_who_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="who_p42_c001",
        text=WHO_CHUNK_TEXT,
        source=WHO_PDF_SOURCE,
        page_number=WHO_PDF_PAGE,
        score=0.82,
        metadata={"page_number": WHO_PDF_PAGE, "source_file": WHO_PDF_SOURCE},
    )


def _make_who_strip(text: str = WHO_CHUNK_TEXT) -> KnowledgeStrip:
    return KnowledgeStrip(
        text=text,
        source=WHO_PDF_SOURCE,
        page_number=WHO_PDF_PAGE,
        parent_chunk_id="who_p42_c001",
        score=0.75,
        position=0,
    )


def _make_unrelated_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="unrelated_p1_c001",
        text="The recipe calls for 200g of flour, two eggs, and a pinch of salt.",
        source="cookbook.pdf",
        page_number=1,
        score=0.05,
        metadata={},
    )


def _make_partial_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="who_p40_c001",
        text=(
            "Life expectancy varies significantly across WHO regions. "
            "High-income countries consistently report higher values than "
            "low-income countries."
        ),
        source=WHO_PDF_SOURCE,
        page_number=40,
        score=0.55,
        metadata={"page_number": 40, "source_file": WHO_PDF_SOURCE},
    )


class _MockLLM:
    def __init__(self, decision: str, reason: str = "test reason") -> None:
        self._response = "{" + '"decision": "' + decision + '", "reason": "' + reason + '"}'
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        return self._response


def _make_routing_decision(action: str) -> RoutingDecision:
    return RoutingDecision(
        action=action,
        similarity_pre_filter_decision=action,
        judge_called=False,
        judge_decision=None,
        judge_reason=None,
        judge_latency=None,
    )


def _build_pipeline(
    retriever_chunks,
    evaluator_scores,
    router_decision,
    refiner_strips=None,
    web_results=None,
    generated_answer="The generated answer.",
) -> CRAGPipeline:
    retriever = MagicMock(spec=VectorRetriever)
    retriever.retrieve.return_value = retriever_chunks

    evaluator = MagicMock(spec=RelevanceEvaluator)
    evaluator.score_batch.return_value = evaluator_scores

    router = MagicMock(spec=ActionRouter)
    router.route.return_value = router_decision

    refiner = MagicMock(spec=KnowledgeRefiner)
    refiner.refine.return_value = refiner_strips if refiner_strips is not None else []

    query_rewriter = MagicMock(spec=QueryRewriter)
    query_rewriter.rewrite.return_value = "WHO life expectancy 2019 2023"

    web_search = MagicMock(spec=WebSearchClient)
    web_search.search.return_value = web_results if web_results is not None else []

    llm_client = MagicMock(spec=LLMProvider)
    llm_client.generate.return_value = generated_answer

    return CRAGPipeline(
        retriever=retriever,
        evaluator=evaluator,
        router=router,
        refiner=refiner,
        query_rewriter=query_rewriter,
        web_search=web_search,
        llm_client=llm_client,
        top_k=5,
    )


# ---------------------------------------------------------------------------
# Scenario 1: Clearly-supported internal factual query -> CORRECT, no web
# ---------------------------------------------------------------------------

class TestWHOCorrectBranch:
    def test_correct_action_returned(self):
        chunk = _make_who_chunk()
        strip = _make_who_strip()
        pipeline = _build_pipeline(
            retriever_chunks=[chunk],
            evaluator_scores=[0.64],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=[strip],
            generated_answer="Global life expectancy was 73 years in 2019 and recovered in 2023.",
        )
        result = pipeline.run(WHO_QUERY)
        assert result.action == "CORRECT"

    def test_no_web_search_when_correct(self):
        chunk = _make_who_chunk()
        strip = _make_who_strip()
        pipeline = _build_pipeline(
            retriever_chunks=[chunk],
            evaluator_scores=[0.64],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=[strip],
        )
        result = pipeline.run(WHO_QUERY)
        assert result.trace.web_search_used is False
        pipeline.web_search.search.assert_not_called()

    def test_internal_strips_used_not_external(self):
        chunk = _make_who_chunk()
        strip = _make_who_strip()
        pipeline = _build_pipeline(
            retriever_chunks=[chunk],
            evaluator_scores=[0.64],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=[strip],
        )
        result = pipeline.run(WHO_QUERY)
        assert len(result.refined_strips) == 1
        assert result.trace.internal_strip_count == 1
        assert result.trace.external_strip_count == 0
        assert result.trace.final_context_source == "internal"

    def test_rewritten_query_is_none_for_correct(self):
        pipeline = _build_pipeline(
            retriever_chunks=[_make_who_chunk()],
            evaluator_scores=[0.64],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=[_make_who_strip()],
        )
        result = pipeline.run(WHO_QUERY)
        assert result.rewritten_query is None


# ---------------------------------------------------------------------------
# Scenario 2: Relevant-but-insufficient query -> AMBIGUOUS
# ---------------------------------------------------------------------------

class TestWHOAmbiguousBranch:
    def test_ambiguous_action_returned(self):
        chunk = _make_partial_chunk()
        strip = _make_who_strip(text=chunk.text)
        web_result = WebSearchResult(
            title="WHO Report 2026",
            url="https://www.who.int/stats2026",
            content="Life expectancy was 73 years in 2019.",
            score=0.8,
        )
        pipeline = _build_pipeline(
            retriever_chunks=[chunk],
            evaluator_scores=[0.3],
            router_decision=_make_routing_decision("AMBIGUOUS"),
            refiner_strips=[strip],
            web_results=[web_result],
        )
        result = pipeline.run(WHO_QUERY)
        assert result.action == "AMBIGUOUS"

    def test_ambiguous_uses_web_search(self):
        chunk = _make_partial_chunk()
        strip = _make_who_strip(text=chunk.text)
        web_result = WebSearchResult(
            title="WHO Report 2026",
            url="https://www.who.int/stats2026",
            content="Life expectancy was 73 years in 2019.",
            score=0.8,
        )
        pipeline = _build_pipeline(
            retriever_chunks=[chunk],
            evaluator_scores=[0.3],
            router_decision=_make_routing_decision("AMBIGUOUS"),
            refiner_strips=[strip],
            web_results=[web_result],
        )
        result = pipeline.run(WHO_QUERY)
        assert result.trace.web_search_used is True
        # AMBIGUOUS calls refiner twice (internal + external)
        assert pipeline.refiner.refine.call_count == 2


# ---------------------------------------------------------------------------
# Scenario 3: Genuinely unsupported query -> INCORRECT / web fallback
# ---------------------------------------------------------------------------

class TestWHOIncorrectBranch:
    def test_incorrect_action_returned(self):
        chunk = _make_unrelated_chunk()
        pipeline = _build_pipeline(
            retriever_chunks=[chunk],
            evaluator_scores=[-0.3],
            router_decision=_make_routing_decision("INCORRECT"),
            refiner_strips=[],
            web_results=[WebSearchResult(
                title="WHO", url="https://who.int", content="73.3 years in 2019.", score=0.9,
            )],
        )
        result = pipeline.run(WHO_QUERY)
        assert result.action == "INCORRECT"

    def test_incorrect_triggers_web_search(self):
        chunk = _make_unrelated_chunk()
        pipeline = _build_pipeline(
            retriever_chunks=[chunk],
            evaluator_scores=[-0.3],
            router_decision=_make_routing_decision("INCORRECT"),
            refiner_strips=[],
            web_results=[WebSearchResult(
                title="WHO", url="https://who.int", content="73.3 years in 2019.", score=0.9,
            )],
        )
        result = pipeline.run(WHO_QUERY)
        assert result.trace.web_search_used is True
        assert pipeline.web_search.search.call_count == 1

    def test_incorrect_has_no_internal_strips(self):
        chunk = _make_unrelated_chunk()
        pipeline = _build_pipeline(
            retriever_chunks=[chunk],
            evaluator_scores=[-0.3],
            router_decision=_make_routing_decision("INCORRECT"),
            refiner_strips=[],
            web_results=[],
        )
        result = pipeline.run(WHO_QUERY)
        assert result.refined_strips == []
        assert result.trace.internal_strip_count == 0


# ---------------------------------------------------------------------------
# Scenario 4: PDF filename + page provenance preserved
# ---------------------------------------------------------------------------

class TestWHOProvenance:
    def test_strip_source_matches_pdf_filename(self):
        pipeline = _build_pipeline(
            retriever_chunks=[_make_who_chunk()],
            evaluator_scores=[0.64],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=[_make_who_strip()],
        )
        result = pipeline.run(WHO_QUERY)
        assert result.refined_strips[0].source == WHO_PDF_SOURCE

    def test_strip_page_number_is_valid_positive_integer(self):
        pipeline = _build_pipeline(
            retriever_chunks=[_make_who_chunk()],
            evaluator_scores=[0.64],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=[_make_who_strip()],
        )
        result = pipeline.run(WHO_QUERY)
        page = result.refined_strips[0].page_number
        assert isinstance(page, int)
        assert page > 0

    def test_strip_page_number_matches_source_chunk(self):
        pipeline = _build_pipeline(
            retriever_chunks=[_make_who_chunk()],
            evaluator_scores=[0.64],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=[_make_who_strip()],
        )
        result = pipeline.run(WHO_QUERY)
        assert result.refined_strips[0].page_number == WHO_PDF_PAGE

    def test_retrieved_chunk_carries_correct_metadata(self):
        pipeline = _build_pipeline(
            retriever_chunks=[_make_who_chunk()],
            evaluator_scores=[0.64],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=[_make_who_strip()],
        )
        result = pipeline.run(WHO_QUERY)
        rc = result.retrieved_chunks[0]
        assert rc.source == WHO_PDF_SOURCE
        assert rc.page_number == WHO_PDF_PAGE

    def test_incorrect_branch_produces_no_internal_provenance(self):
        """Confirms the INCORRECT branch intentionally has no internal strip
        provenance -- this is expected behaviour for the web-fallback path,
        not a bug. The 'page--' display was caused by the router incorrectly
        routing to INCORRECT, not by provenance code itself."""
        pipeline = _build_pipeline(
            retriever_chunks=[_make_unrelated_chunk()],
            evaluator_scores=[-0.3],
            router_decision=_make_routing_decision("INCORRECT"),
            refiner_strips=[],
            web_results=[],
        )
        result = pipeline.run(WHO_QUERY)
        assert result.refined_strips == []
        assert result.trace.internal_strip_count == 0


# ---------------------------------------------------------------------------
# Scenario 5: Judge prompt content and ActionRouter wiring
# ---------------------------------------------------------------------------

class TestJudgePromptAndWiring:
    def test_judge_prompt_contains_approximate_guidance(self):
        """After fix: the prompt must explicitly say approximate figures are sufficient."""
        assert "approximate" in JUDGE_PROMPT_TEMPLATE.lower(), (
            "JUDGE_PROMPT_TEMPLATE must mention approximate figures as sufficient for CORRECT"
        )

    def test_judge_prompt_not_require_exact_numbers(self):
        """After fix: the prompt must explicitly say INCORRECT is not for approximate numbers."""
        prompt_lower = JUDGE_PROMPT_TEMPLATE.lower()
        assert "exact numbers" in prompt_lower or "exact numerical" in prompt_lower, (
            "JUDGE_PROMPT_TEMPLATE must clarify that approximate numbers are NOT INCORRECT"
        )

    def test_borderline_score_triggers_judge_not_prefilter(self):
        """s_max=0.64 is between beta=-0.1 and alpha=0.7 so judge must fire."""
        llm = _MockLLM("CORRECT")
        router = ActionRouter(
            clearly_relevant_threshold=0.7,
            clearly_irrelevant_threshold=-0.1,
            llm_client=llm,
        )
        decision = router.route(WHO_QUERY, [_make_who_chunk()], [0.64])
        assert decision.similarity_pre_filter_decision == "AMBIGUOUS"
        assert decision.judge_called is True

    def test_judge_correct_used_for_approximate_evidence(self):
        """Judge returning CORRECT on approximate evidence must yield CORRECT action."""
        llm = _MockLLM("CORRECT", "Evidence covers life expectancy for the relevant years.")
        router = ActionRouter(
            clearly_relevant_threshold=0.7,
            clearly_irrelevant_threshold=-0.1,
            llm_client=llm,
        )
        decision = router.route(WHO_QUERY, [_make_who_chunk()], [0.64])
        assert decision.action == "CORRECT"
        assert decision.judge_called is True
        assert llm.call_count == 1

    def test_judge_incorrect_on_unrelated_still_routes_incorrectly(self):
        """Fix must not break the genuine INCORRECT path."""
        llm = _MockLLM("INCORRECT", "Evidence is about cooking, not health statistics.")
        router = ActionRouter(
            clearly_relevant_threshold=0.7,
            clearly_irrelevant_threshold=-0.1,
            llm_client=llm,
        )
        decision = router.route(WHO_QUERY, [_make_unrelated_chunk()], [0.3])
        assert decision.action == "INCORRECT"
        assert decision.judge_called is True

    def test_clearly_relevant_score_bypasses_judge(self):
        """s_max >= alpha must short-circuit without calling the judge."""
        llm = _MockLLM("INCORRECT")  # would be wrong if called
        router = ActionRouter(
            clearly_relevant_threshold=0.7,
            clearly_irrelevant_threshold=-0.1,
            llm_client=llm,
        )
        decision = router.route(WHO_QUERY, [_make_who_chunk()], [0.75])
        assert decision.action == "CORRECT"
        assert decision.judge_called is False
        assert llm.call_count == 0

    def test_clearly_irrelevant_score_bypasses_judge(self):
        """s_max <= beta must short-circuit without calling the judge."""
        llm = _MockLLM("CORRECT")  # would be wrong if called
        router = ActionRouter(
            clearly_relevant_threshold=0.7,
            clearly_irrelevant_threshold=-0.1,
            llm_client=llm,
        )
        decision = router.route("unrelated query", [_make_unrelated_chunk()], [-0.2])
        assert decision.action == "INCORRECT"
        assert decision.judge_called is False
        assert llm.call_count == 0


# ---------------------------------------------------------------------------
# Scenario 6: End-to-end WHO pipeline integration
# ---------------------------------------------------------------------------

class TestWHOEndToEndIntegration:
    def test_who_query_correct_end_to_end(self):
        """Full pipeline run: CORRECT action, PDF provenance, no Tavily."""
        who_chunk = _make_who_chunk()
        who_strip = _make_who_strip()
        expected_answer = (
            "Global life expectancy at birth was 73 years in 2019 "
            "and recovered to 73 years in 2023 after the COVID-19 dip."
        )
        pipeline = _build_pipeline(
            retriever_chunks=[who_chunk],
            evaluator_scores=[0.64],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=[who_strip],
            web_results=[],
            generated_answer=expected_answer,
        )
        result = pipeline.run(WHO_QUERY)

        assert result.action == "CORRECT"
        assert result.trace.web_search_used is False
        pipeline.web_search.search.assert_not_called()
        assert result.trace.internal_strip_count == 1
        assert result.trace.external_strip_count == 0
        assert result.trace.final_context_source == "internal"
        assert result.refined_strips[0].source == WHO_PDF_SOURCE
        assert isinstance(result.refined_strips[0].page_number, int)
        assert result.refined_strips[0].page_number == WHO_PDF_PAGE
        assert result.answer == expected_answer
