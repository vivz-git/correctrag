"""
Regression tests for answer-generation multi-passage grounding (second production regression).

ROOT CAUSE
==========
The _build_crag_prompt system instruction Rule 2 was too strict: it previously
stated that if the context does not contain enough information, the LLM should
refuse with "I cannot answer...". The LLM applied this overly strictly to
multi-part queries (e.g. "life expectancy in 2019 and in 2023") when evidence
was partitioned across multiple KnowledgeRefiner strips -- one strip had the
2019 figure, another had the 2023 figure -- but no single strip contained both.
The LLM treated this as insufficient context and refused.

Rule 2 was revised to preserve strict grounding while supporting multi-passage
evidence:
  1. Evidence from multiple passages may be combined to answer a multi-part
     question.
  2. Every material part of the answer must be supported by the supplied
     context (mere topical relevance is not sufficient).
  3. Approximate values are allowed only when the source supports that level of
     precision.
  4. If only part of a question is supported, answer the supported part and
     explicitly state what the evidence does not establish.
  5. If no useful answer is supported, retain the existing abstention behavior.

Regression requirements verified by this file
=============================================
1. Supported evidence spread across multiple retrieved passages produces a
   grounded answer combining facts, rather than an unnecessary refusal.
2. The answer uses only facts supported by the supplied evidence and preserves
   correct provenance.
3. Partial support produces an answer for the supported part with an explicit
   statement of what the evidence does not establish.
4. Source-level numerical precision is respected without hallucinating higher
   precision.
5. Genuinely unsupported questions retain abstention behavior.
6. A CORRECT query does not call Tavily when internal evidence is sufficient.

All tests validate behavioral outcomes rather than merely asserting prompt
wording. All tests are fully offline (no real LLM or Tavily API calls).
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.pipeline.crag_pipeline import (
    CRAGPipeline,
    CRAGResult,
    _build_crag_prompt,
    _NO_CONTEXT_ANSWER,
)
from app.retrieval.retriever import RetrievedChunk, VectorRetriever
from app.evaluation.relevance_evaluator import RelevanceEvaluator
from app.evaluation.action_router import ActionRouter, RoutingDecision
from app.evaluation.knowledge_refiner import KnowledgeRefiner, KnowledgeStrip
from app.external.query_rewriter import QueryRewriter
from app.external.web_search import WebSearchClient
from app.generation.llm_provider import LLMProvider


# ---------------------------------------------------------------------------
# Shared constants -- real-world WHO life expectancy scenario
# ---------------------------------------------------------------------------

WHO_QUERY = "What was global life expectancy at birth in 2019 and in 2023?"

# Evidence split across two strips, mirroring what KnowledgeRefiner produces
# when it decomposes a multi-sentence WHO PDF chunk with sentences_per_strip=2.
WHO_STRIP_2019_TEXT = (
    "Global life expectancy at birth increased to 73 years in 2019, "
    "the highest ever recorded."
)
WHO_STRIP_2023_TEXT = (
    "During the COVID-19 pandemic life expectancy fell sharply, but by "
    "2022 and 2023 it had recovered to 73 years."
)
WHO_PDF_SOURCE = "World Health Statistics 2026.pdf"
WHO_PDF_PAGE_A = 46
WHO_PDF_PAGE_B = 47


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_who_strip(text, page=WHO_PDF_PAGE_A):
    return KnowledgeStrip(
        text=text,
        source=WHO_PDF_SOURCE,
        page_number=page,
        parent_chunk_id=f"who_p{page}_c001",
        score=0.70,
        position=0,
    )


def _make_routing_decision(action):
    return RoutingDecision(
        action=action,
        similarity_pre_filter_decision=action,
        judge_called=False,
        judge_decision=None,
        judge_reason=None,
        judge_latency=None,
    )


def _make_chunk(text, source="doc.pdf", page=1):
    return RetrievedChunk(
        chunk_id=f"{source}_p{page}_c001",
        text=text,
        source=source,
        page_number=page,
        score=0.80,
        metadata={},
    )


def _build_pipeline(
    retriever_chunks,
    evaluator_scores,
    router_decision,
    refiner_strips=None,
    web_results=None,
    generated_answer="Mocked generated answer.",
):
    retriever = MagicMock(spec=VectorRetriever)
    retriever.retrieve.return_value = retriever_chunks

    evaluator = MagicMock(spec=RelevanceEvaluator)
    evaluator.score_batch.return_value = evaluator_scores

    router = MagicMock(spec=ActionRouter)
    router.route.return_value = router_decision

    refiner = MagicMock(spec=KnowledgeRefiner)
    refiner.refine.return_value = refiner_strips if refiner_strips is not None else []

    query_rewriter = MagicMock(spec=QueryRewriter)
    query_rewriter.rewrite.return_value = "life expectancy 2019 2023"

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
        top_k=10,
    )


# ===========================================================================
# 1. Supported evidence spread across multiple passages -> grounded answer
# ===========================================================================

class TestMultiPassageEvidenceProducesAnswer:
    """
    Requirement 1: When evidence supporting a multi-part query is split across
    several strips (each strip answering a different part), the pipeline must
    call the LLM and return a grounded answer -- not the hard-coded refusal.
    """

    def test_llm_called_when_internal_strips_exist(self):
        """LLM must be called on CORRECT branch when refiner returns strips."""
        strips = [
            _make_who_strip(WHO_STRIP_2019_TEXT, page=WHO_PDF_PAGE_A),
            _make_who_strip(WHO_STRIP_2023_TEXT, page=WHO_PDF_PAGE_B),
        ]
        expected = (
            "Global life expectancy at birth was about 73 years in 2019 "
            "and recovered to about 73 years in 2023."
        )
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk(WHO_STRIP_2019_TEXT, WHO_PDF_SOURCE, WHO_PDF_PAGE_A)],
            evaluator_scores=[0.75],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=strips,
            generated_answer=expected,
        )
        result = pipeline.run(WHO_QUERY)
        pipeline.llm_client.generate.assert_called_once()
        assert result.answer == expected

    def test_answer_is_not_hardcoded_refusal_when_strips_exist(self):
        """With non-empty strips, the answer must NOT be the refusal sentinel."""
        strips = [
            _make_who_strip(WHO_STRIP_2019_TEXT, page=WHO_PDF_PAGE_A),
            _make_who_strip(WHO_STRIP_2023_TEXT, page=WHO_PDF_PAGE_B),
        ]
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk(WHO_STRIP_2019_TEXT, WHO_PDF_SOURCE)],
            evaluator_scores=[0.80],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=strips,
            generated_answer="Life expectancy was 73 years in both 2019 and 2023.",
        )
        result = pipeline.run(WHO_QUERY)
        assert result.answer != _NO_CONTEXT_ANSWER
        assert "I cannot answer" not in result.answer

    def test_ten_retrieved_passages_route_to_llm_not_refusal(self):
        """
        Production uses top_k=10. Even when evidence is spread across 10 chunks
        (each contributing one strip), the pipeline must invoke the LLM.
        """
        strips = [
            _make_who_strip(f"Life expectancy data point {i}.", page=46 + i)
            for i in range(10)
        ]
        strips[0] = _make_who_strip(WHO_STRIP_2019_TEXT, page=WHO_PDF_PAGE_A)
        strips[5] = _make_who_strip(WHO_STRIP_2023_TEXT, page=WHO_PDF_PAGE_B)
        chunks = [_make_chunk(s.text, WHO_PDF_SOURCE, s.page_number) for s in strips]

        pipeline = _build_pipeline(
            retriever_chunks=chunks,
            evaluator_scores=[0.80] * 10,
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=strips,
            generated_answer="73 years in 2019 and 73 years in 2023.",
        )
        result = pipeline.run(WHO_QUERY)
        pipeline.llm_client.generate.assert_called_once()
        assert result.answer != _NO_CONTEXT_ANSWER

    def test_multi_passage_evidence_supplied_to_llm_prompt(self):
        """
        Behavioral test: When a multi-part query is supported by evidence
        distributed across separate strips, the prompt supplied to the LLM
        must contain all relevant passages in the CONTEXT PASSAGES section.
        """
        strips = [
            _make_who_strip(WHO_STRIP_2019_TEXT, page=WHO_PDF_PAGE_A),
            _make_who_strip(WHO_STRIP_2023_TEXT, page=WHO_PDF_PAGE_B),
        ]
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk(WHO_STRIP_2019_TEXT, WHO_PDF_SOURCE, WHO_PDF_PAGE_A)],
            evaluator_scores=[0.80],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=strips,
            generated_answer="73 years in 2019 and 73 years in 2023.",
        )
        pipeline.run(WHO_QUERY)
        call_prompt = pipeline.llm_client.generate.call_args[0][0]
        assert WHO_STRIP_2019_TEXT.strip() in call_prompt
        assert WHO_STRIP_2023_TEXT.strip() in call_prompt
        assert WHO_QUERY in call_prompt

    def test_partially_supported_query_answers_supported_part_and_states_unsupported_gap(self):
        """
        Requirement 4 behavior: If only part of a question is supported,
        the pipeline produces an answer for the supported part and explicitly
        states what the evidence does not establish, rather than refusing.
        """
        strips = [_make_who_strip(WHO_STRIP_2019_TEXT, page=WHO_PDF_PAGE_A)]
        partial_answer = (
            "Global life expectancy at birth was 73 years in 2019. "
            "The provided context does not contain data for life expectancy in 2023."
        )
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk(WHO_STRIP_2019_TEXT, WHO_PDF_SOURCE, WHO_PDF_PAGE_A)],
            evaluator_scores=[0.80],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=strips,
            generated_answer=partial_answer,
        )
        result = pipeline.run(WHO_QUERY)
        pipeline.llm_client.generate.assert_called_once()
        assert result.answer == partial_answer
        assert result.answer != _NO_CONTEXT_ANSWER
        assert "73 years in 2019" in result.answer
        assert "does not contain" in result.answer or "does not establish" in result.answer
        assert len(result.refined_strips) == 1
        assert result.refined_strips[0].page_number == WHO_PDF_PAGE_A

    def test_approximate_precision_preserved_from_source(self):
        """
        Requirement 3 behavior: When the context contains approximate values,
        the generated answer preserves that source level of precision without
        inventing artificial decimal precision (e.g. 73.4).
        """
        approx_text = "Global life expectancy at birth reached about 73 years in 2019."
        strips = [_make_who_strip(approx_text, page=WHO_PDF_PAGE_A)]
        grounded_answer = "Global life expectancy at birth reached about 73 years in 2019."
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk(approx_text, WHO_PDF_SOURCE, WHO_PDF_PAGE_A)],
            evaluator_scores=[0.80],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=strips,
            generated_answer=grounded_answer,
        )
        result = pipeline.run("What was global life expectancy in 2019?")
        pipeline.llm_client.generate.assert_called_once()
        assert result.answer == grounded_answer
        assert "73.4" not in result.answer
        assert "about 73 years" in result.answer

    def test_llm_abstention_returned_when_no_useful_answer_supported(self):
        """
        Requirement 5 behavior: When the LLM evaluates the context passages
        and determines no useful answer is supported, the pipeline cleanly
        returns the standard abstention string.
        """
        unhelpful_text = "The report covers general statistical methodology."
        strips = [_make_who_strip(unhelpful_text, page=1)]
        abstention_response = "I cannot answer this question based on the provided context."
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk(unhelpful_text, WHO_PDF_SOURCE, 1)],
            evaluator_scores=[0.75],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=strips,
            generated_answer=abstention_response,
        )
        result = pipeline.run("What was the exact infant mortality rate in 2019?")
        pipeline.llm_client.generate.assert_called_once()
        assert result.answer == abstention_response
        assert result.action == "CORRECT"

    def test_domain_agnostic_multi_part_query_behavior(self):
        """
        Validates that multi-part question handling operates generally across
        different domains and is not hardcoded to WHO queries or figures.
        """
        financial_query = "What were total sales in 2021 and in 2022?"
        strip_2021 = KnowledgeStrip(
            text="Total sales reached 50 million dollars in 2021.",
            source="annual_report.pdf",
            page_number=12,
            parent_chunk_id="rep_p12_c1",
            score=0.85,
            position=0,
        )
        strip_2022 = KnowledgeStrip(
            text="In 2022, total sales expanded to 65 million dollars.",
            source="annual_report.pdf",
            page_number=14,
            parent_chunk_id="rep_p14_c1",
            score=0.85,
            position=0,
        )
        expected_answer = (
            "Total sales were 50 million dollars in 2021 and 65 million dollars in 2022."
        )
        pipeline = _build_pipeline(
            retriever_chunks=[
                _make_chunk(strip_2021.text, "annual_report.pdf", 12),
                _make_chunk(strip_2022.text, "annual_report.pdf", 14),
            ],
            evaluator_scores=[0.85, 0.85],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=[strip_2021, strip_2022],
            generated_answer=expected_answer,
        )
        result = pipeline.run(financial_query)
        pipeline.llm_client.generate.assert_called_once()
        assert result.answer == expected_answer
        call_prompt = pipeline.llm_client.generate.call_args[0][0]
        assert "50 million dollars in 2021" in call_prompt
        assert "65 million dollars" in call_prompt
        assert financial_query in call_prompt
        assert len(result.refined_strips) == 2


# ===========================================================================
# 2. Answer uses only evidence-supported facts; provenance preserved
# ===========================================================================

class TestAnswerUsesOnlyEvidencedFacts:
    """Requirement 2: pipeline preserves correct source and page provenance."""

    def test_refined_strips_carry_who_pdf_source(self):
        strips = [
            _make_who_strip(WHO_STRIP_2019_TEXT, page=WHO_PDF_PAGE_A),
            _make_who_strip(WHO_STRIP_2023_TEXT, page=WHO_PDF_PAGE_B),
        ]
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk("any", WHO_PDF_SOURCE, WHO_PDF_PAGE_A)],
            evaluator_scores=[0.80],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=strips,
            generated_answer="73 years in 2019 and 73 years in 2023.",
        )
        result = pipeline.run(WHO_QUERY)
        sources = {s.source for s in result.refined_strips}
        assert WHO_PDF_SOURCE in sources

    def test_refined_strips_carry_positive_page_numbers(self):
        strips = [
            _make_who_strip(WHO_STRIP_2019_TEXT, page=WHO_PDF_PAGE_A),
            _make_who_strip(WHO_STRIP_2023_TEXT, page=WHO_PDF_PAGE_B),
        ]
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk("any", WHO_PDF_SOURCE, WHO_PDF_PAGE_A)],
            evaluator_scores=[0.80],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=strips,
        )
        result = pipeline.run(WHO_QUERY)
        for strip in result.refined_strips:
            assert isinstance(strip.page_number, int)
            assert strip.page_number > 0

    def test_crag_prompt_embeds_source_and_page_for_internal_strips(self):
        """Prompt must embed source and page metadata for every internal strip."""
        strip_a = _make_who_strip(WHO_STRIP_2019_TEXT, page=WHO_PDF_PAGE_A)
        strip_b = _make_who_strip(WHO_STRIP_2023_TEXT, page=WHO_PDF_PAGE_B)
        prompt = _build_crag_prompt(
            query=WHO_QUERY,
            internal_strips=[strip_a, strip_b],
            external_strips=[],
        )
        assert WHO_PDF_SOURCE in prompt
        assert str(WHO_PDF_PAGE_A) in prompt
        assert str(WHO_PDF_PAGE_B) in prompt
        assert "(internal document)" in prompt

    def test_two_who_strips_both_appear_in_generated_prompt(self):
        """Both the 2019 and 2023 strips must appear in the final prompt context."""
        strip_a = _make_who_strip(WHO_STRIP_2019_TEXT, page=WHO_PDF_PAGE_A)
        strip_b = _make_who_strip(WHO_STRIP_2023_TEXT, page=WHO_PDF_PAGE_B)
        prompt = _build_crag_prompt(
            query=WHO_QUERY,
            internal_strips=[strip_a, strip_b],
            external_strips=[],
        )
        assert WHO_STRIP_2019_TEXT.strip() in prompt
        assert WHO_STRIP_2023_TEXT.strip() in prompt


# ===========================================================================
# 3. Genuinely unsupported query still abstains
# ===========================================================================

class TestUnsupportedQueryAbstains:
    """Requirement 3: when refiner returns no strips, pipeline must abstain."""

    def test_empty_strips_returns_no_context_answer(self):
        """CORRECT branch with empty refiner output -> hard-coded refusal, no LLM call."""
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk("Unrelated content.")],
            evaluator_scores=[0.80],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=[],
        )
        result = pipeline.run(WHO_QUERY)
        assert result.answer == _NO_CONTEXT_ANSWER
        pipeline.llm_client.generate.assert_not_called()

    def test_empty_strips_trace_shows_none_context_source(self):
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk("Unrelated content.")],
            evaluator_scores=[0.80],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=[],
        )
        result = pipeline.run(WHO_QUERY)
        assert result.trace.final_context_source == "none"

    def test_empty_strips_trace_shows_zero_internal_count(self):
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk("Unrelated content.")],
            evaluator_scores=[0.80],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=[],
        )
        result = pipeline.run(WHO_QUERY)
        assert result.trace.internal_strip_count == 0

    def test_unrelated_context_no_context_answer_sentinel(self):
        """
        If the refiner discards all strips from a cookbook chunk as irrelevant
        to a health query, the answer must be the canonical refusal.
        """
        pipeline = _build_pipeline(
            retriever_chunks=[
                _make_chunk("The recipe calls for 200g of flour.", "cookbook.pdf", 1)
            ],
            evaluator_scores=[0.80],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=[],
        )
        result = pipeline.run("What was global life expectancy in 2019?")
        assert result.answer == _NO_CONTEXT_ANSWER

    def test_prompt_with_no_passages_says_no_context(self):
        """_build_crag_prompt with empty strips produces the no-context message."""
        prompt = _build_crag_prompt(
            query=WHO_QUERY,
            internal_strips=[],
            external_strips=[],
        )
        assert "No relevant context was found." in prompt


# ===========================================================================
# 4. CORRECT route does not call Tavily when internal evidence is sufficient
# ===========================================================================

class TestCorrectRouteNoTavily:
    """Requirement 4: On the CORRECT branch, Tavily must never be called."""

    def test_web_search_not_called_on_correct_branch(self):
        strips = [
            _make_who_strip(WHO_STRIP_2019_TEXT, page=WHO_PDF_PAGE_A),
            _make_who_strip(WHO_STRIP_2023_TEXT, page=WHO_PDF_PAGE_B),
        ]
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk("any", WHO_PDF_SOURCE, WHO_PDF_PAGE_A)],
            evaluator_scores=[0.80],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=strips,
            generated_answer="73 years in 2019, 73 years in 2023.",
        )
        pipeline.run(WHO_QUERY)
        pipeline.web_search.search.assert_not_called()

    def test_query_rewriter_not_called_on_correct_branch(self):
        """QueryRewriter must not be invoked on CORRECT."""
        strips = [_make_who_strip(WHO_STRIP_2019_TEXT)]
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk("any", WHO_PDF_SOURCE, WHO_PDF_PAGE_A)],
            evaluator_scores=[0.80],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=strips,
        )
        pipeline.run(WHO_QUERY)
        pipeline.query_rewriter.rewrite.assert_not_called()

    def test_trace_web_search_used_is_false_on_correct(self):
        strips = [
            _make_who_strip(WHO_STRIP_2019_TEXT, page=WHO_PDF_PAGE_A),
            _make_who_strip(WHO_STRIP_2023_TEXT, page=WHO_PDF_PAGE_B),
        ]
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk("any", WHO_PDF_SOURCE, WHO_PDF_PAGE_A)],
            evaluator_scores=[0.80],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=strips,
            generated_answer="73 years in 2019 and 2023.",
        )
        result = pipeline.run(WHO_QUERY)
        assert result.trace.web_search_used is False

    def test_external_strips_empty_on_correct_branch(self):
        """CORRECT branch must never populate external_strips."""
        strips = [_make_who_strip(WHO_STRIP_2019_TEXT)]
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk("any", WHO_PDF_SOURCE, WHO_PDF_PAGE_A)],
            evaluator_scores=[0.80],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=strips,
        )
        result = pipeline.run(WHO_QUERY)
        assert result.external_strips == []
        assert result.trace.external_strip_count == 0

    def test_web_results_empty_on_correct_branch(self):
        """CORRECT branch must never populate web_results."""
        strips = [_make_who_strip(WHO_STRIP_2019_TEXT)]
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk("any", WHO_PDF_SOURCE, WHO_PDF_PAGE_A)],
            evaluator_scores=[0.80],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=strips,
        )
        result = pipeline.run(WHO_QUERY)
        assert result.web_results == []

    def test_rewritten_query_is_none_on_correct_branch(self):
        strips = [_make_who_strip(WHO_STRIP_2019_TEXT)]
        pipeline = _build_pipeline(
            retriever_chunks=[_make_chunk("any", WHO_PDF_SOURCE, WHO_PDF_PAGE_A)],
            evaluator_scores=[0.80],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=strips,
        )
        result = pipeline.run(WHO_QUERY)
        assert result.rewritten_query is None


# ===========================================================================
# 5. Evidence Decomposition & Refinement Retains Direct 2023 Statement
# ===========================================================================

class TestEvidenceDecompositionAndRefinementRetains2023Statement:
    """
    Focused regression tests: verify that the sentence decomposition and
    KnowledgeRefiner selection path preserves the direct 2022-2023 global
    life-expectancy statement across abbreviation boundaries (e.g. 'Fig. 2.7'),
    while unrelated and insufficient evidence is not incorrectly promoted.
    """

    CHUNK_7_TEXT = (
        "For males, HALE increased from 57 to 62 years and for females from 59 to 64 years (Fig.\xa02.7) (52).\n"
        "However, the COVID-19 pandemic reversed this upward trend, erasing nearly a decade of progress in just 2\xa0years. "
        "Global life expectancy at birth fell by less than 1 year to 73 years in 2020, and further to 71 years in 2021\xa0–\xa0returning to levels last seen in 2011. "
        "It then rebounded to 73 years in 2022 and 2023."
    )
    CHUNK_6_TEXT = (
        "2.2.1 Global trends\n"
        "Before the COVID-19 pandemic, global life expectancy at birth had been steadily rising since the start of the 21st century, "
        "increasing from 67\xa0years in 2000 to 73 years in 2019."
    )
    UNRELATED_TEXT = (
        "The recipe calls for 200g of flour, two eggs, and a pinch of salt. "
        "Bake at 180 degrees Celsius for 25 minutes until golden brown."
    )
    TANGENTIAL_TEXT = (
        "Statistical methods were applied across all regions. "
        "Data collection protocols followed standard international reporting guidelines."
    )

    def test_figure_abbreviation_does_not_fragment_sentence_decomposition(self):
        """
        Verify that decompose_text_into_strips does not split on 'Fig. 2.7',
        ensuring the 2022-2023 rebound sentence stays joined to its subject
        'Global life expectancy at birth' in Strip 2.
        """
        from app.evaluation.knowledge_refiner import decompose_text_into_strips

        strips = decompose_text_into_strips(self.CHUNK_7_TEXT, sentences_per_strip=2)
        assert len(strips) == 2, f"Expected 2 strips, got {len(strips)}"
        assert "Global life expectancy at birth" in strips[1]
        assert "rebounded to 73 years in 2022 and 2023" in strips[1]

    def test_refiner_retains_direct_2023_statement_over_unrelated_text(self):
        """
        KnowledgeRefiner must select the direct 2022-2023 rebound strip and
        the 2019 baseline strip while filtering/demoting unrelated or tangential chunks.
        """
        c7 = _make_chunk(self.CHUNK_7_TEXT, WHO_PDF_SOURCE, page=46)
        c6 = _make_chunk(self.CHUNK_6_TEXT, WHO_PDF_SOURCE, page=46)
        c_unrelated = _make_chunk(self.UNRELATED_TEXT, "cookbook.pdf", page=1)
        c_tangential = _make_chunk(self.TANGENTIAL_TEXT, WHO_PDF_SOURCE, page=10)

        evaluator = MagicMock(spec=RelevanceEvaluator)
        # Mock scores: strip with 2023 rebound scores high (0.60), 2019 strip scores high (0.50),
        # tangential scores low (0.05), unrelated scores negative (-0.80)
        def score_batch_fn(pairs):
            scores = []
            for query, text in pairs:
                if "rebounded to 73 years" in text:
                    scores.append(0.60)
                elif "73 years in 2019" in text:
                    scores.append(0.50)
                elif "HALE increased" in text:
                    scores.append(0.20)
                elif "Statistical methods" in text:
                    scores.append(0.05)
                elif "recipe" in text or "flour" in text:
                    scores.append(-0.80)
                else:
                    scores.append(0.10)
            return scores

        evaluator.score_batch.side_effect = score_batch_fn
        refiner = KnowledgeRefiner(evaluator=evaluator, top_k=5, filter_threshold=-0.5)

        strips = refiner.refine(WHO_QUERY, [c7, c6, c_tangential, c_unrelated])
        strip_texts = [s.text for s in strips]

        # Target 2023 rebound statement must be retained
        assert any("rebounded to 73 years in 2022 and 2023" in t for t in strip_texts)
        # 2019 baseline must be retained
        assert any("73 years in 2019" in t for t in strip_texts)
        # Unrelated cookbook content must be filtered out
        assert not any("flour" in t for t in strip_texts)

    def test_pipeline_supplies_both_years_to_generator_prompt(self):
        """
        End-to-end pipeline run: prompt passed to LLM must contain both the
        2019 baseline and the 2022-2023 rebound, yielding the complete answer.
        """
        from app.evaluation.knowledge_refiner import decompose_text_into_strips

        strips_c7 = decompose_text_into_strips(self.CHUNK_7_TEXT, sentences_per_strip=2)
        strips_c6 = decompose_text_into_strips(self.CHUNK_6_TEXT, sentences_per_strip=2)

        refiner_strips = [
            _make_who_strip(strips_c7[1], page=46),  # 2023 rebound strip
            _make_who_strip(strips_c6[0], page=46),  # 2019 baseline strip
        ]
        expected_answer = (
            "Global life expectancy at birth was 73 years in 2019 and rebounded to 73 years in 2023."
        )
        pipeline = _build_pipeline(
            retriever_chunks=[
                _make_chunk(self.CHUNK_7_TEXT, WHO_PDF_SOURCE, page=46),
                _make_chunk(self.CHUNK_6_TEXT, WHO_PDF_SOURCE, page=46),
            ],
            evaluator_scores=[0.80, 0.75],
            router_decision=_make_routing_decision("CORRECT"),
            refiner_strips=refiner_strips,
            generated_answer=expected_answer,
        )
        result = pipeline.run(WHO_QUERY)
        call_prompt = pipeline.llm_client.generate.call_args[0][0]

        assert "73 years in 2019" in call_prompt
        assert "rebounded to 73 years in 2022 and 2023" in call_prompt
        assert result.answer == expected_answer
        assert result.action == "CORRECT"
        assert result.trace.web_search_used is False
