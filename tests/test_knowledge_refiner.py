"""
Offline unit tests for CRAG Knowledge Refinement.

All tests are fully offline using mocked RelevanceEvaluator.
No external API calls or model downloads occur during testing.

Coverage:
  - Text decomposition into strips (short vs. multi-sentence)
  - Provenance preservation (source, page_number, parent_chunk_id, position)
  - Strip scoring with RelevanceEvaluator
  - Hard filtering below filter_threshold
  - Top-K selection of highest-scoring strips
  - Natural source order restoration (recomposition)
  - Input validation and error handling (query, thresholds, top_k)
  - Empty input handling
  - Determinism across repeated executions
"""

import pytest
from unittest.mock import MagicMock

from app.retrieval.retriever import RetrievedChunk
from app.evaluation.relevance_evaluator import RelevanceEvaluator
from app.evaluation.knowledge_refiner import (
    KnowledgeStrip,
    KnowledgeRefiner,
    decompose_text_into_strips,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test Fixtures & Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_chunk(
    chunk_id: str = "chunk_001",
    text: str = "Sentence one. Sentence two.",
    source: str = "doc.pdf",
    page_number: int = 1,
    score: float = 0.8,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        page_number=page_number,
        score=score,
        metadata={},
    )


def make_mock_evaluator(scores: list[float] | None = None) -> RelevanceEvaluator:
    """Create a mock RelevanceEvaluator returning predetermined scores."""
    evaluator = MagicMock(spec=RelevanceEvaluator)
    if scores is not None:
        evaluator.score_batch.return_value = scores
    return evaluator


# ─────────────────────────────────────────────────────────────────────────────
# 1. Text Decomposition Unit Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDecomposeTextIntoStrips:
    """Test text decomposition into fine-grained strips."""

    def test_short_document_remains_one_strip(self):
        text = "This is a single short sentence."
        strips = decompose_text_into_strips(text, sentences_per_strip=2)
        assert len(strips) == 1
        assert strips[0] == text

    def test_two_sentences_remain_one_strip_when_target_is_two(self):
        text = "First sentence here. Second sentence follows."
        strips = decompose_text_into_strips(text, sentences_per_strip=2)
        assert len(strips) == 1
        assert strips[0] == text

    def test_long_document_decomposes_into_multiple_strips(self):
        text = (
            "Sentence one is here. Sentence two is next. "
            "Sentence three follows. Sentence four is fourth. "
            "Sentence five concludes."
        )
        strips = decompose_text_into_strips(text, sentences_per_strip=2)
        assert len(strips) == 3
        assert strips[0] == "Sentence one is here. Sentence two is next."
        assert strips[1] == "Sentence three follows. Sentence four is fourth."
        assert strips[2] == "Sentence five concludes."

    def test_empty_or_whitespace_text_returns_empty_list(self):
        assert decompose_text_into_strips("") == []
        assert decompose_text_into_strips("   \n\t  ") == []

    def test_handles_exclamation_and_question_marks(self):
        text = "Is this RAG? Yes it is! Let us refine it."
        strips = decompose_text_into_strips(text, sentences_per_strip=1)
        assert len(strips) == 3
        assert strips[0] == "Is this RAG?"
        assert strips[1] == "Yes it is!"
        assert strips[2] == "Let us refine it."


# ─────────────────────────────────────────────────────────────────────────────
# 2. KnowledgeRefiner Initialization & Parameter Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestKnowledgeRefinerInit:
    """Test configuration and parameter validation."""

    def test_valid_initialization_with_defaults(self):
        mock_eval = make_mock_evaluator()
        refiner = KnowledgeRefiner(evaluator=mock_eval)
        assert refiner.filter_threshold == -0.5
        assert refiner.top_k == 5
        assert refiner.sentences_per_strip == 2

    def test_valid_custom_parameters(self):
        mock_eval = make_mock_evaluator()
        refiner = KnowledgeRefiner(
            evaluator=mock_eval,
            filter_threshold=0.2,
            top_k=3,
            sentences_per_strip=1,
        )
        assert refiner.filter_threshold == 0.2
        assert refiner.top_k == 3
        assert refiner.sentences_per_strip == 1

    def test_none_evaluator_raises_type_error(self):
        with pytest.raises(TypeError, match="evaluator cannot be None"):
            KnowledgeRefiner(evaluator=None)  # type: ignore

    def test_invalid_filter_threshold_bounds_raises_value_error(self):
        mock_eval = make_mock_evaluator()
        with pytest.raises(ValueError, match="filter_threshold must be within"):
            KnowledgeRefiner(evaluator=mock_eval, filter_threshold=1.5)
        with pytest.raises(ValueError, match="filter_threshold must be within"):
            KnowledgeRefiner(evaluator=mock_eval, filter_threshold=-1.2)

    def test_non_numeric_filter_threshold_raises_type_error(self):
        mock_eval = make_mock_evaluator()
        with pytest.raises(TypeError, match="filter_threshold must be numeric"):
            KnowledgeRefiner(evaluator=mock_eval, filter_threshold="high")  # type: ignore

    def test_boolean_filter_threshold_raises_type_error(self):
        mock_eval = make_mock_evaluator()
        with pytest.raises(TypeError, match="filter_threshold must be numeric"):
            KnowledgeRefiner(evaluator=mock_eval, filter_threshold=False)  # type: ignore

    def test_invalid_top_k_raises_error(self):
        mock_eval = make_mock_evaluator()
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            KnowledgeRefiner(evaluator=mock_eval, top_k=0)
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            KnowledgeRefiner(evaluator=mock_eval, top_k=-3)
        with pytest.raises(TypeError, match="top_k must be an integer"):
            KnowledgeRefiner(evaluator=mock_eval, top_k=2.5)  # type: ignore

    def test_invalid_sentences_per_strip_raises_error(self):
        mock_eval = make_mock_evaluator()
        with pytest.raises(ValueError, match="sentences_per_strip must be a positive integer"):
            KnowledgeRefiner(evaluator=mock_eval, sentences_per_strip=0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. KnowledgeRefiner Refinement Workflow Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKnowledgeRefinerWorkflow:
    """Test the full decompose -> score -> filter -> rank -> recompose pipeline."""

    def test_refine_preserves_source_page_and_parent_metadata(self):
        chunk = make_chunk(
            chunk_id="chunk_p4_c002",
            text="CRAG evaluates retrieved documents. It refines knowledge strips.",
            source="paper_spec.pdf",
            page_number=4,
        )
        mock_eval = make_mock_evaluator(scores=[0.75])
        refiner = KnowledgeRefiner(evaluator=mock_eval, filter_threshold=0.0)

        refined_strips = refiner.refine("What is CRAG?", [chunk])

        assert len(refined_strips) == 1
        strip = refined_strips[0]
        assert strip.source == "paper_spec.pdf"
        assert strip.page_number == 4
        assert strip.parent_chunk_id == "chunk_p4_c002"
        assert strip.score == 0.75
        assert strip.position == 0

    def test_evaluator_is_called_with_all_decomposed_strips(self):
        chunk = make_chunk(
            text="Sentence 1. Sentence 2. Sentence 3. Sentence 4."
        )
        # With sentences_per_strip=1, 4 strips are generated
        mock_eval = make_mock_evaluator(scores=[0.8, 0.6, 0.4, 0.2])
        refiner = KnowledgeRefiner(
            evaluator=mock_eval,
            sentences_per_strip=1,
            filter_threshold=-0.5,
        )

        refiner.refine("query text", [chunk])

        mock_eval.score_batch.assert_called_once()
        pairs = mock_eval.score_batch.call_args[0][0]
        assert len(pairs) == 4
        assert pairs[0] == ("query text", "Sentence 1.")
        assert pairs[1] == ("query text", "Sentence 2.")
        assert pairs[2] == ("query text", "Sentence 3.")
        assert pairs[3] == ("query text", "Sentence 4.")

    def test_low_scoring_strips_are_filtered(self):
        chunk = make_chunk(
            text="Relevant sentence. Irrelevant sentence. Another relevant sentence."
        )
        # 3 strips: scores are 0.8, -0.6 (below threshold 0.0), 0.5
        mock_eval = make_mock_evaluator(scores=[0.8, -0.6, 0.5])
        refiner = KnowledgeRefiner(
            evaluator=mock_eval,
            sentences_per_strip=1,
            filter_threshold=0.0,
        )

        refined = refiner.refine("query", [chunk])

        assert len(refined) == 2
        texts = [s.text for s in refined]
        assert "Relevant sentence." in texts
        assert "Another relevant sentence." in texts
        assert "Irrelevant sentence." not in texts

    def test_strips_exactly_at_filter_threshold_are_retained(self):
        chunk = make_chunk(text="Strip one. Strip two.")
        # Filter threshold is 0.2. Scores: 0.2 (equal), 0.19 (below)
        mock_eval = make_mock_evaluator(scores=[0.2, 0.19])
        refiner = KnowledgeRefiner(
            evaluator=mock_eval,
            sentences_per_strip=1,
            filter_threshold=0.2,
        )

        refined = refiner.refine("query", [chunk])
        assert len(refined) == 1
        assert refined[0].text == "Strip one."
        assert refined[0].score == 0.2

    def test_all_strips_below_threshold_returns_empty_list(self):
        chunk = make_chunk(text="Irrelevant text.")
        mock_eval = make_mock_evaluator(scores=[-0.8])
        refiner = KnowledgeRefiner(evaluator=mock_eval, filter_threshold=-0.5)

        refined = refiner.refine("query", [chunk])
        assert refined == []

    def test_top_k_limits_number_of_retained_strips(self):
        chunk = make_chunk(
            text="Sentence A. Sentence B. Sentence C. Sentence D. Sentence E."
        )
        # 5 strips, all above threshold (0.0). Scores: [0.9, 0.3, 0.8, 0.4, 0.7]
        # Top 2 scores: 0.9 (Sentence A, pos 0) and 0.8 (Sentence C, pos 2)
        mock_eval = make_mock_evaluator(scores=[0.9, 0.3, 0.8, 0.4, 0.7])
        refiner = KnowledgeRefiner(
            evaluator=mock_eval,
            sentences_per_strip=1,
            filter_threshold=0.0,
            top_k=2,
        )

        refined = refiner.refine("query", [chunk])

        assert len(refined) == 2
        # Scores present must be the top 2 (0.9 and 0.8)
        assert {s.score for s in refined} == {0.9, 0.8}

    def test_surviving_strips_are_recomposed_in_original_source_order(self):
        """Verify that ranking selects top-K, but output order preserves source sequence."""
        chunk = make_chunk(
            text="First part. Second part. Third part. Fourth part."
        )
        # Positions: 0 (First), 1 (Second), 2 (Third), 3 (Fourth)
        # Scores:    0.4,          0.9,          0.2,         0.85
        # Top 2 by score are: pos 1 (0.9) and pos 3 (0.85)
        # After recomposition, order must be pos 1 then pos 3 (natural reading order)
        mock_eval = make_mock_evaluator(scores=[0.4, 0.9, 0.2, 0.85])
        refiner = KnowledgeRefiner(
            evaluator=mock_eval,
            sentences_per_strip=1,
            filter_threshold=0.0,
            top_k=2,
        )

        refined = refiner.refine("query", [chunk])

        assert len(refined) == 2
        assert refined[0].text == "Second part."
        assert refined[0].position == 1
        assert refined[0].score == 0.9

        assert refined[1].text == "Fourth part."
        assert refined[1].position == 3
        assert refined[1].score == 0.85

    def test_multi_document_decomposition_and_order_preservation(self):
        chunk1 = make_chunk(
            chunk_id="chunk_1",
            text="Doc1 sentence A. Doc1 sentence B.",
            source="doc1.pdf",
            page_number=1,
        )
        chunk2 = make_chunk(
            chunk_id="chunk_2",
            text="Doc2 sentence C. Doc2 sentence D.",
            source="doc2.pdf",
            page_number=2,
        )
        # 4 strips across 2 documents
        # Scores: Doc1 A (0.9), Doc1 B (-0.7 filtered), Doc2 C (0.85), Doc2 D (0.2 filtered)
        mock_eval = make_mock_evaluator(scores=[0.9, -0.7, 0.85, 0.2])
        refiner = KnowledgeRefiner(
            evaluator=mock_eval,
            sentences_per_strip=1,
            filter_threshold=0.5,
            top_k=5,
        )

        refined = refiner.refine("query", [chunk1, chunk2])

        assert len(refined) == 2
        assert refined[0].source == "doc1.pdf"
        assert refined[0].parent_chunk_id == "chunk_1"
        assert refined[0].text == "Doc1 sentence A."
        assert refined[0].position == 0

        assert refined[1].source == "doc2.pdf"
        assert refined[1].parent_chunk_id == "chunk_2"
        assert refined[1].text == "Doc2 sentence C."
        assert refined[1].position == 2

    def test_empty_document_list_returns_empty_list(self):
        mock_eval = make_mock_evaluator()
        refiner = KnowledgeRefiner(evaluator=mock_eval)
        assert refiner.refine("query", []) == []

    def test_empty_query_raises_value_error(self):
        mock_eval = make_mock_evaluator()
        refiner = KnowledgeRefiner(evaluator=mock_eval)
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            refiner.refine("", [make_chunk()])
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            refiner.refine("   ", [make_chunk()])

    def test_non_string_query_raises_type_error(self):
        mock_eval = make_mock_evaluator()
        refiner = KnowledgeRefiner(evaluator=mock_eval)
        with pytest.raises(TypeError, match="query must be a string"):
            refiner.refine(123, [make_chunk()])  # type: ignore

    def test_non_list_documents_raises_type_error(self):
        mock_eval = make_mock_evaluator()
        refiner = KnowledgeRefiner(evaluator=mock_eval)
        with pytest.raises(TypeError, match="documents must be a list"):
            refiner.refine("query", "not a list")  # type: ignore

    def test_refinement_is_deterministic(self):
        chunk = make_chunk(text="Sentence 1. Sentence 2. Sentence 3.")
        scores = [0.8, 0.4, 0.7]
        mock_eval = make_mock_evaluator(scores=scores)
        refiner = KnowledgeRefiner(
            evaluator=mock_eval, sentences_per_strip=1, filter_threshold=0.0
        )

        res1 = refiner.refine("query", [chunk])
        mock_eval.score_batch.return_value = scores
        res2 = refiner.refine("query", [chunk])

        assert len(res1) == len(res2)
        for s1, s2 in zip(res1, res2):
            assert s1.text == s2.text
            assert s1.score == s2.score
            assert s1.position == s2.position
