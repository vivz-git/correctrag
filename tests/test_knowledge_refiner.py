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

    def test_table_like_multi_line_block_preserved_as_one_strip(self):
        """Q1 structured-content fallback: >= 6 non-empty lines with >= 60%
        of lines containing <= 3 words must be kept as a single strip,
        not sentence-split."""
        text = (
            "Type\nInput\nOutput\nDefinitions\n"
            "Retrieve\nx / x, y\n{yes, no, continue}\n"
            "Decides when to retrieve with R.\n"
            "Table 1: Four types of tokens used. Each type uses several tokens."
        )
        strips = decompose_text_into_strips(text, sentences_per_strip=2)
        assert len(strips) == 1
        assert strips[0] == text

    def test_real_self_rag_table_content_stays_together(self):
        """Real Self-RAG_p4_c001-style table: row data (Retrieve/ISREL/ISSUP/
        ISUSE) and the 'Table 1:' caption must land in the same strip so the
        caption cannot displace the row definitions during top-k selection."""
        text = (
            "Preprint.\nType\nInput\nOutput\nDefinitions\n"
            "Retrieve\nx / x, y\n{yes, no, continue}\nDecides when to retrieve with R\n"
            "ISREL\nx, d\n{relevant, irrelevant}\nd provides useful information to solve x.\n"
            "ISSUP\nx, d, y\n{fully supported, partially\nsupported, no support}\n"
            "All of the verification-worthy statement in y\nis supported by d.\n"
            "ISUSE\nx, y\n{5, 4, 3, 2, 1}\ny is a useful response to x.\n"
            "Table 1: Four types of reflection tokens used in SELF-RAG. Each type uses several tokens to represent\n"
            "its output values."
        )
        strips = decompose_text_into_strips(text, sentences_per_strip=2)
        assert len(strips) == 1
        strip = strips[0]
        assert "Retrieve" in strip
        assert "ISREL" in strip
        assert "ISSUP" in strip
        assert "ISUSE" in strip
        assert "Table 1:" in strip

    def test_structured_prefix_with_trailing_narrative_run_is_split(self):
        """Candidate D: a structured block (>= 6 short lines) followed by a
        trailing run of >= 4 consecutive narrative lines (each > 3 words)
        must split into a preserved structured prefix strip plus a
        sentence-decomposed narrative suffix -- not stay one giant strip."""
        text = (
            "Type\nInput\nOutput\nDefinitions\n"
            "Retrieve\nx, y\n{yes, no, continue}\n"
            "Figure 2: An overview of the proposed CRAG at inference.\n"
            "A retrieval evaluator is constructed to evaluate relevance.\n"
            "It estimates a confidence degree for each retrieved document.\n"
            "Different knowledge actions are triggered based on that degree."
        )
        strips = decompose_text_into_strips(text, sentences_per_strip=2)

        assert len(strips) >= 2
        prefix = strips[0]
        assert "Type" in prefix
        assert "Retrieve" in prefix
        assert "Figure 2" not in prefix

        narrative = " ".join(strips[1:])
        assert "Figure 2" in narrative
        assert "confidence degree" in narrative

    def test_structured_block_with_no_trailing_narrative_run_stays_one_strip(self):
        """When no qualifying trailing narrative run exists, the whole
        structured block is preserved as a single strip (unchanged from the
        pre-Candidate-D behavior)."""
        text = (
            "Type\nInput\nOutput\nDefinitions\n"
            "Retrieve\nx / x, y\n{yes, no, continue}\n"
            "Decides when to retrieve with R.\n"
            "Table 1: Four types of tokens used. Each type uses several tokens."
        )
        strips = decompose_text_into_strips(text, sentences_per_strip=2)
        assert len(strips) == 1
        assert strips[0] == text

    def test_normal_prose_still_decomposes_as_before(self):
        """A normal prose chunk (few short lines) must not trigger the
        structured-content fallback -- it decomposes exactly as before."""
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

    def test_coverage_floor_prevents_one_chunk_from_starving_another(self):
        """A chunk with many high-scoring strips must not crowd out every
        strip from another retrieved chunk that also survived filtering."""
        chunk_a = make_chunk(
            chunk_id="chunk_a",
            text="A one. A two. A three. A four.",
        )
        chunk_b = make_chunk(
            chunk_id="chunk_b",
            text="B one.",
        )
        # Chunk A strips: 0.95, 0.9, 0.85, 0.8 (all high)
        # Chunk B strip:  0.3  (lower, but still above threshold 0.0)
        mock_eval = make_mock_evaluator(scores=[0.95, 0.9, 0.85, 0.8, 0.3])
        refiner = KnowledgeRefiner(
            evaluator=mock_eval,
            sentences_per_strip=1,
            filter_threshold=0.0,
            top_k=2,
        )

        refined = refiner.refine("query", [chunk_a, chunk_b])

        parent_ids = {s.parent_chunk_id for s in refined}
        assert "chunk_b" in parent_ids, "chunk_b must receive a representative slot"
        assert len(refined) == 2

    def test_coverage_floor_then_global_fill(self):
        """Representatives are picked first (one per parent chunk); any
        remaining top_k slots are filled by highest global score."""
        chunk_a = make_chunk(chunk_id="chunk_a", text="A one. A two. A three.")
        chunk_b = make_chunk(chunk_id="chunk_b", text="B one. B two.")
        # Chunk A: 0.9, 0.7, 0.6   Chunk B: 0.8, 0.5
        mock_eval = make_mock_evaluator(scores=[0.9, 0.7, 0.6, 0.8, 0.5])
        refiner = KnowledgeRefiner(
            evaluator=mock_eval,
            sentences_per_strip=1,
            filter_threshold=0.0,
            top_k=3,
        )

        refined = refiner.refine("query", [chunk_a, chunk_b])

        scores = {s.score for s in refined}
        # Representatives: A's best (0.9), B's best (0.8) -> 2 slots used.
        # 1 remaining slot filled by next highest global score: A two (0.7).
        assert scores == {0.9, 0.8, 0.7}
        assert len(refined) == 3

    def test_top_k_still_strictly_respected_with_more_chunks_than_top_k(self):
        """When there are more surviving parent chunks than top_k,
        only top_k representatives (by score) are kept."""
        chunks = [
            make_chunk(chunk_id=f"chunk_{i}", text=f"Sentence {i}.")
            for i in range(5)
        ]
        # One strip per chunk; scores strictly descending by chunk index reversed
        scores = [0.1, 0.9, 0.5, 0.7, 0.3]
        mock_eval = make_mock_evaluator(scores=scores)
        refiner = KnowledgeRefiner(
            evaluator=mock_eval,
            sentences_per_strip=1,
            filter_threshold=0.0,
            top_k=2,
        )

        refined = refiner.refine("query", chunks)

        assert len(refined) == 2
        assert {s.score for s in refined} == {0.9, 0.7}

    def test_relevant_strip_survives_sentence_pair_boundary_crowding(self):
        """Regression for the parity/crowding bug class: a relevant strip
        must survive top_k selection regardless of which sentence-pair
        boundary it falls on, as long as its parent chunk is represented."""
        chunk_a = make_chunk(
            chunk_id="chunk_a",
            text="Alpha filler one. Alpha filler two. Alpha filler three. Alpha filler four.",
        )
        chunk_b = make_chunk(
            chunk_id="chunk_b",
            text="Beta filler one. Beta highly relevant answer.",
        )
        # Chunk A dominates on raw score for every strip.
        # Chunk B's second strip (the actually relevant answer) is its best.
        mock_eval = make_mock_evaluator(scores=[0.99, 0.98, 0.97, 0.96, 0.4, 0.6])
        refiner = KnowledgeRefiner(
            evaluator=mock_eval,
            sentences_per_strip=1,
            filter_threshold=0.0,
            top_k=3,
        )

        refined = refiner.refine("query", [chunk_a, chunk_b])

        texts = [s.text for s in refined]
        assert "Beta highly relevant answer." in texts

    # ─────────────────────────────────────────────────────────────────────
    # Document-level coverage floor (Q4-style source starvation)
    # ─────────────────────────────────────────────────────────────────────

    def test_document_floor_recovers_fully_starved_source(self):
        """Q4-style bug: many parent chunks from one source can outscore every
        chunk from another surviving source, dropping that source entirely
        once representatives are cut to top_k. The document floor must admit
        a representative for the starved source without changing top_k size."""
        dominant_chunks = [
            make_chunk(chunk_id=f"dom_{i}", source="Dominant.pdf", text=f"Dominant sentence {i}.")
            for i in range(6)
        ]
        starved_chunks = [
            make_chunk(chunk_id=f"starved_{i}", source="Starved.pdf", text=f"Starved sentence {i}.")
            for i in range(2)
        ]
        # Dominant.pdf's 6 chunks all outscore Starved.pdf's 2 chunks.
        scores = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6]
        mock_eval = make_mock_evaluator(scores=scores)
        refiner = KnowledgeRefiner(
            evaluator=mock_eval,
            sentences_per_strip=1,
            filter_threshold=0.0,
            top_k=5,
        )

        refined = refiner.refine("query", dominant_chunks + starved_chunks)

        assert len(refined) == 5
        sources = {s.source for s in refined}
        assert "Starved.pdf" in sources, "Starved.pdf must gain a representative slot"
        assert "Dominant.pdf" in sources

    def test_document_floor_is_noop_when_sources_already_diverse(self):
        """When every surviving source already has a selected slot, the
        document floor must not alter the selection."""
        chunk_a = make_chunk(chunk_id="chunk_a", source="A.pdf", text="A one. A two.")
        chunk_b = make_chunk(chunk_id="chunk_b", source="B.pdf", text="B one. B two.")
        # Representatives: A's best (0.9), B's best (0.7) -> both sources already
        # covered before the document floor runs, so it must be a strict no-op.
        mock_eval = make_mock_evaluator(scores=[0.9, 0.8, 0.7, 0.6])
        refiner = KnowledgeRefiner(
            evaluator=mock_eval,
            sentences_per_strip=1,
            filter_threshold=0.0,
            top_k=2,
        )

        refined = refiner.refine("query", [chunk_a, chunk_b])

        assert len(refined) == 2
        assert {s.source for s in refined} == {"A.pdf", "B.pdf"}
        assert {s.score for s in refined} == {0.9, 0.7}

    def test_document_floor_is_noop_for_single_source_input(self):
        """With only one surviving source, there is nothing to starve."""
        chunk = make_chunk(
            chunk_id="chunk_1",
            source="Solo.pdf",
            text="Sentence A. Sentence B. Sentence C.",
        )
        mock_eval = make_mock_evaluator(scores=[0.9, 0.5, 0.3])
        refiner = KnowledgeRefiner(
            evaluator=mock_eval,
            sentences_per_strip=1,
            filter_threshold=0.0,
            top_k=2,
        )

        refined = refiner.refine("query", [chunk])

        assert len(refined) == 2
        assert {s.source for s in refined} == {"Solo.pdf"}
        assert {s.score for s in refined} == {0.9, 0.5}

    def test_document_floor_cannot_fully_starve_another_source(self):
        """Displacement may only draw from a source holding >= 2 selected
        representatives, so an already-1-slot source is never zeroed out."""
        chunk_a = make_chunk(chunk_id="chunk_a", source="A.pdf", text="A one.")
        chunk_b = make_chunk(chunk_id="chunk_b", source="B.pdf", text="B one.")
        chunk_c = make_chunk(chunk_id="chunk_c", source="C.pdf", text="C one.")
        # top_k=2 forces a cutoff: representatives sorted desc are A(0.9), B(0.8), C(0.2).
        # A and B each hold exactly 1 selected slot -> neither is displaceable,
        # so C stays starved (best effort, not guaranteed).
        mock_eval = make_mock_evaluator(scores=[0.9, 0.8, 0.2])
        refiner = KnowledgeRefiner(
            evaluator=mock_eval,
            sentences_per_strip=1,
            filter_threshold=0.0,
            top_k=2,
        )

        refined = refiner.refine("query", [chunk_a, chunk_b, chunk_c])

        assert len(refined) == 2
        sources = {s.source for s in refined}
        assert sources == {"A.pdf", "B.pdf"}
        # Neither A.pdf nor B.pdf was reduced to zero representatives.
        assert "A.pdf" in sources and "B.pdf" in sources

    def test_document_floor_is_noop_in_below_top_k_branch(self):
        """When len(representatives) < top_k, every surviving parent chunk
        (and therefore every surviving source) is already included before
        the global-fill step runs, so the document floor has nothing to do.
        This confirms the floor applies safely -- as a no-op -- in that
        branch too, per requirement to cover both branches."""
        chunk_a = make_chunk(chunk_id="chunk_a", source="A.pdf", text="A one. A two. A three.")
        chunk_b = make_chunk(chunk_id="chunk_b", source="B.pdf", text="B one.")
        # Representatives: A's best (0.9), B's best (0.1) -> both sources already
        # covered. top_k=3 > 2 representatives, so 1 remaining slot is filled by
        # global score (A's second-best, 0.7). No source is ever fully absent
        # in this branch, so the floor cannot and does not alter the selection.
        mock_eval = make_mock_evaluator(scores=[0.9, 0.7, 0.05, 0.1])
        refiner = KnowledgeRefiner(
            evaluator=mock_eval,
            sentences_per_strip=1,
            filter_threshold=0.0,
            top_k=3,
        )

        refined = refiner.refine("query", [chunk_a, chunk_b])

        assert len(refined) == 3
        assert {s.score for s in refined} == {0.9, 0.1, 0.7}
        sources = {s.source for s in refined}
        assert "A.pdf" in sources and "B.pdf" in sources

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
