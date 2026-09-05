"""
Diagnostic and fix-verification tests for structured-content (table/caption)
behavior in CorrectRAG, against the real evaluation corpus.

  1. Chunk-level: does the CRAG-frozen chunking (chunk_size=500,
     chunk_overlap=100) place a table's data rows and its caption ("Table 1:
     ...") in the SAME parent chunk, or does it split them across chunks?
  2. Strip-level: given that parent chunk's real text, does
     `decompose_text_into_strips` (sentences_per_strip=2) keep the table rows
     and their caption in the same strip?
  3. Refinement-level: with the structured-content fallback in place, can the
     caption's own strip still displace the row-data strip during top-k
     selection? (It cannot -- they are now one strip.)

Gated on the local, gitignored `data/evaluation_corpus/Self-RAG.pdf` file --
it is not committed to the repository, so these tests skip cleanly wherever
that corpus is not present.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.evaluation.knowledge_refiner import (
    KnowledgeRefiner,
    decompose_text_into_strips,
)
from app.evaluation.relevance_evaluator import RelevanceEvaluator
from app.ingestion.pdf_loader import load_pdf
from app.retrieval.retriever import RetrievedChunk

REPO_ROOT = Path(__file__).resolve().parent.parent
SELF_RAG_PDF = REPO_ROOT / "data" / "evaluation_corpus" / "Self-RAG.pdf"
CRAG_PDF = REPO_ROOT / "data" / "evaluation_corpus" / "CRAG.pdf"

requires_self_rag_corpus = pytest.mark.skipif(
    not SELF_RAG_PDF.is_file(),
    reason="Local isolated real-corpus file data/evaluation_corpus/Self-RAG.pdf not present.",
)

requires_crag_corpus = pytest.mark.skipif(
    not CRAG_PDF.is_file(),
    reason="Local isolated real-corpus file data/evaluation_corpus/CRAG.pdf not present.",
)


@pytest.fixture(scope="module")
def self_rag_table_chunk():
    """The real parent chunk containing Self-RAG's reflection-token table (Table 1)."""
    chunks = load_pdf(SELF_RAG_PDF, chunk_size=500, chunk_overlap=100)
    # Anchor on the literal Table 1 caption text (not just "Table 1", which is
    # also a substring of "Table 10".."Table 19" elsewhere in the paper).
    table_chunks = [
        c
        for c in chunks
        if "ISREL" in c.text and "Table 1: Four types of reflection tokens" in c.text
    ]
    assert len(table_chunks) == 1, (
        "Expected exactly one real parent chunk to contain both the ISREL table "
        "row and the 'Table 1: Four types of reflection tokens' caption under "
        "frozen chunk_size=500/overlap=100."
    )
    return table_chunks[0]


@requires_self_rag_corpus
def test_q1_table_rows_and_caption_share_the_same_parent_chunk(self_rag_table_chunk):
    """Diagnostic: table-row evidence (e.g. ISREL) and its 'Table 1:' caption
    belong to the SAME parent_chunk_id on the real corpus.

    This rules out cross-parent-chunk retrieval/coverage as the cause of any
    Q1 table-vs-caption evidence problem -- the document-level coverage floor
    (which operates across parent chunks/sources) cannot be the fix, because
    there is nothing to redistribute between chunks here.
    """
    assert "ISREL" in self_rag_table_chunk.text
    assert "Table 1:" in self_rag_table_chunk.text
    assert self_rag_table_chunk.source == "Self-RAG.pdf"


@requires_self_rag_corpus
def test_q1_structured_fallback_keeps_caption_with_table_rows(self_rag_table_chunk):
    """Fix verification: with the structured-content fallback in place,
    `decompose_text_into_strips` keeps the 'Table 1:' caption in the SAME
    strip as the ISREL/ISSUP/ISUSE row data on the real Self-RAG parent
    chunk, instead of isolating the caption into its own strip.
    """
    strips = decompose_text_into_strips(
        self_rag_table_chunk.text, sentences_per_strip=2
    )

    assert len(strips) == 1, (
        "The structured-layout heuristic must treat this real table chunk "
        "as one bounded strip rather than sentence-splitting it."
    )
    strip = strips[0]
    assert "Retrieve" in strip
    assert "ISREL" in strip
    assert "ISSUP" in strip
    assert "ISUSE" in strip
    assert "Table 1:" in strip


@requires_self_rag_corpus
def test_q1_caption_cannot_displace_table_rows_during_refinement(self_rag_table_chunk):
    """Fix verification: because the caption and row data are now one strip,
    a low-relevance-scoring caption strip can no longer independently
    displace the row-data strip during top-k selection -- there is only one
    strip for the evaluator to score, and if it survives, all four token
    types (Retrieve/ISREL/ISSUP/ISUSE) survive with it.
    """
    mock_evaluator = MagicMock(spec=RelevanceEvaluator)
    mock_evaluator.score_batch.return_value = [0.8]

    refiner = KnowledgeRefiner(
        evaluator=mock_evaluator, filter_threshold=0.0, top_k=1
    )
    chunk = RetrievedChunk(
        chunk_id=self_rag_table_chunk.chunk_id,
        text=self_rag_table_chunk.text,
        source=self_rag_table_chunk.source,
        page_number=self_rag_table_chunk.page_number,
        score=0.8,
    )

    refined = refiner.refine(
        "What are the four types of reflection tokens in Self-RAG?", [chunk]
    )

    assert len(refined) == 1
    strip_text = refined[0].text
    for token in ("Retrieve", "ISREL", "ISSUP", "ISUSE"):
        assert token in strip_text


# ─────────────────────────────────────────────────────────────────────────────
# Candidate D: CRAG_p4_c002 -- structured diagram noise + narrative suffix
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def crag_p4_c002_chunk():
    """The real CRAG_p4_c002 parent chunk: Figure 2's diagram labels
    (Correction/Generation/Correct/Ambiguous/Incorrect/etc.) followed by the
    figure caption and narrative body text ("Our objective is to correct...")
    in the same 500-char/100-overlap chunk."""
    chunks = load_pdf(CRAG_PDF, chunk_size=500, chunk_overlap=100)
    matches = [c for c in chunks if c.chunk_id == "CRAG_p4_c002"]
    assert len(matches) == 1, (
        "Expected exactly one real parent chunk with id CRAG_p4_c002 under "
        "frozen chunk_size=500/overlap=100."
    )
    return matches[0]


@requires_crag_corpus
def test_crag_p4_c002_has_structured_diagram_noise_and_narrative_body(crag_p4_c002_chunk):
    """Diagnostic: this real chunk genuinely mixes single-word diagram/figure
    labels with multi-sentence narrative body text in one parent chunk."""
    text = crag_p4_c002_chunk.text
    assert "Figure 2" in text
    assert "Our objective is to correct the retrieved documents" in text


@requires_crag_corpus
def test_crag_p4_c002_narrative_split_from_structured_noise(crag_p4_c002_chunk):
    """Fix verification (Candidate D): the structured diagram-label prefix is
    preserved as one strip, separate from the sentence-decomposed narrative
    suffix (the figure caption and evaluator/action discussion), instead of
    the whole chunk collapsing into a single undifferentiated strip."""
    strips = decompose_text_into_strips(crag_p4_c002_chunk.text, sentences_per_strip=2)

    assert len(strips) >= 2, (
        "A qualifying trailing narrative run must split the structured "
        "diagram-label prefix from the narrative suffix."
    )

    narrative_text = " ".join(strips[1:])
    assert "Figure 2" in narrative_text
    assert "retrieval evaluator" in narrative_text
