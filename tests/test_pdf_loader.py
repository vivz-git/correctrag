"""
Unit and regression tests for the PDF ingestion module.
"""

from pathlib import Path
import pytest
import pymupdf

from app.ingestion.pdf_loader import (
    DocumentChunk,
    EmptyPDFError,
    InvalidPDFError,
    PDFNotFoundError,
    clean_text,
    chunk_text,
    load_pdf,
)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a temporary multi-page PDF with known sample text."""
    pdf_path = tmp_path / "sample_document.pdf"
    doc = pymupdf.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_text(
        (50, 72),
        "Corrective Retrieval Augmented Generation improves RAG robustness.\n"
        "It uses a retrieval evaluator to score retrieved documents.\n"
        "When retrieval is incorrect, external web searches are used.",
    )

    # Page 2
    page2 = doc.new_page()
    page2.insert_text(
        (50, 72),
        "The decompose-then-recompose algorithm extracts key information strips.\n"
        "Irrelevant strips are filtered out using calibrated relevance scores.",
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    """Create a temporary PDF with pages but no text."""
    pdf_path = tmp_path / "empty_document.pdf"
    doc = pymupdf.open()
    doc.new_page()  # Blank page
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def corrupt_pdf(tmp_path: Path) -> Path:
    """Create a temporary corrupted non-PDF file."""
    pdf_path = tmp_path / "corrupt_file.pdf"
    pdf_path.write_bytes(b"This is not a valid PDF header or binary data.")
    return pdf_path


# ============================================================================
# Text Cleaning Tests
# ============================================================================


def test_clean_text_normalizes_line_endings_and_whitespace():
    raw = "Line 1\r\nLine 2\rLine 3   with   extra   spaces\t\ttabs."
    cleaned = clean_text(raw)
    assert "Line 1\nLine 2\nLine 3 with extra spaces tabs." == cleaned


def test_clean_text_removes_zero_width_and_control_chars():
    raw = "Text\u200b with\ufeff zero\u200c width\x00 chars."
    cleaned = clean_text(raw)
    assert "Text with zero width chars." == cleaned


def test_clean_text_recombines_hyphenated_linebreaks():
    raw = "Corrective re-\ntrieval augmented gen-\neration framework."
    cleaned = clean_text(raw)
    assert "Corrective retrieval augmented generation framework." == cleaned


def test_clean_text_handles_empty_or_whitespace():
    assert clean_text("") == ""
    assert clean_text("   \n\n\t  ") == ""


# ============================================================================
# Chunking & Length Invariant Tests
# ============================================================================


def test_chunk_text_deterministic():
    text = (
        "Corrective Retrieval Augmented Generation (CRAG) is a framework designed to "
        "mitigate hallucinations in language models. It evaluates retrieved documents "
        "and triggers corrective actions accordingly."
    )
    chunks_1 = chunk_text(text, chunk_size=80, chunk_overlap=20)
    chunks_2 = chunk_text(text, chunk_size=80, chunk_overlap=20)
    assert chunks_1 == chunks_2
    assert len(chunks_1) > 1
    for chunk in chunks_1:
        assert len(chunk) <= 80


def test_chunk_text_strictly_enforces_max_chunk_size_regression():
    """Regression test: boundary-snapping must NEVER produce a chunk exceeding chunk_size.

    Tests edge cases including:
    - Long unbroken tokens with no whitespace or punctuation
    - Punctuation positioned immediately at/around boundary cutoffs
    - Multiple paragraph breaks and whitespace sequences
    """
    chunk_size = 50
    chunk_overlap = 15

    # Case A: Monolithic continuous string without whitespace or punctuation
    monolithic_text = "A" * 250
    chunks_a = chunk_text(monolithic_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(chunks_a) > 1
    for c in chunks_a:
        assert (
            len(c) <= chunk_size
        ), f"Chunk exceeded max size {chunk_size}: len={len(c)}, content={c}"

    # Case B: Sentence punctuation specifically placed at end-of-window boundaries
    tricky_punctuation_text = (
        "Word word word. Word word word! Word word? Word word word word word. "
        "Another sentence with punctuation right around boundary marks. End."
    )
    chunks_b = chunk_text(
        tricky_punctuation_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    assert len(chunks_b) > 1
    for c in chunks_b:
        assert (
            len(c) <= chunk_size
        ), f"Chunk exceeded max size {chunk_size}: len={len(c)}, content={c}"

    # Case C: Punctuation clusters and trailing whitespace
    cluster_text = "Cluster one... Next sentence??? Final part!! With extra spaces     "
    chunks_c = chunk_text(cluster_text, chunk_size=30, chunk_overlap=10)
    for c in chunks_c:
        assert len(c) <= 30, f"Chunk exceeded max size 30: len={len(c)}, content={c}"


def test_chunk_text_sweep_guarantees_length_bound():
    """Parametric sweep over multiple chunk_sizes and overlaps to assert the <= chunk_size invariant."""
    sample_text = (
        "Large language models (LLMs) inevitably exhibit hallucinations since the accuracy "
        "of generated texts cannot be secured solely by the parametric knowledge they encapsulate. "
        "Although retrieval-augmented generation (RAG) is a practicable complement to LLMs, it relies "
        "heavily on the relevance of retrieved documents, raising concerns about model behavior if "
        "retrieval goes wrong. To this end, Corrective Retrieval Augmented Generation (CRAG) is proposed.\n\n"
        "Specifically, a lightweight retrieval evaluator is designed to assess the overall quality of "
        "retrieved documents for a query, returning a confidence degree based on which different knowledge "
        "retrieval actions can be triggered: Correct, Incorrect, or Ambiguous."
    )

    test_configs = [
        (30, 5),
        (50, 10),
        (80, 20),
        (100, 30),
        (150, 40),
        (250, 50),
    ]

    for size, overlap in test_configs:
        chunks = chunk_text(sample_text, chunk_size=size, chunk_overlap=overlap)
        assert len(chunks) > 0
        for idx, chunk in enumerate(chunks):
            assert (
                len(chunk) <= size
            ), f"Config (size={size}, overlap={overlap}) failed on chunk {idx}: len={len(chunk)}"


def test_chunk_text_invalid_parameters():
    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        chunk_text("Sample text", chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        chunk_text("Sample text", chunk_size=100, chunk_overlap=-10)

    with pytest.raises(ValueError, match="must be strictly less than chunk_size"):
        chunk_text("Sample text", chunk_size=100, chunk_overlap=100)


def test_chunk_text_empty():
    assert chunk_text("", chunk_size=100, chunk_overlap=20) == []


# ============================================================================
# PDF Loading & Metadata Tests
# ============================================================================


def test_load_pdf_success(sample_pdf: Path):
    chunks = load_pdf(sample_pdf, chunk_size=120, chunk_overlap=30)

    assert len(chunks) > 0
    assert all(isinstance(c, DocumentChunk) for c in chunks)

    # Check page numbers
    page_numbers = {c.page_number for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers

    # Check source
    assert all(c.source == "sample_document.pdf" for c in chunks)

    # Check deterministic chunk ID formatting
    assert chunks[0].chunk_id == "sample_document_p1_c001"
    assert chunks[0].metadata["chunk_index"] == 1
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[0].metadata["source_file"] == "sample_document.pdf"

    # Check that each chunk adheres strictly to max chunk_size
    for c in chunks:
        assert len(c.text) <= 120


def test_load_pdf_preserves_page_structure(sample_pdf: Path):
    chunks = load_pdf(sample_pdf, chunk_size=500, chunk_overlap=50)

    # With large chunk size, we should have 1 chunk for page 1 and 1 chunk for page 2
    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert "Corrective Retrieval Augmented Generation" in chunks[0].text
    assert chunks[1].page_number == 2
    assert "decompose-then-recompose" in chunks[1].text


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_load_pdf_missing_file(tmp_path: Path):
    missing_path = tmp_path / "non_existent_file.pdf"
    with pytest.raises(PDFNotFoundError) as exc_info:
        load_pdf(missing_path)
    assert "PDF file not found" in str(exc_info.value)


def test_load_pdf_corrupted_file(corrupt_pdf: Path):
    with pytest.raises(InvalidPDFError) as exc_info:
        load_pdf(corrupt_pdf)
    assert "Failed to parse PDF" in str(exc_info.value)


def test_load_pdf_empty_file(empty_pdf: Path):
    with pytest.raises(EmptyPDFError) as exc_info:
        load_pdf(empty_pdf)
    assert "contains no extractable text" in str(exc_info.value)
