"""
PDF Ingestion Module for CorrectRAG.

Handles PDF text extraction via PyMuPDF, artifact cleaning,
deterministic chunking with configurable size/overlap, and structured metadata generation.
"""

from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, Field

def _get_pymupdf():
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        import fitz
        return fitz


MAX_PDF_COUNT = 5
MAX_PDF_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


class DocumentChunk(BaseModel):
    """Structured representation of a document chunk."""

    chunk_id: str = Field(..., description="Deterministic unique identifier for the chunk")
    text: str = Field(..., description="Cleaned chunk text content")
    source: str = Field(..., description="Source filename or identifier")
    page_number: int = Field(..., description="1-indexed source page number")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary chunk metadata (e.g. index, length, character offsets)",
    )


class PDFIngestionError(Exception):
    """Base exception for PDF ingestion errors."""

    pass


class PDFNotFoundError(PDFIngestionError, FileNotFoundError):
    """Raised when the specified PDF file does not exist."""

    pass


class InvalidPDFError(PDFIngestionError, ValueError):
    """Raised when the file is not a valid or readable PDF."""

    pass


class EmptyPDFError(PDFIngestionError, ValueError):
    """Raised when the PDF has no pages or contains no extractable text."""

    pass


class PDFLimitExceededError(PDFIngestionError, ValueError):
    """Raised when the number of PDFs exceeds the allowed maximum of 5."""

    pass


class PDFSizeLimitExceededError(PDFIngestionError, ValueError):
    """Raised when a PDF file exceeds the allowed size limit of 25 MB."""

    pass


def clean_text(raw_text: str) -> str:
    """Clean common extraction artifacts from PDF text.

    - Normalizes line breaks to LF.
    - Strips non-printable and zero-width characters.
    - Recombines hyphenated line breaks (e.g., 're-\\ntrieval' -> 'retrieval').
    - Normalizes excessive whitespace while preserving paragraph structure.
    """
    if not raw_text:
        return ""

    # Normalize line endings
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove zero-width characters and null bytes
    text = re.sub(r"[\x00\u200b\u200c\u200d\ufeff]", "", text)

    # Recombine hyphenated words split across line breaks
    text = re.sub(r"(\b\w+)-\n(\w+\b)", r"\1\2", text)

    # Normalize horizontal whitespace (tabs, consecutive spaces) within lines
    text = re.sub(r"[ \t]+", " ", text)

    # Clean leading/trailing spaces on individual lines
    lines = [line.strip() for line in text.split("\n")]
    cleaned_text = "\n".join(lines)

    # Collapse more than two consecutive newlines into a standard paragraph break
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """Split text into deterministic chunks with configurable size and overlap.

    Guarantees that len(chunk) <= chunk_size for every returned chunk.
    Attempts to snap chunk boundaries to paragraph breaks, sentence endings,
    or word boundaries within the search window [max(start, end - chunk_overlap), end].

    Args:
        text: Input text string to be chunked.
        chunk_size: Maximum character length allowed for any chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of non-empty string chunks, each satisfying len(chunk) <= chunk_size.
    """
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})."
        )

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end >= text_len:
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Search window is strictly bounded within [start, end] where end == start + chunk_size
        search_window_start = max(start, end - chunk_overlap)
        candidate_segment = text[search_window_start:end]

        # Priority 1: Paragraph break (\n\n)
        para_idx = candidate_segment.rfind("\n\n")
        if para_idx != -1:
            split_point = search_window_start + para_idx + 2
        else:
            # Priority 2: Sentence boundary (. ! ? followed by whitespace or end of candidate)
            sentence_matches = list(re.finditer(r"[.!?](?:\s+|$)", candidate_segment))
            if sentence_matches:
                split_point = search_window_start + sentence_matches[-1].end()
            else:
                # Priority 3: Word boundary (whitespace)
                space_matches = list(re.finditer(r"\s+", candidate_segment))
                if space_matches:
                    split_point = search_window_start + space_matches[-1].end()
                else:
                    # Priority 4: Hard cutoff at exact boundary
                    split_point = end

        # Fallback to prevent non-advancing zero-length slices
        if split_point <= start:
            split_point = end

        # Ensure split_point never exceeds start + chunk_size
        split_point = min(split_point, end)

        chunk = text[start:split_point].strip()
        if chunk:
            chunks.append(chunk)

        # Advance start position ensuring strict forward progress
        start = max(split_point - chunk_overlap, start + 1)

    return chunks


def sanitize_doc_slug(filename: str) -> str:
    """Generate a clean, deterministic document slug for unique chunk ID prefixing."""
    stem = Path(filename).stem
    cleaned = re.sub(r"[^\w\-]", "_", stem).strip("_")
    return cleaned or "doc"


def load_pdf(
    file_path: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    doc_id_override: str | None = None,
) -> list[DocumentChunk]:
    """Load a PDF document, extract text per page, clean artifacts, and chunk deterministically.

    Args:
        file_path: Local filesystem path to the PDF file.
        chunk_size: Target maximum character length for each chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
        doc_id_override: Optional explicit document identifier for chunk ID prefixing.

    Returns:
        A list of DocumentChunk instances with deterministic IDs and page metadata.

    Raises:
        PDFNotFoundError: If the file does not exist.
        PDFSizeLimitExceededError: If the file exceeds 25 MB.
        InvalidPDFError: If the file is not a valid PDF or is corrupted.
        EmptyPDFError: If the PDF contains no extractable text across all pages.
    """
    path = Path(file_path)

    if not path.is_file():
        raise PDFNotFoundError(f"PDF file not found at path: '{path.resolve()}'")

    file_size = path.stat().st_size
    if file_size > MAX_PDF_SIZE_BYTES:
        raise PDFSizeLimitExceededError(
            f"PDF file '{path.name}' size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed limit of 25 MB."
        )

    try:
        pymupdf_lib = _get_pymupdf()
        doc = pymupdf_lib.open(str(path))
    except Exception as exc:
        raise InvalidPDFError(f"Failed to parse PDF file '{path.name}': {exc}") from exc

    try:
        if len(doc) == 0:
            raise EmptyPDFError(f"PDF file '{path.name}' contains 0 pages.")

        all_chunks: list[DocumentChunk] = []
        has_any_text = False
        source_name = path.name
        stem = doc_id_override or sanitize_doc_slug(path.name)

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_number = page_idx + 1  # 1-indexed

            raw_page_text = page.get_text("text")
            cleaned_page_text = clean_text(raw_page_text)

            if not cleaned_page_text:
                continue

            has_any_text = True
            page_chunks = chunk_text(
                cleaned_page_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for chunk_idx, chunk_content in enumerate(page_chunks):
                chunk_id = f"{stem}_p{page_number}_c{chunk_idx + 1:03d}"
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_content,
                        source=source_name,
                        page_number=page_number,
                        metadata={
                            "chunk_index": chunk_idx + 1,
                            "char_length": len(chunk_content),
                            "page_number": page_number,
                            "source_file": source_name,
                            "doc_id": stem,
                        },
                    )
                )

        if not has_any_text:
            raise EmptyPDFError(
                f"PDF file '{path.name}' contains no extractable text across all {len(doc)} pages."
            )

        return all_chunks

    finally:
        doc.close()


def load_multiple_pdfs(
    file_paths: list[str | Path],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:
    """Load, clean, and chunk multiple PDF documents with collision-resistant chunk IDs.

    Args:
        file_paths: List of file paths to PDF documents. Maximum 5 PDFs allowed.
        chunk_size: Target maximum character length for each chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        Combined list of DocumentChunk instances across all documents.

    Raises:
        PDFLimitExceededError: If len(file_paths) > 5.
    """
    if len(file_paths) > MAX_PDF_COUNT:
        raise PDFLimitExceededError(
            f"Cannot load {len(file_paths)} PDFs. Maximum allowed is {MAX_PDF_COUNT}."
        )

    all_chunks: list[DocumentChunk] = []
    seen_stems: dict[str, int] = {}

    for fp in file_paths:
        path = Path(fp)
        base_stem = sanitize_doc_slug(path.name)
        seen_stems[base_stem] = seen_stems.get(base_stem, 0) + 1
        if seen_stems[base_stem] > 1:
            doc_id = f"{base_stem}_{seen_stems[base_stem]}"
        else:
            doc_id = base_stem

        chunks = load_pdf(
            path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            doc_id_override=doc_id,
        )
        all_chunks.extend(chunks)

    return all_chunks


def load_documents_dir(
    dir_path: str | Path,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:
    """Discover and ingest all PDF files in a directory up to the 5-PDF limit.

    Args:
        dir_path: Local filesystem directory containing PDF files.
        chunk_size: Target maximum character length for each chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        Combined list of DocumentChunk instances across all discovered PDF documents.
    """
    path = Path(dir_path)
    if not path.is_dir():
        return []

    pdf_files = sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"],
        key=lambda p: p.name.lower(),
    )
    if not pdf_files:
        return []

    return load_multiple_pdfs(
        pdf_files,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
