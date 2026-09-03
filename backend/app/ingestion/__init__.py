"""
Document Ingestion Package for CorrectRAG.
"""

from app.ingestion.pdf_loader import (
    DocumentChunk,
    EmptyPDFError,
    InvalidPDFError,
    PDFIngestionError,
    PDFLimitExceededError,
    PDFNotFoundError,
    PDFSizeLimitExceededError,
    MAX_PDF_COUNT,
    MAX_PDF_SIZE_BYTES,
    clean_text,
    chunk_text,
    load_pdf,
    load_multiple_pdfs,
    load_documents_dir,
    sanitize_doc_slug,
)

__all__ = [
    "DocumentChunk",
    "PDFIngestionError",
    "PDFNotFoundError",
    "InvalidPDFError",
    "EmptyPDFError",
    "PDFLimitExceededError",
    "PDFSizeLimitExceededError",
    "MAX_PDF_COUNT",
    "MAX_PDF_SIZE_BYTES",
    "clean_text",
    "chunk_text",
    "load_pdf",
    "load_multiple_pdfs",
    "load_documents_dir",
    "sanitize_doc_slug",
]
