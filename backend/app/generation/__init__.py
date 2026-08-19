"""
Generation Package for CorrectRAG (Baseline RAG).

Provides:
  - GeminiClient  — Google GenAI SDK wrapper
  - build_rag_prompt / format_context_block — prompt construction
  - BaselineRAG   — Retrieve-then-Generate pipeline
  - RAGResult / SourceCitation — structured output models
"""

from app.generation.gemini_client import GeminiClient, GeminiAPIError
from app.generation.prompt import build_rag_prompt, format_context_block
from app.generation.rag_pipeline import BaselineRAG, RAGResult, SourceCitation

__all__ = [
    "GeminiClient",
    "GeminiAPIError",
    "build_rag_prompt",
    "format_context_block",
    "BaselineRAG",
    "RAGResult",
    "SourceCitation",
]
