"""
Generation Package for CorrectRAG (Baseline RAG).

Provides:
  - LLMProvider   — protocol defining generate(prompt) -> str
  - GeminiClient  — Google GenAI SDK wrapper (primary production)
  - GroqClient    — Groq SDK wrapper (alternate evaluation)
  - GeminiAPIError / GroqAPIError — provider exception classes
  - build_rag_prompt / format_context_block — prompt construction
  - BaselineRAG   — Retrieve-then-Generate pipeline
  - RAGResult / SourceCitation — structured output models
"""

from app.generation.llm_provider import LLMProvider
from app.generation.gemini_client import GeminiClient, GeminiAPIError
from app.generation.groq_client import GroqClient, GroqAPIError
from app.generation.prompt import build_rag_prompt, format_context_block
from app.generation.rag_pipeline import BaselineRAG, RAGResult, SourceCitation

__all__ = [
    "LLMProvider",
    "GeminiClient",
    "GeminiAPIError",
    "GroqClient",
    "GroqAPIError",
    "build_rag_prompt",
    "format_context_block",
    "BaselineRAG",
    "RAGResult",
    "SourceCitation",
]
