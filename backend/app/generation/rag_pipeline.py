"""
Baseline RAG Pipeline for CorrectRAG.

Implements the standard Retrieve-then-Generate flow:

    user question
      → VectorRetriever.retrieve(top_k)
      → build_rag_prompt(question, chunks)
      → GeminiClient.generate(prompt)
      → RAGResult (answer + sources + chunks)

This is the BASELINE pipeline.
CRAG (Corrective RAG with evaluator, routing, and knowledge refinement)
is NOT implemented here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.retrieval.retriever import RetrievedChunk, VectorRetriever
from app.generation.llm_provider import LLMProvider
from app.generation.prompt import build_rag_prompt


# ──────────────────────────────────────────────────────────────────────────────
# Output models
# ──────────────────────────────────────────────────────────────────────────────

class SourceCitation(BaseModel):
    """Structured citation for a single retrieved source passage."""

    source: str = Field(..., description="Source document filename or path")
    page_number: int = Field(..., description="1-indexed page number")
    chunk_id: str = Field(..., description="Unique chunk identifier")


class RAGResult(BaseModel):
    """Structured output of the Baseline RAG pipeline."""

    answer: str = Field(..., description="Generated answer text")
    sources: list[SourceCitation] = Field(
        default_factory=list,
        description="Deduplicated citations for retrieved passages",
    )
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Raw retrieved chunks (includes score and full text)",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fallback answer when no context is available
# ──────────────────────────────────────────────────────────────────────────────
_NO_CONTEXT_ANSWER = (
    "I cannot answer this question because no relevant context "
    "was retrieved from the knowledge base."
)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline class
# ──────────────────────────────────────────────────────────────────────────────

class BaselineRAG:
    """Standard Retrieve-then-Generate pipeline (no CRAG logic).

    Args:
        retriever:     Initialised VectorRetriever.
        llm_client:    GeminiClient (or any object with a .generate(str)->str method).
        top_k:         Number of chunks to retrieve per query (default: 5).
        skip_llm_on_empty: When True (default), return a deterministic fallback
                           answer without calling the LLM if no chunks are found.
                           Set to False only if you want the LLM to handle
                           the empty-context case directly.
    """

    def __init__(
        self,
        retriever: VectorRetriever,
        llm_client: LLMProvider,
        top_k: int = 5,
        skip_llm_on_empty: bool = True,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.top_k = top_k
        self.skip_llm_on_empty = skip_llm_on_empty

    def run(self, query: str) -> RAGResult:
        """Execute the full Retrieve → Prompt → Generate pipeline.

        Args:
            query: User question string.

        Returns:
            RAGResult with answer, citations, and raw retrieved chunks.
        """
        query = query.strip()
        if not query:
            raise ValueError("query must be a non-empty string.")

        # ── Step 1: Retrieve ──────────────────────────────────────────────────
        chunks: list[RetrievedChunk] = self.retriever.retrieve(
            query, top_k=self.top_k
        )

        # ── Step 2: Short-circuit on empty retrieval ──────────────────────────
        if not chunks and self.skip_llm_on_empty:
            return RAGResult(
                answer=_NO_CONTEXT_ANSWER,
                sources=[],
                retrieved_chunks=[],
            )

        # ── Step 3: Build prompt ──────────────────────────────────────────────
        prompt = build_rag_prompt(question=query, chunks=chunks)

        # ── Step 4: Generate ─────────────────────────────────────────────────
        answer: str = self.llm_client.generate(prompt)

        # ── Step 5: Build deduplicated citations ──────────────────────────────
        seen: set[str] = set()
        citations: list[SourceCitation] = []
        for chunk in chunks:
            key = chunk.chunk_id
            if key not in seen:
                seen.add(key)
                citations.append(
                    SourceCitation(
                        source=chunk.source,
                        page_number=chunk.page_number,
                        chunk_id=chunk.chunk_id,
                    )
                )

        return RAGResult(
            answer=answer,
            sources=citations,
            retrieved_chunks=chunks,
        )
