#!/usr/bin/env python
"""
CorrectRAG — Baseline RAG Demo Script

End-to-end demonstration:
  1. Ingest a PDF (or use synthetic data).
  2. Index chunks into ChromaDB.
  3. Retrieve top-K chunks for a question.
  4. Generate a grounded answer via Gemini.
  5. Print question, answer, sources, and chunk scores.

Usage (from the project root):
    python scripts/demo_rag.py [path/to/document.pdf]

Environment variables:
    GEMINI_API_KEY   — required for real Gemini calls
    GEMINI_MODEL     — optional, defaults to gemini-3.6-flash
"""

import os
import sys
import textwrap

# ── Path setup so we can import from backend/app ─────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "backend"))

from app.ingestion.pdf_loader import load_pdf
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import InMemoryVectorStore
from app.retrieval.retriever import VectorRetriever
from app.generation.gemini_client import GeminiClient, GeminiAPIError
from app.generation.rag_pipeline import BaselineRAG

# ─────────────────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "What is the CORRECT action in CRAG when retrieval confidence is high?",
    "How does CRAG handle the AMBIGUOUS case?",
    "What is the role of the retrieval evaluator in CRAG?",
    "How does knowledge refinement work in CRAG?",
]

DEMO_DOCUMENTS = [
    ("Corrective RAG overview",
     "Corrective Retrieval Augmented Generation (CRAG) introduces a corrective "
     "mechanism that evaluates the quality of retrieved documents before generation. "
     "It triggers three actions: CORRECT when retrieval is confident, INCORRECT "
     "when retrieval fails, and AMBIGUOUS when confidence is mixed."),
    ("Knowledge Refinement in CRAG",
     "CRAG refines retrieved knowledge by decomposing documents into fine-grained "
     "knowledge strips, filtering irrelevant strips, and recomposing the remaining "
     "strips into a clean context for the generator."),
    ("CRAG web search fallback",
     "When retrieval quality is poor (INCORRECT action), CRAG rewrites the query "
     "and performs an external web search. Retrieved web documents then undergo "
     "the same refinement process as internal corpus documents."),
]

SEPARATOR = "─" * 70


def print_section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def run_demo(pdf_path: str | None) -> None:
    print_section("CorrectRAG — Baseline RAG Demo (Standard RAG, no CRAG evaluator)")

    # ── 1. Check API key ──────────────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print(
            "\n[WARNING] GEMINI_API_KEY is not set.\n"
            "          Real Gemini generation will not run.\n"
            "          Set the variable and re-run to get actual answers.\n"
        )
        api_available = False
    else:
        api_available = True

    # ── 2. Initialize retrieval stack ─────────────────────────────────────────
    print("\n[1/4] Initializing Embedding Model & ChromaDB...")
    embedding_model = EmbeddingModel()
    vector_store = InMemoryVectorStore(
        embedding_model=embedding_model,
        persist_directory=":memory:",   # ephemeral — demo only
    )
    retriever = VectorRetriever(vector_store=vector_store)

    # ── 3. Load & index document(s) ───────────────────────────────────────────
    if pdf_path:
        print(f"\n[2/4] Loading PDF: {pdf_path}")
        chunks = load_pdf(pdf_path)
        print(f"      Extracted {len(chunks)} chunks.")
    else:
        print("\n[2/4] No PDF supplied — using synthetic sample documents.")
        from app.ingestion.pdf_loader import DocumentChunk
        chunks = [
            DocumentChunk(
                chunk_id=f"synth_p1_c00{i + 1}",
                text=text,
                source="synthetic",
                page_number=1,
                metadata={"synthetic": True},
            )
            for i, (_, text) in enumerate(DEMO_DOCUMENTS)
        ]

    print(f"\n[3/4] Indexing {len(chunks)} chunks into ChromaDB...")
    vector_store.upsert(chunks)
    print(f"      Vector store size: {vector_store.count()} items.")

    # ── 4. Init Gemini (if key available) ─────────────────────────────────────
    if api_available:
        try:
            llm = GeminiClient()
            print(f"\n      Gemini model : {llm.model}")
        except GeminiAPIError as exc:
            print(f"\n[ERROR] Could not initialize Gemini client: {exc}")
            api_available = False
            llm = None
    else:
        llm = None

    # ── 5. Run queries ────────────────────────────────────────────────────────
    print_section("4/4 — Baseline RAG Queries")

    for question in DEMO_QUESTIONS:
        print(f"\nQUESTION: {question}")

        # Always show retrieved chunks (no API needed)
        raw_chunks = retriever.retrieve(question, top_k=3)
        print(f"\nRetrieved chunks ({len(raw_chunks)}):")
        for i, c in enumerate(raw_chunks, 1):
            excerpt = textwrap.shorten(c.text, width=100, placeholder="...")
            print(f"  [{i}] score={c.score:.4f} | page={c.page_number} | {c.source}")
            print(f"       {excerpt}")

        if api_available and llm is not None:
            rag = BaselineRAG(retriever=retriever, llm_client=llm, top_k=3)
            result = rag.run(question)
            print(f"\nANSWER:\n{textwrap.fill(result.answer, width=80)}")
            if result.sources:
                print("\nSOURCES:")
                for s in result.sources:
                    print(f"  • {s.source} | page {s.page_number} | chunk {s.chunk_id}")
        else:
            print("\n[SKIPPED] Gemini API call skipped — GEMINI_API_KEY not set.")

        print()

    print_section("Demo complete")


if __name__ == "__main__":
    pdf_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_demo(pdf_arg)
