#!/usr/bin/env python
"""
Corpus Re-indexing Script for CorrectRAG.

Loads CRAG.pdf, extracts 168 chunks, computes embeddings via Jina AI
Embedding API (model: jina-embeddings-v5-text-small, task: retrieval.passage),
and saves the updated vector store to chroma_data/correctrag.pkl.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Path setup so we can import from backend/app
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

load_dotenv(_PROJECT_ROOT / ".env")

from app.ingestion.pdf_loader import load_documents_dir, load_pdf
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import InMemoryVectorStore


def reindex_corpus(target_path: str | None = None) -> None:
    api_key = os.environ.get("JINA_API_KEY")
    if not api_key:
        print("[ERROR] JINA_API_KEY environment variable is missing.")
        sys.exit(1)

    persist_dir = _PROJECT_ROOT / "chroma_data"
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Determine files to load
    target = Path(target_path) if target_path else None
    if not target:
        docs_dir_env = os.environ.get("DOCUMENTS_DIR")
        target = Path(docs_dir_env) if docs_dir_env else (_PROJECT_ROOT / "data" / "documents")

    chunks = []
    if target.is_dir():
        print(f"[1/4] Scanning directory {target} for PDF files...")
        chunks = load_documents_dir(target, chunk_size=500, chunk_overlap=100)
    elif target.is_file():
        print(f"[1/4] Loading single PDF {target.name}...")
        chunks = load_pdf(target, chunk_size=500, chunk_overlap=100)

    if not chunks:
        fallback = _PROJECT_ROOT / "CRAG.pdf"
        if fallback.exists():
            print(f"[1/4] Fallback: Loading PDF chunks from {fallback.name}...")
            chunks = load_pdf(str(fallback), chunk_size=500, chunk_overlap=100)
        else:
            print("[ERROR] No PDF documents found to index.")
            sys.exit(1)

    print(f"      Extracted {len(chunks)} document chunks.")

    print(f"[2/4] Initializing Jina EmbeddingModel (model: jina-embeddings-v5-text-small, dim: 1024)...")
    embedding_model = EmbeddingModel()

    # Ephemeral vector store to avoid reading stale embeddings into cache
    vector_store = InMemoryVectorStore(
        persist_directory=None,
        collection_name="correctrag",
        embedding_model=embedding_model,
    )

    print(f"[3/4] Embedding {len(chunks)} chunks with Jina API (task: retrieval.passage)...")
    vector_store.add_chunks(chunks)
    doc_counts = vector_store.get_indexed_documents()
    print(f"      Successfully embedded {vector_store.count()} chunks across {len(doc_counts)} document(s): {doc_counts}")

    # Point persist path to destination and save
    vector_store.persist_path = persist_dir / "correctrag.pkl"
    print(f"[4/4] Persisting vector store to {vector_store.persist_path}...")
    vector_store._save()

    # Validate output
    sample_chunk = next(iter(vector_store.chunks.values()))
    emb_dim = len(sample_chunk["embedding"])
    print(f"[SUCCESS] Re-indexing complete! Documents: {len(doc_counts)}, Total Chunks: {vector_store.count()}, Embedding Dimension: {emb_dim}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    reindex_corpus(arg)
