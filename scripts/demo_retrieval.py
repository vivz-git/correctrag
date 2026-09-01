"""
Demo script for testing Ingest -> Index -> Semantic Retrieval manually.

Usage:
    python scripts/demo_retrieval.py [optional_pdf_path]
"""

import sys
from pathlib import Path

# Ensure backend directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.ingestion.pdf_loader import DocumentChunk, load_pdf
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import InMemoryVectorStore
from app.retrieval.retriever import VectorRetriever


def main():
    print("=" * 70)
    print("CorrectRAG: Manual Ingestion & Semantic Retrieval Demo")
    print("=" * 70)

    # 1. Initialize Components
    print("\n[1/4] Initializing Embedding Model & ChromaDB Vector Store...")
    embedding_model = EmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = InMemoryVectorStore(
        persist_directory=str(root_dir / "chroma_data"),
        collection_name="demo_collection",
        embedding_model=embedding_model,
    )
    vector_store.clear()
    retriever = VectorRetriever(vector_store=vector_store, embedding_model=embedding_model)

    # 2. Ingest PDF or synthetic sample chunks
    pdf_candidate = Path(sys.argv[1]) if len(sys.argv) > 1 else root_dir / "CRAG.pdf"

    if pdf_candidate.is_file():
        print(f"\n[2/4] Ingesting PDF file: '{pdf_candidate.name}'...")
        chunks = load_pdf(pdf_candidate, chunk_size=500, chunk_overlap=100)
        print(f"      Extracted {len(chunks)} document chunks across pages.")
    else:
        print("\n[2/4] No PDF specified. Using synthetic document chunks...")
        chunks = [
            DocumentChunk(
                chunk_id="synthetic_p1_c001",
                text="Corrective Retrieval Augmented Generation (CRAG) evaluates retrieval quality.",
                source="crag_overview.pdf",
                page_number=1,
            ),
            DocumentChunk(
                chunk_id="synthetic_p1_c002",
                text="When retrieval is incorrect, CRAG discards internal documents and triggers web search.",
                source="crag_overview.pdf",
                page_number=1,
            ),
            DocumentChunk(
                chunk_id="synthetic_p2_c001",
                text="The decompose-then-recompose algorithm extracts key information strips.",
                source="crag_refinement.pdf",
                page_number=2,
            ),
        ]

    # 3. Index Chunks
    print(f"\n[3/4] Indexing {len(chunks)} chunks into ChromaDB...")
    vector_store.add_chunks(chunks)
    print(f"      Total items in vector store: {vector_store.count()}")

    # 4. Perform Sample Queries
    sample_queries = [
        "What happens when retrieval is incorrect?",
        "How does knowledge refinement decompose documents?",
        "What is the role of the retrieval evaluator?",
    ]

    print("\n[4/4] Executing Semantic Retrieval Queries...")
    for query in sample_queries:
        print("\n" + "-" * 70)
        print(f"Query: \"{query}\"")
        results = retriever.retrieve(query=query, top_k=2)
        if not results:
            print("  No matching chunks found.")
            continue

        for rank, r in enumerate(results, start=1):
            print(f"  [{rank}] Score: {r.score:.4f} | Page: {r.page_number} | ID: {r.chunk_id}")
            print(f"      Source: {r.source}")
            first_line = r.text.replace("\n", " ")[:120]
            print(f"      Excerpt: {first_line}...")

    print("\n" + "=" * 70)
    print("Demo completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
