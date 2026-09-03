"""
Tests for user-facing Multi-PDF Upload and Inventory API endpoints.

Tests:
1. POST /documents with one valid PDF.
2. POST /documents with two valid PDFs.
3. Six PDFs rejected (HTTP 400).
4. >25 MB rejected (HTTP 400).
5. Non-PDF rejected (HTTP 400).
6. Existing CRAG.pdf chunks preserved.
7. GET /documents returns correct inventory.
8. Uploaded PDF source filename preserved.
9. Uploaded PDF page number preserved.
10. Identical content across PDFs does not collide.
11. Existing query endpoint still works after indexing.
"""

from unittest.mock import MagicMock
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_crag_pipeline
from app.main import app
from app.pipeline.crag_pipeline import CRAGPipeline, CRAGResult, ExecutionTrace
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.retriever import RetrievedChunk, VectorRetriever
from app.retrieval.vector_store import InMemoryVectorStore
from app.ingestion.pdf_loader import DocumentChunk


def _create_mock_pdf_bytes(pages_text: list[str]) -> bytes:
    """Generate in-memory real PDF bytes with specified text per page."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def mock_embedding_model() -> EmbeddingModel:
    """EmbeddingModel with fast dummy embeddings."""
    class DummyEmbeddingModel(EmbeddingModel):
        def __init__(self):
            super().__init__()
            self._dim = 1024

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            res = []
            for t in texts:
                cache_key = f"retrieval.passage:{t}"
                if cache_key not in self._cache:
                    val = hash(t) % 1000 / 1000.0
                    self._cache[cache_key] = [val] * self._dim
                res.append(self._cache[cache_key])
            return res

        def embed_query(self, text: str) -> list[float]:
            val = hash(text) % 1000 / 1000.0
            return [val] * self._dim

    return DummyEmbeddingModel()


@pytest.fixture
def test_vector_store(mock_embedding_model: EmbeddingModel) -> InMemoryVectorStore:
    """Ephemeral in-memory vector store."""
    return InMemoryVectorStore(
        persist_directory=None,
        collection_name="test_api_docs",
        embedding_model=mock_embedding_model,
    )


@pytest.fixture
def test_pipeline(test_vector_store: InMemoryVectorStore, mock_embedding_model: EmbeddingModel) -> MagicMock:
    """Mock CRAGPipeline using real in-memory vector store and mock embeddings."""
    pipeline = MagicMock(spec=CRAGPipeline)
    retriever = VectorRetriever(vector_store=test_vector_store, embedding_model=mock_embedding_model)
    pipeline.retriever = retriever

    # Default query run result
    def _mock_run(query_str: str):
        chunks = retriever.retrieve(query_str, top_k=5)
        return CRAGResult(
            query=query_str,
            retrieved_chunks=chunks,
            relevance_scores=[0.85] * len(chunks),
            action="CORRECT",
            refined_strips=[],
            external_strips=[],
            web_results=[],
            answer="Mock answer referencing uploaded documents.",
            trace=ExecutionTrace(
                retrieved_count=len(chunks),
                action="CORRECT",
                max_relevance_score=0.85,
                web_search_used=False,
                rewritten_query=None,
                internal_strip_count=0,
                external_strip_count=0,
                final_context_source="internal",
            ),
        )

    pipeline.run.side_effect = _mock_run
    return pipeline


@pytest.fixture
def client(test_pipeline: MagicMock):
    """TestClient with overridden get_crag_pipeline dependency."""
    app.dependency_overrides[get_crag_pipeline] = lambda: test_pipeline
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# 1. POST /documents with one valid PDF
def test_post_documents_one_valid_pdf(client: TestClient, test_pipeline: MagicMock):
    pdf_bytes = _create_mock_pdf_bytes(["Introduction to Quantum Computing on page 1."])
    response = client.post(
        "/documents",
        files=[("files", ("quantum.pdf", pdf_bytes, "application/pdf"))],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["added_chunks"] == 1
    assert data["total_chunks"] == 1
    assert "quantum.pdf" in data["indexed_documents"]
    assert data["indexed_documents"]["quantum.pdf"] == 1


# 2. POST /documents with two valid PDFs
def test_post_documents_two_valid_pdfs(client: TestClient, test_pipeline: MagicMock):
    pdf1 = _create_mock_pdf_bytes(["Quantum mechanics and qubits."])
    pdf2 = _create_mock_pdf_bytes(["Deep learning and neural networks."])

    response = client.post(
        "/documents",
        files=[
            ("files", ("quantum.pdf", pdf1, "application/pdf")),
            ("files", ("ml.pdf", pdf2, "application/pdf")),
        ],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["added_chunks"] == 2
    assert data["total_chunks"] == 2
    assert "quantum.pdf" in data["indexed_documents"]
    assert "ml.pdf" in data["indexed_documents"]


# 3. Six PDFs rejected (HTTP 400)
def test_post_documents_six_pdfs_rejected(client: TestClient):
    files = [
        ("files", (f"doc_{i}.pdf", _create_mock_pdf_bytes([f"Page {i}"]), "application/pdf"))
        for i in range(6)
    ]
    response = client.post("/documents", files=files)
    assert response.status_code == 400
    assert "Maximum allowed is 5" in response.json()["detail"]


# 4. >25 MB rejected (HTTP 400)
def test_post_documents_oversized_rejected(client: TestClient):
    oversized_dummy = b"%PDF-1.4 " + b"0" * (25 * 1024 * 1024 + 100)
    response = client.post(
        "/documents",
        files=[("files", ("large.pdf", oversized_dummy, "application/pdf"))],
    )
    assert response.status_code == 400
    assert "exceeds maximum allowed size of 25 MB" in response.json()["detail"]


# 5. Non-PDF rejected (HTTP 400)
def test_post_documents_non_pdf_rejected(client: TestClient):
    response = client.post(
        "/documents",
        files=[("files", ("notes.txt", b"plain text notes", "text/plain"))],
    )
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


# 6. Existing CRAG.pdf chunks preserved
def test_post_documents_preserves_crag_chunks(client: TestClient, test_pipeline: MagicMock):
    store = test_pipeline.retriever.vector_store
    crag_chunk = DocumentChunk(
        chunk_id="CRAG_p1_c001",
        text="Corrective Retrieval Augmented Generation baseline text.",
        source="CRAG.pdf",
        page_number=1,
        metadata={"source_file": "CRAG.pdf"},
    )
    store.add_chunks([crag_chunk])
    assert store.count() == 1
    assert "CRAG.pdf" in store.get_indexed_documents()

    new_pdf = _create_mock_pdf_bytes(["New uploaded research document."])
    response = client.post(
        "/documents",
        files=[("files", ("research.pdf", new_pdf, "application/pdf"))],
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_chunks"] == 2
    assert data["indexed_documents"]["CRAG.pdf"] == 1
    assert data["indexed_documents"]["research.pdf"] == 1
    assert "CRAG_p1_c001" in store.chunks


# 7. GET /documents returns correct inventory
def test_get_documents_inventory(client: TestClient, test_pipeline: MagicMock):
    store = test_pipeline.retriever.vector_store
    doc_chunk = DocumentChunk(
        chunk_id="alpha_p1_c001",
        text="Alpha document text content.",
        source="alpha.pdf",
        page_number=1,
        metadata={},
    )
    store.add_chunks([doc_chunk])

    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["total_chunks"] == 1
    assert data["documents"] == {"alpha.pdf": 1}


# 8. Uploaded PDF source filename preserved
def test_uploaded_pdf_source_filename_preserved(client: TestClient, test_pipeline: MagicMock):
    pdf = _create_mock_pdf_bytes(["Test provenance text."])
    response = client.post(
        "/documents",
        files=[("files", ("Custom_Report_2026.pdf", pdf, "application/pdf"))],
    )
    assert response.status_code == 200
    store = test_pipeline.retriever.vector_store
    sources = [c["metadata"]["source"] for c in store.chunks.values()]
    assert "Custom_Report_2026.pdf" in sources


# 9. Uploaded PDF page number preserved
def test_uploaded_pdf_page_number_preserved(client: TestClient, test_pipeline: MagicMock):
    pdf = _create_mock_pdf_bytes(["Page one text content.", "Page two text content."])
    response = client.post(
        "/documents",
        files=[("files", ("two_pager.pdf", pdf, "application/pdf"))],
    )
    assert response.status_code == 200
    store = test_pipeline.retriever.vector_store
    page_numbers = {c["metadata"]["page_number"] for c in store.chunks.values()}
    assert page_numbers == {1, 2}


# 10. Identical content across PDFs does not collide
def test_identical_content_across_pdfs_no_collision(client: TestClient, test_pipeline: MagicMock):
    shared_text = "Shared paragraph text appearing in multiple papers exactly the same."
    pdf_a = _create_mock_pdf_bytes([shared_text])
    pdf_b = _create_mock_pdf_bytes([shared_text])

    response = client.post(
        "/documents",
        files=[
            ("files", ("DocA.pdf", pdf_a, "application/pdf")),
            ("files", ("DocB.pdf", pdf_b, "application/pdf")),
        ],
    )
    assert response.status_code == 200
    store = test_pipeline.retriever.vector_store
    assert store.count() == 2
    chunk_ids = list(store.chunks.keys())
    assert len(set(chunk_ids)) == 2
    assert any(cid.startswith("DocA") for cid in chunk_ids)
    assert any(cid.startswith("DocB") for cid in chunk_ids)


# 11. Existing query endpoint still works after indexing
def test_query_endpoint_works_after_indexing(client: TestClient, test_pipeline: MagicMock):
    pdf = _create_mock_pdf_bytes(["Corrective Retrieval Augmented Generation knowledge base."])
    upload_res = client.post(
        "/documents",
        files=[("files", ("crag_guide.pdf", pdf, "application/pdf"))],
    )
    assert upload_res.status_code == 200

    query_res = client.post(
        "/query",
        json={"question": "What is Corrective Retrieval Augmented Generation?"},
    )
    assert query_res.status_code == 200, query_res.json()
    data = query_res.json()
    assert data["action"] == "CORRECT"
    assert len(data["retrieved_chunks"]) > 0
    assert data["retrieved_chunks"][0]["source"] == "crag_guide.pdf"
