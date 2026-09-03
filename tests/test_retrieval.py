"""
Unit tests for embeddings, ChromaDB vector store, and semantic retriever.
"""

import pytest

from app.ingestion.pdf_loader import DocumentChunk
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import InMemoryVectorStore
from app.retrieval.retriever import RetrievedChunk, VectorRetriever


class MockEmbeddingModel(EmbeddingModel):
    @property
    def dimension(self) -> int:
        return 1024
    def embed_query(self, text: str) -> list[float]:
        # Use text length to create slightly different vectors
        val = 0.1 + (len(text) % 10) * 0.01
        if "Eiffel" in text:
            val = 0.9
        elif "Colosseum" in text:
            val = 0.8
        return [val] * self.dimension
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

@pytest.fixture(scope="module")
def shared_embedding_model() -> EmbeddingModel:
    return MockEmbeddingModel()


@pytest.fixture
def memory_vector_store(shared_embedding_model: EmbeddingModel) -> InMemoryVectorStore:
    """Create a clean in-memory InMemoryVectorStore for each test."""
    store = InMemoryVectorStore(
        persist_directory=":memory:",
        collection_name="test_collection",
        embedding_model=shared_embedding_model,
    )
    store.clear()
    return store


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Create controlled, synthetic test document chunks."""
    return [
        DocumentChunk(
            chunk_id="doc1_p1_c001",
            text="The Eiffel Tower is a famous wrought-iron lattice tower located in Paris, France.",
            source="paris_guide.pdf",
            page_number=1,
            metadata={"topic": "landmarks", "city": "Paris"},
        ),
        DocumentChunk(
            chunk_id="doc1_p2_c001",
            text="The Colosseum is an ancient amphitheatre located in the centre of Rome, Italy.",
            source="rome_guide.pdf",
            page_number=2,
            metadata={"topic": "landmarks", "city": "Rome"},
        ),
        DocumentChunk(
            chunk_id="doc2_p1_c001",
            text="Photosynthesis is the process by which green plants use sunlight to synthesize nutrients.",
            source="biology.pdf",
            page_number=1,
            metadata={"topic": "science"},
        ),
    ]


# ============================================================================
# Embedding Tests
# ============================================================================


def test_embeddings_return_correct_shape_and_type(shared_embedding_model: EmbeddingModel):
    texts = [
        "First document text for testing embeddings.",
        "Second document about machine learning systems.",
    ]
    embeddings = shared_embedding_model.embed_documents(texts)

    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(isinstance(vec, list) for vec in embeddings)
    assert len(embeddings[0]) == shared_embedding_model.dimension
    assert len(embeddings[1]) == shared_embedding_model.dimension
    assert shared_embedding_model.dimension == 1024


def test_embed_query_returns_consistent_vector(shared_embedding_model: EmbeddingModel):
    query = "Where is the Eiffel Tower?"
    query_vec = shared_embedding_model.embed_query(query)

    assert isinstance(query_vec, list)
    assert len(query_vec) == shared_embedding_model.dimension
    assert any(val != 0.0 for val in query_vec)


def test_embed_documents_empty_list(shared_embedding_model: EmbeddingModel):
    assert shared_embedding_model.embed_documents([]) == []


def test_jina_embedding_tasks_and_payload(monkeypatch):
    """Verify that embed_query and embed_documents send the correct task tags to Jina."""
    from unittest.mock import MagicMock
    import httpx

    captured_requests = []

    def mock_post(url, headers=None, json=None):
        captured_requests.append({"url": url, "headers": headers, "json": json})
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        input_items = json.get("input", [])
        resp.json.return_value = {
            "model": json.get("model"),
            "data": [
                {"object": "embedding", "index": i, "embedding": [0.1] * 1024}
                for i in range(len(input_items))
            ],
            "usage": {"total_tokens": 10},
        }
        resp.raise_for_status = MagicMock()
        return resp

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, **kwargs))

    model = EmbeddingModel(api_key="test-key")
    assert model.dimension == 1024

    # Test embed_query task
    q_vec = model.embed_query("test query")
    assert len(q_vec) == 1024
    assert len(captured_requests) == 1
    assert captured_requests[0]["json"]["task"] == "retrieval.query"
    assert captured_requests[0]["json"]["input"] == ["test query"]
    assert captured_requests[0]["json"]["model"] == "jina-embeddings-v5-text-small"
    assert captured_requests[0]["json"]["dimensions"] == 1024
    assert captured_requests[0]["headers"]["Authorization"] == "Bearer test-key"

    # Test embed_documents task
    d_vecs = model.embed_documents(["doc text 1", "doc text 2"])
    assert len(d_vecs) == 2
    assert len(d_vecs[0]) == 1024
    assert len(captured_requests) == 2
    assert captured_requests[1]["json"]["task"] == "retrieval.passage"
    assert captured_requests[1]["json"]["input"] == ["doc text 1", "doc text 2"]
    assert captured_requests[1]["json"]["dimensions"] == 1024


# ============================================================================
# Vector Store Tests
# ============================================================================


def test_vector_store_creation_and_insertion(
    memory_vector_store: InMemoryVectorStore,
    sample_chunks: list[DocumentChunk],
):
    assert memory_vector_store.count() == 0

    inserted_count = memory_vector_store.add_chunks(sample_chunks)
    assert inserted_count == 3
    assert memory_vector_store.count() == 3


def test_vector_store_avoids_duplicate_inserts(
    memory_vector_store: InMemoryVectorStore,
    sample_chunks: list[DocumentChunk],
):
    # Insert chunks first time
    memory_vector_store.add_chunks(sample_chunks)
    assert memory_vector_store.count() == 3

    # Insert same chunks with identical chunk_ids (should update/upsert, not duplicate)
    memory_vector_store.add_chunks(sample_chunks)
    assert memory_vector_store.count() == 3


def test_vector_store_clear(
    memory_vector_store: InMemoryVectorStore,
    sample_chunks: list[DocumentChunk],
):
    memory_vector_store.add_chunks(sample_chunks)
    assert memory_vector_store.count() == 3

    memory_vector_store.clear()
    assert memory_vector_store.count() == 0


# ============================================================================
# Retriever Tests
# ============================================================================


def test_retriever_finds_relevant_chunk(
    memory_vector_store: InMemoryVectorStore,
    sample_chunks: list[DocumentChunk],
):
    memory_vector_store.add_chunks(sample_chunks)
    retriever = VectorRetriever(vector_store=memory_vector_store)

    results = retriever.retrieve(query="Tell me about Paris monument and Eiffel Tower", top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, RetrievedChunk) for r in results)

    # Top result should be the Paris Eiffel Tower chunk
    top_result = results[0]
    assert top_result.chunk_id == "doc1_p1_c001"
    assert "Eiffel Tower" in top_result.text
    assert top_result.source == "paris_guide.pdf"
    assert top_result.page_number == 1
    assert top_result.score > 0.4  # High cosine similarity
    assert top_result.metadata.get("city") == "Paris"


def test_retriever_respects_top_k(
    memory_vector_store: InMemoryVectorStore,
    sample_chunks: list[DocumentChunk],
):
    memory_vector_store.add_chunks(sample_chunks)
    retriever = VectorRetriever(vector_store=memory_vector_store)

    results_k1 = retriever.retrieve(query="monuments in Europe", top_k=1)
    assert len(results_k1) == 1

    results_k3 = retriever.retrieve(query="monuments in Europe", top_k=3)
    assert len(results_k3) == 3


def test_retriever_empty_collection_behaves_safely(
    memory_vector_store: InMemoryVectorStore,
):
    assert memory_vector_store.count() == 0
    retriever = VectorRetriever(vector_store=memory_vector_store)

    results = retriever.retrieve(query="any query string", top_k=5)
    assert results == []


def test_retriever_empty_query_returns_empty_list(
    memory_vector_store: InMemoryVectorStore,
    sample_chunks: list[DocumentChunk],
):
    memory_vector_store.add_chunks(sample_chunks)
    retriever = VectorRetriever(vector_store=memory_vector_store)

    assert retriever.retrieve(query="", top_k=5) == []
    assert retriever.retrieve(query="   ", top_k=5) == []
