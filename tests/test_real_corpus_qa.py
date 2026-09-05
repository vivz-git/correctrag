"""
Tests for real-corpus document Q&A capabilities in CorrectRAG.

Validates:
1. Multi-document inventory and chunk counts
2. Cross-document retrieval across multiple real-world PDF sources
3. KnowledgeRefiner strip extraction across multiple documents
4. Multi-document citation and provenance formatting in generation prompts
5. ActionRouter correct identification of out-of-corpus queries
"""

from unittest.mock import MagicMock
import pytest

from app.evaluation.action_router import ActionRouter
from app.evaluation.knowledge_refiner import KnowledgeRefiner, KnowledgeStrip
from app.evaluation.relevance_evaluator import RelevanceEvaluator
from app.ingestion.pdf_loader import DocumentChunk
from app.pipeline.crag_pipeline import _build_crag_prompt
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.retriever import RetrievedChunk, VectorRetriever
from app.retrieval.vector_store import InMemoryVectorStore


class MockDeterministicEmbeddingModel(EmbeddingModel):
    """Offline embedding model providing distinct clusters for multi-doc testing."""

    def __init__(self, dimension: int = 1024):
        super().__init__(model_name="mock-model", api_key="dummy", dimension=dimension)
        self._dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            vec = [0.0] * self.dimension
            if "self-rag" in text.lower():
                vec[0] = 1.0
            elif "lewis" in text.lower() or "bart" in text.lower():
                vec[1] = 1.0
            elif "crag" in text.lower():
                vec[2] = 1.0
            else:
                vec[3] = 0.5
            embeddings.append(vec)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        lower = text.lower()
        if "self-rag" in lower:
            vec[0] += 0.8
        if "crag" in lower:
            vec[2] += 0.8
        if "lewis" in lower or "bart" in lower:
            vec[1] += 0.8
        if sum(vec) == 0:
            vec[4] = 1.0
        return vec


@pytest.fixture
def multi_doc_store() -> InMemoryVectorStore:
    """Vector store populated with chunks from three distinct documents."""
    emb_model = MockDeterministicEmbeddingModel()
    store = InMemoryVectorStore(
        persist_directory=None,
        collection_name="test_real_qa",
        embedding_model=emb_model,
    )
    chunks = [
        DocumentChunk(
            chunk_id="crag_p001_c001",
            text="CRAG evaluates retrieved documents using a retrieval evaluator.",
            source="CRAG.pdf",
            page_number=1,
        ),
        DocumentChunk(
            chunk_id="crag_p003_c002",
            text="CRAG introduces three action triggers: Correct, Incorrect, and Ambiguous.",
            source="CRAG.pdf",
            page_number=3,
        ),
        DocumentChunk(
            chunk_id="self_rag_p002_c001",
            text="Self-RAG introduces four types of reflection tokens: Retrieve, IsRel, IsSup, IsUse.",
            source="Self-RAG.pdf",
            page_number=2,
        ),
        DocumentChunk(
            chunk_id="self_rag_p005_c002",
            text="Self-RAG trains critic and generator models using GPT-4 feedback.",
            source="Self-RAG.pdf",
            page_number=5,
        ),
        DocumentChunk(
            chunk_id="lewis_p003_c001",
            text="Lewis et al. propose RAG-Sequence and RAG-Token using a BART generator.",
            source="Lewis_RAG.pdf",
            page_number=3,
        ),
    ]
    store.add_chunks(chunks)
    return store


def test_real_corpus_inventory_tracking(multi_doc_store: InMemoryVectorStore):
    """Verify that multiple documents are tracked accurately in inventory."""
    inventory = multi_doc_store.get_indexed_documents()
    assert inventory == {
        "CRAG.pdf": 2,
        "Self-RAG.pdf": 2,
        "Lewis_RAG.pdf": 1,
    }
    assert multi_doc_store.count() == 5


def test_cross_document_retrieval(multi_doc_store: InMemoryVectorStore):
    """Verify that a cross-document query retrieves chunks from multiple distinct documents."""
    emb_model = multi_doc_store.embedding_model
    retriever = VectorRetriever(vector_store=multi_doc_store, embedding_model=emb_model)

    # Query mentioning both CRAG and Self-RAG
    results = retriever.retrieve("Compare CRAG evaluator with Self-RAG reflection tokens", top_k=4)
    assert len(results) >= 2
    sources = set(r.source for r in results)
    assert "CRAG.pdf" in sources
    assert "Self-RAG.pdf" in sources

    # Check provenance
    for r in results:
        assert r.source in ["CRAG.pdf", "Self-RAG.pdf", "Lewis_RAG.pdf"]
        assert isinstance(r.page_number, int)
        assert r.page_number > 0


def test_multi_document_strip_refinement():
    """Verify KnowledgeRefiner preserves multiple sources and pages during strip selection."""
    mock_evaluator = MagicMock(spec=RelevanceEvaluator)
    mock_evaluator.score_batch.side_effect = lambda pairs: [0.85] * len(pairs)

    refiner = KnowledgeRefiner(evaluator=mock_evaluator, top_k=4)
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="CRAG decomposes documents into strips. Fine-grained filtering occurs.",
            source="CRAG.pdf",
            page_number=4,
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="Self-RAG critiques generation dynamically. Four reflection tokens are used.",
            source="Self-RAG.pdf",
            page_number=2,
            score=0.88,
        ),
    ]

    strips = refiner.refine("CRAG and Self-RAG mechanism", chunks)
    assert len(strips) > 0
    sources = set(s.source for s in strips)
    assert "CRAG.pdf" in sources
    assert "Self-RAG.pdf" in sources


def test_cross_document_prompt_citation_formatting():
    """Verify that _build_crag_prompt properly formats citations for multiple documents."""
    strips = [
        KnowledgeStrip(
            text="CRAG confidence thresholds determine actions.",
            source="CRAG.pdf",
            page_number=3,
            parent_chunk_id="c1",
            position=0,
            score=0.92,
        ),
        KnowledgeStrip(
            text="Self-RAG uses [IsRel] to test passage relevance.",
            source="Self-RAG.pdf",
            page_number=4,
            parent_chunk_id="c2",
            position=1,
            score=0.89,
        ),
    ]

    prompt = _build_crag_prompt("Compare CRAG and Self-RAG", internal_strips=strips, external_strips=[])
    assert "[1] Source: CRAG.pdf | Page: 3 (internal document)" in prompt
    assert "[2] Source: Self-RAG.pdf | Page: 4 (internal document)" in prompt
    assert "Sources: [<source>, page <page>]" in prompt


def test_out_of_corpus_routing_to_incorrect():
    """Verify that low-relevance queries route to INCORRECT without web fallback suppression."""
    router = ActionRouter(clearly_relevant_threshold=0.7, clearly_irrelevant_threshold=-0.1)
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="RAG-Sequence generates the entire sequence with one document.",
            source="Lewis_RAG.pdf",
            page_number=3,
            score=-0.4,
        )
    ]
    scores = [-0.45]
    decision = router.route("Who won the 2024 cricket world cup?", chunks, scores)
    assert decision.action == "INCORRECT"