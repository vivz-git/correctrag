"""
Comprehensive tests for multi-PDF support in CorrectRAG.

Validates:
- Two different PDFs loading and coexisting
- Identical content in different PDFs without collision
- Duplicate filenames from different directories disambiguated
- Directory discovery of PDF documents
- 5-PDF count limit enforcement
- 25 MB file size limit enforcement
- Unique document-aware chunk IDs
- Correct filename and page number provenance
- Multi-document vector store coexistence and persistence
- Task-namespaced embedding cache behavior
- Cross-document retrieval
- Multi-document prompt source rendering
"""

from pathlib import Path
import pytest
import pymupdf

from app.ingestion.pdf_loader import (
    DocumentChunk,
    MAX_PDF_COUNT,
    MAX_PDF_SIZE_BYTES,
    PDFLimitExceededError,
    PDFSizeLimitExceededError,
    load_documents_dir,
    load_multiple_pdfs,
    load_pdf,
    sanitize_doc_slug,
)
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.retriever import RetrievedChunk, VectorRetriever
from app.retrieval.vector_store import InMemoryVectorStore
from app.evaluation.knowledge_refiner import KnowledgeStrip
from app.pipeline.crag_pipeline import _build_crag_prompt


def _create_mock_pdf(path: Path, pages_text: list[str]) -> Path:
    """Helper to generate a real PDF file with specified text per page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    doc.save(str(path))
    doc.close()
    return path


class DummyEmbeddingModel(EmbeddingModel):
    """Deterministic offline embedding model for testing without API calls."""

    def __init__(self, dimension: int = 1024):
        super().__init__(model_name="dummy-jina", api_key="dummy", dimension=dimension)
        self._dimension = dimension
        self._cache = {}

    def embed_query(self, text: str) -> list[float]:
        cache_key = f"retrieval.query:{text}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        vec = [0.0] * self._dimension
        vec[hash(text) % self._dimension] = 1.0
        self._cache[cache_key] = vec
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        for t in texts:
            cache_key = f"retrieval.passage:{t}"
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
            else:
                vec = [0.0] * self._dimension
                vec[hash(t) % self._dimension] = 1.0
                self._cache[cache_key] = vec
                results.append(vec)
        return results


def test_sanitize_doc_slug():
    assert sanitize_doc_slug("CRAG.pdf") == "CRAG"
    assert sanitize_doc_slug("Attention-Is-All-You-Need.pdf") == "Attention-Is-All-You-Need"
    assert sanitize_doc_slug("paper (v2) [final].pdf") == "paper__v2___final"
    assert sanitize_doc_slug("___") == "doc"


def test_load_two_different_pdfs(tmp_path: Path):
    pdf_a = _create_mock_pdf(
        tmp_path / "paper_a.pdf",
        ["Introduction to Quantum Computing on page one.", "Quantum algorithms on page two."],
    )
    pdf_b = _create_mock_pdf(
        tmp_path / "paper_b.pdf",
        ["Introduction to CRAG and KnowledgeRefiner.", "Action trigger decisions in CRAG."],
    )

    chunks = load_multiple_pdfs([pdf_a, pdf_b])
    assert len(chunks) == 4

    sources = {c.source for c in chunks}
    assert sources == {"paper_a.pdf", "paper_b.pdf"}

    a_chunks = [c for c in chunks if c.source == "paper_a.pdf"]
    b_chunks = [c for c in chunks if c.source == "paper_b.pdf"]

    assert len(a_chunks) == 2
    assert len(b_chunks) == 2

    assert a_chunks[0].chunk_id.startswith("paper_a_p1_c")
    assert a_chunks[1].chunk_id.startswith("paper_a_p2_c")
    assert b_chunks[0].chunk_id.startswith("paper_b_p1_c")
    assert b_chunks[1].chunk_id.startswith("paper_b_p2_c")

    assert a_chunks[0].page_number == 1
    assert a_chunks[1].page_number == 2
    assert b_chunks[0].page_number == 1
    assert b_chunks[1].page_number == 2


def test_identical_content_in_different_pdfs_no_collision(tmp_path: Path):
    shared_text = "This is a shared abstract describing modern artificial intelligence methods."
    pdf_1 = _create_mock_pdf(tmp_path / "doc1.pdf", [shared_text])
    pdf_2 = _create_mock_pdf(tmp_path / "doc2.pdf", [shared_text])

    chunks = load_multiple_pdfs([pdf_1, pdf_2])
    assert len(chunks) == 2

    c1, c2 = chunks[0], chunks[1]
    assert c1.text == c2.text
    assert c1.chunk_id != c2.chunk_id
    assert c1.chunk_id == "doc1_p1_c001"
    assert c2.chunk_id == "doc2_p1_c001"
    assert c1.source == "doc1.pdf"
    assert c2.source == "doc2.pdf"

    # Verify coexistence in vector store
    store = InMemoryVectorStore(persist_directory=None, embedding_model=DummyEmbeddingModel())
    store.add_chunks(chunks)

    assert store.count() == 2
    assert "doc1_p1_c001" in store.chunks
    assert "doc2_p1_c001" in store.chunks
    assert store.chunks["doc1_p1_c001"]["metadata"]["source"] == "doc1.pdf"
    assert store.chunks["doc2_p1_c001"]["metadata"]["source"] == "doc2.pdf"


def test_duplicate_filenames_from_different_directories(tmp_path: Path):
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    pdf1 = _create_mock_pdf(dir1 / "report.pdf", ["Report from first branch."])
    pdf2 = _create_mock_pdf(dir2 / "report.pdf", ["Report from second branch."])

    chunks = load_multiple_pdfs([pdf1, pdf2])
    assert len(chunks) == 2

    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == 2
    assert ids[0] == "report_p1_c001"
    assert ids[1] == "report_2_p1_c001"


def test_directory_discovery(tmp_path: Path):
    docs_dir = tmp_path / "documents"
    _create_mock_pdf(docs_dir / "alpha.pdf", ["Alpha content."])
    _create_mock_pdf(docs_dir / "beta.pdf", ["Beta content."])
    (docs_dir / "notes.txt").write_text("Not a pdf file.")

    chunks = load_documents_dir(docs_dir)
    assert len(chunks) == 2
    sources = [c.source for c in chunks]
    assert sources == ["alpha.pdf", "beta.pdf"]


def test_directory_discovery_empty_or_missing(tmp_path: Path):
    assert load_documents_dir(tmp_path / "non_existent") == []
    empty_dir = tmp_path / "empty_folder"
    empty_dir.mkdir()
    assert load_documents_dir(empty_dir) == []


def test_max_pdf_count_limit(tmp_path: Path):
    pdfs = []
    for i in range(6):
        pdfs.append(_create_mock_pdf(tmp_path / f"doc_{i}.pdf", [f"Page {i}"]))

    with pytest.raises(PDFLimitExceededError) as exc_info:
        load_multiple_pdfs(pdfs)
    assert "Maximum allowed is 5" in str(exc_info.value)


def test_max_pdf_size_limit(tmp_path: Path, monkeypatch):
    pdf = _create_mock_pdf(tmp_path / "oversized.pdf", ["Some text."])

    orig_stat = Path.stat

    def mock_stat(self):
        real = orig_stat(self)

        class MockStat:
            st_mode = real.st_mode
            st_size = 26 * 1024 * 1024  # 26 MB

        return MockStat()

    monkeypatch.setattr(Path, "stat", mock_stat)

    with pytest.raises(PDFSizeLimitExceededError) as exc_info:
        load_pdf(pdf)
    assert "exceeds maximum allowed limit of 25 MB" in str(exc_info.value)


def test_unique_chunk_ids_across_all_documents(tmp_path: Path):
    pdfs = [
        _create_mock_pdf(tmp_path / f"paper_{i}.pdf", [f"Page {p} for doc {i}" for p in range(1, 4)])
        for i in range(3)
    ]
    chunks = load_multiple_pdfs(pdfs)
    assert len(chunks) == 9
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_multi_document_vector_store_coexistence_and_persistence(tmp_path: Path):
    pdf_a = _create_mock_pdf(tmp_path / "doc_a.pdf", ["Content of Doc A."])
    pdf_b = _create_mock_pdf(tmp_path / "doc_b.pdf", ["Content of Doc B."])

    chunks = load_multiple_pdfs([pdf_a, pdf_b])
    persist_dir = tmp_path / "store"

    model = DummyEmbeddingModel()
    store = InMemoryVectorStore(
        persist_directory=persist_dir,
        collection_name="test_multi",
        embedding_model=model,
    )
    store.add_chunks(chunks)

    assert store.count() == 2
    docs_map = store.get_indexed_documents()
    assert docs_map == {"doc_a.pdf": 1, "doc_b.pdf": 1}

    # Verify reloading from persisted store
    reloaded_store = InMemoryVectorStore(
        persist_directory=persist_dir,
        collection_name="test_multi",
        embedding_model=DummyEmbeddingModel(),
    )
    assert reloaded_store.count() == 2
    assert reloaded_store.get_indexed_documents() == {"doc_a.pdf": 1, "doc_b.pdf": 1}


def test_task_namespaced_cache_behavior():
    model = EmbeddingModel(api_key="mock_key")
    text = "identical query and document text"

    # Prepopulate namespaced cache keys
    query_vec = [0.2] * model.dimension
    passage_vec = [0.8] * model.dimension

    model._cache[f"retrieval.query:{text}"] = query_vec
    model._cache[f"retrieval.passage:{text}"] = passage_vec

    # Query uses retrieval.query cache
    assert model.embed_query(text) == query_vec

    # Documents use retrieval.passage cache
    doc_results = model.embed_documents([text])
    assert doc_results[0] == passage_vec


def test_cross_document_retrieval(tmp_path: Path):
    pdf_a = _create_mock_pdf(tmp_path / "physics.pdf", ["Quantum mechanics and wave functions."])
    pdf_b = _create_mock_pdf(tmp_path / "history.pdf", ["Ancient Roman Empire and the Senate."])

    chunks = load_multiple_pdfs([pdf_a, pdf_b])

    model = DummyEmbeddingModel()
    store = InMemoryVectorStore(persist_directory=None, embedding_model=model)
    store.add_chunks(chunks)

    retriever = VectorRetriever(vector_store=store, embedding_model=model)

    # Retrieval across all indexed documents returns top-k
    results = retriever.retrieve("Quantum mechanics", top_k=2)
    assert len(results) == 2
    retrieved_sources = [r.source for r in results]
    assert "physics.pdf" in retrieved_sources
    assert "history.pdf" in retrieved_sources


def test_multi_document_prompt_source_rendering():
    strip_a = KnowledgeStrip(
        text="Quantum entanglement allows state correlation.",
        source="physics.pdf",
        page_number=3,
        parent_chunk_id="physics_p3_c001",
        score=0.85,
        position=0,
    )
    strip_b = KnowledgeStrip(
        text="Julius Caesar crossed the Rubicon in 49 BC.",
        source="history.pdf",
        page_number=12,
        parent_chunk_id="history_p12_c002",
        score=0.75,
        position=1,
    )

    prompt = _build_crag_prompt("Explain historical and physical discoveries", [strip_a, strip_b], [])

    assert "[1] Source: physics.pdf | Page: 3 (internal document)" in prompt
    assert "[2] Source: history.pdf | Page: 12 (internal document)" in prompt
    assert "Quantum entanglement allows state correlation." in prompt
    assert "Julius Caesar crossed the Rubicon in 49 BC." in prompt
