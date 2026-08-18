# CorrectRAG

CorrectRAG is a production-oriented implementation of the Corrective Retrieval Augmented Generation framework designed to mitigate hallucinations in retrieval-augmented language models. By evaluating retrieved document quality prior to generation, the system dynamically routes queries across three confidence states—refining high-confidence internal documents into clean knowledge strips, discarding low-confidence results in favor of rewritten external web searches, or combining both when retrieval relevance is ambiguous.

## 📄 Research Paper

- **Paper Title**: Corrective Retrieval Augmented Generation
- **Authors**: Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling (USTC / UCLA / Google DeepMind)
- **Reference**: [arXiv:2401.15884v3](https://arxiv.org/abs/2401.15884) [cs.CL]
- **Document Reference**: [paper_spec.md](paper_spec.md)

## 📌 Project Status

**Current Milestone**: `Semantic Retrieval Pipeline Implemented`

Implemented Pipeline:
$$\text{PDF Document} \longrightarrow \text{Page Extraction \& Cleaning} \longrightarrow \text{Deterministic Chunking} \longrightarrow \text{Sentence Transformers Embeddings} \longrightarrow \text{ChromaDB Indexing} \longrightarrow \text{Semantic Vector Retrieval}$$

## 🏗️ Architecture & Modules

### 1. Ingestion Layer (`app.ingestion`)
- **`pdf_loader.py`**: Extracts text page-by-page with PyMuPDF, cleans extraction artifacts (zero-width characters, line breaks, hyphenated breaks), and applies deterministic sliding-window chunking with boundary snapping (`chunk_id = {stem}_p{page}_c{idx}`).

### 2. Retrieval Layer (`app.retrieval`)
- **`embeddings.py`**: Centralized wrapper for `sentence-transformers/all-MiniLM-L6-v2` with lazy model caching and batch document / single query embedding methods.
- **`vector_store.py`**: ChromaDB persistence wrapper supporting document deduplication via deterministic `chunk_id` keys, metadata preservation, and collection clearing.
- **`retriever.py`**: `VectorRetriever` scoring and returning ranked `RetrievedChunk` results with cosine similarity scores and complete page/source provenance.

### 3. Planned Next Steps (CRAG Pipeline)
- **Retrieval Evaluator**: Confidence scoring module estimating relevance degrees in $[-1, 1]$.
- **Action Trigger**: `{CORRECT, INCORRECT, AMBIGUOUS}` three-way decision routing.
- **Knowledge Refinement**: Fine-grained strip decomposition, filtering, and recomposition.
- **Web Search & Query Rewriting**: External fallback augmentation for missing knowledge.
- **Generation Layer**: Final conditioned LLM answering with audit trails.

---

## 🛠️ Frozen Production Stack

- **PDF Ingestion**: PyMuPDF (`pymupdf`) + Pydantic
- **Vector Store & Embeddings**: ChromaDB + `sentence-transformers/all-MiniLM-L6-v2`
- **Backend Service**: FastAPI + Uvicorn

---

## 🚀 Running the Project

### 1. Run Unit & Integration Tests
```bash
pytest -v
```

### 2. Run Manual Ingestion & Retrieval Demo
```bash
python scripts/demo_retrieval.py [optional_path_to_pdf]
```
