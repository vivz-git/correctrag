# CorrectRAG

CorrectRAG is a production-oriented implementation of the **Corrective Retrieval Augmented Generation** framework, designed to mitigate hallucinations in retrieval-augmented language models. By evaluating retrieved document quality prior to generation, the system dynamically routes queries across three confidence states: refining high-confidence internal documents, discarding low-confidence results in favor of rewritten web searches, or combining both when retrieval relevance is ambiguous.

> **Status**: Baseline RAG pipeline implemented. CRAG evaluator, routing, and knowledge refinement are **not yet implemented**.

---

## 📄 Research Paper

- **Title**: Corrective Retrieval Augmented Generation
- **Authors**: Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling (USTC / UCLA / Google DeepMind)
- **Reference**: [arXiv:2401.15884v3](https://arxiv.org/abs/2401.15884)
- **Spec Document**: [paper_spec.md](paper_spec.md)

---

## 🏗️ Architecture

### Baseline RAG (Current)

```
PDF Document
  → Page Extraction & Cleaning   (PyMuPDF)
  → Deterministic Chunking       (page-bounded, boundary-snapped)
  → Sentence Transformer Embeds  (all-MiniLM-L6-v2 / 384-dim)
  → ChromaDB Indexing            (cosine similarity)
  → Semantic Vector Retrieval    (top-K chunks + scores)
  → Prompt Construction          (grounded, source-cited)
  → Gemini 3.6 Flash             (Google GenAI SDK)
  → Grounded Answer + Citations
```

> This is **standard RAG**. The CRAG evaluator, CORRECT/INCORRECT/AMBIGUOUS routing, knowledge refinement strips, and web search fallback are the **next milestones**.

### Planned CRAG Pipeline (Next)

```
Query
  → Retrieval Evaluator (confidence scoring)
      ├── CORRECT    → Knowledge Refinement → Generation
      ├── INCORRECT  → Query Rewriting → Web Search → Generation
      └── AMBIGUOUS  → Both paths combined → Generation
```

---

## 🛠️ Stack

| Layer | Technology |
|---|---|
| PDF Ingestion | PyMuPDF (`pymupdf`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (persistent + ephemeral) |
| LLM | Google Gemini via `google-genai` SDK |
| API Framework | FastAPI + Uvicorn |
| Data Models | Pydantic v2 |

---

## 📁 Project Structure

```
correctrag/
├── paper_spec.md               # CRAG paper analysis & engineering spec
├── .env.example                # Environment variable template
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py             # FastAPI app (GET /health)
│       ├── ingestion/
│       │   └── pdf_loader.py   # PDF → DocumentChunk pipeline
│       ├── retrieval/
│       │   ├── embeddings.py   # EmbeddingModel (SentenceTransformers)
│       │   ├── vector_store.py # ChromaVectorStore wrapper
│       │   └── retriever.py    # VectorRetriever → RetrievedChunk
│       └── generation/
│           ├── gemini_client.py  # Google GenAI SDK wrapper
│           ├── prompt.py         # RAG prompt builder
│           └── rag_pipeline.py   # BaselineRAG pipeline + RAGResult
├── scripts/
│   ├── demo_retrieval.py       # Retrieval-only demo
│   └── demo_rag.py             # Full RAG demo (retrieval + generation)
├── tests/
│   ├── test_pdf_loader.py      # 14 ingestion tests
│   ├── test_retrieval.py       # 10 retrieval tests
│   └── test_generation.py      # 16 generation tests (mocked LLM)
├── evaluation/                 # Placeholder for CRAG evaluation harness
├── configs/
├── docs/
└── frontend/
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env — set GEMINI_API_KEY to your Google AI Studio key
```

### 3. Run Tests (No API Key Required)

```bash
pytest -v
```

### 4. Run Manual Demos

```bash
# Retrieval only (no API key needed)
python scripts/demo_retrieval.py CRAG.pdf

# Full Baseline RAG (requires GEMINI_API_KEY)
python scripts/demo_rag.py CRAG.pdf

# Without a PDF — uses synthetic sample documents
python scripts/demo_rag.py
```

---

## 🔑 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes (for generation) | — | Google AI Studio API key |
| `GEMINI_MODEL` | No | `gemini-3.6-flash` | Gemini model identifier |

---

## 📊 Data Flow (Baseline RAG)

```
User Query
    │
    ▼
VectorRetriever.retrieve(top_k=5)
    │  embeds query → ChromaDB cosine search
    │
    ▼
[RetrievedChunk × top_k]
    │  text, source, page_number, similarity score
    │
    ▼
build_rag_prompt(question, chunks)
    │  numbered context passages + system instruction
    │
    ▼
GeminiClient.generate(prompt)
    │  Google GenAI SDK → gemini-3.6-flash
    │
    ▼
RAGResult
    ├── answer          (generated text)
    ├── sources         (deduplicated citations: source, page, chunk_id)
    └── retrieved_chunks (raw chunks with scores)
```

---

## ✅ Test Coverage

| Suite | Tests | Coverage |
|---|---|---|
| `test_pdf_loader.py` | 14 | PDF extraction, cleaning, chunking, edge cases |
| `test_retrieval.py` | 10 | Embeddings, vector store, retriever ranking |
| `test_generation.py` | 16 | Prompt formatting, Gemini config, RAG pipeline (mocked) |
| **Total** | **40** | |
