# CorrectRAG

CorrectRAG is a production-oriented implementation of the **Corrective Retrieval Augmented Generation** framework, designed to mitigate hallucinations in retrieval-augmented language models. By evaluating retrieved document quality prior to generation, the system dynamically routes queries across three confidence states: refining high-confidence internal documents, discarding low-confidence results in favor of rewritten web searches, or combining both when retrieval relevance is ambiguous.

> **Status**: Baseline RAG + Retrieval Evaluator + Action Router + Knowledge Refinement implemented. Web search fallback and full CRAG pipeline orchestration are **not yet implemented**.

---

## 📄 Research Paper

- **Title**: Corrective Retrieval Augmented Generation
- **Authors**: Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling (USTC / UCLA / Google DeepMind)
- **Reference**: [arXiv:2401.15884v3](https://arxiv.org/abs/2401.15884)
- **Spec Document**: [paper_spec.md](paper_spec.md)

---

## 🏗️ Architecture

### Baseline RAG

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

### CRAG Pipeline (In Progress)

```
Query
  → Semantic Retrieval (top-K chunks)
  → Relevance Evaluator (score each chunk in [-1, 1])
  → Action Router (s_max vs. alpha/beta)
      ├── CORRECT    → Knowledge Refinement → Generation
      ├── INCORRECT  → Query Rewriting → Web Search → Generation
      └── AMBIGUOUS  → Combined Internal + Web Knowledge → Generation
```

---

## 🔍 CRAG Retrieval Evaluator

### Why an evaluator?

Standard RAG blindly trusts retrieved documents. CRAG adds a **retrieval evaluator** that scores each retrieved document for relevance to the query before deciding how to use it. A low-confidence retrieval triggers a different action (web search fallback) than a high-confidence one.

### What the paper uses

> **[PAPER]** The original CRAG paper (arXiv:2401.15884v3) trains a **T5-large** model as the retrieval evaluator, fine-tuned specifically to classify query-document relevance into three confidence bands: *Correct*, *Incorrect*, and *Ambiguous*.

### What we use (our adaptation)

> **[OUR ADAPTATION]** We use the frozen cross-encoder **`cross-encoder/ms-marco-MiniLM-L-6-v2`** (Sentence-Transformers) as a practical production substitute. This model was not fine-tuned for CRAG — it is a general MS MARCO passage-ranking model. Raw logits are mapped to **[-1, 1]** via a temperature-scaled sigmoid:
>
> ```
> p     = sigmoid(logit / temperature)
> score = 2 * p - 1          →  [-1.0, +1.0]
> ```
>
> This is an approximation. The resulting scores are **not equivalent** to the paper's T5 evaluator output.

### Calibration status

Calibration parameters (`temperature`) are **development defaults only** — they have not been fitted on a validation set. Scientific calibration requires held-out query-document pairs with human-labelled relevance judgements and is a future milestone.

---

## 🔀 CRAG Action Router

### Role of the Action Router

The **Action Router** takes the individual relevance scores assigned by the evaluator to all retrieved documents and applies the CRAG decision rule to select one of three actions:

$$\text{Action} = \begin{cases} \text{CORRECT}, & \text{if } s_{\max} > \alpha \\ \text{INCORRECT}, & \text{if } s_{\max} < \beta \\ \text{AMBIGUOUS}, & \text{if } \beta \le s_{\max} \le \alpha \end{cases}$$

Where $s_{\max} = \max_{i} (\text{score}_i)$, $\alpha$ is the upper confidence threshold, and $\beta$ is the lower confidence threshold (with $-1.0 \le \beta < \alpha \le 1.0$).

### Pure Decision Logic

The router contains **pure decision logic** only:
- It receives a list of document relevance scores and outputs `"CORRECT"`, `"INCORRECT"`, or `"AMBIGUOUS"`.
- It does not perform retrieval, refinement, or generation.

---

## ✂️ CRAG Knowledge Refinement

### Why refine retrieved knowledge?

Coarse retrieved chunks often contain irrelevant sentences alongside relevant facts. Feeding noisy chunks directly into the generator can cause hallucinations. CRAG refines internal knowledge through a multi-stage filtering and recomposition process:

$$\text{Retrieved Chunks} \longrightarrow \text{Decompose into Strips} \longrightarrow \text{Score Strips} \longrightarrow \text{Filter by } \gamma \longrightarrow \text{Rank Top-}K \longrightarrow \text{Recompose Order}$$

### Step-by-step workflow

1. **Decompose**: Splits retrieved document chunks into fine-grained strips (targeting 1–3 sentences).
2. **Score**: Evaluates each strip independently using `RelevanceEvaluator` to assign $s_i = \text{Evaluator}(\text{query}, \text{strip}_i)$.
3. **Filter**: Discards any strip where $s_i < \gamma$ (where $\gamma$ is the configurable `filter_threshold`).
4. **Rank & Select**: Sorts surviving strips by score and selects at most `top_k` strips.
5. **Recompose**: Restores the selected strips to their original document sequence order (`position`), preserving natural reading flow.

> **[NOTE]** The paper used $\gamma = -0.5$ tuned for their T5 evaluator. In our implementation, `filter_threshold` is configurable because our cross-encoder evaluator operates with a different score distribution.

---

## 🛠️ Stack

| Layer | Technology |
|---|---|
| PDF Ingestion | PyMuPDF (`pymupdf`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieval Evaluator | `cross-encoder/ms-marco-MiniLM-L-6-v2` \[OUR ADAPTATION\] |
| Action Router | Pure Python Threshold Decision Logic \[PAPER\] |
| Knowledge Refiner | Fine-Grained Strip Extraction & Filtering \[PAPER / ADAPTATION\] |
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
│       ├── generation/
│       │   ├── gemini_client.py  # Google GenAI SDK wrapper
│       │   ├── prompt.py         # RAG prompt builder
│       │   └── rag_pipeline.py   # BaselineRAG pipeline + RAGResult
│       └── evaluation/
│           ├── relevance_evaluator.py  # Query-document relevance scorer
│           ├── action_router.py        # Three-way action trigger
│           └── knowledge_refiner.py    # Strip decomposition & refinement
├── scripts/
│   ├── demo_retrieval.py       # Retrieval-only demo
│   └── demo_rag.py             # Full RAG demo (retrieval + generation)
├── tests/
│   ├── test_pdf_loader.py          # PDF extraction, cleaning, chunking
│   ├── test_retrieval.py           # Embeddings, vector store, retriever
│   ├── test_generation.py          # Prompt formatting, Gemini config, RAG pipeline
│   ├── test_relevance_evaluator.py # Evaluator score mapping & caching
│   ├── test_action_router.py       # Three-way action routing & threshold rules
│   └── test_knowledge_refiner.py   # Strip decomposition, filtering, recomposition
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

| Suite | Focus Area |
|---|---|
| `test_pdf_loader.py` | PDF extraction, cleaning, deterministic chunking, edge cases |
| `test_retrieval.py` | Embeddings, vector store persistence/deduplication, retriever ranking |
| `test_generation.py` | Prompt formatting, Gemini config, RAG pipeline (mocked LLM) |
| `test_relevance_evaluator.py` | Score mapping, input validation, batch processing, model caching |
| `test_action_router.py` | Threshold validation, boundary logic, multi-score routing, determinism |
| `test_knowledge_refiner.py` | Strip decomposition, score evaluation, threshold filtering, order recomposition |
