# CorrectRAG

CorrectRAG is a practical, lightweight engineering adaptation of the **Corrective Retrieval Augmented Generation (CRAG)** framework. 

It is designed to be a clear, explainable, and production-oriented portfolio project that demonstrates how to implement intelligent semantic routing in RAG pipelines.

> **Note**: This project does NOT attempt to reproduce the original CRAG research paper exactly. Instead, it adapts the core concepts (evaluate retrieval, route actions, use web fallback) into a modern, lightweight, API-driven application architecture without heavy PyTorch dependencies.

## Key Features & Architecture

Our adaptation replaces the heavy local ML dependencies (PyTorch, sentence-transformers, CrossEncoders, ChromaDB) with a fast, lightweight stack:

1. **Lightweight Hosted Embeddings**: Uses the Gemini Embedding API (`gemini-embedding-2`) instead of heavy local models.
2. **In-Memory Retrieval**: Uses a pure Python deterministic cosine similarity vector store.
3. **Cheap Similarity Pre-filter**: Instantly routes clearly relevant or irrelevant queries using fast embedding math.
4. **LLM Judgment for Borderline Cases**: Calls a `openai/gpt-oss-120b` LLM Judge (via Groq) *only* when retrieval quality is borderline, saving cost and latency.
5. **Corrective Routing**: Routes to CORRECT, INCORRECT, or AMBIGUOUS branches based on the two-stage evaluation.
6. **Optional Web Fallback**: Seamlessly queries Tavily to rewrite searches and inject web knowledge on INCORRECT or AMBIGUOUS routes.
7. **Grounded Generation**: Uses Groq, OpenAI, or Gemini to synthesize the final grounded response.

## CRAG Pipeline Workflow

```
Query
  → Semantic Retrieval (top-K chunks from PDF)
  → Two-Stage Relevance Evaluator
      1. Cheap Similarity Filter (Gemini embeddings)
      2. LLM Judge (GPT-OSS 120B for borderline cases)
  → Action Router (uses Judge decision)
      ├── CORRECT    → Knowledge Refinement → Generation
      ├── INCORRECT  → Query Rewriting → Web Search → Generation
      └── AMBIGUOUS  → Combine Internal + Web Search → Generation
```

The LLM judge returns a strict JSON decision with a reason. It *only* evaluates retrieval quality; it does not generate the final answer.

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, Pydantic, pytest
- **Embeddings**: Gemini (`gemini-embedding-2`)
- **Judge LLM**: Groq (`openai/gpt-oss-120b` by default, configurable via `GROQ_MODEL`)
- **Generation LLM**: Gemini / Groq / OpenAI (configurable)
- **Web Search**: Tavily API
- **Deployment**: Docker Compose (multi-architecture)

## Local Development Setup

### 1. Requirements
- Python 3.10+
- Valid API keys for Gemini, Groq, and Tavily

### 2. Environment Variables
Create a `.env` file in the project root:
```bash
GEMINI_API_KEY="your-gemini-key"
GROQ_API_KEY="your-groq-key"
TAVILY_API_KEY="your-tavily-key"
```

### 3. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 4. Run the Server
```bash
uvicorn backend.app.main:app --reload --port 8000
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

## Docker Deployment

CorrectRAG includes a containerized multi-service setup.

```bash
docker compose up --build
```
This starts the backend API on port 8000 and the browser UI on port 3000.
The resulting container is heavily optimized (linux/arm64 is ~208MB) and offers instant `/health` readiness.

## Test Coverage

The project is backed by a robust, 303-test suite covering everything from PDF ingestion and vector similarity math to LLM parsing and pipeline orchestrations.

To run the tests locally:
```bash
pytest -v tests
```

---

*This is a portfolio project intended to demonstrate professional API engineering, testing, and modern GenAI patterns.*
