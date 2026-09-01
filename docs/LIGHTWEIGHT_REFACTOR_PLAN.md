# Lightweight Refactor Plan

## Current Behavior Audit

### Current APIs
- **Embeddings**: Local PyTorch model `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Store**: ChromaDB (local SQLite/HNSW vector database).
- **Relevance Evaluator**: Local PyTorch CrossEncoder `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Generation**: Google GenAI SDK (Gemini) or Groq.
- **Web Search**: Tavily API.

### Current Data Flow
1. **Ingestion**: PyMuPDF -> DocumentChunks -> Embedded via local `sentence-transformers` -> Stored in ChromaDB.
2. **Retrieval**: Query -> Embedded via local `sentence-transformers` -> ChromaDB cosine distance search -> Top K `RetrievedChunk`s.
3. **Evaluation**: Query + each chunk text -> local CrossEncoder -> Logits mapped to [-1.0, 1.0] relevance scores.
4. **Action Routing**: `s_max` compared against `alpha` and `beta` thresholds -> CORRECT, INCORRECT, or AMBIGUOUS.
5. **Knowledge Refiner**: Breaks chunks into sentence strips -> Scores each strip with CrossEncoder -> Filters out strips below `filter_threshold` -> Recomposes.
6. **Generation**: Prompt -> Gemini/Groq API.

### Current Thresholds
- **Action Router `alpha` / `beta`**: Not hardcoded in the class (passed from outside, usually alpha=0.5, beta=-0.5 or similar depending on pipeline instantiation in `main.py`).
- **Knowledge Refiner `filter_threshold`**: Default `-0.5`.

### Current Dependencies
- `torch`
- `sentence-transformers`
- `chromadb`
- `fastapi`, `pydantic`, `pytest`, `google-genai`, `tavily-python`, `groq`, `pymupdf`

## Refactor Plan

1. **Implement Lightweight Embeddings**:
   - Replace `EmbeddingModel` in `embeddings.py` to use `google-genai` with the `gemini-embedding-2` model.
2. **Replace ChromaDB**:
   - Replace `ChromaVectorStore` with `InMemoryVectorStore` using simple Python lists/dicts and NumPy for cosine similarity.
3. **Replace CrossEncoder**:
   - Update `RelevanceEvaluator` to use the pre-calculated embedding similarity score (which is in `[0, 1]`) mapped or shifted to the expected `[-1.0, 1.0]` range, or rewrite the pipeline to just use similarity scores directly. To preserve the existing CRAG behavior seamlessly, we will map similarity scores into `[-1.0, 1.0]` or just evaluate them based on new similarity-appropriate thresholds. 
4. **Remove Heavy Dependencies**:
   - Remove `torch`, `sentence-transformers`, `chromadb` from `requirements.txt`.
   - Remove model preloading from `Dockerfile`.
5. **Tests & Run**:
   - Update tests to reflect new embedding API and in-memory store.
   - Re-build Docker image.
