# CorrectRAG Deployment Lightening Audit

## A. Current heavy components
The build process takes ~41+ minutes and results in a very large image primarily due to:
1. `torch` (PyTorch, pulled via `--extra-index-url` for CPU)
2. `sentence-transformers`
3. `chromadb`
4. Pre-downloaded local ML models in the Dockerfile (`all-MiniLM-L6-v2` and `ms-marco-MiniLM-L-6-v2`)

## B. Their actual purpose
1. **`sentence-transformers/all-MiniLM-L6-v2`**: Used by `EmbeddingModel` to generate dense vector embeddings for documents and queries.
2. **`cross-encoder/ms-marco-MiniLM-L-6-v2`**: Used by `RelevanceEvaluator` to score the relevance of retrieved chunks to the query, acting as the core CRAG evaluator to route actions (CORRECT, INCORRECT, AMBIGUOUS).
3. **`torch` & `sentence-transformers`**: The underlying framework dependencies required to run the local embedding and cross-encoder models.
4. **`chromadb`**: A full-featured vector database used to store embeddings and perform semantic similarity searches.

## C. What can be removed
1. **PyTorch (`torch`)**: Can be completely removed from `requirements.txt`.
2. **`sentence-transformers`**: Can be completely removed.
3. **`chromadb`**: Can be completely removed. For a single PDF (CRAG.pdf), a heavy vector DB is massive overkill. 
4. **Local model caching**: The `RUN python -c "from sentence_transformers..."` pre-loading step in the `Dockerfile` can be entirely deleted.

## D. What can be replaced with APIs
1. **Embeddings**: Replace the local `all-MiniLM-L6-v2` model with an API-based embedding model (e.g., Google's `text-embedding-004`). Since `google-genai` is already integrated for text generation, reusing it for embeddings adds zero new dependencies.
2. **Relevance Evaluator (CrossEncoder)**: Replace the local `ms-marco-MiniLM-L-6-v2` model with an **LLM-as-a-judge**. A lightweight prompt to Gemini asking it to score the relevance of a chunk to the query can replicate the bounded [-1, 1] evaluation required by the CRAG Action Router without needing any local machine learning models.
3. **Vector Store**: Replace ChromaDB with a lightweight, pure-Python in-memory exact search (using standard library lists and math, or at most a lightweight `numpy` array). A single PDF yields ~100-200 chunks, which can be cosine-searched in-memory in less than a millisecond.

## E. Recommended final architecture
The smallest sensible production architecture that preserves all core CRAG functionality (RAG, evaluation, corrective routing, grounded generation) is:

- **Ingestion**: PyMuPDF (extracts and chunks PDF).
- **Embedding Layer**: Google GenAI Embedding API (`text-embedding-004` or similar).
- **Vector Store**: In-memory Python `list`/`dict` (optionally serialized to a simple JSON file for persistence).
- **Retrieval Evaluator**: LLM-as-a-judge via Google GenAI API (few-shot prompted to return a relevance score).
- **Query Rewriter**: Google GenAI API (unchanged).
- **Web Search**: Tavily API (unchanged).
- **Generation**: Google GenAI API (unchanged).
- **Backend Framework**: FastAPI + Pydantic (unchanged).

## F. Estimated impact on image size
- **Current**: Likely 2 GB to 3 GB due to PyTorch, ChromaDB, and pre-downloaded model weights.
- **Projected**: ~250 MB to 350 MB. 
- **Reduction**: ~80-90% reduction in image size. Build time will drop from 41+ minutes to under 2 minutes.

## G. Estimated impact on RAM
- **Current**: Loading a CrossEncoder, SentenceTransformer, and ChromaDB service simultaneously can consume 1.5 GB - 2 GB of RAM at runtime.
- **Projected**: A pure API-based FastAPI app will consume < 150 MB of RAM at idle, spiking only slightly during request handling.

## H. AWS deployment recommendation
With the dramatically lightened architecture, the application becomes highly suitable for cost-effective, serverless container environments:
- **AWS App Runner**: Ideal for simple web services. A basic instance (1 vCPU, 2 GB RAM) will comfortably run this API.
- **AWS ECS (Fargate)**: You can deploy on the smallest Fargate task size (0.25 vCPU, 0.5 GB RAM) for minimal cost.
- **AWS Lambda**: Using AWS Web Adapter, the FastAPI app could even run as a serverless Lambda function (charged only per request), which is perfect for portfolio/demo deployments.

## I. Any functionality that would be lost
1. **Local Offline Capabilities**: The system will become fully reliant on external APIs (Gemini & Tavily) for embeddings and evaluation. It will no longer function in an offline environment.
2. **Evaluator Score Distribution**: An LLM-as-a-judge will have a different score distribution compared to the deterministic logits of the MS MARCO cross-encoder. Thresholds ($\alpha$, $\beta$, $\gamma$) in the Action Router and Knowledge Refiner may need minor recalibration to match the LLM's output behavior.
3. **Slightly Increased API Latency**: Making two additional network API calls per request (one for embeddings, one for LLM evaluation) will slightly increase the overall response time compared to fast local CPU inference, though this is an excellent tradeoff for a massive reduction in RAM and build complexity.

---

**FINAL VERDICT:**
LIGHTEN BEFORE AWS
