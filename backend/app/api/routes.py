"""
HTTP API routes for CorrectRAG.
"""

import os
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.schemas import (
    ChunkSource,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    StripSource,
    TraceSchema,
    WebSource,
)
from app.evaluation.action_router import ActionRouter
from app.evaluation.knowledge_refiner import KnowledgeRefiner
from app.evaluation.relevance_evaluator import RelevanceEvaluator
from app.external.query_rewriter import QueryRewriter
from app.external.web_search import WebSearchClient
from app.generation import GeminiClient, GroqClient, LLMProvider
from app.pipeline.crag_pipeline import CRAGPipeline, CRAGResult
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.retriever import VectorRetriever
from app.retrieval.vector_store import ChromaVectorStore

router = APIRouter()


def _get_llm_client() -> LLMProvider:
    """Resolve and construct the configured LLMProvider.

    Supported LLM_PROVIDER values:
        - "groq" (default): GroqClient with GROQ_API_KEY / GROQ_MODEL
        - "gemini": GeminiClient with GEMINI_API_KEY / GEMINI_MODEL
    """
    provider = os.environ.get("LLM_PROVIDER", "groq").lower().strip()

    if provider == "groq":
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")
        groq_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        return GroqClient(api_key=groq_key, model=groq_model)

    if provider == "gemini":
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
        return GeminiClient(api_key=gemini_key, model=gemini_model)

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{provider}'. Supported providers are: 'groq', 'gemini'."
    )


@lru_cache()
def get_crag_pipeline() -> CRAGPipeline:
    """Dependency factory to construct and cache the production CRAGPipeline.

    Uses LLM_PROVIDER (defaults to Groq; supports Gemini) for generation
    and Tavily (TAVILY_API_KEY / TAVILY_MAX_RESULTS) for web search.
    """
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY environment variable is missing.")
    tavily_max_results = int(os.environ.get("TAVILY_MAX_RESULTS", 5))

    # Resolve ChromaDB directory
    chroma_dir = os.environ.get("CHROMA_PERSIST_DIR")
    if not chroma_dir:
        # Default to <workspace_root>/chroma_data
        root = Path(__file__).resolve().parent.parent.parent.parent
        chroma_dir = str(root / "chroma_data")

    llm_client = _get_llm_client()
    embedding_model = EmbeddingModel()
    vector_store = ChromaVectorStore(
        embedding_model=embedding_model,
        collection_name="correctrag",
        persist_directory=chroma_dir,
    )
    retriever = VectorRetriever(vector_store=vector_store)
    evaluator = RelevanceEvaluator()
    router_component = ActionRouter(alpha=0.5, beta=-0.2)
    refiner = KnowledgeRefiner(evaluator=evaluator)
    query_rewriter = QueryRewriter(llm_client=llm_client)
    web_search = WebSearchClient(api_key=tavily_key, max_results=tavily_max_results)

    return CRAGPipeline(
        retriever=retriever,
        evaluator=evaluator,
        router=router_component,
        refiner=refiner,
        query_rewriter=query_rewriter,
        web_search=web_search,
        llm_client=llm_client,
        top_k=5,
    )


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    """Check API health status."""
    return HealthResponse(status="ok", service="correctrag-api")


@router.post("/query", response_model=QueryResponse, tags=["Query"])
def query(
    request: QueryRequest,
    pipeline: CRAGPipeline = Depends(get_crag_pipeline),
) -> QueryResponse:
    """Execute a Corrective RAG query.

    Args:
        request: QueryRequest containing user question.
        pipeline: Injected CRAGPipeline instance.

    Returns:
        Structured QueryResponse with answer, action, sources, and execution trace.
    """
    try:
        result: CRAGResult = pipeline.run(request.question)
    except Exception as exc:  # noqa: BLE001
        # Safe structured error without exposing secrets or raw internal stack traces
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the query: {exc.__class__.__name__}",
        ) from exc

    # Map CRAGResult to clean API response model
    action_str = (
        result.action.value if hasattr(result.action, "value") else str(result.action)
    )
    trace_action_str = (
        result.trace.action.value
        if hasattr(result.trace.action, "value")
        else str(result.trace.action)
    )

    return QueryResponse(
        answer=result.answer,
        action=action_str,
        query=result.query,
        rewritten_query=result.rewritten_query,
        retrieved_chunks=[
            ChunkSource(
                chunk_id=c.chunk_id,
                source=c.source,
                page_number=c.page_number,
                score=c.score,
                text_snippet=c.text[:200] if c.text else None,
                metadata=c.metadata,
            )
            for c in result.retrieved_chunks
        ],
        relevance_scores=result.relevance_scores,
        refined_strips=[
            StripSource(
                text=s.text,
                source=s.source,
                page_number=s.page_number,
                parent_chunk_id=s.parent_chunk_id,
                position=s.position,
                score=s.score,
                origin="internal",
            )
            for s in result.refined_strips
        ],
        external_strips=[
            StripSource(
                text=s.text,
                source=s.source,
                page_number=s.page_number,
                parent_chunk_id=s.parent_chunk_id,
                position=s.position,
                score=s.score,
                origin="external",
            )
            for s in result.external_strips
        ],
        web_results=[
            WebSource(
                url=w.url,
                title=w.title,
                snippet=w.content[:200] if w.content else "",
                score=float(w.score) if w.score is not None else None,
            )
            for w in result.web_results
        ],
        execution_trace=TraceSchema(
            retrieved_count=result.trace.retrieved_count,
            action=trace_action_str,
            max_relevance_score=result.trace.max_relevance_score,
            web_search_used=result.trace.web_search_used,
            rewritten_query=result.trace.rewritten_query,
            internal_strip_count=result.trace.internal_strip_count,
            external_strip_count=result.trace.external_strip_count,
            final_context_source=result.trace.final_context_source,
        ),
    )
