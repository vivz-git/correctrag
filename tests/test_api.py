"""
Offline unit tests for CorrectRAG FastAPI HTTP endpoints.

All tests use fully mocked CRAGPipeline / dependencies.
No real Gemini, Groq, or Tavily calls occur during testing.
"""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_crag_pipeline
from app.evaluation.knowledge_refiner import KnowledgeStrip
from app.external.web_search import WebSearchResult
from app.main import app
from app.pipeline.crag_pipeline import CRAGPipeline, CRAGResult, ExecutionTrace
from app.retrieval.retriever import RetrievedChunk


@pytest.fixture
def mock_pipeline() -> MagicMock:
    """Mock CRAGPipeline fixture."""
    return MagicMock(spec=CRAGPipeline)


@pytest.fixture(autouse=True)
def override_pipeline_dependency(mock_pipeline: MagicMock):
    """Automatically mock the get_crag_pipeline dependency for all tests."""
    app.dependency_overrides[get_crag_pipeline] = lambda: mock_pipeline
    yield mock_pipeline
    app.dependency_overrides.clear()


@pytest.fixture
def client() -> TestClient:
    """TestClient fixture for FastAPI app."""
    return TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Health Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_get_health(client: TestClient) -> None:
    """GET /health returns 200 with status ok and service identifier."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "correctrag-api"


def test_cors_headers_wildcard_no_credentials(client: TestClient) -> None:
    """CORS middleware returns wildcard origin and does not set allow-credentials."""
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"
    assert "access-control-allow-credentials" not in response.headers


# ─────────────────────────────────────────────────────────────────────────────
# 2. Query Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_question_rejected(client: TestClient) -> None:
    """POST /query with empty string is rejected with 422 validation error."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_whitespace_question_rejected(client: TestClient) -> None:
    """POST /query with whitespace-only question is rejected with 422."""
    response = client.post("/query", json={"question": "   \n\t   "})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_missing_question_field_rejected(client: TestClient) -> None:
    """POST /query without question field is rejected with 422."""
    response = client.post("/query", json={})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_non_string_question_rejected(client: TestClient) -> None:
    """POST /query with non-string question is rejected with 422."""
    response = client.post("/query", json={"question": 12345})
    assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 3. Successful Query Execution Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_successful_query_correct_action(client: TestClient, mock_pipeline: MagicMock) -> None:
    """POST /query returns structured answer, action, and trace for CORRECT action."""
    mock_result = CRAGResult(
        answer="Corrective Retrieval-Augmented Generation (CRAG) is a framework...",
        action="CORRECT",
        query="What is CRAG?",
        rewritten_query=None,
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="chunk-001",
                text="CRAG is designed to improve RAG robustness.",
                source="CRAG.pdf",
                page_number=1,
                score=0.85,
                metadata={"title": "CRAG Overview"},
            )
        ],
        relevance_scores=[0.85],
        refined_strips=[
            KnowledgeStrip(
                text="CRAG improves RAG robustness.",
                source="CRAG.pdf",
                page_number=1,
                parent_chunk_id="chunk-001",
                position=0,
                score=0.85,
            )
        ],
        external_strips=[],
        web_results=[],
        trace=ExecutionTrace(
            retrieved_count=1,
            action="CORRECT",
            max_relevance_score=0.85,
            web_search_used=False,
            rewritten_query=None,
            internal_strip_count=1,
            external_strip_count=0,
            final_context_source="internal",
        ),
    )
    mock_pipeline.run.return_value = mock_result

    response = client.post("/query", json={"question": "What is CRAG?"})
    assert response.status_code == 200
    data = response.json()

    assert data["answer"] == mock_result.answer
    assert data["action"] == "CORRECT"
    assert data["query"] == "What is CRAG?"
    assert data["rewritten_query"] is None
    assert len(data["retrieved_chunks"]) == 1
    assert data["retrieved_chunks"][0]["chunk_id"] == "chunk-001"
    assert data["retrieved_chunks"][0]["source"] == "CRAG.pdf"
    assert data["relevance_scores"] == [0.85]
    assert len(data["refined_strips"]) == 1
    assert data["refined_strips"][0]["text"] == "CRAG improves RAG robustness."
    assert data["execution_trace"]["action"] == "CORRECT"
    assert data["execution_trace"]["web_search_used"] is False
    assert data["execution_trace"]["final_context_source"] == "internal"

    mock_pipeline.run.assert_called_once_with("What is CRAG?")


def test_successful_query_ambiguous_action(client: TestClient, mock_pipeline: MagicMock) -> None:
    """POST /query preserves web results and combined source trace for AMBIGUOUS action."""
    mock_result = CRAGResult(
        answer="ChatGPT is an LLM combined with RAG systems...",
        action="AMBIGUOUS",
        query="What is ChatGPT and how does it relate to RAG?",
        rewritten_query="ChatGPT, RAG systems, relationship",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="chunk-007",
                text="ChatGPT is listed alongside LLaMA2...",
                source="CRAG.pdf",
                page_number=7,
                score=0.31,
                metadata={},
            )
        ],
        relevance_scores=[0.31],
        refined_strips=[
            KnowledgeStrip(
                text="ChatGPT is listed alongside LLaMA2...",
                source="CRAG.pdf",
                page_number=7,
                parent_chunk_id="chunk-007",
                position=0,
                score=0.31,
            )
        ],
        external_strips=[
            KnowledgeStrip(
                text="ChatGPT can use retrieval to augment answers.",
                source="https://example.com/chatgpt-rag",
                page_number=1,
                parent_chunk_id="web-chunk-001",
                position=0,
                score=0.75,
            )
        ],
        web_results=[
            WebSearchResult(
                title="ChatGPT and RAG",
                url="https://example.com/chatgpt-rag",
                content="ChatGPT can use retrieval to augment answers.",
                score=0.75,
            )
        ],
        trace=ExecutionTrace(
            retrieved_count=1,
            action="AMBIGUOUS",
            max_relevance_score=0.31,
            web_search_used=True,
            rewritten_query="ChatGPT, RAG systems, relationship",
            internal_strip_count=1,
            external_strip_count=1,
            final_context_source="combined",
        ),
    )
    mock_pipeline.run.return_value = mock_result

    response = client.post(
        "/query", json={"question": "What is ChatGPT and how does it relate to RAG?"}
    )
    assert response.status_code == 200
    data = response.json()

    assert data["action"] == "AMBIGUOUS"
    assert data["rewritten_query"] == "ChatGPT, RAG systems, relationship"
    assert len(data["web_results"]) == 1
    assert data["web_results"][0]["url"] == "https://example.com/chatgpt-rag"
    assert len(data["external_strips"]) == 1
    assert data["execution_trace"]["web_search_used"] is True
    assert data["execution_trace"]["final_context_source"] == "combined"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Error Handling Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_error_becomes_safe_http_500(
    client: TestClient, mock_pipeline: MagicMock
) -> None:
    """Pipeline runtime errors are caught and converted to safe 500 responses."""
    mock_pipeline.run.side_effect = RuntimeError("Upstream rate limit: secret_api_key_12345")

    response = client.post("/query", json={"question": "Any question?"})
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    # Ensure secret API keys / sensitive messages are NOT leaked
    assert "secret_api_key" not in data["detail"]
    assert "RuntimeError" in data["detail"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. LLM Provider Factory Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_provider_selection_groq_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default LLM_PROVIDER (or explicitly 'groq') resolves to GroqClient."""
    from app.api.routes import _get_llm_client
    from app.generation.groq_client import GroqClient

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_12345")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    client = _get_llm_client()
    assert isinstance(client, GroqClient)
    assert client.model == "openai/gpt-oss-120b"

    # Explicit 'groq' with custom model
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    client_custom = _get_llm_client()
    assert isinstance(client_custom, GroqClient)
    assert client_custom.model == "llama-3.3-70b-versatile"


def test_provider_selection_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM_PROVIDER=gemini resolves to GeminiClient."""
    from app.api.routes import _get_llm_client
    from app.generation.gemini_client import GeminiClient

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key_67890")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.6-flash")

    client = _get_llm_client()
    assert isinstance(client, GeminiClient)
    assert client.model == "gemini-3.6-flash"


def test_provider_selection_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported LLM_PROVIDER value raises a clear ValueError."""
    from app.api.routes import _get_llm_client

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER 'anthropic'"):
        _get_llm_client()


def test_provider_selection_groq_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM_PROVIDER=groq without GROQ_API_KEY raises a clear ValueError."""
    from app.api.routes import _get_llm_client

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GROQ_API_KEY environment variable is missing"):
        _get_llm_client()


def test_provider_selection_gemini_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM_PROVIDER=gemini without GEMINI_API_KEY raises a clear ValueError."""
    from app.api.routes import _get_llm_client

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is missing"):
        _get_llm_client()

