"""
Tests for production hardening: per-IP rate limiting and request timeout
protection on POST /query.

The rate-limiting tests build a fresh FastAPI app instance (via
`create_app`) instead of reusing the shared `app` singleton imported by
other test modules. This keeps the limiter's in-memory state isolated per
test and avoids interference with (or from) other tests that also exercise
POST /query against the shared app.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.api.middleware import (
    QUERY_TIMEOUT_SECONDS,
    RATE_LIMIT_MAX_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    FixedWindowRateLimiter,
    RateLimitMiddleware,
    TimeoutMiddleware,
    get_client_ip,
)
from app.api.routes import get_crag_pipeline
from app.main import create_app
from app.pipeline.crag_pipeline import CRAGPipeline, CRAGResult, ExecutionTrace


def _mock_crag_result() -> CRAGResult:
    return CRAGResult(
        answer="Test answer.",
        action="CORRECT",
        query="Test question?",
        rewritten_query=None,
        retrieved_chunks=[],
        relevance_scores=[],
        refined_strips=[],
        external_strips=[],
        web_results=[],
        trace=ExecutionTrace(
            retrieved_count=0,
            action="CORRECT",
            max_relevance_score=0.0,
            web_search_used=False,
            rewritten_query=None,
            internal_strip_count=0,
            external_strip_count=0,
            final_context_source="internal",
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. FixedWindowRateLimiter unit tests (pure Python, no HTTP layer)
# ─────────────────────────────────────────────────────────────────────────────


def test_limiter_allows_requests_within_limit() -> None:
    """Requests up to and including max_requests are allowed."""
    limiter = FixedWindowRateLimiter(max_requests=3, window_seconds=60)
    assert limiter.hit("1.2.3.4") is True
    assert limiter.hit("1.2.3.4") is True
    assert limiter.hit("1.2.3.4") is True


def test_limiter_rejects_requests_exceeding_limit() -> None:
    """The (max_requests + 1)-th request in the same window is rejected."""
    limiter = FixedWindowRateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        assert limiter.hit("1.2.3.4") is True
    assert limiter.hit("1.2.3.4") is False
    # Continues to reject further requests in the same window.
    assert limiter.hit("1.2.3.4") is False


def test_limiter_tracks_clients_independently() -> None:
    """Two different keys have independent budgets."""
    limiter = FixedWindowRateLimiter(max_requests=1, window_seconds=60)
    assert limiter.hit("client-a") is True
    assert limiter.hit("client-a") is False
    assert limiter.hit("client-b") is True


def test_limiter_resets_on_new_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """A client that was rate-limited is allowed again once the window rolls over."""
    limiter = FixedWindowRateLimiter(max_requests=1, window_seconds=60)

    monkeypatch.setattr("app.api.middleware.time.time", lambda: 0.0)
    assert limiter.hit("1.2.3.4") is True
    assert limiter.hit("1.2.3.4") is False

    # Advance past the 60s window.
    monkeypatch.setattr("app.api.middleware.time.time", lambda: 61.0)
    assert limiter.hit("1.2.3.4") is True


def test_limiter_bucket_count_does_not_grow_unbounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Buckets from expired windows are pruned, bounding memory growth."""
    limiter = FixedWindowRateLimiter(max_requests=5, window_seconds=60)

    monkeypatch.setattr("app.api.middleware.time.time", lambda: 0.0)
    for i in range(200):
        limiter.hit(f"client-{i}")
    assert limiter.bucket_count() == 200

    # Move to a new window and make one more request; all 200 stale
    # per-client buckets from the previous window should be dropped.
    monkeypatch.setattr("app.api.middleware.time.time", lambda: 61.0)
    limiter.hit("client-new")
    assert limiter.bucket_count() == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. get_client_ip resolution
# ─────────────────────────────────────────────────────────────────────────────


def test_get_client_ip_prefers_forwarded_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """X-Forwarded-For (set by the trusted Caddy proxy) is used when present."""

    class DummyRequest:
        headers = {"x-forwarded-for": "203.0.113.5, 10.0.0.1"}
        client = None

    assert get_client_ip(DummyRequest()) == "203.0.113.5"


def test_get_client_ip_falls_back_to_peer_address() -> None:
    """Without X-Forwarded-For, the raw connection peer address is used."""

    class DummyClient:
        host = "127.0.0.1"

    class DummyRequest:
        headers: dict = {}
        client = DummyClient()

    assert get_client_ip(DummyRequest()) == "127.0.0.1"


# ─────────────────────────────────────────────────────────────────────────────
# 3. HTTP-level behavior against an isolated app instance
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_client() -> TestClient:
    """A fresh FastAPI app (own middleware state) with a mocked CRAG pipeline."""
    from unittest.mock import MagicMock

    app = create_app()
    mock_pipeline = MagicMock(spec=CRAGPipeline)
    mock_pipeline.run.return_value = _mock_crag_result()
    app.dependency_overrides[get_crag_pipeline] = lambda: mock_pipeline
    return TestClient(app)


def test_query_within_rate_limit_succeeds(isolated_client: TestClient) -> None:
    """A single POST /query well within the limit succeeds."""
    response = isolated_client.post("/query", json={"question": "What is CRAG?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Test answer."


def test_query_exceeding_rate_limit_returns_429(isolated_client: TestClient) -> None:
    """Exceeding RATE_LIMIT_MAX_REQUESTS in the same window returns HTTP 429."""
    for _ in range(RATE_LIMIT_MAX_REQUESTS):
        response = isolated_client.post("/query", json={"question": "What is CRAG?"})
        assert response.status_code == 200

    limited_response = isolated_client.post("/query", json={"question": "What is CRAG?"})
    assert limited_response.status_code == 429
    assert "Rate limit exceeded" in limited_response.json()["detail"]


def test_health_not_rate_limited(isolated_client: TestClient) -> None:
    """GET /health remains accessible even after exhausting the /query rate limit."""
    for _ in range(RATE_LIMIT_MAX_REQUESTS + 5):
        isolated_client.post("/query", json={"question": "What is CRAG?"})

    response = isolated_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rate_limit_response_has_cors_headers(isolated_client: TestClient) -> None:
    """The 429 response still carries CORS headers (CORS wraps the guards)."""
    for _ in range(RATE_LIMIT_MAX_REQUESTS):
        isolated_client.post(
            "/query", json={"question": "What is CRAG?"}, headers={"Origin": "http://localhost:3000"}
        )

    limited_response = isolated_client.post(
        "/query", json={"question": "What is CRAG?"}, headers={"Origin": "http://localhost:3000"}
    )
    assert limited_response.status_code == 429
    assert limited_response.headers.get("access-control-allow-origin") == "*"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Timeout middleware
# ─────────────────────────────────────────────────────────────────────────────


def test_timeout_returns_controlled_504_response() -> None:
    """A guarded request that runs longer than the timeout gets a 504, not a hang."""

    async def slow_call_next(request):
        await asyncio.sleep(10)
        return None  # pragma: no cover - never reached

    class DummyURL:
        path = "/query"

    class DummyRequest:
        method = "POST"
        url = DummyURL()

    middleware = TimeoutMiddleware(app=None, timeout_seconds=0.05)

    async def run():
        return await middleware.dispatch(DummyRequest(), slow_call_next)

    response = asyncio.run(run())
    assert response.status_code == 504
    import json

    body = json.loads(bytes(response.body))
    assert "timed out" in body["detail"]


def test_normal_query_path_still_works(isolated_client: TestClient) -> None:
    """A normal, fast query completes successfully under the timeout guard."""
    response = isolated_client.post("/query", json={"question": "What is CRAG?"})
    assert response.status_code == 200
    assert response.json()["action"] == "CORRECT"


def test_default_timeout_is_reasonable() -> None:
    """Sanity check on the configured default timeout value."""
    assert QUERY_TIMEOUT_SECONDS == 30
    assert RATE_LIMIT_MAX_REQUESTS == 10
    assert RATE_LIMIT_WINDOW_SECONDS == 60
