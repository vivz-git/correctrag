"""
Production hardening middleware for the public CorrectRAG API.

Two concerns, both dependency-free and single-process by design (this
service runs on a single t4g.micro instance with no Redis/queue
infrastructure, so state simply lives in process memory):

1. RateLimitMiddleware — a fixed-window per-client-IP request counter that
   protects POST /query from accidental scripted abuse.
2. TimeoutMiddleware — bounds how long a POST /query request is allowed to
   take before the client gets a controlled response instead of a hang.
"""

import asyncio
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Conservative demo-friendly default: a real user issuing queries one at a
# time will never come close to this; it exists to stop a runaway script or
# a mistaken retry loop from hammering the paid Jina/Groq/Tavily APIs behind
# this endpoint on a free/low-tier portfolio deployment.
RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60

# The CRAG pipeline can call embeddings, an LLM judge, generation, and
# optionally Tavily web search in sequence; 30s comfortably covers a normal
# run while still guaranteeing a bounded worst case for the caller.
QUERY_TIMEOUT_SECONDS = 30

# Only the expensive, publicly-writable endpoint is protected. /health and
# static frontend assets (served separately by Vercel) are intentionally
# excluded.
GUARDED_PATHS = {"/query"}


class FixedWindowRateLimiter:
    """Fixed-window per-key request counter with bounded memory.

    Each key (typically a client IP) gets a request count for the current
    time window (`window_seconds` wide). Any bucket belonging to a window
    older than the current one is dropped on every call, so memory is
    bounded by the number of distinct clients active in the current window,
    not by total requests received over the process lifetime.
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, tuple[int, int]] = {}  # key -> (window_index, count)

    def _current_window(self) -> int:
        return int(time.time() // self.window_seconds)

    def hit(self, key: str) -> bool:
        """Register one request for `key`. Returns True if allowed, False if over the limit."""
        now_window = self._current_window()

        stale_keys = [k for k, (window, _count) in self._buckets.items() if window < now_window]
        for stale_key in stale_keys:
            del self._buckets[stale_key]

        window, count = self._buckets.get(key, (now_window, 0))
        if window != now_window:
            window, count = now_window, 0
        count += 1
        self._buckets[key] = (window, count)
        return count <= self.max_requests

    def bucket_count(self) -> int:
        """Number of distinct keys currently tracked (for tests/diagnostics)."""
        return len(self._buckets)


def get_client_ip(request: Request) -> str:
    """Resolve the client IP for rate-limiting purposes.

    Port 8000 is bound to 127.0.0.1 and is never exposed publicly; Caddy is
    the only process that can reach this API, making it a trusted
    single-hop reverse proxy. Caddy sets X-Forwarded-For with the real
    client IP, so it is safe to trust here. Without it, every request would
    appear to originate from Caddy's loopback connection, making per-client
    limiting meaningless. Fall back to the raw peer address only if the
    header is absent (e.g. direct local testing).
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window per-IP rate limiter for guarded (write/expensive) endpoints."""

    def __init__(self, app, limiter: FixedWindowRateLimiter | None = None) -> None:
        super().__init__(app)
        self.limiter = limiter or FixedWindowRateLimiter(
            RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS
        )

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method == "POST" and request.url.path in GUARDED_PATHS:
            client_ip = get_client_ip(request)
            if not self.limiter.hit(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": (
                            f"Rate limit exceeded: max {self.limiter.max_requests} "
                            f"requests per {self.limiter.window_seconds} seconds. "
                            "Please try again shortly."
                        )
                    },
                )
        return await call_next(request)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Bounds processing time for guarded (expensive) endpoints."""

    def __init__(self, app, timeout_seconds: float = QUERY_TIMEOUT_SECONDS) -> None:
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.method == "POST" and request.url.path in GUARDED_PATHS:
            try:
                return await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
            except asyncio.TimeoutError:
                return JSONResponse(
                    status_code=504,
                    content={
                        "detail": (
                            f"Request timed out after {self.timeout_seconds} seconds. "
                            "Please try again."
                        )
                    },
                )
        return await call_next(request)
