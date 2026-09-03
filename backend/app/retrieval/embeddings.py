"""
Embedding Model Wrapper for CorrectRAG.

Wraps Jina AI Embedding API with a centralized, reusable model instance.
"""

import os
import time
from typing import Any, Optional
import httpx


class EmbeddingModel:
    """Wrapper for Jina AI embedding models."""

    DEFAULT_MODEL = "jina-embeddings-v5-text-small"
    DEFAULT_API_URL = "https://api.jina.ai/v1/embeddings"
    DEFAULT_DIMENSION = 1024

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        dimension: int = DEFAULT_DIMENSION,
    ) -> None:
        """Initialize the embedding model wrapper.

        Args:
            model_name: Jina AI embedding model name.
            api_key: Optional Jina API key (defaults to JINA_API_KEY environment variable).
            dimension: Output embedding vector dimension (defaults to 1024).
        """
        self.model_name = (
            model_name
            or os.environ.get("JINA_MODEL")
            or self.DEFAULT_MODEL
        )
        self.api_key = api_key or os.environ.get("JINA_API_KEY", "")
        self.api_url = os.environ.get("JINA_API_URL", self.DEFAULT_API_URL)
        self._dimension = dimension
        self._cache: dict[str, list[float]] = {}

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._dimension

    def _get_api_key(self) -> str:
        """Resolve and validate the Jina API key."""
        key = self.api_key or os.environ.get("JINA_API_KEY", "")
        if not key:
            raise RuntimeError(
                "JINA_API_KEY is not set. "
                "Export JINA_API_KEY environment variable or pass api_key to EmbeddingModel."
            )
        return key

    def _call_api(self, texts: list[str], task: str) -> list[list[float]]:
        """Execute a batch request to the Jina AI embeddings endpoint with retries."""
        if not texts:
            return []

        api_key = self._get_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model_name,
            "task": task,
            "dimensions": self.dimension,
            "input": texts,
        }

        retries = 5
        timeout = httpx.Timeout(30.0, connect=10.0)

        for attempt in range(retries):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(self.api_url, headers=headers, json=payload)

                if (response.status_code == 429 or response.status_code >= 500) and attempt < retries - 1:
                    sleep_time = min(20.0, 1.5 * (2 ** attempt))
                    time.sleep(sleep_time)
                    continue

                response.raise_for_status()
                data = response.json()
                records = data.get("data", [])
                records_sorted = sorted(records, key=lambda r: r.get("index", 0))
                return [r["embedding"] for r in records_sorted]

            except httpx.HTTPStatusError as exc:
                if (exc.response.status_code == 429 or exc.response.status_code >= 500) and attempt < retries - 1:
                    time.sleep(min(20.0, 1.5 * (2 ** attempt)))
                    continue
                raise RuntimeError(
                    f"Jina AI API request failed with status {exc.response.status_code}: {exc.response.text}"
                ) from exc
            except (httpx.RequestError, Exception) as exc:
                if attempt < retries - 1:
                    time.sleep(min(20.0, 1.5 * (2 ** attempt)))
                    continue
                raise RuntimeError(f"Failed to compute embeddings via Jina AI API: {exc}") from exc

        raise RuntimeError("Failed to compute embeddings after all retry attempts.")

    def embed_query(self, text: str) -> list[float]:
        """Compute embedding for a single query text string using retrieval.query task.

        Args:
            text: Query string.

        Returns:
            Float vector embedding.
        """
        if not text or not text.strip():
            return [0.0] * self.dimension

        if text in self._cache:
            return self._cache[text]

        embeddings = self._call_api(texts=[text], task="retrieval.query")
        if not embeddings:
            raise RuntimeError(f"Jina AI API returned empty embedding list for query: {text!r}")

        emb = embeddings[0]
        self._cache[text] = emb
        return emb

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a list of document texts using retrieval.passage task.

        Args:
            texts: List of document text strings.

        Returns:
            List of float vector embeddings.
        """
        if not texts:
            return []

        results: list[Optional[list[float]]] = []
        uncached: list[str] = []
        uncached_indices: list[int] = []

        for i, t in enumerate(texts):
            if not t or not t.strip():
                results.append([0.0] * self.dimension)
            elif t in self._cache:
                results.append(self._cache[t])
            else:
                results.append(None)
                uncached.append(t)
                uncached_indices.append(i)

        if uncached:
            batch_size = 64
            all_uncached_embs: list[list[float]] = []
            for start in range(0, len(uncached), batch_size):
                batch = uncached[start : start + batch_size]
                batch_embs = self._call_api(texts=batch, task="retrieval.passage")
                all_uncached_embs.extend(batch_embs)

            for idx, emb, t in zip(uncached_indices, all_uncached_embs, uncached):
                results[idx] = emb
                self._cache[t] = emb

        return [r for r in results if r is not None]
