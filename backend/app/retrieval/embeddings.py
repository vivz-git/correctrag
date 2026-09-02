"""
Embedding Model Wrapper for CorrectRAG.

Wraps Google GenAI embedding API with a centralized, reusable model instance.
"""

from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any, Optional
from google import genai
from google.genai import errors


class EmbeddingModel:
    """Wrapper for Google GenAI embedding models."""

    def __init__(self, model_name: str = "gemini-embedding-2", api_key: Optional[str] = None) -> None:
        """Initialize the embedding model wrapper.

        Args:
            model_name: Google GenAI embedding model name.
            api_key: Optional Gemini API key.
        """
        self.model_name = model_name
        self.api_key = api_key
        self._cache: dict[str, list[float]] = {}

    def _get_client(self) -> genai.Client:
        """Create a thread-safe GenAI client instance."""
        resolved_key = self.api_key or os.environ.get("GEMINI_API_KEY", "")
        if resolved_key:
            return genai.Client(api_key=resolved_key)
        return genai.Client()

    @property
    def client(self) -> genai.Client:
        """Legacy access to GenAI client."""
        return self._get_client()

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        # gemini-embedding-2 produces 3072-dimensional embeddings by default
        return 3072

    def embed_query(self, text: str) -> list[float]:
        """Compute embedding for a single query text string.

        Args:
            text: Query string.

        Returns:
            Float vector embedding.
        """
        if not text:
            return [0.0] * self.dimension

        if text in self._cache:
            return self._cache[text]

        import time
        import re
        retries = 8
        for attempt in range(retries):
            try:
                client = self._get_client()
                response = client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                )
                emb = list(response.embeddings[0].values)
                self._cache[text] = emb
                return emb
            except Exception as exc:
                if ("429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)) and attempt < retries - 1:
                    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc)) or re.search(r"retryDelay': '(\d+)s", str(exc))
                    if match:
                        sleep_time = float(match.group(1)) + 1.5
                    else:
                        sleep_time = min(30.0, 5.0 * (2 ** min(attempt, 3)))
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(f"Failed to compute query embedding: {exc}") from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a list of document texts.

        Args:
            texts: List of document text strings.

        Returns:
            List of float vector embeddings.
        """
        if not texts:
            return []

        # Partition into cached vs uncached
        results: list[Optional[list[float]]] = []
        uncached: list[str] = []
        uncached_indices: list[int] = []

        for i, t in enumerate(texts):
            if t in self._cache:
                results.append(self._cache[t])
            else:
                results.append(None)
                uncached.append(t)
                uncached_indices.append(i)

        if uncached:
            max_workers = min(16, len(uncached))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                uncached_embs = list(executor.map(self.embed_query, uncached))

            for idx, emb, t in zip(uncached_indices, uncached_embs, uncached):
                results[idx] = emb
                self._cache[t] = emb

        return [r for r in results if r is not None]
