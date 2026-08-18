"""
Embedding Model Wrapper for CorrectRAG.

Wraps sentence-transformers with a centralized, reusable model instance.
"""

from typing import Any
import numpy as np


class EmbeddingModel:
    """Wrapper for sentence-transformers embedding models."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Initialize the embedding model wrapper.

        Args:
            model_name: Hugging Face repository or local path for sentence-transformers.
        """
        self.model_name = model_name
        self._model: Any = None

    @property
    def model(self) -> Any:
        """Lazy-load and cache the underlying sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
        else:
            dim = self.model.get_sentence_embedding_dimension()
        return int(dim) if dim is not None else 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a list of document texts.

        Args:
            texts: List of document text strings.

        Returns:
            List of float vector embeddings.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return [list(e) for e in embeddings]

    def embed_query(self, text: str) -> list[float]:
        """Compute embedding for a single query text string.

        Args:
            text: Query string.

        Returns:
            Float vector embedding.
        """
        if not text:
            return [0.0] * self.dimension

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return list(embedding)
