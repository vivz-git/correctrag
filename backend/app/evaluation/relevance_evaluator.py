"""
Relevance Evaluator for CorrectRAG (Lightweight).

Uses a deterministic, similarity-based
relevance evaluator. Maps embedding cosine similarity into the [-1.0, 1.0]
range to seamlessly integrate with the existing ActionRouter and KnowledgeRefiner.
"""

import math
from typing import Optional

from app.retrieval.embeddings import EmbeddingModel


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class RelevanceEvaluator:
    """Query-document relevance scorer based on embedding similarity."""

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        mapping_scale: float = 2.0,
        mapping_shift: float = -1.0,
    ) -> None:
        """Initialize the evaluator.

        Args:
            embedding_model: Instance of EmbeddingModel. Defaults to new instance.
            mapping_scale: Multiplier for similarity score (default 2.0).
            mapping_shift: Additive shift for similarity score (default -1.0).
                           This maps [0, 1] similarity to [-1, 1].
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.mapping_scale = mapping_scale
        self.mapping_shift = mapping_shift

    # ── Input validation ──────────────────────────────────────────────────────

    @staticmethod
    def _validate_pair(query: str, document: str, index: Optional[int] = None) -> None:
        loc = f" at index {index}" if index is not None else ""
        if not query or not query.strip():
            raise ValueError(f"query{loc} must be a non-empty string.")
        if not document or not document.strip():
            raise ValueError(f"document{loc} must be a non-empty string.")

    # ── Public interface ──────────────────────────────────────────────────────

    def _map_to_score(self, similarity: float) -> float:
        score = (similarity * self.mapping_scale) + self.mapping_shift
        return float(max(-1.0, min(1.0, score)))

    def score(self, query: str, document: str) -> float:
        self._validate_pair(query, document)
        v_query = self.embedding_model.embed_query(query)
        v_doc = self.embedding_model.embed_query(document)
        sim = _cosine_similarity(v_query, v_doc)
        return self._map_to_score(sim)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            raise ValueError("pairs must be a non-empty list.")
        for i, (query, document) in enumerate(pairs):
            self._validate_pair(query, document, index=i)

        queries = [p[0] for p in pairs]
        docs = [p[1] for p in pairs]

        # Optimize for common CRAG case where query is identical across the batch
        unique_queries = list(set(queries))
        query_embs = {}
        if len(unique_queries) == 1:
            query_embs[unique_queries[0]] = self.embedding_model.embed_query(unique_queries[0])
        else:
            for q in unique_queries:
                query_embs[q] = self.embedding_model.embed_query(q)

        # Batch embed all documents
        doc_embs = self.embedding_model.embed_documents(docs)

        scores = []
        for i, (q, d) in enumerate(pairs):
            sim = _cosine_similarity(query_embs[q], doc_embs[i])
            scores.append(self._map_to_score(sim))

        return scores
