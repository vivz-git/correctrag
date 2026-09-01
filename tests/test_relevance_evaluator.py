"""Tests for the lightweight, similarity-based RelevanceEvaluator."""

import pytest
from app.evaluation.relevance_evaluator import RelevanceEvaluator

class MockEmbeddingModel:
    def embed_query(self, text: str) -> list[float]:
        # Return dummy vector for testing
        if text == "query":
            return [1.0, 0.0]
        elif text == "exact match":
            return [1.0, 0.0]
        elif text == "orthogonal":
            return [0.0, 1.0]
        elif text == "opposite":
            return [-1.0, 0.0]
        return [0.5, 0.5]
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

def test_relevance_evaluator_exact_match():
    ev = RelevanceEvaluator(embedding_model=MockEmbeddingModel())
    score = ev.score("query", "exact match")
    assert score == 1.0  # similarity=1.0 -> 1.0 * 2.0 - 1.0 = 1.0

def test_relevance_evaluator_orthogonal():
    ev = RelevanceEvaluator(embedding_model=MockEmbeddingModel())
    score = ev.score("query", "orthogonal")
    assert score == -1.0 # similarity=0.0 -> 0.0 * 2.0 - 1.0 = -1.0

def test_relevance_evaluator_opposite():
    ev = RelevanceEvaluator(embedding_model=MockEmbeddingModel())
    score = ev.score("query", "opposite")
    assert score == -1.0 # similarity=-1.0 -> -1.0 * 2.0 - 1.0 = -3.0 -> clamped to -1.0

def test_relevance_evaluator_batch():
    ev = RelevanceEvaluator(embedding_model=MockEmbeddingModel())
    scores = ev.score_batch([("query", "exact match"), ("query", "orthogonal")])
    assert scores == [1.0, -1.0]
