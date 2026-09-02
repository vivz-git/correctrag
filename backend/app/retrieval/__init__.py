"""
Semantic Retrieval Package for CorrectRAG.

Provides embedding model management and vector retrieval.
"""

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import InMemoryVectorStore
from app.retrieval.retriever import RetrievedChunk, VectorRetriever

__all__ = [
    "EmbeddingModel",
    "InMemoryVectorStore",
    "VectorRetriever",
    "RetrievedChunk",
]
