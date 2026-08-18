"""
Semantic Retrieval Package for CorrectRAG.

Provides embedding model management, ChromaDB persistence, and vector retrieval.
"""

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import ChromaVectorStore
from app.retrieval.retriever import RetrievedChunk, VectorRetriever

__all__ = [
    "EmbeddingModel",
    "ChromaVectorStore",
    "VectorRetriever",
    "RetrievedChunk",
]
