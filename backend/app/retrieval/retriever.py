"""
Vector Retriever Module for CorrectRAG.

Embeds incoming queries, retrieves nearest chunks from vector store,
and returns structured RetrievedChunk instances with metadata and similarity scores.
"""

from typing import Any
from pydantic import BaseModel, Field

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import InMemoryVectorStore


class RetrievedChunk(BaseModel):
    """Structured search result chunk returned from retrieval."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Retrieved chunk text content")
    source: str = Field(..., description="Source document name or path")
    page_number: int = Field(..., description="1-indexed source page number")
    score: float = Field(..., description="Cosine similarity score (higher is more similar)")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Preserved auxiliary chunk metadata",
    )


class VectorRetriever:
    """Semantic vector retriever using in-memory store and Gemini embeddings."""

    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        """Initialize the vector retriever.

        Args:
            vector_store: Instantiated InMemoryVectorStore.
            embedding_model: EmbeddingModel instance (shares vector_store's model by default).
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model or vector_store.embedding_model

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve top-K most relevant chunks for a text query.

        Args:
            query: User or pipeline query string.
            top_k: Number of nearest candidate chunks to retrieve.

        Returns:
            List of RetrievedChunk instances sorted by relevance score descending.
        """
        clean_query = query.strip()
        if not clean_query or self.vector_store.count() == 0:
            return []

        # 1. Embed query
        query_vector = self.embedding_model.embed_query(clean_query)

        # 2. Query vector store
        results = self.vector_store.query(query_embedding=query_vector, top_k=top_k)

        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved_chunks: list[RetrievedChunk] = []

        for idx in range(len(ids)):
            chunk_id = ids[idx]
            text = docs[idx]
            meta = metas[idx] if idx < len(metas) and metas[idx] else {}
            dist = distances[idx] if idx < len(distances) and distances[idx] is not None else 1.0

            # Convert cosine distance to cosine similarity: similarity = 1 - distance
            similarity_score = float(max(-1.0, min(1.0, 1.0 - dist)))

            source = meta.get("source", "unknown")
            page_number = int(meta.get("page_number", 1))

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    source=source,
                    page_number=page_number,
                    score=round(similarity_score, 4),
                    metadata=meta,
                )
            )

        # Sort descending by similarity score
        retrieved_chunks.sort(key=lambda c: c.score, reverse=True)
        return retrieved_chunks
