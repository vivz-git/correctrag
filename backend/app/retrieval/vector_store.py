"""
In-Memory Vector Store Wrapper for CorrectRAG.

Manages persistent or in-memory vector storage, indexing DocumentChunk objects,
and preventing duplicate insertions using deterministic chunk IDs.
Replaces ChromaDB with a lightweight pure-Python + math similarity implementation.
"""

import math
import pickle
from pathlib import Path
from typing import Any

from app.ingestion.pdf_loader import DocumentChunk
from app.retrieval.embeddings import EmbeddingModel


class InMemoryVectorStore:
    """Lightweight pure-Python in-memory vector store."""

    def __init__(
        self,
        persist_directory: str | Path | None = "chroma_data",
        collection_name: str = "correctrag_documents",
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        """Initialize the in-memory vector store.

        Args:
            persist_directory: Local directory for persistence. If None or ':memory:',
                               runs an in-memory ephemeral client.
            collection_name: Name of the collection.
            embedding_model: Optional EmbeddingModel instance for computing embeddings.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model or EmbeddingModel()
        
        self.chunks: dict[str, dict[str, Any]] = {}
        
        if self.persist_directory and str(self.persist_directory) != ":memory:":
            self.persist_path = Path(self.persist_directory) / f"{self.collection_name}.pkl"
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._load()
        else:
            self.persist_path = None

    def _load(self) -> None:
        """Load chunks from disk if available."""
        if self.persist_path and self.persist_path.exists():
            try:
                with open(self.persist_path, "rb") as f:
                    self.chunks = pickle.load(f)
            except Exception:
                self.chunks = {}

    def _save(self) -> None:
        """Save chunks to disk."""
        if self.persist_path:
            with open(self.persist_path, "wb") as f:
                pickle.dump(self.chunks, f)

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Add or update a list of DocumentChunk instances in the vector store.

        Uses chunk_id as the primary document key with upsert semantics
        to prevent duplicate entries.

        Args:
            chunks: List of DocumentChunk instances.

        Returns:
            Number of chunks added or updated.
        """
        if not chunks:
            return 0

        documents = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(documents)
        
        added = 0
        for i, chunk in enumerate(chunks):
            if chunk.chunk_id not in self.chunks:
                added += 1
                
            meta: dict[str, Any] = {
                "source": chunk.source,
                "page_number": int(chunk.page_number),
            }
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                elif v is not None:
                    meta[k] = str(v)
            
            self.chunks[chunk.chunk_id] = {
                "document": chunk.text,
                "metadata": meta,
                "embedding": embeddings[i]
            }
        
        if added > 0 or len(chunks) > 0:
            self._save()
            
        return len(chunks)

    def _cosine_distance(self, v1: list[float], v2: list[float]) -> float:
        """Compute cosine distance between two vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0 or norm2 == 0:
            return 1.0
        similarity = dot / (norm1 * norm2)
        # Match ChromaDB's distance metric (1 - similarity)
        return float(max(0.0, min(2.0, 1.0 - similarity)))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Query the collection using a pre-computed query embedding vector.

        Args:
            query_embedding: Dense embedding vector for the query.
            top_k: Maximum number of closest candidate results to return.

        Returns:
            Raw dictionary result mimicking ChromaDB containing ids, documents, metadatas, distances.
        """
        if not self.chunks:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        scored = []
        for chunk_id, data in self.chunks.items():
            dist = self._cosine_distance(query_embedding, data["embedding"])
            scored.append((dist, chunk_id, data))
            
        scored.sort(key=lambda x: x[0])  # Sort by distance (lowest first)
        top_results = scored[:top_k]
        
        ids = [res[1] for res in top_results]
        docs = [res[2]["document"] for res in top_results]
        metas = [res[2]["metadata"] for res in top_results]
        distances = [res[0] for res in top_results]

        return {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [distances],
        }

    def count(self) -> int:
        """Return the total number of documents in the collection."""
        return len(self.chunks)

    def clear(self) -> None:
        """Delete all documents in the current collection and reset."""
        self.chunks = {}
        if self.persist_path and self.persist_path.exists():
            try:
                self.persist_path.unlink()
            except Exception:
                pass
