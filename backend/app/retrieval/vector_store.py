"""
ChromaDB Vector Store Wrapper for CorrectRAG.

Manages persistent or in-memory vector storage, indexing DocumentChunk objects,
and preventing duplicate insertions using deterministic chunk IDs.
"""

from pathlib import Path
from typing import Any
import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.ingestion.pdf_loader import DocumentChunk
from app.retrieval.embeddings import EmbeddingModel


class ChromaVectorStore:
    """Vector storage wrapper around ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path | None = "chroma_data",
        collection_name: str = "correctrag_documents",
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        """Initialize the ChromaDB vector store.

        Args:
            persist_directory: Local directory for persistence. If None or ':memory:',
                               runs an in-memory ephemeral client.
            collection_name: Name of the Chroma collection.
            embedding_model: Optional EmbeddingModel instance for computing embeddings.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model or EmbeddingModel()

        if persist_directory is None or str(persist_directory) == ":memory:":
            self.client: ClientAPI = chromadb.EphemeralClient()
        else:
            persist_path = Path(persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            self.client: ClientAPI = chromadb.PersistentClient(path=str(persist_path))

        self.collection: Collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create a new one with cosine distance."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

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

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)

            # Build sanitized metadata dictionary (Chroma requires primitive types)
            meta: dict[str, Any] = {
                "source": chunk.source,
                "page_number": int(chunk.page_number),
            }
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                elif v is not None:
                    meta[k] = str(v)
            metadatas.append(meta)

        # Compute dense embeddings via centralized embedding model
        embeddings = self.embedding_model.embed_documents(documents)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

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
            Raw dictionary result from ChromaDB containing ids, documents, metadatas, distances.
        """
        total_items = self.collection.count()
        if total_items == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        n_results = min(top_k, total_items)
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        """Return the total number of documents in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete all documents in the current collection and reset."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self._get_or_create_collection()
