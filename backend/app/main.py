"""
Main FastAPI Application Entrypoint for CorrectRAG.
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure backend root is in sys.path when running via `uvicorn backend.app.main:app`
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    from app.api.routes import get_crag_pipeline
    from app.ingestion.pdf_loader import load_documents_dir, load_pdf

    async def _index_background():
        try:
            pipeline = get_crag_pipeline()
            vector_store = pipeline.retriever.vector_store

            if vector_store.count() == 0:
                root_dir = Path(__file__).resolve().parent.parent.parent
                docs_dir_env = os.environ.get("DOCUMENTS_DIR")
                docs_dir = Path(docs_dir_env) if docs_dir_env else (root_dir / "data" / "documents")

                chunks = []
                if docs_dir.is_dir():
                    chunks = load_documents_dir(docs_dir, chunk_size=500, chunk_overlap=100)

                if not chunks:
                    pdf_path = root_dir / "CRAG.pdf"
                    if pdf_path.exists():
                        print(f"Vector store is empty. Indexing {pdf_path.name}...")
                        chunks = load_pdf(pdf_path, chunk_size=500, chunk_overlap=100)
                    else:
                        print(f"WARNING: No documents found in {docs_dir} or {pdf_path}. Knowledge base remains empty.")

                if chunks:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, vector_store.add_chunks, chunks)
                    doc_counts = vector_store.get_indexed_documents()
                    print(f"Indexed {len(chunks)} chunks across {len(doc_counts)} document(s) into vector store: {doc_counts}")
        except Exception as e:
            print(f"Error during background indexing: {e}")

    # Launch background indexing task so server starts listening immediately
    asyncio.create_task(_index_background())
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="CorrectRAG API",
        description="Production HTTP API for Corrective Retrieval Augmented Generation (CRAG)",
        version="1.0.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS configuration for browser frontends
    cors_origins_env = os.environ.get("CORS_ALLOW_ORIGINS", "")
    if cors_origins_env:
        allow_origins = [origin.strip() for origin in cors_origins_env.split(",")]
    else:
        allow_origins = ["*"]

    application.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    application.include_router(api_router)

    return application


app = create_app()
