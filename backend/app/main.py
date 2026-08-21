"""
Main FastAPI Application Entrypoint for CorrectRAG.
"""

import sys
from pathlib import Path

# Ensure backend root is in sys.path when running via `uvicorn backend.app.main:app`
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="CorrectRAG API",
        description="Production HTTP API for Corrective Retrieval Augmented Generation (CRAG)",
        version="1.0.0",
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    # CORS configuration for browser frontends
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    application.include_router(api_router)

    return application


app = create_app()
