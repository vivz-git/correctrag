# ─────────────────────────────────────────────────────────────────────────────
# CorrectRAG Backend Dockerfile
# Python 3.10 runtime for FastAPI & CRAGPipeline
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# Set environment variables for Python runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend:/app

WORKDIR /app

# Install system dependencies (build-essential for C extensions, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python package dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Set up non-root user (UID 1000) for Hugging Face Spaces
RUN useradd -m -u 1000 user
RUN mkdir -p /app/chroma_data /app/.cache/huggingface && chown -R user:user /app
ENV HF_HOME=/app/.cache/huggingface

# Switch to non-root user
USER user

# Copy application source code
COPY --chown=user:user backend/ /app/backend/

# No local ML models to preload in lightweight architecture
# Copy or download CRAG.pdf (safely under non-root)
COPY --chown=user:user README.m[d] CRAG.pd[f] /tmp/
RUN if [ -f /tmp/CRAG.pdf ]; then \
        mv /tmp/CRAG.pdf /app/CRAG.pdf; \
    else \
        echo "CRAG.pdf not found in build context. Downloading from arXiv..."; \
        curl -L -f -o /app/CRAG.pdf https://arxiv.org/pdf/2401.15884.pdf || exit 1; \
    fi

# Expose FastAPI HTTP port (HF overrides this via app_port: 8000 in README.md)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start uvicorn server
CMD ["sh", "-c", "uvicorn app.main:app --app-dir /app/backend --host 0.0.0.0 --port ${PORT:-8000}"]
