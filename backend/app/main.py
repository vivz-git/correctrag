from fastapi import FastAPI

app = FastAPI(
    title="CorrectRAG API",
    description="API for Corrective Retrieval Augmented Generation (CRAG)",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "correctrag-api",
    }
