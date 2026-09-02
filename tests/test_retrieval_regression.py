"""
Regression tests for internal retrieval and vector store persistence.
"""
from pathlib import Path
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import InMemoryVectorStore
from app.evaluation.action_router import ActionRouter

def test_embedding_model_cache_avoids_duplicate_calls():
    em = EmbeddingModel()
    dummy_vec = [0.1] * em.dimension
    em._cache["sample query"] = dummy_vec
    assert em.embed_query("sample query") == dummy_vec

def test_vector_store_loads_pre_persisted_chunks():
    root_dir = Path(__file__).resolve().parent.parent
    persist_dir = root_dir / "chroma_data"
    if (persist_dir / "correctrag.pkl").exists():
        em = EmbeddingModel()
        vs = InMemoryVectorStore(persist_directory=persist_dir, collection_name="correctrag", embedding_model=em)
        assert vs.count() == 168
        assert len(em._cache) == 168

def test_action_router_calibrated_thresholds():
    router = ActionRouter(clearly_relevant_threshold=0.7, clearly_irrelevant_threshold=-0.1)
    assert router.alpha == 0.7
    assert router.beta == -0.1
