"""
Before vs CorrectRAG Evaluation Runner.
Evaluates 12 controlled questions across Plain RAG and CorrectRAG.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add backend to path
root_dir = Path(__file__).resolve().parent.parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
load_dotenv(root_dir / ".env")

from app.ingestion.pdf_loader import load_pdf
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import InMemoryVectorStore
from app.retrieval.retriever import VectorRetriever
from app.evaluation.relevance_evaluator import RelevanceEvaluator
from app.evaluation.action_router import ActionRouter
from app.evaluation.knowledge_refiner import KnowledgeRefiner
from app.external.query_rewriter import QueryRewriter
from app.external.web_search import WebSearchClient
from app.generation.rag_pipeline import BaselineRAG
from app.pipeline.crag_pipeline import CRAGPipeline
from app.generation.groq_client import GroqClient
from app.generation.gemini_client import GeminiClient


def build_llm_client():
    provider = os.environ.get("LLM_PROVIDER", "groq").lower().strip()
    if provider == "groq":
        groq_key = os.environ.get("GROQ_API_KEY")
        groq_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        return GroqClient(api_key=groq_key, model=groq_model)
    elif provider == "gemini":
        gemini_key = os.environ.get("GEMINI_API_KEY")
        gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        return GeminiClient(api_key=gemini_key, model=gemini_model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def main():
    print("==================================================")
    print("CorrectRAG Before/After Evaluation")
    print("==================================================")

    questions_file = Path(__file__).parent / "questions.json"
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} evaluation questions from {questions_file}")

    # 1. Setup Vector Store with CRAG.pdf
    pdf_path = root_dir / "CRAG.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    print(f"Chunking {pdf_path.name}...")
    chunks = load_pdf(pdf_path, chunk_size=500, chunk_overlap=100)
    print(f"Extracted {len(chunks)} chunks.")

    embedding_model = EmbeddingModel()
    vector_store = InMemoryVectorStore(embedding_model=embedding_model)
    
    cache_path = Path(__file__).parent / "cached_embeddings.json"
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        for chunk, emb in zip(chunks, cached_data):
            meta = {
                "source": chunk.source,
                "page_number": int(chunk.page_number),
            }
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                elif v is not None:
                    meta[k] = str(v)
            vector_store.chunks[chunk.chunk_id] = {
                "document": chunk.text,
                "metadata": meta,
                "embedding": emb,
            }
        print(f"Loaded {vector_store.count()} chunks from cache.")
    else:
        print("Embedding chunks with rate limiting...")
        all_embeddings = []
        for i, chunk in enumerate(chunks):
            retries = 5
            for attempt in range(retries):
                try:
                    emb = embedding_model.embed_query(chunk.text)
                    all_embeddings.append(emb)
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        time.sleep(3 * (attempt + 1))
                    else:
                        raise e
            if (i + 1) % 20 == 0:
                print(f"  Embedded {i+1}/{len(chunks)} chunks...")
                time.sleep(1.0)
                
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(all_embeddings, f)
            
        for chunk, emb in zip(chunks, all_embeddings):
            meta = {
                "source": chunk.source,
                "page_number": int(chunk.page_number),
            }
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                elif v is not None:
                    meta[k] = str(v)
            vector_store.chunks[chunk.chunk_id] = {
                "document": chunk.text,
                "metadata": meta,
                "embedding": emb,
            }
        print(f"Indexed and cached {vector_store.count()} chunks.")

    # 2. Setup Shared Components
    llm_client = build_llm_client()
    retriever = VectorRetriever(vector_store=vector_store)
    evaluator = RelevanceEvaluator()
    router = ActionRouter(clearly_relevant_threshold=0.5, clearly_irrelevant_threshold=-0.2, llm_client=llm_client)
    refiner = KnowledgeRefiner(evaluator=evaluator)
    query_rewriter = QueryRewriter(llm_client=llm_client)

    tavily_key = os.environ.get("TAVILY_API_KEY")
    tavily_max_results = int(os.environ.get("TAVILY_MAX_RESULTS", 5))
    web_search = WebSearchClient(api_key=tavily_key, max_results=tavily_max_results)

    # 3. Setup Pipelines
    baseline_rag = BaselineRAG(
        retriever=retriever,
        llm_client=llm_client,
        top_k=5,
    )

    crag_pipeline = CRAGPipeline(
        retriever=retriever,
        evaluator=evaluator,
        router=router,
        refiner=refiner,
        query_rewriter=query_rewriter,
        web_search=web_search,
        llm_client=llm_client,
        top_k=5,
    )

    results = []

    for idx, q_item in enumerate(questions, 1):
        q_id = q_item["id"]
        cat = q_item["category"]
        q_text = q_item["question"]

        print(f"\n[{idx}/12] ({q_id}) [{cat}] {q_text}")

        # Plain RAG
        t0 = time.time()
        plain_res = baseline_rag.run(q_text)
        plain_time = time.time() - t0

        # CorrectRAG
        t0 = time.time()
        crag_res = crag_pipeline.run(q_text)
        crag_time = time.time() - t0

        record = {
            "id": q_id,
            "category": cat,
            "question": q_text,
            "plain_rag": {
                "answer": plain_res.answer,
                "retrieved_count": len(plain_res.retrieved_chunks),
                "citations": [c.model_dump() for c in plain_res.sources],
                "latency_seconds": round(plain_time, 3)
            },
            "correctrag": {
                "answer": crag_res.answer,
                "action": crag_res.action,
                "rewritten_query": crag_res.rewritten_query,
                "web_search_used": crag_res.trace.web_search_used,
                "judge_called": crag_res.trace.judge_called,
                "judge_decision": crag_res.trace.judge_decision,
                "judge_reason": crag_res.trace.judge_reason,
                "internal_strip_count": crag_res.trace.internal_strip_count,
                "external_strip_count": crag_res.trace.external_strip_count,
                "final_context_source": crag_res.trace.final_context_source,
                "latency_seconds": round(crag_time, 3)
            }
        }
        results.append(record)
        print(f"  -> Plain RAG: {len(plain_res.answer)} chars ({plain_time:.2f}s)")
        print(f"  -> CRAG: action={crag_res.action}, web={crag_res.trace.web_search_used}, judge={crag_res.trace.judge_called} ({crag_time:.2f}s)")
        time.sleep(2.0)

    output_path = Path(__file__).parent / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nCompleted! Saved results to {output_path}")


if __name__ == "__main__":
    main()
