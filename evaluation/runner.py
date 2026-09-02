"""
Evaluation Runner for CorrectRAG.

Runs Baseline RAG and/or the CRAG pipeline against the evaluation dataset
and collects structured results.

Usage (from the correctrag/ project root):

    # Run both systems
    python evaluation/runner.py

    # Run only Baseline RAG
    python evaluation/runner.py --system baseline

    # Run only CRAG
    python evaluation/runner.py --system crag

    # Use a custom dataset or output path
    python evaluation/runner.py --dataset evaluation/dataset.json --output evaluation/results.json

Prerequisites:
    - GEMINI_API_KEY must be set in the environment (or .env file)
    - TAVILY_API_KEY must be set for CRAG web search (INCORRECT / AMBIGUOUS actions)
    - A PDF must have been ingested into ChromaDB before running

IMPORTANT: This runner performs REAL API calls (Gemini + Tavily).
           Do NOT run this during pytest — tests mock both systems.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Ensure backend/ is importable when running as a script ────────────────────
_root = Path(__file__).resolve().parent.parent
_backend = _root / "backend"
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

# ── Import after path fixup ───────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(_root / ".env")
except ImportError:
    pass  # python-dotenv is optional; rely on shell env vars


# ──────────────────────────────────────────────────────────────────────────────
# Result schema
# ──────────────────────────────────────────────────────────────────────────────

def make_eval_record(
    question_id: str,
    category: str,
    question: str,
    system: str,
    answer: str | None = None,
    action: str | None = None,
    retrieved_count: int = 0,
    relevance_scores: list[float] | None = None,
    citations: list[str] | None = None,
    latency_seconds: float | None = None,
    status: str = "success",
    error_message: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured evaluation record for one question/system pair.

    Fields:
        question_id:      Matches dataset id (e.g., 'q001').
        category:         INTERNAL_SUPPORTED / INTERNAL_IRRELEVANT / INTERNAL_PARTIAL.
        question:         Original question text.
        system:           'baseline' or 'crag'.
        answer:           Generated answer text (None on error).
        action:           CRAG action taken (None for Baseline).
        retrieved_count:  Number of internal chunks retrieved.
        relevance_scores: Per-chunk relevance scores (CRAG only).
        citations:        List of citation source strings.
        latency_seconds:  Wall-clock time for this question.
        status:           'success' or 'error'.
        error_message:    Exception message when status == 'error'.
        extra:            Optional system-specific metadata.
    """
    return {
        "question_id": question_id,
        "category": category,
        "question": question,
        "system": system,
        "answer": answer,
        "action": action,
        "retrieved_count": retrieved_count,
        "relevance_scores": relevance_scores or [],
        "citations": citations or [],
        "latency_seconds": latency_seconds,
        "status": status,
        "error_message": error_message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "extra": extra or {},
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loader
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str | Path) -> dict[str, Any]:
    """Load and validate the evaluation dataset from a JSON file.

    Returns:
        Parsed dataset dict with 'metadata' and 'questions' keys.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError:        If required keys are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, encoding="utf-8") as f:
        dataset = json.load(f)

    if "questions" not in dataset:
        raise ValueError("Dataset must contain a 'questions' key.")
    for q in dataset["questions"]:
        for required in ("id", "category", "question"):
            if required not in q:
                raise ValueError(f"Question missing required field '{required}': {q}")

    return dataset


# ──────────────────────────────────────────────────────────────────────────────
# Citation extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _citations_from_rag_result(result: Any) -> list[str]:
    """Extract citation strings from a BaselineRAG RAGResult."""
    citations: list[str] = []
    for src in getattr(result, "sources", []):
        citations.append(f"{getattr(src, 'source', '')} p.{getattr(src, 'page_number', '?')}")
    return citations


def _citations_from_crag_result(result: Any) -> list[str]:
    """Extract citation strings from a CRAGPipeline CRAGResult.

    Collects sources from refined internal strips and refined external strips.
    """
    citations: list[str] = []
    seen: set[str] = set()
    for strip in list(getattr(result, "refined_strips", [])) + list(
        getattr(result, "external_strips", [])
    ):
        src = getattr(strip, "source", "")
        page = getattr(strip, "page_number", "?")
        key = f"{src}:{page}"
        if key not in seen:
            seen.add(key)
            citations.append(f"{src} p.{page}")
    return citations


# ──────────────────────────────────────────────────────────────────────────────
# Runner class
# ──────────────────────────────────────────────────────────────────────────────

class EvaluationRunner:
    """Runs Baseline RAG and/or CRAG against an evaluation dataset.

    Components are injected so they can be mocked in tests.

    Args:
        baseline_rag:   Initialised BaselineRAG instance (or None to skip).
        crag_pipeline:  Initialised CRAGPipeline instance (or None to skip).
        verbose:        Print progress to stdout when True.
    """

    def __init__(
        self,
        baseline_rag: Any = None,
        crag_pipeline: Any = None,
        verbose: bool = True,
    ) -> None:
        self.baseline_rag = baseline_rag
        self.crag_pipeline = crag_pipeline
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def run_baseline(self, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run Baseline RAG against all questions.

        Args:
            questions: List of question dicts from the dataset.

        Returns:
            List of EvalRecord dicts.
        """
        if self.baseline_rag is None:
            raise RuntimeError("baseline_rag component is not initialised.")

        records: list[dict[str, Any]] = []
        total = len(questions)

        for i, q in enumerate(questions, 1):
            self._log(f"[Baseline {i}/{total}] {q['id']}: {q['question'][:60]}...")
            t0 = time.perf_counter()
            try:
                result = self.baseline_rag.run(q["question"])
                latency = round(time.perf_counter() - t0, 3)

                record = make_eval_record(
                    question_id=q["id"],
                    category=q["category"],
                    question=q["question"],
                    system="baseline",
                    answer=result.answer,
                    action=None,
                    retrieved_count=len(result.retrieved_chunks),
                    relevance_scores=[],
                    citations=_citations_from_rag_result(result),
                    latency_seconds=latency,
                    status="success",
                )
            except Exception as exc:  # noqa: BLE001
                latency = round(time.perf_counter() - t0, 3)
                self._log(f"  ERROR: {exc}")
                record = make_eval_record(
                    question_id=q["id"],
                    category=q["category"],
                    question=q["question"],
                    system="baseline",
                    latency_seconds=latency,
                    status="error",
                    error_message=str(exc),
                )

            records.append(record)
            self._log(f"  status={record['status']} latency={record['latency_seconds']}s")

        return records

    def run_crag(self, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run CRAG pipeline against all questions.

        Args:
            questions: List of question dicts from the dataset.

        Returns:
            List of EvalRecord dicts.
        """
        if self.crag_pipeline is None:
            raise RuntimeError("crag_pipeline component is not initialised.")

        records: list[dict[str, Any]] = []
        total = len(questions)

        for i, q in enumerate(questions, 1):
            self._log(f"[CRAG {i}/{total}] {q['id']}: {q['question'][:60]}...")
            t0 = time.perf_counter()
            try:
                result = self.crag_pipeline.run(q["question"])
                latency = round(time.perf_counter() - t0, 3)

                trace = getattr(result, "trace", None)
                record = make_eval_record(
                    question_id=q["id"],
                    category=q["category"],
                    question=q["question"],
                    system="crag",
                    answer=result.answer,
                    action=str(result.action),
                    retrieved_count=len(result.retrieved_chunks),
                    relevance_scores=list(result.relevance_scores),
                    citations=_citations_from_crag_result(result),
                    latency_seconds=latency,
                    status="success",
                    extra={
                        "rewritten_query": result.rewritten_query,
                        "internal_strip_count": len(result.refined_strips),
                        "external_strip_count": len(result.external_strips),
                        "max_relevance_score": (
                            getattr(trace, "max_relevance_score", None)
                            if trace
                            else None
                        ),
                        "final_context_source": (
                            getattr(trace, "final_context_source", None)
                            if trace
                            else None
                        ),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                latency = round(time.perf_counter() - t0, 3)
                self._log(f"  ERROR: {exc}")
                record = make_eval_record(
                    question_id=q["id"],
                    category=q["category"],
                    question=q["question"],
                    system="crag",
                    latency_seconds=latency,
                    status="error",
                    error_message=str(exc),
                )

            records.append(record)
            self._log(
                f"  status={record['status']} action={record['action']} "
                f"latency={record['latency_seconds']}s"
            )

        return records

    def run(
        self,
        questions: list[dict[str, Any]],
        systems: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run the specified systems against all questions.

        Args:
            questions: List of question dicts from the dataset.
            systems:   Which systems to run. Default: ['baseline', 'crag'].

        Returns:
            Combined list of EvalRecord dicts for all systems.
        """
        if systems is None:
            systems = ["baseline", "crag"]

        all_records: list[dict[str, Any]] = []
        if "baseline" in systems:
            all_records.extend(self.run_baseline(questions))
        if "crag" in systems:
            all_records.extend(self.run_crag(questions))
        return all_records


# ──────────────────────────────────────────────────────────────────────────────
# Results I/O
# ──────────────────────────────────────────────────────────────────────────────

def save_results(
    records: list[dict[str, Any]],
    dataset_metadata: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save evaluation records to a JSON file.

    Args:
        records:          List of EvalRecord dicts.
        dataset_metadata: The 'metadata' key from dataset.json.
        output_path:      Destination file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "run_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_version": dataset_metadata.get("version"),
            "document_context": dataset_metadata.get("document_context"),
            "total_records": len(records),
            "systems": sorted({r["system"] for r in records}),
        },
        "records": records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")


def load_results(path: str | Path) -> dict[str, Any]:
    """Load previously saved evaluation results from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline factory (real components, not for tests)
# ──────────────────────────────────────────────────────────────────────────────

def _build_llm_client(provider: str = "gemini") -> Any:
    """Instantiate the requested LLMProvider (gemini or groq)."""
    if provider == "groq":
        from app.generation.groq_client import GroqClient
        groq_key = os.environ.get("GROQ_API_KEY")
        if not groq_key:
            raise EnvironmentError("GROQ_API_KEY not set. Cannot run real Groq provider.")
        return GroqClient(api_key=groq_key)
    else:
        from app.generation.gemini_client import GeminiClient
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            raise EnvironmentError("GEMINI_API_KEY not set. Cannot run real Baseline RAG / CRAG.")
        return GeminiClient(api_key=gemini_key)


def _build_baseline_rag(provider: str = "gemini") -> Any:
    """Instantiate BaselineRAG with real components from environment."""
    from app.retrieval.embeddings import EmbeddingModel
    from app.retrieval.vector_store import InMemoryVectorStore
    from app.retrieval.retriever import VectorRetriever
    from app.generation.rag_pipeline import BaselineRAG

    embedding_model = EmbeddingModel()
    vector_store = InMemoryVectorStore(
        embedding_model=embedding_model,
        collection_name="correctrag",
        persist_directory=str(_root / "chroma_data"),
    )
    retriever = VectorRetriever(vector_store=vector_store)
    llm_client = _build_llm_client(provider=provider)
    return BaselineRAG(retriever=retriever, llm_client=llm_client, top_k=5)


def _build_crag_pipeline(provider: str = "gemini") -> Any:
    """Instantiate CRAGPipeline with real components from environment."""
    from app.retrieval.embeddings import EmbeddingModel
    from app.retrieval.vector_store import InMemoryVectorStore
    from app.retrieval.retriever import VectorRetriever
    from app.evaluation.relevance_evaluator import RelevanceEvaluator
    from app.evaluation.action_router import ActionRouter
    from app.evaluation.knowledge_refiner import KnowledgeRefiner
    from app.external.query_rewriter import QueryRewriter
    from app.external.web_search import WebSearchClient
    from app.pipeline.crag_pipeline import CRAGPipeline

    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        raise EnvironmentError("TAVILY_API_KEY not set. CRAG requires web search.")

    llm_client = _build_llm_client(provider=provider)

    embedding_model = EmbeddingModel()
    vector_store = InMemoryVectorStore(
        embedding_model=embedding_model,
        collection_name="correctrag",
        persist_directory=str(_root / "chroma_data"),
    )
    retriever = VectorRetriever(vector_store=vector_store)
    evaluator = RelevanceEvaluator()
    router = ActionRouter(clearly_relevant_threshold=0.5, clearly_irrelevant_threshold=-0.2, llm_client=llm_client)
    refiner = KnowledgeRefiner(evaluator=evaluator)
    query_rewriter = QueryRewriter(llm_client=llm_client)
    web_search = WebSearchClient(api_key=tavily_key)

    return CRAGPipeline(
        retriever=retriever,
        evaluator=evaluator,
        router=router,
        refiner=refiner,
        query_rewriter=query_rewriter,
        web_search=web_search,
        llm_client=llm_client,
        top_k=5,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CorrectRAG Evaluation Runner — Baseline RAG vs CRAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--system",
        choices=["baseline", "crag", "both"],
        default="both",
        help="Which system(s) to evaluate (default: both)",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "groq"],
        default="gemini",
        help="LLM provider for generation (default: gemini; alternate: groq)",
    )
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "dataset.json"),
        help="Path to evaluation dataset JSON (default: evaluation/dataset.json)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "results.json"),
        help="Output path for results JSON (default: evaluation/results.json)",
    )
    parser.add_argument(
        "--category",
        choices=["INTERNAL_SUPPORTED", "INTERNAL_IRRELEVANT", "INTERNAL_PARTIAL"],
        default=None,
        help="Filter to a single question category (default: all categories)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of questions to run (default: all)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-question progress output",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Load dataset
    print(f"Loading dataset from: {args.dataset}")
    dataset = load_dataset(args.dataset)
    questions = dataset["questions"]

    # Filter by category if requested
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]
        print(f"Filtered to {len(questions)} questions in category: {args.category}")

    # Apply limit
    if args.limit:
        questions = questions[: args.limit]
        print(f"Limited to first {len(questions)} question(s)")

    print(f"Running evaluation on {len(questions)} question(s) using LLM provider: {args.provider}...")

    # Determine systems to run
    systems = ["baseline", "crag"] if args.system == "both" else [args.system]

    # Build real components
    baseline_rag = None
    crag_pipeline = None
    if "baseline" in systems:
        print(f"Initialising Baseline RAG ({args.provider})...")
        baseline_rag = _build_baseline_rag(provider=args.provider)
    if "crag" in systems:
        print(f"Initialising CRAG pipeline ({args.provider})...")
        crag_pipeline = _build_crag_pipeline(provider=args.provider)

    # Run evaluation
    runner = EvaluationRunner(
        baseline_rag=baseline_rag,
        crag_pipeline=crag_pipeline,
        verbose=not args.quiet,
    )
    records = runner.run(questions, systems=systems)

    # Save results
    save_results(records, dataset.get("metadata", {}), args.output)

    # Print brief summary
    success = sum(1 for r in records if r["status"] == "success")
    errors = sum(1 for r in records if r["status"] == "error")
    print(f"\nSummary: {success} succeeded, {errors} failed out of {len(records)} total records.")
    print("Run metrics.py or generate a report to see full analysis.")


if __name__ == "__main__":
    main()
