"""
Offline unit tests for the CorrectRAG evaluation framework.

All tests in this module are offline and mock both Baseline RAG and CRAG.
No model downloads, no vector store queries, no Gemini API calls, and no
Tavily API calls occur during testing.

Test groups:
  - TestDatasetLoading          — dataset.json structure, validity, schema checks
  - TestResultSchema            — make_eval_record format and fields
  - TestMetricsAnswerCorrectness — keyword_overlap_score and F1 logic
  - TestMetricsCitationRecall   — citation_recall_score behavior
  - TestMetricsActionDist       — action_distribution and category breakdowns
  - TestMetricsLatencyAndError  — mean latency, percentiles, error rate
  - TestMetricsScoringAndAgg    — score_single_result, aggregate_by_category, compute_summary
  - TestEvaluationRunner        — mock runner executing Baseline and CRAG
  - TestResultsIO               — save_results and load_results functionality
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation.metrics import (
    action_distribution,
    action_distribution_by_category,
    aggregate_by_category,
    citation_recall_score,
    compute_summary,
    contains_reference_keywords,
    error_rate,
    keyword_overlap_score,
    latency_percentile,
    mean_latency_seconds,
    score_single_result,
)
from evaluation.runner import (
    EvaluationRunner,
    _citations_from_crag_result,
    _citations_from_rag_result,
    load_dataset,
    load_results,
    make_eval_record,
    save_results,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def dataset_path() -> Path:
    return Path(__file__).resolve().parent.parent / "evaluation" / "dataset.json"


@pytest.fixture
def sample_questions() -> list[dict]:
    return [
        {
            "id": "q001",
            "category": "INTERNAL_SUPPORTED",
            "question": "What is CRAG?",
            "reference_answer": "Corrective Retrieval Augmented Generation",
            "expected_sources": ["CRAG.pdf"],
            "notes": "Supported internally",
        },
        {
            "id": "q002",
            "category": "INTERNAL_IRRELEVANT",
            "question": "What is the capital of France?",
            "reference_answer": "Paris",
            "expected_sources": ["https://paris.org"],
            "notes": "External world knowledge",
        },
        {
            "id": "q003",
            "category": "INTERNAL_PARTIAL",
            "question": "What is BM25 versus dense retrieval?",
            "reference_answer": "BM25 is sparse and dense is vector embeddings",
            "expected_sources": None,
            "notes": "Partial internal evidence",
        },
    ]


# ──────────────────────────────────────────────────────────────────────────────
# TestDatasetLoading
# ──────────────────────────────────────────────────────────────────────────────

class TestDatasetLoading:
    """Validate loading and schema of evaluation/dataset.json."""

    def test_load_dataset_succeeds(self, dataset_path: Path):
        data = load_dataset(dataset_path)
        assert "metadata" in data
        assert "questions" in data
        assert len(data["questions"]) == 30

    def test_dataset_contains_all_three_categories(self, dataset_path: Path):
        data = load_dataset(dataset_path)
        categories = {q["category"] for q in data["questions"]}
        assert categories == {
            "INTERNAL_SUPPORTED",
            "INTERNAL_IRRELEVANT",
            "INTERNAL_PARTIAL",
        }

    def test_each_category_has_ten_questions(self, dataset_path: Path):
        data = load_dataset(dataset_path)
        for cat in ("INTERNAL_SUPPORTED", "INTERNAL_IRRELEVANT", "INTERNAL_PARTIAL"):
            cat_qs = [q for q in data["questions"] if q["category"] == cat]
            assert len(cat_qs) == 10

    def test_each_question_has_required_fields(self, dataset_path: Path):
        data = load_dataset(dataset_path)
        for q in data["questions"]:
            assert "id" in q and isinstance(q["id"], str) and q["id"].startswith("q")
            assert "category" in q
            assert "question" in q and len(q["question"]) > 5
            assert "reference_answer" in q and len(q["reference_answer"]) > 0
            assert "notes" in q

    def test_load_dataset_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "non_existent.json")

    def test_load_dataset_invalid_schema_raises(self, tmp_path: Path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="Dataset must contain a 'questions' key"):
            load_dataset(bad_json)

    def test_load_dataset_missing_question_field_raises(self, tmp_path: Path):
        bad_json = tmp_path / "bad_q.json"
        bad_json.write_text(
            json.dumps({"questions": [{"category": "INTERNAL_SUPPORTED"}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing required field"):
            load_dataset(bad_json)


# ──────────────────────────────────────────────────────────────────────────────
# TestResultSchema
# ──────────────────────────────────────────────────────────────────────────────

class TestResultSchema:
    """Validate make_eval_record output schema and defaults."""

    def test_make_eval_record_defaults(self):
        record = make_eval_record(
            question_id="q001",
            category="INTERNAL_SUPPORTED",
            question="What is CRAG?",
            system="crag",
            answer="Corrective RAG",
            action="CORRECT",
            retrieved_count=3,
            relevance_scores=[0.8, 0.6],
            citations=["doc.pdf p.1"],
            latency_seconds=1.23,
            status="success",
        )
        assert record["question_id"] == "q001"
        assert record["category"] == "INTERNAL_SUPPORTED"
        assert record["system"] == "crag"
        assert record["answer"] == "Corrective RAG"
        assert record["action"] == "CORRECT"
        assert record["retrieved_count"] == 3
        assert record["relevance_scores"] == [0.8, 0.6]
        assert record["citations"] == ["doc.pdf p.1"]
        assert record["latency_seconds"] == 1.23
        assert record["status"] == "success"
        assert record["error_message"] is None
        assert "timestamp" in record
        assert isinstance(record["extra"], dict)

    def test_make_eval_record_error_status(self):
        record = make_eval_record(
            question_id="q002",
            category="INTERNAL_IRRELEVANT",
            question="Query?",
            system="baseline",
            status="error",
            error_message="API Timeout",
        )
        assert record["status"] == "error"
        assert record["error_message"] == "API Timeout"
        assert record["answer"] is None


# ──────────────────────────────────────────────────────────────────────────────
# TestMetricsAnswerCorrectness
# ──────────────────────────────────────────────────────────────────────────────

class TestMetricsAnswerCorrectness:
    """Test keyword overlap and F1 calculations."""

    def test_exact_match_yields_perfect_f1(self):
        ans = "Corrective Retrieval Augmented Generation"
        ref = "Corrective Retrieval Augmented Generation"
        assert keyword_overlap_score(ans, ref) == 1.0
        assert contains_reference_keywords(ans, ref) is True

    def test_partial_match_yields_bounded_f1(self):
        ans = "Corrective Retrieval system"
        ref = "Corrective Retrieval Augmented Generation"
        score = keyword_overlap_score(ans, ref)
        assert 0.0 < score < 1.0

    def test_no_overlap_yields_zero_f1(self):
        ans = "The weather is sunny today"
        ref = "Corrective Retrieval Augmented Generation"
        assert keyword_overlap_score(ans, ref) == 0.0
        assert contains_reference_keywords(ans, ref, threshold=0.1) is False

    def test_empty_string_yields_zero_f1(self):
        assert keyword_overlap_score("", "reference") == 0.0
        assert keyword_overlap_score("answer", "") == 0.0
        assert keyword_overlap_score("   ", "   ") == 0.0

    def test_punctuation_and_casing_ignored(self):
        ans = "Paris, France!"
        ref = "paris france"
        assert keyword_overlap_score(ans, ref) == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# TestMetricsCitationRecall
# ──────────────────────────────────────────────────────────────────────────────

class TestMetricsCitationRecall:
    """Test citation recall metric."""

    def test_perfect_citation_recall(self):
        citations = ["CRAG.pdf p.1", "other_doc.pdf p.2"]
        expected = ["CRAG.pdf", "other_doc.pdf"]
        assert citation_recall_score(citations, expected) == 1.0

    def test_partial_citation_recall(self):
        citations = ["CRAG.pdf p.1"]
        expected = ["CRAG.pdf", "missing_doc.pdf"]
        assert citation_recall_score(citations, expected) == 0.5

    def test_zero_citation_recall(self):
        citations = ["unrelated.pdf p.1"]
        expected = ["CRAG.pdf"]
        assert citation_recall_score(citations, expected) == 0.0

    def test_empty_expected_sources_returns_one(self):
        assert citation_recall_score(["doc.pdf p.1"], []) == 1.0

    def test_empty_citations_with_nonempty_expected_returns_zero(self):
        assert citation_recall_score([], ["CRAG.pdf"]) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# TestMetricsActionDist
# ──────────────────────────────────────────────────────────────────────────────

class TestMetricsActionDist:
    """Test action distribution computation."""

    def test_action_distribution_counts_and_percentages(self):
        results = [
            {"action": "CORRECT"},
            {"action": "CORRECT"},
            {"action": "INCORRECT"},
            {"action": "AMBIGUOUS"},
            {"action": None},  # Baseline, should be ignored
        ]
        dist = action_distribution(results)
        assert dist["total_crag_runs"] == 4
        assert dist["counts"]["CORRECT"] == 2
        assert dist["counts"]["INCORRECT"] == 1
        assert dist["counts"]["AMBIGUOUS"] == 1
        assert dist["percentages"]["CORRECT"] == 50.0
        assert dist["percentages"]["INCORRECT"] == 25.0
        assert dist["percentages"]["AMBIGUOUS"] == 25.0

    def test_action_distribution_empty_results(self):
        dist = action_distribution([])
        assert dist["total_crag_runs"] == 0
        assert dist["counts"]["CORRECT"] == 0

    def test_action_distribution_by_category(self):
        results = [
            {"category": "INTERNAL_SUPPORTED", "action": "CORRECT"},
            {"category": "INTERNAL_SUPPORTED", "action": "CORRECT"},
            {"category": "INTERNAL_IRRELEVANT", "action": "INCORRECT"},
        ]
        breakdown = action_distribution_by_category(results)
        assert "INTERNAL_SUPPORTED" in breakdown
        assert breakdown["INTERNAL_SUPPORTED"]["counts"]["CORRECT"] == 2
        assert breakdown["INTERNAL_IRRELEVANT"]["counts"]["INCORRECT"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# TestMetricsLatencyAndError
# ──────────────────────────────────────────────────────────────────────────────

class TestMetricsLatencyAndError:
    """Test latency and error rate metrics."""

    def test_mean_latency_success_only(self):
        results = [
            {"status": "success", "latency_seconds": 1.0},
            {"status": "success", "latency_seconds": 3.0},
            {"status": "error", "latency_seconds": 10.0},  # error ignored
        ]
        assert mean_latency_seconds(results) == 2.0

    def test_latency_percentile(self):
        results = [
            {"status": "success", "latency_seconds": float(i)}
            for i in range(1, 101)
        ]
        p95 = latency_percentile(results, 95.0)
        assert p95 == 95.0

    def test_error_rate_calculation(self):
        results = [
            {"status": "success"},
            {"status": "success"},
            {"status": "error"},
            {"status": "success"},
        ]
        assert error_rate(results) == 0.25

    def test_error_rate_empty_returns_zero(self):
        assert error_rate([]) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# TestMetricsScoringAndAgg
# ──────────────────────────────────────────────────────────────────────────────

class TestMetricsScoringAndAgg:
    """Test score_single_result, aggregate_by_category, and compute_summary."""

    def test_score_single_result_with_reference(self):
        res = {
            "question_id": "q001",
            "system": "crag",
            "status": "success",
            "answer": "Paris is the capital",
            "citations": ["https://paris.org p.1"],
        }
        scored = score_single_result(
            result=res,
            reference_answer="Paris",
            expected_sources=["https://paris.org"],
        )
        assert scored["question_id"] == "q001"
        assert isinstance(scored["keyword_f1"], float)
        assert scored["citation_recall"] == 1.0
        assert scored["requires_manual_review"] is False

    def test_score_single_result_no_sources_flags_manual_review(self):
        res = {
            "question_id": "q001",
            "system": "baseline",
            "status": "success",
            "answer": "Paris",
            "citations": [],
        }
        scored = score_single_result(
            result=res,
            reference_answer="Paris",
            expected_sources=None,
        )
        assert scored["citation_recall"] == "manual_review_required"
        assert scored["requires_manual_review"] is True

    def test_aggregate_by_category(self):
        scored = [
            {"category": "INTERNAL_SUPPORTED", "keyword_f1": 0.8},
            {"category": "INTERNAL_SUPPORTED", "keyword_f1": 0.6},
            {"category": "INTERNAL_IRRELEVANT", "keyword_f1": "manual_review_required"},
        ]
        agg = aggregate_by_category(scored)
        assert agg["INTERNAL_SUPPORTED"]["count"] == 2
        assert agg["INTERNAL_SUPPORTED"]["mean_keyword_f1"] == 0.7
        assert agg["INTERNAL_SUPPORTED"]["n_manual_review"] == 0
        assert agg["INTERNAL_IRRELEVANT"]["count"] == 1
        assert agg["INTERNAL_IRRELEVANT"]["mean_keyword_f1"] is None
        assert agg["INTERNAL_IRRELEVANT"]["n_manual_review"] == 1

    def test_compute_summary(self, sample_questions: list[dict]):
        questions_by_id = {q["id"]: q for q in sample_questions}
        results = [
            {
                "question_id": "q001",
                "system": "crag",
                "status": "success",
                "answer": "Corrective Retrieval Augmented Generation",
                "action": "CORRECT",
                "latency_seconds": 1.5,
                "citations": ["CRAG.pdf p.1"],
            },
            {
                "question_id": "q002",
                "system": "crag",
                "status": "success",
                "answer": "Paris",
                "action": "INCORRECT",
                "latency_seconds": 2.5,
                "citations": ["https://paris.org p.1"],
            },
        ]
        summary = compute_summary(results, questions_by_id, system="crag")
        assert summary["system"] == "crag"
        assert summary["total_questions"] == 2
        assert summary["error_rate"] == 0.0
        assert summary["mean_latency_seconds"] == 2.0
        assert summary["mean_keyword_f1"] == 1.0
        assert summary["action_distribution"]["counts"]["CORRECT"] == 1
        assert summary["action_distribution"]["counts"]["INCORRECT"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# TestEvaluationRunner
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluationRunner:
    """Test EvaluationRunner execution with mocked BaselineRAG and CRAGPipeline."""

    def test_run_baseline_mocked(self, sample_questions: list[dict]):
        mock_baseline = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "Baseline Answer"
        mock_result.retrieved_chunks = [MagicMock(), MagicMock()]
        mock_result.sources = [MagicMock(source="doc.pdf", page_number=1)]
        mock_baseline.run.return_value = mock_result

        runner = EvaluationRunner(baseline_rag=mock_baseline, verbose=False)
        records = runner.run_baseline(sample_questions)

        assert len(records) == 3
        assert mock_baseline.run.call_count == 3
        for r in records:
            assert r["system"] == "baseline"
            assert r["status"] == "success"
            assert r["answer"] == "Baseline Answer"
            assert r["action"] is None
            assert r["retrieved_count"] == 2

    def test_run_crag_mocked(self, sample_questions: list[dict]):
        mock_crag = MagicMock()
        mock_result = MagicMock()
        mock_result.answer = "CRAG Answer"
        mock_result.action = "CORRECT"
        mock_result.retrieved_chunks = [MagicMock()]
        mock_result.relevance_scores = [0.85]
        mock_result.refined_strips = [MagicMock(source="CRAG.pdf", page_number=1)]
        mock_result.external_strips = []
        mock_result.rewritten_query = None
        mock_result.trace = MagicMock(
            max_relevance_score=0.85,
            final_context_source="internal",
        )
        mock_crag.run.return_value = mock_result

        runner = EvaluationRunner(crag_pipeline=mock_crag, verbose=False)
        records = runner.run_crag(sample_questions)

        assert len(records) == 3
        assert mock_crag.run.call_count == 3
        for r in records:
            assert r["system"] == "crag"
            assert r["status"] == "success"
            assert r["answer"] == "CRAG Answer"
            assert r["action"] == "CORRECT"
            assert r["relevance_scores"] == [0.85]
            assert r["extra"]["final_context_source"] == "internal"

    def test_runner_handles_system_exceptions_gracefully(self, sample_questions: list[dict]):
        mock_baseline = MagicMock()
        mock_baseline.run.side_effect = RuntimeError("Chroma connection error")

        runner = EvaluationRunner(baseline_rag=mock_baseline, verbose=False)
        records = runner.run_baseline(sample_questions[:1])

        assert len(records) == 1
        assert records[0]["status"] == "error"
        assert "Chroma connection error" in records[0]["error_message"]
        assert records[0]["answer"] is None

    def test_run_both_systems(self, sample_questions: list[dict]):
        mock_baseline = MagicMock()
        mock_baseline.run.return_value = MagicMock(
            answer="B", retrieved_chunks=[], sources=[]
        )
        mock_crag = MagicMock()
        mock_crag.run.return_value = MagicMock(
            answer="C",
            action="CORRECT",
            retrieved_chunks=[],
            relevance_scores=[],
            refined_strips=[],
            external_strips=[],
            rewritten_query=None,
            trace=None,
        )

        runner = EvaluationRunner(
            baseline_rag=mock_baseline,
            crag_pipeline=mock_crag,
            verbose=False,
        )
        records = runner.run(sample_questions, systems=["baseline", "crag"])
        assert len(records) == 6
        systems = [r["system"] for r in records]
        assert systems.count("baseline") == 3
        assert systems.count("crag") == 3

    def test_citations_extraction_helpers(self):
        # Baseline RAG result
        mock_rag_res = MagicMock()
        mock_rag_res.sources = [
            MagicMock(source="doc.pdf", page_number=2),
            MagicMock(source="doc2.pdf", page_number=3),
        ]
        cites = _citations_from_rag_result(mock_rag_res)
        assert cites == ["doc.pdf p.2", "doc2.pdf p.3"]

        # CRAG result
        mock_crag_res = MagicMock()
        mock_crag_res.refined_strips = [MagicMock(source="internal.pdf", page_number=1)]
        mock_crag_res.external_strips = [MagicMock(source="https://web.com", page_number=1)]
        cites_crag = _citations_from_crag_result(mock_crag_res)
        assert "internal.pdf p.1" in cites_crag
        assert "https://web.com p.1" in cites_crag


# ──────────────────────────────────────────────────────────────────────────────
# TestResultsIO
# ──────────────────────────────────────────────────────────────────────────────

class TestResultsIO:
    """Test saving and loading evaluation results."""

    def test_save_and_load_results(self, tmp_path: Path):
        output_file = tmp_path / "eval" / "results.json"
        records = [
            make_eval_record(
                question_id="q001",
                category="INTERNAL_SUPPORTED",
                question="What is CRAG?",
                system="crag",
                answer="Answer",
                status="success",
            )
        ]
        metadata = {"version": "1.0", "document_context": "CRAG.pdf"}

        save_results(records, metadata, output_file)
        assert output_file.exists()

        loaded = load_results(output_file)
        assert loaded["run_metadata"]["total_records"] == 1
        assert loaded["run_metadata"]["systems"] == ["crag"]
        assert len(loaded["records"]) == 1
        assert loaded["records"][0]["question_id"] == "q001"

    def test_load_non_existent_results_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_results(tmp_path / "does_not_exist.json")
