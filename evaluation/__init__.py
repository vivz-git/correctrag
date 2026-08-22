"""
Evaluation Package for CorrectRAG.

Provides dataset loader, runner, and metrics for comparing Baseline RAG vs CRAG.
"""

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
    load_dataset,
    load_results,
    make_eval_record,
    save_results,
)

__all__ = [
    "EvaluationRunner",
    "action_distribution",
    "action_distribution_by_category",
    "aggregate_by_category",
    "citation_recall_score",
    "compute_summary",
    "contains_reference_keywords",
    "error_rate",
    "keyword_overlap_score",
    "latency_percentile",
    "load_dataset",
    "load_results",
    "make_eval_record",
    "mean_latency_seconds",
    "save_results",
    "score_single_result",
]
