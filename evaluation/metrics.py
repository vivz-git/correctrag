"""
Evaluation Metrics for CorrectRAG Baseline vs CRAG Comparison.

All metrics in this module are:
- Transparent: no black-box LLM scoring; all logic is explicit
- Reproducible: same inputs always produce same outputs
- Conservative: when exact computation is impossible, the metric is
  flagged as "manual review required" rather than estimating a score

Metric functions operate on plain Python dicts and lists; they do not
import any RAG or CRAG pipeline code.
"""

from __future__ import annotations

import re
import math
from collections import Counter
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Text normalisation
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    """Return a list of whitespace-separated tokens after normalisation."""
    return _normalize(text).split()


# ──────────────────────────────────────────────────────────────────────────────
# Answer correctness
# ──────────────────────────────────────────────────────────────────────────────

def keyword_overlap_score(answer: str, reference: str) -> float:
    """Token-level F1 between the generated answer and the reference answer.

    This is the same bag-of-words F1 used as an automatic metric in many QA
    benchmarks (e.g., SQuAD, TriviaQA).  It is transparent, cheap to compute,
    and reproducible.  It is NOT a semantic similarity measure — high lexical
    overlap does not guarantee correctness, and paraphrastic answers will score
    lower than they should.  Use alongside manual review for critical items.

    Returns:
        F1 score in [0.0, 1.0].  Returns 0.0 when either string is empty.
    """
    if not answer.strip() or not reference.strip():
        return 0.0

    pred_tokens = Counter(_tokenize(answer))
    ref_tokens = Counter(_tokenize(reference))

    common = sum((pred_tokens & ref_tokens).values())
    if common == 0:
        return 0.0

    precision = common / sum(pred_tokens.values())
    recall = common / sum(ref_tokens.values())
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


def contains_reference_keywords(answer: str, reference: str, threshold: float = 0.3) -> bool:
    """Return True when keyword_overlap_score meets or exceeds threshold."""
    return keyword_overlap_score(answer, reference) >= threshold


# ──────────────────────────────────────────────────────────────────────────────
# Citation / source correctness
# ──────────────────────────────────────────────────────────────────────────────

def citation_recall_score(citations: list[str], expected_sources: list[str]) -> float:
    """Fraction of expected sources that appear (substring match) in citations.

    Args:
        citations:        List of citation strings from the system output.
        expected_sources: Ground-truth sources the answer should cite.

    Returns:
        Recall in [0.0, 1.0].  Returns 1.0 when expected_sources is empty
        (nothing expected, nothing missed).
    """
    if not expected_sources:
        return 1.0
    if not citations:
        return 0.0

    citations_lower = [c.lower() for c in citations]
    matched = sum(
        1
        for src in expected_sources
        if any(src.lower() in c for c in citations_lower)
    )
    return round(matched / len(expected_sources), 4)


# ──────────────────────────────────────────────────────────────────────────────
# CRAG-specific metrics
# ──────────────────────────────────────────────────────────────────────────────

def action_distribution(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute CORRECT / INCORRECT / AMBIGUOUS counts and percentages.

    Args:
        results: List of EvalRecord dicts containing an 'action' key.
                 Records where action is None (e.g., Baseline results) are ignored.

    Returns:
        Dict with keys 'counts' and 'percentages' for each action value,
        plus 'total_crag_runs'.
    """
    crag_results = [r for r in results if r.get("action") is not None]
    total = len(crag_results)
    if total == 0:
        return {
            "counts": {"CORRECT": 0, "INCORRECT": 0, "AMBIGUOUS": 0},
            "percentages": {"CORRECT": 0.0, "INCORRECT": 0.0, "AMBIGUOUS": 0.0},
            "total_crag_runs": 0,
        }

    counts: dict[str, int] = {"CORRECT": 0, "INCORRECT": 0, "AMBIGUOUS": 0}
    for r in crag_results:
        action = r.get("action", "").upper()
        if action in counts:
            counts[action] += 1

    percentages = {
        k: round(v / total * 100, 1) for k, v in counts.items()
    }
    return {
        "counts": counts,
        "percentages": percentages,
        "total_crag_runs": total,
    }


def action_distribution_by_category(results: list[dict[str, Any]]) -> dict[str, Any]:
    """CRAG action distribution broken down by question category."""
    categories = sorted({r.get("category", "UNKNOWN") for r in results})
    breakdown: dict[str, Any] = {}
    for cat in categories:
        cat_results = [r for r in results if r.get("category") == cat]
        breakdown[cat] = action_distribution(cat_results)
    return breakdown


# ──────────────────────────────────────────────────────────────────────────────
# Latency
# ──────────────────────────────────────────────────────────────────────────────

def mean_latency_seconds(results: list[dict[str, Any]]) -> float:
    """Mean latency in seconds across all successful results.

    Returns float('nan') when no successful results are present.
    """
    latencies = [
        r["latency_seconds"]
        for r in results
        if r.get("status") == "success" and r.get("latency_seconds") is not None
    ]
    if not latencies:
        return float("nan")
    return round(sum(latencies) / len(latencies), 3)


def latency_percentile(results: list[dict[str, Any]], percentile: float = 95.0) -> float:
    """Return the p-th percentile latency (e.g., p95) in seconds."""
    latencies = sorted(
        r["latency_seconds"]
        for r in results
        if r.get("status") == "success" and r.get("latency_seconds") is not None
    )
    if not latencies:
        return float("nan")
    idx = math.ceil(percentile / 100 * len(latencies)) - 1
    return round(latencies[max(0, idx)], 3)


# ──────────────────────────────────────────────────────────────────────────────
# Error rate
# ──────────────────────────────────────────────────────────────────────────────

def error_rate(results: list[dict[str, Any]]) -> float:
    """Fraction of results with status == 'error'.

    Returns 0.0 when results is empty.
    """
    if not results:
        return 0.0
    errors = sum(1 for r in results if r.get("status") == "error")
    return round(errors / len(results), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Per-result scoring
# ──────────────────────────────────────────────────────────────────────────────

def score_single_result(
    result: dict[str, Any],
    reference_answer: str | None,
    expected_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Compute all available metrics for a single EvalRecord.

    Args:
        result:           EvalRecord dict from the runner.
        reference_answer: Ground-truth answer string (may be None).
        expected_sources: Expected citation sources (may be None or empty).

    Returns:
        Dict with:
          - keyword_f1: float or "manual_review_required"
          - citation_recall: float or "manual_review_required"
          - status: from the result
    """
    scores: dict[str, Any] = {
        "question_id": result.get("question_id"),
        "system": result.get("system"),
        "status": result.get("status"),
        "keyword_f1": None,
        "citation_recall": None,
        "requires_manual_review": False,
    }

    answer = result.get("answer", "")

    # Answer correctness
    if result.get("status") != "success":
        scores["keyword_f1"] = "manual_review_required"
        scores["requires_manual_review"] = True
    elif reference_answer:
        scores["keyword_f1"] = keyword_overlap_score(answer, reference_answer)
    else:
        scores["keyword_f1"] = "manual_review_required"
        scores["requires_manual_review"] = True

    # Citation recall
    citations = result.get("citations", [])
    if expected_sources:
        scores["citation_recall"] = citation_recall_score(citations, expected_sources)
    else:
        # No ground-truth sources to check against
        scores["citation_recall"] = "manual_review_required"
        scores["requires_manual_review"] = True

    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_by_category(
    scored_results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Group scored results by category and compute per-category statistics.

    Args:
        scored_results: List of dicts from score_single_result(), each must
                        have a 'category' key added from the question record.

    Returns:
        Dict mapping category name → {mean_keyword_f1, n_manual_review, count}.
    """
    categories = sorted({r.get("category", "UNKNOWN") for r in scored_results})
    agg: dict[str, dict[str, Any]] = {}

    for cat in categories:
        cat_results = [r for r in scored_results if r.get("category") == cat]
        f1_values = [
            r["keyword_f1"]
            for r in cat_results
            if isinstance(r.get("keyword_f1"), float)
        ]
        n_manual = sum(
            1
            for r in cat_results
            if r.get("keyword_f1") == "manual_review_required"
        )
        agg[cat] = {
            "count": len(cat_results),
            "mean_keyword_f1": (
                round(sum(f1_values) / len(f1_values), 4) if f1_values else None
            ),
            "n_manual_review": n_manual,
        }

    return agg


def compute_summary(
    results: list[dict[str, Any]],
    questions_by_id: dict[str, dict[str, Any]],
    system: str,
) -> dict[str, Any]:
    """Compute the full metric summary for one system's results.

    Args:
        results:          All EvalRecord dicts for this system.
        questions_by_id:  Lookup dict from question_id → question record.
        system:           'baseline' or 'crag'.

    Returns:
        Summary dict suitable for inclusion in results.json.
    """
    sys_results = [r for r in results if r.get("system") == system]

    # Score each result
    scored: list[dict[str, Any]] = []
    for r in sys_results:
        qid = r.get("question_id", "")
        q = questions_by_id.get(qid, {})
        s = score_single_result(
            result=r,
            reference_answer=q.get("reference_answer"),
            expected_sources=q.get("expected_sources") or [],
        )
        s["category"] = q.get("category", "UNKNOWN")
        scored.append(s)

    return {
        "system": system,
        "total_questions": len(sys_results),
        "error_rate": error_rate(sys_results),
        "mean_latency_seconds": mean_latency_seconds(sys_results),
        "p95_latency_seconds": latency_percentile(sys_results, 95.0),
        "mean_keyword_f1": (
            round(
                sum(s["keyword_f1"] for s in scored if isinstance(s.get("keyword_f1"), float))
                / max(1, sum(1 for s in scored if isinstance(s.get("keyword_f1"), float))),
                4,
            )
            if any(isinstance(s.get("keyword_f1"), float) for s in scored)
            else None
        ),
        "n_requires_manual_review": sum(1 for s in scored if s.get("requires_manual_review")),
        "action_distribution": action_distribution(sys_results) if system == "crag" else None,
        "action_by_category": (
            action_distribution_by_category(sys_results) if system == "crag" else None
        ),
        "by_category": aggregate_by_category(scored),
    }
