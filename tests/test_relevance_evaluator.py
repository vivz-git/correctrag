"""
Offline tests for the CRAG Retrieval Evaluator.

All tests are fully offline. The CrossEncoder model is mocked throughout —
no model downloads or GPU/CPU inference occur during pytest.

Test groups:
  - TestScoreMapping           — map_logit_to_score helper (pure math, no mocking needed)
  - TestRelevanceEvaluatorInit — construction and configuration
  - TestRelevanceEvaluatorScore — score() method with mocked CrossEncoder
  - TestRelevanceEvaluatorBatch — score_batch() method with mocked CrossEncoder
  - TestModelCaching           — verify model is initialized once, not per-call
"""

import math
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from app.evaluation.relevance_evaluator import (
    RelevanceEvaluator,
    map_logit_to_score,
    _sigmoid,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_evaluator(
    logit_values: list[float],
    temperature: float = 1.0,
    model_name: str = "mock-cross-encoder",
) -> RelevanceEvaluator:
    """Build a RelevanceEvaluator whose CrossEncoder is fully mocked.

    The mock's predict() returns the given logit_values as a list of floats,
    matching the real CrossEncoder.predict() behaviour.
    """
    evaluator = RelevanceEvaluator(model_name=model_name, temperature=temperature)
    mock_model = MagicMock()
    mock_model.predict.return_value = logit_values
    # Inject the mock directly into the lazy cache slot
    evaluator._model = mock_model
    return evaluator


# ─────────────────────────────────────────────────────────────────────────────
# TestScoreMapping — pure math tests for map_logit_to_score, no mocking needed
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreMapping:

    def test_mapping_returns_float(self):
        result = map_logit_to_score(0.0)
        assert isinstance(result, float)

    def test_mapping_output_is_bounded_at_zero_logit(self):
        score = map_logit_to_score(0.0)
        assert -1.0 <= score <= 1.0

    def test_mapping_output_is_bounded_at_large_positive_logit(self):
        # Very high logit → should be clamped at or near +1
        score = map_logit_to_score(1000.0)
        assert -1.0 <= score <= 1.0

    def test_mapping_output_is_bounded_at_large_negative_logit(self):
        # Very low logit → should be clamped at or near -1
        score = map_logit_to_score(-1000.0)
        assert -1.0 <= score <= 1.0

    def test_mapping_zero_logit_maps_to_zero_score(self):
        # sigmoid(0) = 0.5 → 2*0.5 - 1 = 0.0
        score = map_logit_to_score(0.0)
        assert abs(score) < 1e-9

    def test_mapping_high_logit_maps_toward_positive_one(self):
        score = map_logit_to_score(10.0)
        assert score > 0.9

    def test_mapping_low_logit_maps_toward_negative_one(self):
        score = map_logit_to_score(-10.0)
        assert score < -0.9

    def test_mapping_is_deterministic(self):
        s1 = map_logit_to_score(3.5, temperature=2.0)
        s2 = map_logit_to_score(3.5, temperature=2.0)
        assert s1 == s2

    def test_higher_temperature_produces_softer_score(self):
        # Same logit, higher temperature → score closer to 0
        score_sharp = map_logit_to_score(5.0, temperature=0.5)
        score_soft = map_logit_to_score(5.0, temperature=5.0)
        assert score_sharp > score_soft

    def test_mapping_raises_on_nonpositive_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            map_logit_to_score(1.0, temperature=0.0)
        with pytest.raises(ValueError, match="temperature"):
            map_logit_to_score(1.0, temperature=-1.0)

    def test_mapping_monotonic_with_logit(self):
        # Larger logit should always produce a larger bounded score
        scores = [map_logit_to_score(float(x)) for x in range(-5, 6)]
        assert scores == sorted(scores)


# ─────────────────────────────────────────────────────────────────────────────
# TestRelevanceEvaluatorInit — construction and configuration
# ─────────────────────────────────────────────────────────────────────────────

class TestRelevanceEvaluatorInit:

    def test_default_model_name(self):
        ev = RelevanceEvaluator.__new__(RelevanceEvaluator)
        ev._model_name = RelevanceEvaluator.DEFAULT_MODEL
        ev._temperature = 1.0
        ev._model = None
        assert ev._model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_custom_model_name_is_stored(self):
        ev = make_evaluator([0.0], model_name="my-custom-encoder")
        assert ev._model_name == "my-custom-encoder"

    def test_temperature_is_stored(self):
        ev = make_evaluator([0.0], temperature=2.5)
        assert ev._temperature == 2.5

    def test_raises_on_zero_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            RelevanceEvaluator(temperature=0.0)

    def test_raises_on_negative_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            RelevanceEvaluator(temperature=-3.0)

    def test_model_is_not_loaded_on_construction(self):
        """Model must remain None until first .score() or .score_batch() call."""
        ev = RelevanceEvaluator.__new__(RelevanceEvaluator)
        ev._model_name = "mock"
        ev._temperature = 1.0
        ev._model = None
        assert ev._model is None


# ─────────────────────────────────────────────────────────────────────────────
# TestRelevanceEvaluatorScore — score() method
# ─────────────────────────────────────────────────────────────────────────────

class TestRelevanceEvaluatorScore:

    def test_score_returns_float(self):
        ev = make_evaluator([2.0])
        result = ev.score("What is RAG?", "RAG combines retrieval and generation.")
        assert isinstance(result, float)

    def test_score_is_bounded(self):
        for logit in [-100.0, -1.0, 0.0, 1.0, 100.0]:
            ev = make_evaluator([logit])
            s = ev.score("query", "document")
            assert -1.0 <= s <= 1.0, f"score {s} out of bounds for logit {logit}"

    def test_score_high_logit_produces_high_score(self):
        ev = make_evaluator([15.0])
        s = ev.score("query", "document")
        assert s > 0.9

    def test_score_low_logit_produces_low_score(self):
        ev = make_evaluator([-15.0])
        s = ev.score("query", "document")
        assert s < -0.9

    def test_score_raises_on_empty_query(self):
        ev = make_evaluator([1.0])
        with pytest.raises(ValueError, match="query"):
            ev.score("", "some document text")

    def test_score_raises_on_whitespace_query(self):
        ev = make_evaluator([1.0])
        with pytest.raises(ValueError, match="query"):
            ev.score("   ", "some document text")

    def test_score_raises_on_empty_document(self):
        ev = make_evaluator([1.0])
        with pytest.raises(ValueError, match="document"):
            ev.score("What is RAG?", "")

    def test_score_raises_on_whitespace_document(self):
        ev = make_evaluator([1.0])
        with pytest.raises(ValueError, match="document"):
            ev.score("What is RAG?", "   ")

    def test_relevant_pair_scores_higher_than_irrelevant(self):
        """Mock two calls: one high-logit (relevant), one low-logit (irrelevant)."""
        ev_relevant = make_evaluator([8.0])
        ev_irrelevant = make_evaluator([-8.0])

        relevant_score = ev_relevant.score("CRAG evaluation", "CRAG uses a retrieval evaluator.")
        irrelevant_score = ev_irrelevant.score("CRAG evaluation", "The weather in Paris is sunny.")

        assert relevant_score > irrelevant_score


# ─────────────────────────────────────────────────────────────────────────────
# TestRelevanceEvaluatorBatch — score_batch() method
# ─────────────────────────────────────────────────────────────────────────────

class TestRelevanceEvaluatorBatch:

    def test_batch_returns_list(self):
        ev = make_evaluator([1.0, -1.0, 0.0])
        results = ev.score_batch([
            ("q1", "d1"),
            ("q2", "d2"),
            ("q3", "d3"),
        ])
        assert isinstance(results, list)

    def test_batch_length_matches_input(self):
        pairs = [("q", "d")] * 4
        ev = make_evaluator([0.5, -0.5, 1.0, -1.0])
        results = ev.score_batch(pairs)
        assert len(results) == 4

    def test_batch_preserves_order(self):
        """Input order must be reflected in output order."""
        logits = [10.0, -10.0, 0.0]
        ev = make_evaluator(logits)
        results = ev.score_batch([("qa", "da"), ("qb", "db"), ("qc", "dc")])
        # Higher logit → higher score; order should match logit order
        assert results[0] > results[2] > results[1]

    def test_batch_all_scores_bounded(self):
        logits = [-100.0, 0.0, 100.0]
        ev = make_evaluator(logits)
        results = ev.score_batch([("q", "d"), ("q", "d"), ("q", "d")])
        for s in results:
            assert -1.0 <= s <= 1.0

    def test_batch_raises_on_empty_list(self):
        ev = make_evaluator([])
        with pytest.raises(ValueError, match="non-empty"):
            ev.score_batch([])

    def test_batch_raises_on_empty_query_in_pair(self):
        ev = make_evaluator([1.0, 1.0])
        with pytest.raises(ValueError, match="query"):
            ev.score_batch([("valid", "doc"), ("", "doc")])

    def test_batch_raises_on_empty_document_in_pair(self):
        ev = make_evaluator([1.0, 1.0])
        with pytest.raises(ValueError, match="document"):
            ev.score_batch([("query", "doc"), ("query", "")])

    def test_batch_single_pair_consistent_with_score(self):
        """score_batch with one pair must return same value as score()."""
        logit = 3.7
        ev = make_evaluator([logit])
        batch_result = ev.score_batch([("query", "document")])[0]

        ev2 = make_evaluator([logit])
        single_result = ev2.score("query", "document")

        assert abs(batch_result - single_result) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# TestModelCaching — model must be loaded once, not once per call
# ─────────────────────────────────────────────────────────────────────────────

class TestModelCaching:

    def test_model_property_is_called_once_across_multiple_scores(self):
        """Verify the CrossEncoder constructor is invoked exactly once."""
        with patch(
            "app.evaluation.relevance_evaluator.CrossEncoder"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = [1.0]
            mock_cls.return_value = mock_instance

            ev = RelevanceEvaluator(model_name="mock-model")

            # Make three separate score calls
            ev.score("q1", "d1")
            ev.score("q2", "d2")
            ev.score("q3", "d3")

            # CrossEncoder() constructor must have been called exactly once
            mock_cls.assert_called_once_with("mock-model")

    def test_model_property_is_called_once_for_batch(self):
        """Verify the CrossEncoder constructor is invoked once for a batch."""
        with patch(
            "app.evaluation.relevance_evaluator.CrossEncoder"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = [1.0, -1.0, 0.5]
            mock_cls.return_value = mock_instance

            ev = RelevanceEvaluator(model_name="mock-model")
            ev.score_batch([("q1", "d1"), ("q2", "d2"), ("q3", "d3")])

            mock_cls.assert_called_once_with("mock-model")

    def test_cached_model_instance_is_reused(self):
        """The same CrossEncoder instance must be reused across calls."""
        with patch(
            "app.evaluation.relevance_evaluator.CrossEncoder"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = [0.0]
            mock_cls.return_value = mock_instance

            ev = RelevanceEvaluator(model_name="mock-model")
            _ = ev.model
            _ = ev.model
            _ = ev.model

            # Still only constructed once
            assert mock_cls.call_count == 1
