"""
Unit tests for the CRAG Action Router.

All tests are pure unit tests with no external dependencies or models.

Test coverage:
  - Routing to CORRECT when s_max > alpha
  - Routing to INCORRECT when s_max < beta
  - Routing to AMBIGUOUS when beta <= s_max <= alpha
  - Exact boundary conditions: s_max == alpha and s_max == beta
  - Multi-score handling (using max score)
  - Mixed scores (single high score triggering CORRECT)
  - Uniform low scores (triggering INCORRECT)
  - Threshold configuration validation (alpha > beta, bounds in [-1, 1], type checks)
  - Score input validation (empty list, bounds in [-1, 1], type checks)
  - Statelessness and determinism
"""

import pytest

from app.evaluation.action_router import ActionRouter, Action


class TestActionRouterInit:
    """Test initialization and threshold validation in ActionRouter."""

    def test_valid_threshold_initialization(self):
        router = ActionRouter(alpha=0.7, beta=0.2)
        assert router.alpha == 0.7
        assert router.beta == 0.2

    def test_valid_negative_thresholds(self):
        router = ActionRouter(alpha=-0.1, beta=-0.6)
        assert router.alpha == -0.1
        assert router.beta == -0.6

    def test_valid_extreme_thresholds(self):
        router = ActionRouter(alpha=1.0, beta=-1.0)
        assert router.alpha == 1.0
        assert router.beta == -1.0

    def test_alpha_equal_to_beta_raises_value_error(self):
        with pytest.raises(ValueError, match="strictly greater than"):
            ActionRouter(alpha=0.5, beta=0.5)

    def test_alpha_less_than_beta_raises_value_error(self):
        with pytest.raises(ValueError, match="strictly greater than"):
            ActionRouter(alpha=0.2, beta=0.7)

    def test_alpha_greater_than_one_raises_value_error(self):
        with pytest.raises(ValueError, match="alpha must be within"):
            ActionRouter(alpha=1.5, beta=0.2)

    def test_alpha_less_than_minus_one_raises_value_error(self):
        with pytest.raises(ValueError, match="alpha must be within"):
            ActionRouter(alpha=-1.5, beta=-1.8)

    def test_beta_less_than_minus_one_raises_value_error(self):
        with pytest.raises(ValueError, match="beta must be within"):
            ActionRouter(alpha=0.5, beta=-1.5)

    def test_beta_greater_than_one_raises_value_error(self):
        with pytest.raises(ValueError, match="beta must be within"):
            ActionRouter(alpha=0.5, beta=1.2)

    def test_non_numeric_alpha_raises_type_error(self):
        with pytest.raises(TypeError, match="alpha must be a numeric"):
            ActionRouter(alpha="high", beta=0.2)  # type: ignore

    def test_boolean_alpha_raises_type_error(self):
        with pytest.raises(TypeError, match="alpha must be a numeric"):
            ActionRouter(alpha=True, beta=0.2)  # type: ignore

    def test_non_numeric_beta_raises_type_error(self):
        with pytest.raises(TypeError, match="beta must be a numeric"):
            ActionRouter(alpha=0.8, beta=None)  # type: ignore

    def test_boolean_beta_raises_type_error(self):
        with pytest.raises(TypeError, match="beta must be a numeric"):
            ActionRouter(alpha=0.8, beta=False)  # type: ignore


class TestActionRouterDecisionRules:
    """Test three-way routing rules and boundary logic."""

    @pytest.fixture
    def router(self) -> ActionRouter:
        # alpha = 0.6, beta = 0.1
        return ActionRouter(alpha=0.6, beta=0.1)

    def test_route_correct_when_single_score_above_alpha(self, router: ActionRouter):
        action = router.route([0.85])
        assert action == "CORRECT"

    def test_route_incorrect_when_single_score_below_beta(self, router: ActionRouter):
        action = router.route([-0.2])
        assert action == "INCORRECT"

    def test_route_ambiguous_when_single_score_between_beta_and_alpha(self, router: ActionRouter):
        action = router.route([0.35])
        assert action == "AMBIGUOUS"

    def test_route_ambiguous_at_exact_upper_boundary_alpha(self, router: ActionRouter):
        # s_max == alpha -> AMBIGUOUS (since s_max > alpha is required for CORRECT)
        action = router.route([0.6])
        assert action == "AMBIGUOUS"

    def test_route_ambiguous_at_exact_lower_boundary_beta(self, router: ActionRouter):
        # s_max == beta -> AMBIGUOUS (since s_max < beta is required for INCORRECT)
        action = router.route([0.1])
        assert action == "AMBIGUOUS"

    def test_route_correct_just_above_alpha(self, router: ActionRouter):
        action = router.route([0.6001])
        assert action == "CORRECT"

    def test_route_incorrect_just_below_beta(self, router: ActionRouter):
        action = router.route([0.0999])
        assert action == "INCORRECT"

    def test_multiple_scores_uses_max_score(self, router: ActionRouter):
        # scores: [-0.5, 0.0, 0.3, 0.75] -> max is 0.75 > 0.6 -> CORRECT
        action = router.route([-0.5, 0.0, 0.3, 0.75])
        assert action == "CORRECT"

    def test_one_high_score_among_low_scores_yields_correct(self, router: ActionRouter):
        # Even if 4 of 5 are very low, if one chunk is highly relevant, route is CORRECT
        action = router.route([-0.9, -0.8, -0.7, 0.95, -0.6])
        assert action == "CORRECT"

    def test_all_low_scores_yields_incorrect(self, router: ActionRouter):
        # All chunks strictly below beta (0.1)
        action = router.route([-0.8, -0.5, -0.3, -0.25])
        assert action == "INCORRECT"

    def test_all_mid_scores_yields_ambiguous(self, router: ActionRouter):
        # All chunks between beta (0.1) and alpha (0.6)
        action = router.route([0.15, 0.25, 0.45, 0.55])
        assert action == "AMBIGUOUS"

    def test_mixed_mid_and_low_scores_uses_max_for_ambiguous(self, router: ActionRouter):
        # Max is 0.45, which falls in [0.1, 0.6] -> AMBIGUOUS
        action = router.route([-0.8, -0.4, 0.45, 0.0])
        assert action == "AMBIGUOUS"


class TestActionRouterValidationAndEdgeCases:
    """Test input validation for score sequences and edge conditions."""

    @pytest.fixture
    def router(self) -> ActionRouter:
        return ActionRouter(alpha=0.5, beta=-0.2)

    def test_empty_scores_list_raises_value_error(self, router: ActionRouter):
        with pytest.raises(ValueError, match="non-empty"):
            router.route([])

    def test_non_list_input_raises_type_error(self, router: ActionRouter):
        with pytest.raises(TypeError, match="scores must be a list"):
            router.route("0.8")  # type: ignore

    def test_out_of_range_high_score_raises_value_error(self, router: ActionRouter):
        with pytest.raises(ValueError, match="out of valid range"):
            router.route([0.2, 1.05])

    def test_out_of_range_low_score_raises_value_error(self, router: ActionRouter):
        with pytest.raises(ValueError, match="out of valid range"):
            router.route([-1.5, 0.1])

    def test_non_numeric_score_element_raises_type_error(self, router: ActionRouter):
        with pytest.raises(TypeError, match="must be numeric"):
            router.route([0.4, "0.8"])  # type: ignore

    def test_boolean_score_element_raises_type_error(self, router: ActionRouter):
        with pytest.raises(TypeError, match="must be numeric"):
            router.route([0.4, True])  # type: ignore

    def test_accepts_tuple_of_scores(self, router: ActionRouter):
        action = router.route((0.1, 0.6))  # type: ignore
        assert action == "CORRECT"

    def test_router_is_stateless_and_deterministic(self, router: ActionRouter):
        scores_seq = [0.2, -0.1, 0.4]
        action_1 = router.route(scores_seq)
        action_2 = router.route(scores_seq)
        action_3 = router.route(scores_seq)
        assert action_1 == action_2 == action_3 == "AMBIGUOUS"
