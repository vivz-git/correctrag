"""
Action Router for CorrectRAG.

[PAPER] Section 3.2 ("Action Trigger") of the CRAG paper (arXiv:2401.15884v3)
defines a three-way confidence trigger based on the maximum relevance score
among all retrieved documents for a query:

    s_max = max(score_i) for i in 1..k

The decision rule is:
    - CORRECT:   s_max > alpha
    - INCORRECT: s_max < beta
    - AMBIGUOUS: beta <= s_max <= alpha

Where:
    - alpha is the upper confidence threshold.
    - beta  is the lower confidence threshold.

[OUR ADAPTATION] In the original paper, alpha and beta were empirical thresholds
specifically tuned for their fine-tuned T5-large evaluator and target benchmarks.
In our implementation, thresholds are fully configurable parameters passed to
the ActionRouter constructor. They are not hardcoded or claimed to be universal.

This module contains only pure decision logic. It is stateless, deterministic,
and does not perform retrieval, refinement, or generation.
"""

from typing import Literal

Action = Literal["CORRECT", "INCORRECT", "AMBIGUOUS"]


class ActionRouter:
    """Evaluates relevance scores against thresholds to determine the CRAG action.

    Attributes:
        alpha: Upper confidence threshold for triggering the CORRECT action.
        beta:  Lower confidence threshold below which INCORRECT is triggered.
    """

    def __init__(self, alpha: float, beta: float) -> None:
        """Initialize the ActionRouter with upper and lower thresholds.

        Args:
            alpha: Upper threshold in [-1.0, 1.0].
            beta:  Lower threshold in [-1.0, 1.0]. Must be strictly less than alpha.

        Raises:
            TypeError:  If alpha or beta is not numeric.
            ValueError: If alpha or beta is out of [-1.0, 1.0], or if alpha <= beta.
        """
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha must be a numeric float, got {type(alpha).__name__}")
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise TypeError(f"beta must be a numeric float, got {type(beta).__name__}")

        alpha_f = float(alpha)
        beta_f = float(beta)

        if not (-1.0 <= alpha_f <= 1.0):
            raise ValueError(f"alpha must be within [-1.0, 1.0], got {alpha_f}")
        if not (-1.0 <= beta_f <= 1.0):
            raise ValueError(f"beta must be within [-1.0, 1.0], got {beta_f}")
        if alpha_f <= beta_f:
            raise ValueError(
                f"alpha ({alpha_f}) must be strictly greater than beta ({beta_f})"
            )

        self.alpha: float = alpha_f
        self.beta: float = beta_f

    def route(self, scores: list[float]) -> Action:
        """Determine the CRAG action given relevance scores of retrieved documents.

        Args:
            scores: Non-empty list of relevance scores, each within [-1.0, 1.0].

        Returns:
            Action: One of "CORRECT", "INCORRECT", or "AMBIGUOUS".

        Raises:
            TypeError:  If scores is not a list or contains non-numeric elements.
            ValueError: If scores is empty, or any score is outside [-1.0, 1.0].
        """
        if not isinstance(scores, (list, tuple)):
            raise TypeError(f"scores must be a list of floats, got {type(scores).__name__}")
        if len(scores) == 0:
            raise ValueError("scores list must be non-empty.")

        validated_scores: list[float] = []
        for idx, score in enumerate(scores):
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError(
                    f"Score at index {idx} must be numeric, got {type(score).__name__}"
                )
            score_f = float(score)
            if not (-1.0 <= score_f <= 1.0):
                raise ValueError(
                    f"Score at index {idx} ({score_f}) is out of valid range [-1.0, 1.0]"
                )
            validated_scores.append(score_f)

        s_max = max(validated_scores)

        if s_max > self.alpha:
            return "CORRECT"
        elif s_max < self.beta:
            return "INCORRECT"
        else:
            # beta <= s_max <= alpha
            return "AMBIGUOUS"
