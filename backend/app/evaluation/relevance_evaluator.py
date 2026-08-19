"""
Relevance Evaluator for CorrectRAG.

[PAPER] The original CRAG paper (arXiv:2401.15884v3) uses a fine-tuned
T5-large model as the retrieval evaluator. The T5-large evaluator was
specifically fine-tuned for query-document relevance classification and
produces a bounded confidence score used by the CRAG action trigger.

[OUR ADAPTATION] We use a frozen cross-encoder
(cross-encoder/ms-marco-MiniLM-L-6-v2) as a practical production
substitute. The cross-encoder was not fine-tuned for CRAG; it is a
general-purpose MS MARCO passage ranking model. Raw model logits are
mapped to [-1, 1] via a temperature-scaled sigmoid transformation.

IMPORTANT: The temperature parameter and any future bias parameters have
NOT been fitted on labelled validation data. The mapping function
produces bounded relevance scores for development use only. Empirical
fitting against human-labelled query-document pairs is a future task.

This module is fully independent of ChromaDB, FastAPI, Gemini, and the
CRAG routing layer. It scores individual query-document pairs only.
Downstream routing (CORRECT / INCORRECT / AMBIGUOUS) is NOT implemented
here.
"""

import math
from typing import Optional

try:
    from sentence_transformers import CrossEncoder
except ImportError as exc:
    raise ImportError(
        "sentence-transformers is required for RelevanceEvaluator. "
        "Run: pip install sentence-transformers"
    ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Bounded relevance score mapping helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function.

    Uses the positive branch for x >= 0 and the negative branch for x < 0
    to avoid overflow in math.exp.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def map_logit_to_score(logit: float, temperature: float = 1.0) -> float:
    """Map a raw cross-encoder logit to a bounded relevance score in [-1, 1].

    [OUR ADAPTATION] This mapping is NOT equivalent to the paper's T5-large
    scoring, and the temperature parameter has NOT been fitted on labelled
    validation data. The output is a bounded relevance indicator suitable
    for development use. Empirical parameter fitting is a future task.

    Transformation pipeline:
        1.  scaled = logit / temperature       (temperature scaling)
        2.  p      = sigmoid(scaled)           (maps to (0, 1))
        3.  score  = 2 * p - 1                 (rescales to (-1, 1))
        4.  score  = clamp(score, -1.0, 1.0)   (safety clamp)

    Interpretation (with default temperature=1.0):
        logit ≫  0  →  score ≈ +1  (highly relevant)
        logit  = 0  →  score  = 0  (neutral)
        logit ≪  0  →  score ≈ -1  (irrelevant)

    Args:
        logit:       Raw model output (unbounded float).
        temperature: Positive scalar. Higher values produce softer (less
                     extreme) scores. Default 1.0 (no scaling).
                     NOTE: This value has not been fitted on validation data.

    Returns:
        Bounded relevance score clamped to [-1.0, 1.0].

    Raises:
        ValueError: If temperature is not positive.
    """
    if temperature <= 0:
        raise ValueError(
            f"temperature must be a positive float, got {temperature!r}."
        )
    p = _sigmoid(logit / temperature)
    score = 2.0 * p - 1.0
    return float(max(-1.0, min(1.0, score)))


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator class
# ─────────────────────────────────────────────────────────────────────────────

class RelevanceEvaluator:
    """Query-document relevance scorer for the CorrectRAG pipeline.

    [PAPER] The original CRAG paper trains a T5-large evaluator to score
    retrieved documents against the query and assigns one of three confidence
    bands: CORRECT, INCORRECT, or AMBIGUOUS.

    [OUR ADAPTATION] This class wraps the frozen cross-encoder
    ``cross-encoder/ms-marco-MiniLM-L-6-v2`` from Sentence-Transformers.
    The model is loaded lazily on first use and cached for the lifetime of
    the instance (never reloaded per request).

    Raw logits are mapped to [-1, 1] via a temperature-scaled sigmoid
    (see ``map_logit_to_score``). The temperature parameter is a development
    default and has NOT been fitted on labelled validation data. The output
    should be treated as a bounded relevance indicator, not a calibrated
    confidence score.

    This component is independent of ChromaDB, FastAPI, Gemini, and the
    CRAG action trigger. It does NOT implement routing.

    Example::

        evaluator = RelevanceEvaluator()
        score = evaluator.score("What is RAG?", "RAG combines retrieval with generation.")
        # score is a float in [-1.0, 1.0]
    """

    DEFAULT_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 1.0,
    ) -> None:
        """Initialise the evaluator.

        Args:
            model_name:  HuggingFace cross-encoder model identifier.
                         Defaults to ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
            temperature: Scaling factor for the logit-to-score mapping.
                         Must be positive. Higher values produce softer scores.
                         NOTE: This is a development default; it has NOT been
                         fitted on labelled validation data.

        Raises:
            ValueError: If temperature is not positive.
        """
        if temperature <= 0:
            raise ValueError(
                f"temperature must be a positive float, got {temperature!r}."
            )
        self._model_name: str = model_name or self.DEFAULT_MODEL
        self._temperature: float = temperature
        self._model: Optional[CrossEncoder] = None   # lazy-loaded on first call

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load and cache the cross-encoder model.

        The model is downloaded and initialised on the first call only.
        Subsequent calls reuse the cached instance.
        """
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

    # ── Input validation ──────────────────────────────────────────────────────

    @staticmethod
    def _validate_pair(query: str, document: str, index: Optional[int] = None) -> None:
        """Validate a single query-document pair.

        Args:
            query:    Query string.
            document: Document/chunk text string.
            index:    Optional batch index for error messages.

        Raises:
            ValueError: If query or document is empty or whitespace-only.
        """
        loc = f" at index {index}" if index is not None else ""
        if not query or not query.strip():
            raise ValueError(f"query{loc} must be a non-empty string.")
        if not document or not document.strip():
            raise ValueError(f"document{loc} must be a non-empty string.")

    # ── Public interface ──────────────────────────────────────────────────────

    def score(self, query: str, document: str) -> float:
        """Score a single query-document pair for relevance.

        Args:
            query:    The user query string.
            document: The retrieved document or chunk text.

        Returns:
            Bounded relevance score in [-1.0, 1.0].
            Values closer to +1.0 indicate higher relevance.
            Values closer to -1.0 indicate irrelevance.
            NOTE: This is not a calibrated confidence score.

        Raises:
            ValueError: If query or document is empty or whitespace-only.
        """
        self._validate_pair(query, document)
        raw_logits = self.model.predict([(query, document)])
        raw_logit = float(raw_logits[0])
        return map_logit_to_score(raw_logit, self._temperature)

    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score a batch of query-document pairs.

        Input order is strictly preserved in the output list.

        Args:
            pairs: Non-empty list of ``(query, document)`` tuples.

        Returns:
            List of bounded relevance scores in [-1.0, 1.0],
            same length and order as ``pairs``.
            NOTE: These are not calibrated confidence scores.

        Raises:
            ValueError: If ``pairs`` is empty.
            ValueError: If any query or document in the batch is empty.
        """
        if not pairs:
            raise ValueError("pairs must be a non-empty list.")
        for i, (query, document) in enumerate(pairs):
            self._validate_pair(query, document, index=i)
        raw_logits = self.model.predict(pairs)
        return [
            map_logit_to_score(float(logit), self._temperature)
            for logit in raw_logits
        ]
