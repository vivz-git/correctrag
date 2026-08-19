"""
Evaluation Package for CorrectRAG.

[OUR ADAPTATION] Provides the CRAG retrieval evaluator using a frozen
cross-encoder in place of the paper's fine-tuned T5-large model.

Exports:
    RelevanceEvaluator  — query-document relevance scorer
    map_logit_to_score  — raw logit → bounded [-1, 1] score mapping helper
                          (temperature parameter is a development default,
                           NOT fitted on validation data)
"""

from app.evaluation.relevance_evaluator import RelevanceEvaluator, map_logit_to_score

__all__ = [
    "RelevanceEvaluator",
    "map_logit_to_score",
]
