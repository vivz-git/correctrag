"""
Evaluation Package for CorrectRAG.

[OUR ADAPTATION] Provides the CRAG retrieval evaluator using a frozen
cross-encoder in place of the paper's fine-tuned T5-large model, and the
CRAG action router implementing the three-way confidence trigger.

Exports:
    RelevanceEvaluator  — query-document relevance scorer
    map_logit_to_score  — raw logit → bounded [-1, 1] score mapping helper
    Action              — Literal["CORRECT", "INCORRECT", "AMBIGUOUS"]
    ActionRouter        — three-way threshold-based decision router
"""

from app.evaluation.relevance_evaluator import RelevanceEvaluator, map_logit_to_score
from app.evaluation.action_router import Action, ActionRouter

__all__ = [
    "RelevanceEvaluator",
    "map_logit_to_score",
    "Action",
    "ActionRouter",
]
