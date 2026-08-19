"""
Evaluation Package for CorrectRAG.

[OUR ADAPTATION] Provides:
- RelevanceEvaluator: query-document relevance scorer using a frozen cross-encoder
- map_logit_to_score: raw logit -> bounded [-1, 1] score mapping helper
- ActionRouter: three-way confidence decision router (CORRECT, INCORRECT, AMBIGUOUS)
- KnowledgeRefiner: fine-grained document strip decomposition, filtering, and recomposition
"""

from app.evaluation.relevance_evaluator import RelevanceEvaluator, map_logit_to_score
from app.evaluation.action_router import Action, ActionRouter
from app.evaluation.knowledge_refiner import (
    KnowledgeStrip,
    KnowledgeRefiner,
    decompose_text_into_strips,
)

__all__ = [
    "RelevanceEvaluator",
    "map_logit_to_score",
    "Action",
    "ActionRouter",
    "KnowledgeStrip",
    "KnowledgeRefiner",
    "decompose_text_into_strips",
]
