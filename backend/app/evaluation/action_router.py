"""
Two-Stage Action Router for CorrectRAG.

Evaluates relevance using a two-stage approach:
1. Stage 1 (Pre-filter): Uses cheap embedding similarity (max score).
   - If s_max >= clearly_relevant_threshold -> CORRECT
   - If s_max <= clearly_irrelevant_threshold -> INCORRECT
2. Stage 2 (LLM Judge): For borderline similarity (middle range), makes exactly
   one LLM call to classify the entire evidence bundle as CORRECT, AMBIGUOUS,
   or INCORRECT, returning a structured JSON response.

This module determines the final CRAG routing action and provides rich
observability metadata.
"""

import json
import time
import re
from typing import Literal, Optional, Any
from pydantic import BaseModel

from app.retrieval.retriever import RetrievedChunk
from app.generation.llm_provider import LLMProvider

Action = Literal["CORRECT", "INCORRECT", "AMBIGUOUS"]

JUDGE_PROMPT_TEMPLATE = """\
You are an expert relevance evaluator for a retrieval-augmented generation (RAG) system.
Your task is to evaluate whether the retrieved evidence is relevant to the user's query.

User Query: {query}

Retrieved Evidence:
{evidence}

Does the retrieved evidence directly answer or strongly support the query?
Is the evidence only partially useful or missing important information?
Is the evidence unrelated or misleading?

Evaluate the quality of the evidence and respond with exactly ONE of these classifications:
CORRECT: The retrieved context is directly relevant and sufficient.
AMBIGUOUS: The context is partially relevant, incomplete, or uncertain.
INCORRECT: The context is irrelevant or misleading.

Also provide a concise one-line reason.

Respond in strict JSON format:
{{
  "decision": "CORRECT" | "AMBIGUOUS" | "INCORRECT",
  "reason": "one concise sentence explaining why"
}}
"""

class RoutingDecision(BaseModel):
    """Encapsulates the final routing decision and its observability metadata."""
    action: Action
    similarity_pre_filter_decision: Optional[Action]
    judge_called: bool
    judge_decision: Optional[Action]
    judge_reason: Optional[str]
    judge_latency: Optional[float]


class ActionRouter:
    """Evaluates relevance scores and evidence to determine the CRAG action."""

    def __init__(self, clearly_relevant_threshold: float, clearly_irrelevant_threshold: float, llm_client: Optional[LLMProvider] = None) -> None:
        """Initialize the ActionRouter.

        Args:
            clearly_relevant_threshold: Upper threshold for the pre-filter.
            clearly_irrelevant_threshold: Lower threshold for the pre-filter.
            llm_client: LLMProvider to use for the Stage 2 judge.
        """
        if isinstance(clearly_relevant_threshold, bool) or not isinstance(clearly_relevant_threshold, (int, float)):
            raise TypeError(f"clearly_relevant_threshold must be a numeric float")
        if isinstance(clearly_irrelevant_threshold, bool) or not isinstance(clearly_irrelevant_threshold, (int, float)):
            raise TypeError(f"clearly_irrelevant_threshold must be a numeric float")

        alpha_f = float(clearly_relevant_threshold)
        beta_f = float(clearly_irrelevant_threshold)

        if alpha_f <= beta_f:
            raise ValueError(
                f"clearly_relevant_threshold ({alpha_f}) must be strictly greater than clearly_irrelevant_threshold ({beta_f})"
            )

        self.alpha: float = alpha_f
        self.beta: float = beta_f
        self.llm_client = llm_client

    def route(
        self, query: str, chunks: list[RetrievedChunk], scores: list[float]
    ) -> RoutingDecision:
        """Determine the CRAG action using the two-stage evaluation process."""
        if not scores:
            raise ValueError("scores list must be non-empty.")
        if len(chunks) != len(scores):
            raise ValueError("chunks and scores lists must have the same length.")

        s_max = max(scores)

        # Stage 1: Similarity Pre-filter
        if s_max >= self.alpha:
            return RoutingDecision(
                action="CORRECT",
                similarity_pre_filter_decision="CORRECT",
                judge_called=False,
                judge_decision=None,
                judge_reason=None,
                judge_latency=None,
            )
        elif s_max <= self.beta:
            return RoutingDecision(
                action="INCORRECT",
                similarity_pre_filter_decision="INCORRECT",
                judge_called=False,
                judge_decision=None,
                judge_reason=None,
                judge_latency=None,
            )

        # Stage 2: LLM Judge (Borderline)
        pre_filter_decision: Action = "AMBIGUOUS"

        if not self.llm_client:
            # Fallback if no LLM configured: just return AMBIGUOUS
            return RoutingDecision(
                action="AMBIGUOUS",
                similarity_pre_filter_decision=pre_filter_decision,
                judge_called=False,
                judge_decision=None,
                judge_reason="LLM judge disabled; defaulted to AMBIGUOUS",
                judge_latency=None,
            )

        # Build evidence block
        evidence_lines = []
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            evidence_lines.append(f"--- Chunk {i+1} (Sim: {score:.3f}) ---")
            evidence_lines.append(chunk.text.strip())
        evidence_text = "\n".join(evidence_lines)

        prompt = JUDGE_PROMPT_TEMPLATE.format(query=query, evidence=evidence_text)

        start_time = time.time()
        try:
            # Tell the provider to generate (ideally it handles JSON mode if supported,
            # or we parse robustly).
            response_text = self.llm_client.generate(prompt)
        except Exception as e:
            # Safe fallback on API failure
            return RoutingDecision(
                action="AMBIGUOUS",
                similarity_pre_filter_decision=pre_filter_decision,
                judge_called=True,
                judge_decision=None,
                judge_reason=f"Judge API failure: {str(e)}",
                judge_latency=time.time() - start_time,
            )
        latency = time.time() - start_time

        # Robust JSON parsing fallback
        parsed_action: Optional[Action] = None
        parsed_reason: Optional[str] = None
        try:
            # Extract JSON block if surrounded by markdown code blocks
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                data = json.loads(response_text)
            
            raw_decision = str(data.get("decision", "")).strip().upper()
            if raw_decision in ["CORRECT", "INCORRECT", "AMBIGUOUS"]:
                parsed_action = raw_decision # type: ignore
            parsed_reason = str(data.get("reason", "")).strip()
        except Exception:
            # If JSON parsing completely fails, try a very generous regex
            raw = response_text.upper()
            if "CORRECT" in raw and "INCORRECT" not in raw:
                parsed_action = "CORRECT"
            elif "INCORRECT" in raw:
                parsed_action = "INCORRECT"
            elif "AMBIGUOUS" in raw:
                parsed_action = "AMBIGUOUS"
            
            parsed_reason = "Failed to parse structured output."

        if not parsed_action:
            parsed_action = "AMBIGUOUS"

        return RoutingDecision(
            action=parsed_action,
            similarity_pre_filter_decision=pre_filter_decision,
            judge_called=True,
            judge_decision=parsed_action,
            judge_reason=parsed_reason or "No reason provided",
            judge_latency=latency,
        )
