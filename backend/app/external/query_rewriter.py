"""
Query Rewriter Module for CorrectRAG.

[PAPER] Section 4.5 ("Web Search") and Appendix A of the CRAG paper
(arXiv:2401.15884v3) describe query rewriting prior to external search:
When retrieval is deemed incorrect or ambiguous, the query rewriter transforms
the original natural-language question into a concise search query composed
of at most three keywords or key phrases separated by commas.

Example from paper:
    Question: "What is Henry Feilden's occupation?"
    Rewritten Query: "Henry Feilden, occupation"

[OUR ADAPTATION] The paper utilized GPT-3.5-Turbo for query rewriting.
Our implementation reuses the centralized GeminiClient abstraction
with a deterministic few-shot prompt to extract at most 3 search terms.
"""

import re
from typing import Any

from app.generation.llm_provider import LLMProvider


REWRITE_PROMPT_TEMPLATE = """\
Transform the following user question into at most {max_keywords} concise web search keywords separated by commas.

Rules:
1. Output ONLY the keywords separated by commas.
2. Do NOT say hello, do NOT explain, and do NOT output conversational filler.

Examples:
Question: What is Henry Feilden's occupation?
Search Query: Henry Feilden, occupation

Question: Where were the 2024 Summer Olympic Games hosted?
Search Query: 2024 Summer Olympics, host city

Question: When did Albert Einstein win the Nobel Prize in Physics?
Search Query: Albert Einstein, Nobel Prize Physics, year

Question: {query}
Search Query:"""


def sanitize_rewritten_query(raw_text: str, max_keywords: int = 3) -> str:
    """Parse and clean LLM response to produce at most max_keywords comma-separated terms.

    Strips prefixes, quotes, bullet points, extra whitespace, and limits to max_keywords.

    Args:
        raw_text: Raw string returned by the LLM.
        max_keywords: Maximum number of comma-separated terms to retain.

    Returns:
        Sanitized, comma-separated search query string.

    Raises:
        ValueError: If sanitized output is empty.
    """
    text = raw_text.strip()
    if not text:
        raise ValueError("LLM returned an empty response for query rewriting.")

    # 1. Remove common preamble/prefixes like "Search Query:", "Query:", "Keywords:", etc.
    text = re.sub(
        r"^(?:search\s+query|query|keywords|search\s+terms|output)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # 2. Strip enclosing quotes (single, double, or backticks)
    text = text.strip("\"'`").strip()

    # 3. Handle newline-separated or bullet-separated outputs if model produced list instead of comma
    if "\n" in text and "," not in text:
        lines = [
            re.sub(r"^(?:[-*•]|\d+[\.\)])\s*", "", line).strip()
            for line in text.splitlines()
        ]
        terms = [l for l in lines if l]
    else:
        # Split by comma
        terms = [
            re.sub(r"^(?:[-*•]|\d+[\.\)])\s*", "", term).strip().strip("\"'`")
            for term in text.split(",")
        ]
        terms = [t for t in terms if t]

    if not terms:
        raise ValueError(f"Could not extract search keywords from LLM response: {raw_text!r}")

    # Limit to at most max_keywords terms
    selected_terms = terms[:max_keywords]

    # Reassemble as standard comma-separated query string
    return ", ".join(selected_terms)


class QueryRewriter:
    """Transforms verbose natural language questions into concise web search queries."""

    DEFAULT_MAX_KEYWORDS: int = 3

    def __init__(
        self,
        llm_client: LLMProvider,
        max_keywords: int = DEFAULT_MAX_KEYWORDS,
    ) -> None:
        """Initialize the QueryRewriter.

        Args:
            llm_client: GeminiClient instance (or any client implementing .generate(prompt)).
            max_keywords: Maximum number of comma-separated search terms (default: 3).

        Raises:
            TypeError:  If llm_client is None or max_keywords is not an int.
            ValueError: If max_keywords is <= 0.
        """
        if llm_client is None:
            raise TypeError("llm_client cannot be None.")
        if isinstance(max_keywords, bool) or not isinstance(max_keywords, int):
            raise TypeError(f"max_keywords must be an integer, got {type(max_keywords).__name__}")
        if max_keywords <= 0:
            raise ValueError(f"max_keywords must be a positive integer > 0, got {max_keywords}")

        self.llm_client = llm_client
        self.max_keywords = max_keywords

    def rewrite(self, query: str) -> str:
        """Rewrite a user query into at most 3 comma-separated search terms.

        If the LLM returns an empty, whitespace-only, or otherwise unusable
        response, the rewriter gracefully falls back to the original cleaned
        query string (query.strip()) rather than crashing the pipeline.

        Real provider/API exceptions from llm_client.generate() are NOT caught
        here and will propagate normally.

        Args:
            query: The original question string.

        Returns:
            Concise, comma-separated web search query, or clean_query as fallback.

        Raises:
            TypeError:  If query is not a string.
            ValueError: If query is empty or whitespace-only.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query must be a non-empty string.")

        prompt = REWRITE_PROMPT_TEMPLATE.format(
            query=clean_query,
            max_keywords=self.max_keywords,
        )

        raw_response = self.llm_client.generate(prompt)
        try:
            return sanitize_rewritten_query(raw_response, max_keywords=self.max_keywords)
        except ValueError:
            return clean_query
