"""
External Knowledge & Search Package for CorrectRAG.

Provides:
- QueryRewriter: transforms verbose questions into concise keywords for external search
- sanitize_rewritten_query: helper parsing and limiting search terms
"""

from app.external.query_rewriter import (
    QueryRewriter,
    sanitize_rewritten_query,
    REWRITE_PROMPT_TEMPLATE,
)

__all__ = [
    "QueryRewriter",
    "sanitize_rewritten_query",
    "REWRITE_PROMPT_TEMPLATE",
]
