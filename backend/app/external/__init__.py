"""
External Knowledge & Search Package for CorrectRAG.

Provides:
- QueryRewriter: transforms verbose questions into concise keywords for external search
- sanitize_rewritten_query: helper parsing and limiting search terms
- WebSearchClient: Tavily search provider adapter
- WebSearchResult: structured external search result model
- WebSearchError: domain exception for external search configuration/runtime errors
"""

from app.external.query_rewriter import (
    QueryRewriter,
    sanitize_rewritten_query,
    REWRITE_PROMPT_TEMPLATE,
)
from app.external.web_search import (
    WebSearchClient,
    WebSearchResult,
    WebSearchError,
)

__all__ = [
    "QueryRewriter",
    "sanitize_rewritten_query",
    "REWRITE_PROMPT_TEMPLATE",
    "WebSearchClient",
    "WebSearchResult",
    "WebSearchError",
]
