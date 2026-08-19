"""
Web Search Adapter Module for CorrectRAG.

[PAPER] Section 4.5 ("Web Search") of the CRAG paper (arXiv:2401.15884v3)
describes using external web search as a corrective knowledge source when
internal document retrieval is incorrect or ambiguous:
1. Query rewriting formulates targeted keywords.
2. Web search retrieves relevant external web documents and snippets.
3. External knowledge is integrated to correct or supplement retrieval.

[OUR ADAPTATION] The original paper used Google Search API.
Our frozen production search stack uses Tavily (via the official tavily-python SDK)
optimized for LLM search retrieval.

This module is strictly an adapter: it accepts queries, queries Tavily,
and returns structured WebSearchResult objects. Knowledge refinement and
routing orchestration are handled by separate components.
"""

import os
from typing import Optional
from pydantic import BaseModel, Field

try:
    from tavily import TavilyClient
except ImportError as exc:
    raise ImportError(
        "tavily-python package is not installed. "
        "Run: pip install tavily-python"
    ) from exc


class WebSearchError(Exception):
    """Raised when the web search API is misconfigured or fails."""


class WebSearchResult(BaseModel):
    """Structured result returned from an external web search."""

    title: str = Field(..., description="Page title of the search result")
    url: str = Field(..., description="Web URL of the search result")
    content: str = Field(..., description="Text snippet or content extracted from the page")
    score: float = Field(default=0.0, description="Relevance score returned by the search provider")
    raw_content: Optional[str] = Field(
        default=None, description="Optional full raw page content if provided"
    )


class WebSearchClient:
    """Production web search client wrapping the Tavily search provider."""

    DEFAULT_MAX_RESULTS: int = 5

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> None:
        """Initialize the WebSearchClient.

        Args:
            api_key: Tavily API key override (defaults to TAVILY_API_KEY environment variable).
            max_results: Maximum number of search results to retrieve (default: 5 or TAVILY_MAX_RESULTS).

        Raises:
            WebSearchError: If TAVILY_API_KEY is not set.
            TypeError: If max_results is not an integer.
            ValueError: If max_results is not a positive integer > 0.
        """
        resolved_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        if not resolved_key:
            raise WebSearchError(
                "TAVILY_API_KEY is not set. "
                "Export the environment variable or pass api_key= explicitly."
            )

        if max_results is not None:
            if isinstance(max_results, bool) or not isinstance(max_results, int):
                raise TypeError(
                    f"max_results must be an integer, got {type(max_results).__name__}"
                )
            if max_results <= 0:
                raise ValueError(
                    f"max_results must be a positive integer > 0, got {max_results}"
                )
            self.max_results: int = max_results
        else:
            env_max = os.environ.get("TAVILY_MAX_RESULTS")
            if env_max:
                try:
                    self.max_results = int(env_max)
                    if self.max_results <= 0:
                        self.max_results = self.DEFAULT_MAX_RESULTS
                except ValueError:
                    self.max_results = self.DEFAULT_MAX_RESULTS
            else:
                self.max_results = self.DEFAULT_MAX_RESULTS

        self._client = TavilyClient(api_key=resolved_key)

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> list[WebSearchResult]:
        """Execute a web search query and return structured search results.

        Args:
            query: The search query string (e.g. rewritten keywords).
            max_results: Optional per-query result limit override.

        Returns:
            List of WebSearchResult instances (at most max_results items).

        Raises:
            TypeError:  If query is not a string.
            ValueError: If query is empty or whitespace-only, or max_results <= 0.
            WebSearchError: If the search API request fails.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query must be a non-empty string.")

        k = self.max_results
        if max_results is not None:
            if isinstance(max_results, bool) or not isinstance(max_results, int):
                raise TypeError(
                    f"max_results must be an integer, got {type(max_results).__name__}"
                )
            if max_results <= 0:
                raise ValueError(
                    f"max_results must be a positive integer > 0, got {max_results}"
                )
            k = max_results

        try:
            response = self._client.search(
                query=clean_query,
                max_results=k,
                search_depth="basic",
            )
        except Exception as exc:
            raise WebSearchError(
                f"Tavily search request failed for query {clean_query!r}: {exc}"
            ) from exc

        raw_results = response.get("results", []) if isinstance(response, dict) else []
        structured_results: list[WebSearchResult] = []

        for item in raw_results[:k]:
            if not isinstance(item, dict):
                continue
            structured_results.append(
                WebSearchResult(
                    title=str(item.get("title", "")).strip(),
                    url=str(item.get("url", "")).strip(),
                    content=str(item.get("content", "")).strip(),
                    score=float(item.get("score", 0.0) or 0.0),
                    raw_content=item.get("raw_content"),
                )
            )

        return structured_results
