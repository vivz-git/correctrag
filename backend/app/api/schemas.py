"""
Pydantic Request and Response Schemas for CorrectRAG HTTP API.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="ok", description="Service health status")
    service: str = Field(default="correctrag-api", description="Service identifier")


class QueryRequest(BaseModel):
    """User query request schema."""

    question: str = Field(
        ...,
        description="The natural language question to answer using CRAG",
        min_length=1,
    )

    @field_validator("question")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Ensure question is not empty or whitespace-only."""
        if not isinstance(v, str):
            raise ValueError("Question must be a string.")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question must not be empty or whitespace-only.")
        return cleaned


class ChunkSource(BaseModel):
    """Safe representation of an internally retrieved document chunk."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    source: str = Field(..., description="Source document filename or URI")
    page_number: Optional[int] = Field(default=None, description="Document page number")
    score: Optional[float] = Field(default=None, description="Retrieval similarity score")
    text_snippet: Optional[str] = Field(default=None, description="Text snippet of the chunk")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")


class StripSource(BaseModel):
    """Safe representation of a refined knowledge strip."""

    text: str = Field(..., description="Knowledge strip text content")
    source: str = Field(..., description="Provenance document or URL")
    page_number: int = Field(..., description="1-indexed source page number")
    parent_chunk_id: str = Field(..., description="Identifier of the parent chunk")
    position: int = Field(..., description="Sequential position index in document")
    score: float = Field(..., description="Evaluator relevance score in [-1, 1]")
    origin: str = Field(default="internal", description="'internal' or 'external'")


class WebSource(BaseModel):
    """Safe representation of an external web search result."""

    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Webpage title")
    snippet: str = Field(..., description="Search snippet text")
    score: Optional[float] = Field(default=None, description="Search relevance score")


class TraceSchema(BaseModel):
    """Operational execution trace without chain-of-thought or internal reasoning."""

    retrieved_count: int = Field(
        ..., description="Number of internal chunks retrieved from vector store"
    )
    action: str = Field(
        ..., description="CRAG action executed: CORRECT, INCORRECT, or AMBIGUOUS"
    )
    max_relevance_score: Optional[float] = Field(
        default=None, description="Maximum relevance score s_max across internal documents"
    )
    web_search_used: bool = Field(
        ..., description="Whether external web search was triggered"
    )
    rewritten_query: Optional[str] = Field(
        default=None, description="Rewritten search query sent to web search"
    )
    internal_strip_count: int = Field(
        ..., description="Count of refined internal knowledge strips used in generation"
    )
    external_strip_count: int = Field(
        ..., description="Count of refined external knowledge strips used in generation"
    )
    final_context_source: str = Field(
        ..., description="Knowledge source used for generation: internal, external, combined, none"
    )


class QueryResponse(BaseModel):
    """Clean API response schema for a CorrectRAG query execution."""

    answer: str = Field(..., description="Generated answer text")
    action: str = Field(..., description="CRAG action taken (CORRECT, INCORRECT, AMBIGUOUS)")
    query: str = Field(..., description="Original user query string")
    rewritten_query: Optional[str] = Field(
        default=None, description="Rewritten search query (for INCORRECT/AMBIGUOUS branches)"
    )
    retrieved_chunks: list[ChunkSource] = Field(
        default_factory=list, description="Internal chunks retrieved before evaluation"
    )
    relevance_scores: list[float] = Field(
        default_factory=list, description="Per-chunk relevance scores in [-1, 1]"
    )
    refined_strips: list[StripSource] = Field(
        default_factory=list, description="Refined internal knowledge strips"
    )
    external_strips: list[StripSource] = Field(
        default_factory=list, description="Refined external knowledge strips"
    )
    web_results: list[WebSource] = Field(
        default_factory=list, description="Raw external web search results"
    )
    execution_trace: TraceSchema = Field(
        ..., description="Operational execution metadata"
    )
