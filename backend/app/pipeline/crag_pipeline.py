"""
CRAG Orchestration Pipeline for CorrectRAG.

[PAPER] Algorithm 1 of the CRAG paper (arXiv:2401.15884v3) defines the
full corrective retrieval-augmented generation inference flow:

    Given: query q, retrieved documents D = {d_1, ..., d_k}

    1. Compute relevance scores: s_i = Evaluator(q, d_i) for all i.
    2. Compute maximum confidence: s_max = max(s_i).
    3. Action trigger:
         CORRECT   if s_max > alpha  → refine best internal documents
         INCORRECT if s_max < beta   → discard internal, use web knowledge
         AMBIGUOUS if beta ≤ s_max ≤ alpha → combine refined internal + web knowledge
    4. Compose final context from chosen knowledge source(s).
    5. Generate answer using the composed context.

[OUR ADAPTATION]
- Retrieval: VectorRetriever uses Gemini embeddings and in-memory store.
  paper's BM25+dense hybrid retriever.
- Evaluator: Two-stage evaluation with similarity filter and LLM Judge.
  [-1, 1] score mapping replaces the paper's fine-tuned T5-large evaluator.
- Thresholds: alpha/beta are configurable parameters, not the paper's
  empirically fitted T5-specific values.
- Query rewriter: GeminiClient prompt-based rewriting replaces GPT-3.5-Turbo.
- Web search: Tavily replaces Google Search API.
- External knowledge refinement: Web search results are converted to
  RetrievedChunk objects and passed through KnowledgeRefiner before generation,
  applying the same strip decomposition, scoring, and filtering that is applied
  to internal documents. The paper does not specify this exact step; it is our
  adaptation to avoid sending raw unfiltered web snippets directly to the LLM.
- Empty-retrieval adaptation: If the internal retriever returns zero documents,
  the pipeline bypasses document scoring and routing — there is no s_max to
  compute under Algorithm 1 when k=0 — and directly follows the external
  correction path (INCORRECT). This behavior is NOT described in Algorithm 1;
  it is our explicit adaptation to handle an empty vector store gracefully.
- No LangChain / LangGraph. Pure-Python orchestration.

The CRAG pipeline is stateless per call; all components are injected via
the constructor and shared across calls.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

from app.retrieval.retriever import RetrievedChunk, VectorRetriever
from app.evaluation.relevance_evaluator import RelevanceEvaluator
from app.evaluation.action_router import Action, ActionRouter, RoutingDecision
from app.evaluation.knowledge_refiner import KnowledgeRefiner, KnowledgeStrip
from app.external.query_rewriter import QueryRewriter
from app.external.web_search import WebSearchClient, WebSearchResult
from app.generation.llm_provider import LLMProvider


# ──────────────────────────────────────────────────────────────────────────────
# Web result → RetrievedChunk adapter
# ──────────────────────────────────────────────────────────────────────────────

def _web_results_to_chunks(web_results: list[WebSearchResult]) -> list[RetrievedChunk]:
    """Convert WebSearchResult objects to RetrievedChunk objects for KnowledgeRefiner.

    [OUR ADAPTATION] KnowledgeRefiner expects RetrievedChunk objects. This adapter
    converts external web search results into that format so that the same
    KnowledgeRefiner component (strip decomposition, relevance scoring, filtering,
    and recomposition) can be applied to external knowledge, not just internal
    documents.

    The URL is stored as the chunk source for citation provenance in the resulting
    KnowledgeStrip objects. Page number is set to 1 (web pages have no page
    concept). The original title is preserved in metadata.

    Args:
        web_results: Raw WebSearchResult objects returned by WebSearchClient.

    Returns:
        List of RetrievedChunk objects suitable for KnowledgeRefiner.refine().
        Returns an empty list if web_results is empty.
    """
    chunks: list[RetrievedChunk] = []
    for i, result in enumerate(web_results):
        chunks.append(
            RetrievedChunk(
                chunk_id=f"web-{i}-{result.url[:48]}",
                text=result.content,
                source=result.url,
                page_number=1,
                score=float(result.score),
                metadata={"title": result.title, "source_type": "web"},
            )
        )
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Output models
# ──────────────────────────────────────────────────────────────────────────────

class ExecutionTrace(BaseModel):
    """Safe operational metadata for a single CRAGPipeline.run() call.

    Contains factual operational statistics only — no chain-of-thought,
    no hidden intermediate reasoning, no model logits. Suitable for logging,
    observability, and debugging.
    """

    retrieved_count: int = Field(
        ...,
        description="Number of internal documents retrieved from vector store",
    )
    action: Action = Field(
        ...,
        description="CRAG action taken: CORRECT, INCORRECT, or AMBIGUOUS",
    )
    max_relevance_score: Optional[float] = Field(
        default=None,
        description=(
            "Maximum relevance score across retrieved documents in [-1, 1]. "
            "None when no documents were retrieved (empty vector store)."
        ),
    )
    similarity_pre_filter_decision: Optional[str] = Field(
        default=None, description="Action determined by cheap embedding similarity pre-filter"
    )
    judge_called: bool = Field(
        default=False, description="Whether the LLM judge was called for borderline evaluation"
    )
    judge_decision: Optional[str] = Field(
        default=None, description="LLM judge final decision"
    )
    judge_reason: Optional[str] = Field(
        default=None, description="LLM judge reasoning"
    )
    judge_latency: Optional[float] = Field(
        default=None, description="Latency of LLM judge call in seconds"
    )
    web_search_used: bool = Field(
        ...,
        description="True when external web search was executed (INCORRECT or AMBIGUOUS)",
    )
    rewritten_query: Optional[str] = Field(
        default=None,
        description="Rewritten search query sent to web search. None for CORRECT branch.",
    )
    internal_strip_count: int = Field(
        ...,
        description="Number of refined internal knowledge strips used in final generation",
    )
    external_strip_count: int = Field(
        ...,
        description="Number of refined external knowledge strips used in final generation",
    )
    final_context_source: str = Field(
        ...,
        description=(
            "Describes the knowledge source(s) used for generation: "
            "'internal' (CORRECT with strips), "
            "'external' (INCORRECT with external strips), "
            "'combined' (AMBIGUOUS with both sources), "
            "or 'none' (all refinement returned empty)."
        ),
    )


class CRAGResult(BaseModel):
    """Structured output of a single CRAGPipeline.run() call."""

    answer: str = Field(
        ...,
        description="Generated answer text",
    )
    action: Action = Field(
        ...,
        description="CRAG action taken: CORRECT, INCORRECT, or AMBIGUOUS",
    )
    query: str = Field(
        ...,
        description="Original user query string (whitespace-stripped)",
    )
    rewritten_query: Optional[str] = Field(
        default=None,
        description="Web-search query produced by QueryRewriter (set for INCORRECT and AMBIGUOUS only)",
    )
    retrieved_chunks: list[RetrievedChunk] = Field(
        default_factory=list,
        description=(
            "Top-K chunks from internal vector retrieval. "
            "Empty when the vector store contains no documents."
        ),
    )
    relevance_scores: list[float] = Field(
        default_factory=list,
        description=(
            "Per-chunk bounded relevance scores in [-1, 1], parallel to retrieved_chunks. "
            "Empty when retrieved_chunks is empty."
        ),
    )
    refined_strips: list[KnowledgeStrip] = Field(
        default_factory=list,
        description=(
            "Refined internal knowledge strips produced by KnowledgeRefiner "
            "from internal documents. Set for CORRECT and AMBIGUOUS branches only."
        ),
    )
    external_strips: list[KnowledgeStrip] = Field(
        default_factory=list,
        description=(
            "Refined external knowledge strips produced by KnowledgeRefiner "
            "applied to converted web search results. "
            "Set for INCORRECT and AMBIGUOUS branches only."
        ),
    )
    web_results: list[WebSearchResult] = Field(
        default_factory=list,
        description=(
            "Raw web search results preserved for provenance. "
            "Generation uses refined external_strips, not these raw results. "
            "Set for INCORRECT and AMBIGUOUS branches only."
        ),
    )
    trace: ExecutionTrace = Field(
        ...,
        description="Operational execution trace with safe metadata about this pipeline run",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fallback answer
# ──────────────────────────────────────────────────────────────────────────────

_NO_CONTEXT_ANSWER = (
    "I cannot answer this question because no relevant context "
    "was found in internal documents or external web search."
)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_crag_prompt(
    query: str,
    internal_strips: list[KnowledgeStrip],
    external_strips: list[KnowledgeStrip],
) -> str:
    """Build a final generation prompt from refined CRAG knowledge sources.

    Both internal and external knowledge has been processed by KnowledgeRefiner
    (decomposed into strips, scored, filtered, recomposed) before being passed
    to this function. Raw web results are NOT passed here directly.

    Supports three knowledge compositions:
    - Internal only (CORRECT):   refined internal strips.
    - External only (INCORRECT): refined external strips (source = URL).
    - Combined (AMBIGUOUS):      refined internal strips + refined external strips.

    Args:
        query: Original user question.
        internal_strips: Refined internal KnowledgeStrips (may be empty).
        external_strips: Refined external (web) KnowledgeStrips (may be empty).
                         Their .source field contains the originating URL.

    Returns:
        Complete prompt string ready for GeminiClient.generate().
    """
    passages: list[str] = []
    counter = 1

    for strip in internal_strips:
        passages.append(
            f"[{counter}] Source: {strip.source} | Page: {strip.page_number} "
            f"(internal document)\n{strip.text.strip()}"
        )
        counter += 1

    for strip in external_strips:
        passages.append(
            f"[{counter}] Source: {strip.source} "
            f"(web)\n{strip.text.strip()}"
        )
        counter += 1

    system_instruction = """\
You are a precise question-answering assistant.

Rules:
1. Answer ONLY using information found in the provided context passages.
2. If the context does not contain enough information to answer, say: \
"I cannot answer this question based on the provided context."
3. Do NOT invent, assume, or extrapolate facts not present in the context.
4. After your answer, list the sources you used in the format:
   Sources: [<source>, page <page>], ...
5. Be concise and direct. Do not repeat the question."""

    context_section = "\n\n".join(passages) if passages else "No relevant context was found."

    return (
        f"{system_instruction}\n\n"
        f"---\n\n"
        f"CONTEXT PASSAGES:\n\n"
        f"{context_section}\n\n"
        f"---\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER:"
    )


# ──────────────────────────────────────────────────────────────────────────────
# CRAG Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class CRAGPipeline:
    """Corrective Retrieval-Augmented Generation orchestrator.

    Wires together all CRAG components following Algorithm 1 of the CRAG paper
    (arXiv:2401.15884v3), with documented adaptations:

        VectorRetriever  → RelevanceEvaluator → ActionRouter
        → (CORRECT)   KnowledgeRefiner(internal) → Generate
        → (INCORRECT) QueryRewriter → WebSearch → KnowledgeRefiner(external) → Generate
        → (AMBIGUOUS) KnowledgeRefiner(internal)
                      + QueryRewriter → WebSearch → KnowledgeRefiner(external)
                      → Generate from combined context

    Components:
        retriever:        VectorRetriever — internal semantic search.
        evaluator:        RelevanceEvaluator — per-chunk relevance scoring.
        router:           ActionRouter — three-way action decision (CORRECT/INCORRECT/AMBIGUOUS).
        refiner:          KnowledgeRefiner — strip decomposition, scoring, and filtering;
                          used for BOTH internal documents and converted external web chunks.
        query_rewriter:   QueryRewriter — natural language → concise search keywords.
        web_search:       WebSearchClient — external Tavily search.
        llm_client:       GeminiClient — final answer generation.
        top_k:            Number of internal chunks to retrieve per query (default: 5).
    """

    def __init__(
        self,
        retriever: VectorRetriever,
        evaluator: RelevanceEvaluator,
        router: ActionRouter,
        refiner: KnowledgeRefiner,
        query_rewriter: QueryRewriter,
        web_search: WebSearchClient,
        llm_client: LLMProvider,
        top_k: int = 10,
    ) -> None:
        """Initialize the CRAGPipeline with all required components.

        Args:
            retriever:       VectorRetriever connected to vector store.
            evaluator:       RelevanceEvaluator for scoring query-document pairs.
            router:          ActionRouter with configured alpha/beta thresholds.
            refiner:         KnowledgeRefiner for strip decomposition and filtering.
                             Applied to both internal documents and converted web results.
            query_rewriter:  QueryRewriter for generating concise web search queries.
            web_search:      WebSearchClient (Tavily) for external knowledge retrieval.
            llm_client:      GeminiClient for final answer generation.
            top_k:           Number of top internal chunks to retrieve (must be > 0).

        Raises:
            TypeError:  If any required component is None, or top_k has wrong type.
            ValueError: If top_k is not a positive integer.
        """
        for name, obj in [
            ("retriever", retriever),
            ("evaluator", evaluator),
            ("router", router),
            ("refiner", refiner),
            ("query_rewriter", query_rewriter),
            ("web_search", web_search),
            ("llm_client", llm_client),
        ]:
            if obj is None:
                raise TypeError(f"{name} cannot be None.")

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}")
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer > 0, got {top_k}")

        self.retriever = retriever
        self.evaluator = evaluator
        self.router = router
        self.refiner = refiner
        self.query_rewriter = query_rewriter
        self.web_search = web_search
        self.llm_client = llm_client
        self.top_k = top_k

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, query: str) -> CRAGResult:
        """Execute the full CRAG inference workflow for a single query.

        Algorithm (per CRAG paper Algorithm 1, with documented adaptations):
            1. Retrieve top-K internal documents.
            2. Score each document: s_i = Evaluator(query, doc_i).
            3. Route: action = ActionRouter.route(scores) using s_max.
            4a. CORRECT:   KnowledgeRefiner(internal docs) → generate.
            4b. INCORRECT: QueryRewriter → WebSearch
                           → KnowledgeRefiner(external chunks) → generate.
            4c. AMBIGUOUS: KnowledgeRefiner(internal docs)
                           + QueryRewriter → WebSearch
                           → KnowledgeRefiner(external chunks)
                           → generate from combined context.

        [OUR ADAPTATION — empty retrieval]: If the retriever returns zero
        documents, there is no s_max to compute, so scoring and routing are
        bypassed entirely. The pipeline falls directly to the INCORRECT
        (external correction) path. This is not described in Algorithm 1;
        it is our explicit adaptation for empty vector stores.

        Args:
            query: User question string.

        Returns:
            CRAGResult with answer, action, provenance, and execution trace.

        Raises:
            TypeError:  If query is not a string.
            ValueError: If query is empty or whitespace-only.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query must be a non-empty string.")

        # ── Step 1: Retrieve top-K internal documents ─────────────────────────
        chunks: list[RetrievedChunk] = self.retriever.retrieve(
            clean_query, top_k=self.top_k
        )

        # ── Empty-retrieval adaptation (NOT Algorithm 1) ──────────────────────
        # If the vector store is empty there is no s_max, so scoring and routing
        # are skipped. We go directly to the external correction path.
        if not chunks:
            return self._run_incorrect(
                query=clean_query,
                chunks=[],
                scores=[],
            )

        # ── Step 2: Score each retrieved document ─────────────────────────────
        pairs = [(clean_query, chunk.text) for chunk in chunks]
        scores: list[float] = self.evaluator.score_batch(pairs)

        # ── Step 3: Action routing ────────────────────────────────────────────
        raw_decision = self.router.route(clean_query, chunks, scores)
        if isinstance(raw_decision, str):
            action: Action = raw_decision  # type: ignore
            decision: Optional[RoutingDecision] = None
        else:
            decision = raw_decision
            action = decision.action

        # ── Step 4: Branch on action ──────────────────────────────────────────
        if action == "CORRECT":
            return self._run_correct(clean_query, chunks, scores, decision=decision)
        elif action == "INCORRECT":
            return self._run_incorrect(clean_query, chunks, scores, decision=decision)
        else:
            # AMBIGUOUS
            return self._run_ambiguous(clean_query, chunks, scores, decision=decision)

    # ── Private branch methods ────────────────────────────────────────────────

    def _run_correct(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        scores: list[float],
        decision: Optional[RoutingDecision] = None,
    ) -> CRAGResult:
        """Handle the CORRECT branch.

        Refines internal documents with KnowledgeRefiner, then generates from
        the refined internal strips only. No web search or external refinement.
        """
        # Refine internal documents → knowledge strips
        internal_strips: list[KnowledgeStrip] = self.refiner.refine(query, chunks)

        # Build prompt from refined internal knowledge only
        prompt = _build_crag_prompt(
            query=query,
            internal_strips=internal_strips,
            external_strips=[],
        )

        if not internal_strips:
            answer = _NO_CONTEXT_ANSWER
        else:
            answer = self.llm_client.generate(prompt)

        trace = ExecutionTrace(
            retrieved_count=len(chunks),
            action="CORRECT",
            max_relevance_score=max(scores) if scores else None,
            similarity_pre_filter_decision=decision.similarity_pre_filter_decision if decision else None,
            judge_called=decision.judge_called if decision else False,
            judge_decision=decision.judge_decision if decision else None,
            judge_reason=decision.judge_reason if decision else None,
            judge_latency=decision.judge_latency if decision else None,
            web_search_used=False,
            rewritten_query=None,
            internal_strip_count=len(internal_strips),
            external_strip_count=0,
            final_context_source="internal" if internal_strips else "none",
        )

        return CRAGResult(
            answer=answer,
            action="CORRECT",
            query=query,
            rewritten_query=None,
            retrieved_chunks=chunks,
            relevance_scores=scores,
            refined_strips=internal_strips,
            external_strips=[],
            web_results=[],
            trace=trace,
        )

    def _run_incorrect(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        scores: list[float],
        decision: Optional[RoutingDecision] = None,
    ) -> CRAGResult:
        """Handle the INCORRECT branch.

        Rewrites the query, executes web search, converts results to chunks,
        refines the external chunks with KnowledgeRefiner, and generates from
        the refined external strips only. Internal documents are discarded.

        [OUR ADAPTATION] Web search results are converted to RetrievedChunk
        objects and passed through KnowledgeRefiner before generation, rather
        than being sent as raw snippets to the LLM.
        """
        # Rewrite query for targeted web search
        rewritten_query: str = self.query_rewriter.rewrite(query)

        # External web search
        web_results: list[WebSearchResult] = self.web_search.search(rewritten_query)

        # Convert web results → chunks → refine with KnowledgeRefiner
        # Guard: skip refiner call when there are no web results to convert.
        external_chunks = _web_results_to_chunks(web_results)
        external_strips: list[KnowledgeStrip] = (
            self.refiner.refine(query, external_chunks) if external_chunks else []
        )

        # Build prompt from refined external knowledge only
        prompt = _build_crag_prompt(
            query=query,
            internal_strips=[],
            external_strips=external_strips,
        )

        if not external_strips:
            answer = _NO_CONTEXT_ANSWER
        else:
            answer = self.llm_client.generate(prompt)

        trace = ExecutionTrace(
            retrieved_count=len(chunks),
            action="INCORRECT",
            max_relevance_score=max(scores) if scores else None,
            similarity_pre_filter_decision=decision.similarity_pre_filter_decision if decision else None,
            judge_called=decision.judge_called if decision else False,
            judge_decision=decision.judge_decision if decision else None,
            judge_reason=decision.judge_reason if decision else None,
            judge_latency=decision.judge_latency if decision else None,
            web_search_used=True,
            rewritten_query=rewritten_query,
            internal_strip_count=0,
            external_strip_count=len(external_strips),
            final_context_source="external" if external_strips else "none",
        )

        return CRAGResult(
            answer=answer,
            action="INCORRECT",
            query=query,
            rewritten_query=rewritten_query,
            retrieved_chunks=chunks,
            relevance_scores=scores,
            refined_strips=[],
            external_strips=external_strips,
            web_results=web_results,
            trace=trace,
        )

    def _run_ambiguous(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        scores: list[float],
        decision: Optional[RoutingDecision] = None,
    ) -> CRAGResult:
        """Handle the AMBIGUOUS branch.

        Refines internal documents AND refines external web knowledge, then
        generates from the combined refined context.

        [OUR ADAPTATION] Web search results are converted to RetrievedChunk
        objects and passed through KnowledgeRefiner (a second, independent
        call) before being combined with internal strips for generation.
        """
        # Refine internal documents → internal knowledge strips
        internal_strips: list[KnowledgeStrip] = self.refiner.refine(query, chunks)

        # Rewrite query for targeted web search
        rewritten_query: str = self.query_rewriter.rewrite(query)

        # External web search
        web_results: list[WebSearchResult] = self.web_search.search(rewritten_query)

        # Convert web results → chunks → refine with KnowledgeRefiner
        # Guard: skip refiner call when there are no web results to convert.
        external_chunks = _web_results_to_chunks(web_results)
        external_strips: list[KnowledgeStrip] = (
            self.refiner.refine(query, external_chunks) if external_chunks else []
        )

        # Build prompt from combined refined context
        prompt = _build_crag_prompt(
            query=query,
            internal_strips=internal_strips,
            external_strips=external_strips,
        )

        if not internal_strips and not external_strips:
            answer = _NO_CONTEXT_ANSWER
        else:
            answer = self.llm_client.generate(prompt)

        # Determine final context source label for trace
        has_internal = bool(internal_strips)
        has_external = bool(external_strips)
        if has_internal and has_external:
            final_context_source = "combined"
        elif has_internal:
            final_context_source = "internal"
        elif has_external:
            final_context_source = "external"
        else:
            final_context_source = "none"

        trace = ExecutionTrace(
            retrieved_count=len(chunks),
            action="AMBIGUOUS",
            max_relevance_score=max(scores) if scores else None,
            similarity_pre_filter_decision=decision.similarity_pre_filter_decision if decision else None,
            judge_called=decision.judge_called if decision else False,
            judge_decision=decision.judge_decision if decision else None,
            judge_reason=decision.judge_reason if decision else None,
            judge_latency=decision.judge_latency if decision else None,
            web_search_used=True,
            rewritten_query=rewritten_query,
            internal_strip_count=len(internal_strips),
            external_strip_count=len(external_strips),
            final_context_source=final_context_source,
        )

        return CRAGResult(
            answer=answer,
            action="AMBIGUOUS",
            query=query,
            rewritten_query=rewritten_query,
            retrieved_chunks=chunks,
            relevance_scores=scores,
            refined_strips=internal_strips,
            external_strips=external_strips,
            web_results=web_results,
            trace=trace,
        )
