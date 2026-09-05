"""
Knowledge Refinement Module for CorrectRAG.

[PAPER] Section 4.4 ("Knowledge Refinement") of the CRAG paper (arXiv:2401.15884v3)
describes a fine-grained knowledge extraction process to filter out irrelevant
or distracting information within retrieved documents:

1. Document Decomposition:
   Each retrieved document is decomposed into fine-grained knowledge strips
   (typically 1–3 sentences).
2. Strip Relevance Scoring:
   The retrieval evaluator calculates a relevance score for each strip:
   s_i = Evaluator(query, strip_i).
3. Hard Filtering:
   Strips with relevance scores below the threshold are discarded:
   discard strip_i if s_i < filter_threshold (gamma = -0.5 in the paper).
4. Top-K Selection & Order Recomposition:
   The surviving strips are ranked by score, the top-K highest-scoring strips
   are selected, and then reordered to their original document sequence.

[OUR ADAPTATION] The original paper used gamma = -0.5 tailored to their
fine-tuned T5-large evaluator. Because our system employs a frozen cross-encoder
with a bounded [-1, 1] relevance score mapping, filter_threshold is exposed as
a configurable parameter and not claimed to be scientifically calibrated.
"""

import re
from typing import Optional
from pydantic import BaseModel, Field

from app.retrieval.retriever import RetrievedChunk
from app.evaluation.relevance_evaluator import RelevanceEvaluator


class KnowledgeStrip(BaseModel):
    """Fine-grained knowledge strip extracted from a retrieved document chunk."""

    text: str = Field(..., description="Knowledge strip text content")
    source: str = Field(..., description="Source document name or path")
    page_number: int = Field(..., description="1-indexed source page number")
    parent_chunk_id: str = Field(..., description="Identifier of the parent RetrievedChunk")
    score: float = Field(default=0.0, description="Relevance score assigned by evaluator")
    position: int = Field(..., description="Original sequential index across document strips")


def _is_structured_layout(text: str) -> bool:
    """Detect layout-dense text (tables, pseudocode, figure/diagram text) via a
    line-density heuristic: at least 6 non-empty lines, with at least 60% of
    those lines containing 3 or fewer whitespace-separated words.

    Sentence splitting fragments this kind of content (e.g. a table caption
    ending in "." gets isolated from the row data above it), so callers use
    this to keep the whole block as one strip instead.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < 6:
        return False
    short_lines = sum(1 for line in lines if len(line.split()) <= 3)
    return (short_lines / len(lines)) >= 0.6


def _find_structured_prefix_split(text: str) -> Optional[int]:
    """Within a structured-layout block, find where a trailing narrative run begins.

    A run qualifies once it reaches >= 4 consecutive non-empty lines that each
    contain more than 3 whitespace-separated words -- e.g. a figure/table block
    that is followed by ordinary prose paragraphs in the same parent chunk.
    Only the first such qualifying run is used, and only when it does not begin
    at the block's very first non-empty line (there would be no structured
    prefix left to preserve in that case).

    Args:
        text: The already-stripped structured-layout block text.

    Returns:
        Raw line index (into text.split("\n")) where the narrative suffix
        begins, or None if no qualifying trailing run exists.
    """
    raw_lines = text.split("\n")
    non_empty = [(i, line.strip()) for i, line in enumerate(raw_lines) if line.strip()]

    run_start_pos: Optional[int] = None
    run_len = 0
    for pos, (raw_idx, line) in enumerate(non_empty):
        if len(line.split()) > 3:
            if run_len == 0:
                run_start_pos = pos
            run_len += 1
            if run_len >= 4:
                if run_start_pos == 0:
                    return None
                return non_empty[run_start_pos][0]
        else:
            run_len = 0
            run_start_pos = None

    return None


def decompose_text_into_strips(text: str, sentences_per_strip: int = 2) -> list[str]:
    """Decompose document text into sentence-based knowledge strips.

    Short text (single sentence or <= sentences_per_strip) remains a single strip.
    Otherwise, consecutive sentences are grouped into strips of size sentences_per_strip.

    Structured/layout-dense text (tables, pseudocode, figure/diagram text) is
    detected via a line-density heuristic and kept as a single bounded strip
    instead, since sentence splitting would separate row data from captions
    that happen to share the same parent chunk. If that structured block ends
    in a trailing run of >= 4 consecutive narrative-looking lines (ordinary
    prose following a figure/table in the same chunk), the structured prefix
    is preserved as one strip and the narrative suffix is decomposed with the
    normal sentence-based logic below.

    Args:
        text: Raw text of the retrieved chunk.
        sentences_per_strip: Number of sentences per strip (default: 2, targeting 1-3 sentences).

    Returns:
        List of non-empty strip text strings.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    if _is_structured_layout(clean_text):
        split_at = _find_structured_prefix_split(clean_text)
        if split_at is not None:
            raw_lines = clean_text.split("\n")
            prefix_text = "\n".join(raw_lines[:split_at]).strip()
            suffix_text = "\n".join(raw_lines[split_at:]).strip()
            return [prefix_text] + decompose_text_into_strips(
                suffix_text, sentences_per_strip=sentences_per_strip
            )
        return [clean_text]

    # Split on sentence-ending punctuation (.!?) followed by whitespace
    raw_sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", clean_text)
        if s.strip()
    ]

    if not raw_sentences:
        return [clean_text]

    if len(raw_sentences) <= sentences_per_strip:
        return [clean_text]

    strips: list[str] = []
    for i in range(0, len(raw_sentences), sentences_per_strip):
        group = raw_sentences[i : i + sentences_per_strip]
        strip_text = " ".join(group).strip()
        if strip_text:
            strips.append(strip_text)

    return strips


class KnowledgeRefiner:
    """Refines retrieved documents into filtered, ranked, and recomposed knowledge strips."""

    DEFAULT_FILTER_THRESHOLD: float = -0.5
    DEFAULT_TOP_K: int = 5
    DEFAULT_SENTENCES_PER_STRIP: int = 2

    def __init__(
        self,
        evaluator: RelevanceEvaluator,
        filter_threshold: float = DEFAULT_FILTER_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        sentences_per_strip: int = DEFAULT_SENTENCES_PER_STRIP,
    ) -> None:
        """Initialize KnowledgeRefiner.

        Args:
            evaluator: RelevanceEvaluator instance for scoring (query, strip) pairs.
            filter_threshold: Threshold below which strips are filtered out (in [-1.0, 1.0]).
            top_k: Maximum number of highest-scoring surviving strips to retain (must be > 0).
            sentences_per_strip: Target sentences per strip for decomposition (must be > 0).

        Raises:
            TypeError:  If parameters have invalid types.
            ValueError: If filter_threshold is outside [-1.0, 1.0], or top_k/sentences <= 0.
        """
        if evaluator is None:
            raise TypeError("evaluator cannot be None.")

        if isinstance(filter_threshold, bool) or not isinstance(filter_threshold, (int, float)):
            raise TypeError(f"filter_threshold must be numeric, got {type(filter_threshold).__name__}")
        filter_threshold_f = float(filter_threshold)
        if not (-1.0 <= filter_threshold_f <= 1.0):
            raise ValueError(
                f"filter_threshold must be within [-1.0, 1.0], got {filter_threshold_f}"
            )

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}")
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer > 0, got {top_k}")

        if isinstance(sentences_per_strip, bool) or not isinstance(sentences_per_strip, int):
            raise TypeError(f"sentences_per_strip must be an integer, got {type(sentences_per_strip).__name__}")
        if sentences_per_strip <= 0:
            raise ValueError(
                f"sentences_per_strip must be a positive integer > 0, got {sentences_per_strip}"
            )

        self.evaluator: RelevanceEvaluator = evaluator
        self.filter_threshold: float = filter_threshold_f
        self.top_k: int = top_k
        self.sentences_per_strip: int = sentences_per_strip

    def refine(
        self,
        query: str,
        documents: list[RetrievedChunk],
    ) -> list[KnowledgeStrip]:
        """Refine retrieved document chunks into high-relevance structured knowledge strips.

        Workflow:
          1. Decompose documents into fine-grained strips with global position indices.
          2. Score each strip with the evaluator: s_i = Evaluator(query, strip_i).
          3. Filter strips where s_i < filter_threshold.
          4. Give each surviving parent chunk a coverage-floor representative
             (its highest-scoring strip), rank representatives by score, and
             fill any remaining top_k slots from the rest of the pool by score.
          4b. Document-level coverage floor: if a distinct surviving source
              document ended up with zero selected strips (because every one
              of its parent-chunk representatives scored below the cutoff),
              admit its best representative by displacing the weakest
              selected strip from a source that still has at least one other
              selected strip.
          5. Recompose chosen top_k strips in their original source order.

        Args:
            query: The user or pipeline query string.
            documents: List of RetrievedChunk objects from semantic retrieval.

        Returns:
            List of structured KnowledgeStrip objects in original source order.

        Raises:
            TypeError:  If query or documents is of invalid type.
            ValueError: If query is empty or whitespace-only.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("query must be a non-empty string.")

        if not isinstance(documents, (list, tuple)):
            raise TypeError(f"documents must be a list of RetrievedChunk, got {type(documents).__name__}")

        if not documents:
            return []

        # ── Step 1: Decompose documents into fine-grained strips ──────────────
        all_strips: list[KnowledgeStrip] = []
        global_position = 0

        for chunk in documents:
            strip_texts = decompose_text_into_strips(
                chunk.text, sentences_per_strip=self.sentences_per_strip
            )
            for text in strip_texts:
                all_strips.append(
                    KnowledgeStrip(
                        text=text,
                        source=chunk.source,
                        page_number=chunk.page_number,
                        parent_chunk_id=chunk.chunk_id,
                        score=0.0,
                        position=global_position,
                    )
                )
                global_position += 1

        if not all_strips:
            return []

        # ── Step 2: Score each strip ──────────────────────────────────────────
        pairs = [(clean_query, strip.text) for strip in all_strips]
        scores = self.evaluator.score_batch(pairs)

        for strip, score in zip(all_strips, scores):
            strip.score = float(score)

        # ── Step 3: Filter irrelevant strips (score < filter_threshold) ───────
        surviving_strips = [
            strip for strip in all_strips if strip.score >= self.filter_threshold
        ]

        if not surviving_strips:
            return []

        # ── Step 4: Per-parent-chunk coverage floor + global fill ─────────────
        # Rationale: pure global top-k can let one chunk with many strong strips
        # crowd out every strip from another retrieved (and possibly still
        # relevant) chunk. We guarantee each surviving parent chunk a shot at
        # a representative slot before filling remaining slots by raw score.
        by_parent: dict[str, list[KnowledgeStrip]] = {}
        for strip in surviving_strips:
            by_parent.setdefault(strip.parent_chunk_id, []).append(strip)

        representatives = [
            max(strips, key=lambda s: (s.score, -s.position))
            for strips in by_parent.values()
        ]
        representatives.sort(key=lambda s: (-s.score, s.position))

        if len(representatives) >= self.top_k:
            top_k_strips = representatives[: self.top_k]
        else:
            selected_ids = {id(s) for s in representatives}
            remaining_pool = [
                s for s in surviving_strips if id(s) not in selected_ids
            ]
            remaining_pool.sort(key=lambda s: (-s.score, s.position))
            remaining_slots = self.top_k - len(representatives)
            top_k_strips = representatives + remaining_pool[:remaining_slots]

        # ── Step 4b: Document-level coverage floor ─────────────────────────────
        top_k_strips = self._apply_document_floor(
            top_k_strips, representatives, surviving_strips
        )

        # ── Step 5: Recompose (restore original source order) ─────────────────
        recomposed_strips = sorted(top_k_strips, key=lambda s: s.position)

        return recomposed_strips

    @staticmethod
    def _apply_document_floor(
        top_k_strips: list[KnowledgeStrip],
        representatives: list[KnowledgeStrip],
        surviving_strips: list[KnowledgeStrip],
    ) -> list[KnowledgeStrip]:
        """Ensure every surviving source document holds a selected slot where possible.

        Rationale: the per-parent-chunk coverage floor guarantees every surviving
        *chunk* a shot at a representative slot, but when a single source contributes
        many parent chunks that collectively outscore every chunk from another
        source, the top_k cutoff on `representatives` can still drop that other
        source entirely. This pass admits one representative per fully-starved
        source by displacing the weakest currently-selected strip from a source
        that has at least one other selected strip -- never reducing any source
        to zero. top_k size and the caller's recomposition step are unaffected.

        Args:
            top_k_strips: Strips selected by the per-parent-chunk floor + fill step.
            representatives: One highest-scoring strip per surviving parent chunk.
            surviving_strips: All strips that passed the relevance filter.

        Returns:
            The (possibly adjusted) list of selected strips, same length as input.
        """
        all_sources = {s.source for s in surviving_strips}
        if len(all_sources) <= 1:
            return top_k_strips

        selected = list(top_k_strips)
        missing_sources = all_sources - {s.source for s in selected}
        if not missing_sources:
            return selected

        reps_by_source: dict[str, list[KnowledgeStrip]] = {}
        for rep in representatives:
            reps_by_source.setdefault(rep.source, []).append(rep)

        # Best available representative per missing source, highest scoring first.
        candidates = [
            max(reps_by_source[source], key=lambda s: (s.score, -s.position))
            for source in missing_sources
            if source in reps_by_source
        ]
        candidates.sort(key=lambda s: (-s.score, s.position))

        for candidate in candidates:
            source_counts: dict[str, int] = {}
            for s in selected:
                source_counts[s.source] = source_counts.get(s.source, 0) + 1

            displaceable = [s for s in selected if source_counts[s.source] >= 2]
            if not displaceable:
                # No source can spare a slot without being fully starved itself.
                continue

            weakest = min(displaceable, key=lambda s: (s.score, -s.position))
            selected = [s for s in selected if s is not weakest] + [candidate]

        return selected
