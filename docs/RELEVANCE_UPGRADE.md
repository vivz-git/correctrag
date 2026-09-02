# CorrectRAG Relevance Evaluation Upgrade Report

## 1. Overview of Changes
Replaced the static threshold-based Action Router with a **Two-Stage Relevance Evaluator**. 
The similarity pre-filter is a cheap routing optimization. The LLM judge handles only borderline retrieval cases.
- Stage 1 uses embedding similarity mapping to instantly route extremely confident decisions.
- Stage 2 uses an LLM judge (`openai/gpt-oss-120b`) via structured JSON output to evaluate borderline relevance. 

## 2. Pre-filter Threshold Math
- **CLEARLY_RELEVANT_THRESHOLD**: Maintained the baseline `alpha = 0.5` (which maps to a raw cosine similarity of `0.75`).
- **CLEARLY_IRRELEVANT_THRESHOLD**: Maintained the baseline `beta = -0.2` (which maps to a raw cosine similarity of `0.40`).
- By keeping the mathematical thresholds identical, we preserved the exact baseline behavior for the fast pre-filter.

## 3. The LLM Judge 
- Borderline queries (where `0.40 < max_similarity < 0.75`) now trigger the LLM judge.
- All K chunks are combined into a single prompt context block.
- The judge evaluates whether the *overall* evidence directly answers the query.
- The judge outputs a strictly formatted JSON decision (`CORRECT`, `AMBIGUOUS`, `INCORRECT`) along with a concise one-line reason.

## 4. Pipeline Constraints Preserved
- The LLM judge is called **exactly once per query** (at most).
- The `KnowledgeRefiner` remains completely decoupled from the judge, ensuring that sentence-level strips rely purely on the fast similarity evaluator, avoiding dozens of LLM calls.
- Downstream actions (Correct, Incorrect, Ambiguous) still fire perfectly based on the judge's new verdict.

## 5. Model Integration (Groq & 120B)
- The judge leverages the existing `LLMProvider` abstraction.
- It seamlessly uses `GroqClient` querying `openai/gpt-oss-120b` without introducing any provider-specific code into `ActionRouter`.

## 6. Observability Schema Updates
- The `TraceSchema` (and internal `ExecutionTrace`) was extended with 5 new fields:
  - `similarity_pre_filter_decision`
  - `judge_called`
  - `judge_decision`
  - `judge_reason`
  - `judge_latency`
- This ensures full observability in the UI and backend logging.

## 7. Error Handling & Fallback
- If the LLM provider fails (e.g., API timeout), the router catches the exception and safely falls back to `AMBIGUOUS` while recording the failure in `judge_reason`.
- A robust regex/JSON fallback parser guarantees safe routing even if the LLM output is malformed.

## 8. Latency Impact
- **Stage 1 (Pre-filter)**: No latency change (~0ms logic overhead after retrieval).
- **Stage 2 (Judge)**: For borderline queries, introduces 1 additional LLM API call. Using Groq's fast LPU inference, this adds roughly ~300-600ms of latency depending on context size.

## 9. Cost Impact
- Embedding queries remain extremely cheap (Gemini API).
- Clear cases skip the judge entirely.
- Borderline cases cost an additional input context window (query + ~4 chunks) and output tokens (JSON decision) on `gpt-oss-120b`. Given the small output size, this is highly economical.

## 10. Regression Check
- `pytest` suite ran successfully (294 tests passing).
- `test_action_router.py` was fully rewritten to test the two-stage logic, JSON parsing, API failures, and pre-filter routing.
- `test_crag_pipeline.py` required modifications to pass the new `RoutingDecision` objects but functionally verified that downstream behavior remains intact.

## 11. Documentation Updates
- `README.md` was updated to accurately reflect the Two-Stage Relevance Evaluator.
- explicitly clarified that the system does not attempt to reproduce a PyTorch CrossEncoder, but rather offers a more explainable, lightweight alternative via an LLM judge.

## 12. Final Verdict
**READY TO COMMIT**

The refactor successfully satisfies all constraints, preserves the semantic routing capabilities of CRAG, introduces rich observability, maintains local performance, and correctly integrates the 120B model as a single-call judge for borderline retrievals.
