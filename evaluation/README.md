# CorrectRAG Evaluation Framework

This directory contains the evaluation framework for comparing **Baseline RAG**
against **CorrectRAG** on a reproducible, locally-defined question set.

---

## Why We Need a Baseline

Standard RAG blindly trusts retrieved documents regardless of their quality.
Without a baseline, we cannot measure whether the added complexity of CRAG
(evaluator + router + refinement + web search) produces better answers.

**The baseline is the honest reference point** — if CRAG does not outperform
simple RAG on at least some question categories, the extra components are
adding latency without value.

---

## Why the Same Questions Must Be Used

Using the same question set for both systems is a fundamental requirement for
a fair comparison:

- Any difference in answer quality is attributable to the system, not the question.
- Both systems use the same internal document collection and the same top-K retrieval.
- Latency and error rates are directly comparable.

Using different questions would make any comparison meaningless.

---

## Evaluation Categories

### INTERNAL_SUPPORTED
Questions that can be **fully answered** from the internal document collection
(the CRAG paper, arXiv:2401.15884v3).

- **Expected CRAG behaviour**: CORRECT action → internal refinement only.
- **Expected Baseline behaviour**: Should also perform well (high-quality retrieval).
- **Purpose**: Verifies that CRAG does not *degrade* performance when the
  internal knowledge base is sufficient.

### INTERNAL_IRRELEVANT
Questions for which the internal documents contain **no useful evidence**
(general world knowledge facts not present in the CRAG paper).

- **Expected CRAG behaviour**: INCORRECT action → web search fallback.
- **Expected Baseline behaviour**: Fail to find relevant chunks; produce
  low-quality or "cannot answer" responses.
- **Purpose**: Verifies that CRAG's corrective routing provides value when
  internal retrieval is genuinely insufficient.

### INTERNAL_PARTIAL
Questions where the internal documents contain **partial evidence** but
additional external sources are needed for a complete answer.

- **Expected CRAG behaviour**: AMBIGUOUS action → combined internal + external.
- **Expected Baseline behaviour**: Partially correct answers.
- **Purpose**: Tests the AMBIGUOUS routing path and combined knowledge composition.

---

## Files

| File | Purpose |
|---|---|
| `dataset.json` | 30-question evaluation dataset (10 per category) |
| `runner.py` | Evaluation runner — executes both systems, collects results |
| `metrics.py` | Metric computation — transparent, reproducible functions |
| `results.json` | Output of a real experiment run (empty until executed) |
| `report.md` | Comparison report template (populated after experiment) |

---

## Metrics

### Answer Correctness — Keyword F1
Token-level F1 between the generated answer and a human-written reference answer.
Same method used in SQuAD and TriviaQA evaluation. Transparent and reproducible.

**Limitations**: Paraphrastic answers score lower than they deserve. Items that
cannot be automatically scored are flagged as `manual_review_required`.

### Citation Correctness
Fraction of expected sources that appear in the system's citations (recall).
Only applies to questions with `expected_sources` in the dataset.
Most current questions require **manual review** for citation correctness.

### CRAG Action Distribution
Percentage of questions routed to CORRECT / INCORRECT / AMBIGUOUS.
This is the most directly interpretable signal for verifying that the
router and evaluator are working correctly.

### Mean Latency
Wall-clock time per question. CRAG will generally have higher latency than
Baseline RAG due to CrossEncoder evaluation, optional web search, and
knowledge refinement.

### Error Rate
Fraction of questions that produced an exception. Errors include API failures,
timeout issues, and component errors.

---

## This Is Not a Paper Benchmark Reproduction

The CRAG paper (arXiv:2401.15884v3) reports evaluation results on:
- PopQA, TriviaQA, Biography datasets
- With a fine-tuned T5-large retrieval evaluator
- With specific fitted alpha/beta thresholds

**Our system differs in every one of these dimensions**:
- Different dataset (our local 30-question set)
- Different evaluator (frozen CrossEncoder, not fine-tuned T5-large)
- Different thresholds (development defaults, not empirically fitted)
- Different web search (Tavily, not Google Search API)

Do not compare our numeric results to the paper's Table 1 or Table 2.
This is an **engineering evaluation** to verify that the system components
work together correctly and to establish a baseline for future improvements.

---

## Running the Evaluation

```bash
# Prerequisites
# 1. Set environment variables
export GEMINI_API_KEY="your-key"
export TAVILY_API_KEY="your-key"

# 2. Ingest documents (if not already done)
python scripts/demo_retrieval.py CRAG.pdf

# 3. Run evaluation
python evaluation/runner.py

# 4. Run only specific system or category
python evaluation/runner.py --system crag --category INTERNAL_SUPPORTED

# 5. Run tests (no API calls required)
pytest tests/test_evaluation.py -v
```

---

## Extending the Dataset

To add questions:
1. Append to the `questions` array in `dataset.json`.
2. Assign a unique `id` (e.g., `q031`).
3. Choose the correct `category`.
4. Write a human `reference_answer`.
5. Add `notes` explaining the category assignment.

Re-run the evaluation after adding questions.
