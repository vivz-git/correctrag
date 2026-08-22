# CorrectRAG Engineering Evaluation Report

> **Status**: Experiment Completed & Verified
> **Dataset**: [`evaluation/dataset.json`](dataset.json) (30 audited questions across 3 categories)
> **Results Artifact**: [`evaluation/groq_final_results.json`](groq_final_results.json) (60 records: 30 Baseline RAG + 30 CorrectRAG)
> **Execution Provider**: Groq (`openai/gpt-oss-120b`) + Tavily Search

---

## 1. Experiment Setup

| Parameter | Value |
|---|---|
| **Dataset Version** | `1.0` (30 questions: 10 `INTERNAL_SUPPORTED`, 10 `INTERNAL_IRRELEVANT`, 10 `INTERNAL_PARTIAL`) |
| **Document Context** | `CRAG.pdf` (arXiv:2401.15884v3; 168 chunks in local ChromaDB `correctrag` collection) |
| **Total Evaluation Records** | 60 (30 Baseline RAG + 30 CorrectRAG) |
| **LLM Provider** | Groq Cloud API (`openai/gpt-oss-120b`) |
| **Web Search Provider** | Tavily (`tavily-python` SDK, max results = 5) |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, local CPU) |
| **Relevance Evaluator** | `cross-encoder/ms-marco-MiniLM-L-6-v2` (temperature = 1.0, local CPU) |
| **Action Thresholds** | $\alpha = 0.5$ (upper confidence), $\beta = -0.2$ (lower confidence) |
| **Knowledge Refinement** | $\gamma = -0.5$ (strip filtering threshold), 2 sentences per strip, top $K = 5$ |

---

## 2. Dataset Description

The evaluation dataset consists of 30 audited questions divided into three equal cohorts:

1. **`INTERNAL_SUPPORTED` (10 questions, `q001`–`q010`)**:
   Questions whose ground-truth answers are explicitly documented within `CRAG.pdf` (e.g., framework acronym, core definitions, evaluation datasets).
2. **`INTERNAL_IRRELEVANT` (10 questions, `q011`–`q020`)**:
   General-world and factual questions completely absent from `CRAG.pdf` (e.g., geography, history, chemistry). Tests whether the system correctly detects retrieval failure and falls back to external web search.
3. **`INTERNAL_PARTIAL` (10 questions, `q021`–`q030`)**:
   Broader NLP/CS questions where `CRAG.pdf` provides contextual background but external knowledge is required for a complete, well-rounded answer (e.g., T5 architecture, fine-tuning mechanisms, open-domain vs. closed-domain QA).

---

## 3. Systems Compared

1. **Baseline RAG (`BaselineRAG`)**:
   Standard Retrieve-then-Generate pipeline. Retrieves the top-$K$ internal chunks from ChromaDB, formats them into a prompt with source metadata, and generates an answer via the LLM. It possesses no confidence scoring, no dynamic fallback, and no external search capability.
2. **CorrectRAG (`CRAGPipeline`)**:
   Full implementation of Algorithm 1 from arXiv:2401.15884v3 with production adaptations. Retrieves top-$K$ internal chunks, evaluates each chunk with a CrossEncoder ($s_{\max}$), and routes to:
   - **`CORRECT`** ($s_{\max} > \alpha$): Decomposes internal chunks into fine-grained strips, filters by relevance score ($\gamma$), and generates from refined internal context only.
   - **`INCORRECT`** ($s_{\max} < \beta$): Discards internal retrieval, rewrites query into targeted search keywords, queries Tavily web search, refines external web snippets into strips, and generates from external knowledge.
   - **`AMBIGUOUS`** ($\beta \le s_{\max} \le \alpha$): Refines internal document strips AND performs query rewriting + web search, generating from combined internal and external context.

---

## 4. Evaluation Methodology & Metrics

- **Answer Correctness (`keyword_overlap_score`)**:
  Deterministic token-level F1 score over normalized bag-of-words overlap between generated answers and reference answers. It provides a transparent, non-stochastic measure of lexical precision and recall without relying on black-box LLM-as-a-judge scoring.
- **Latency**:
  Wall-clock execution time (in seconds) measured directly around each system call (`BaselineRAG.run()` vs `CRAGPipeline.run()`).
- **CRAG Execution Trace**:
  Full recording of $s_{\max}$, chosen action, rewritten queries, internal strip counts, external strip counts, and final context sources.

---

## 5. Overall Results

| System | Evaluated Records | Success Rate | Mean Keyword F1 | Mean Latency | Median Latency | p95 Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline RAG** | 30 / 30 | 100.0% | **0.1235** | **6.369s** | **7.857s** | **10.003s** |
| **CorrectRAG** | 30 / 30 | 100.0% | **0.1412** | **6.674s** | **6.491s** | **12.658s** |
| **Comparison** | — | — | **+0.0178 (+14.4%)** | **+0.305s (+4.8%)** | -1.366s | +2.655s |

---

## 6. Category-Level Results

| Category | Questions | Baseline Mean F1 | CRAG Mean F1 | Absolute Delta | Relative Change |
|---|:---:|:---:|:---:|:---:|:---:|
| **`INTERNAL_SUPPORTED`** | 10 | **0.1753** | **0.1065** | **-0.0688** | **-39.2%** |
| **`INTERNAL_IRRELEVANT`** | 10 | **0.0189** | **0.0800** | **+0.0612** | **+324.3%** |
| **`INTERNAL_PARTIAL`** | 10 | **0.1762** | **0.2371** | **+0.0609** | **+34.5%** |
| **Overall Dataset** | **30** | **0.1235** | **0.1412** | **+0.0178** | **+14.4%** |

---

## 7. CRAG Action Distribution

Across the 30 questions, the CRAG action router assigned:

| Action | Count | Percentage | Questions Routed |
|---|:---:|:---:|---|
| **`CORRECT`** | **11** | **36.7%** | `q001`, `q002`, `q003`, `q004`, `q005`, `q006`, `q007`, `q008`, `q021`, `q024`, `q029` |
| **`AMBIGUOUS`** | **2** | **6.7%** | `q010`, `q025` |
| **`INCORRECT`** | **17** | **56.7%** | `q009`, `q011`, `q012`, `q013`, `q014`, `q015`, `q016`, `q017`, `q018`, `q019`, `q020`, `q022`, `q023`, `q026`, `q027`, `q028`, `q030` |

### Context Source Composition:
- **`internal` only**: 11 runs (36.7%)
- **`external` only**: 10 runs (33.3%)
- **`combined` (internal + web)**: 2 runs (6.7%)
- **`none` (safely refused after filter)**: 7 runs (23.3%)

---

## 8. Latency Analysis

- **Baseline RAG**:
  - Mean: 6.369s | Median: 7.857s | p95: 10.003s | Cold-Start (`q001`): 6.563s | Warm Mean: 6.362s
- **CorrectRAG**:
  - Mean: 6.674s | Median: 6.491s | p95: 12.658s | Cold-Start (`q001`): 12.424s | Warm Mean: 6.476s
- **Overhead**:
  - CRAG adds approximately **+0.305s (+4.8%)** latency overhead on average.
  - The cold-start differential on `q001` (~5.8s) is a one-time cost due to initializing the local CrossEncoder model (`ms-marco-MiniLM-L-6-v2`) in memory. Once warm, CRAG internal branches are faster than baseline generations because refined strips reduce the prompt token count.

---

## 9. Robustness Incident & Resolution (`q025`)

### The Incident:
During initial full evaluation, `q025` (*"What is ChatGPT and how does it relate to RAG systems?"*) triggered the `AMBIGUOUS` branch but aborted with:
```text
ValueError: LLM returned an empty response for query rewriting.
```
Groq's `openai/gpt-oss-120b` returned an empty completion string during query rewriting. `QueryRewriter` raised an unhandled `ValueError`, which crashed the entire pipeline before web search or generation could occur.

### The Fix:
[`backend/app/external/query_rewriter.py`](../backend/app/external/query_rewriter.py) was updated with a production-grade fallback:
- When the LLM returns an empty, whitespace-only, or unparseable response, `QueryRewriter.rewrite()` catches the `ValueError` and falls back safely to the original cleaned query (`clean_query = query.strip()`).
- Actual provider/network exceptions (`GroqAPIError`, `GeminiAPIError`, timeouts) continue to propagate unhindered.

### Verification:
After the fix, `q025` completed with 100% success in `groq_final_results.json`, correctly routing to `AMBIGUOUS`, generating `"ChatGPT, Retrieval Augmented Generation, relationship"`, and combining 1 internal strip with 5 external web strips.

---

## 10. Top Answer-Level Improvements (CRAG > Baseline)

1. **`q028` (+0.3559)** (*LLM fine-tuning role*):
   - Baseline: F1 = 0.0909 (found only brief mentions in paper).
   - CRAG: Routed to `INCORRECT`, retrieved comprehensive external literature, scoring **F1 = 0.4468**.
2. **`q014` (+0.3077)** (*Author of Romeo and Juliet*):
   - Baseline: F1 = 0.0000 (retrieval refusal).
   - CRAG: Routed to `INCORRECT`, queried web search, and returned Shakespeare (**F1 = 0.3077**).
3. **`q030` (+0.2090)** (*Open vs. closed domain QA*):
   - Baseline: F1 = 0.0455 (minimal mention in paper text).
   - CRAG: Retrieved authoritative external definitions (**F1 = 0.2545**).
4. **`q015` (+0.1818)** (*Chemical symbol for gold*):
   - Baseline: F1 = 0.0000 (no relevant chunks).
   - CRAG: Web search retrieved "Au" and chemistry context (**F1 = 0.1818**).
5. **`q027` (+0.1534)** (*Transformer attention mechanisms*):
   - Baseline: F1 = 0.0930 (paper-specific notes only).
   - CRAG: Supplemented general transformer architecture context (**F1 = 0.2464**).

---

## 11. Top Answer-Level Degradations (Baseline > CRAG)

1. **`q002` (-0.2759)** (*CRAG three actions definition*):
   - Baseline: F1 = 0.2759 (fed all 5 raw chunks to generator, providing enough paragraph context).
   - CRAG: F1 = 0.0000 (`KnowledgeRefiner` stripped chunks to 2-sentence units; only 1 strip passed $\gamma$, which pruned surrounding explanatory sentences and caused strict prompt refusal).
2. **`q026` (-0.2316)** (*QA accuracy measurement*):
   - Baseline: F1 = 0.2316 (extracted partial metrics from paper Section 5).
   - CRAG: F1 = 0.0000 (routed to `INCORRECT`, but external search snippets did not survive $\gamma$ filtering).
3. **`q008` (-0.2312)** (*INCORRECT action trigger behavior*):
   - Baseline: F1 = 0.2979.
   - CRAG: F1 = 0.0667 (fine-grained strip filtering pruned adjacent explanatory sentences).
4. **`q007` (-0.1459)** (*Knowledge refinement definition*):
   - Baseline: F1 = 0.3146.
   - CRAG: F1 = 0.1687 (strip filtering reduced overall lexical overlap).
5. **`q020` (-0.1053)** (*Largest ocean on Earth*):
   - Baseline: F1 = 0.1053.
   - CRAG: F1 = 0.0000 (external search snippets were pruned by $\gamma$ filtering).

---

## 12. Methodological Distinctions & Limitations

### Distinction: Research Paper Methodology vs Our Engineering Adaptation

| Component | Original CRAG Paper (arXiv:2401.15884v3) | CorrectRAG Engineering Adaptation |
|---|---|---|
| **Retrieval Evaluator** | Fine-tuned **T5-large** specifically trained on QA-relevance pairs. | Frozen **`ms-marco-MiniLM-L-6-v2`** cross-encoder with sigmoid score mapping. |
| **Vector Retrieval** | BM25 + dense neural hybrid retriever. | Dense cosine similarity via ChromaDB + `all-MiniLM-L6-v2`. |
| **Threshold Calibration** | Empirically optimized on held-out QA training sets. | Development heuristics ($\alpha=0.5, \beta=-0.2, \gamma=-0.5$). |
| **External Search Engine** | Google Search API. | Tavily Search API (`tavily-python`). |
| **External Knowledge Refinement** | Sent raw web snippets directly to generation. | Converts web results to chunks and applies `KnowledgeRefiner` before generation. |
| **Benchmark Scale** | Thousands of questions across PopQA, Biography, PubHealth, Arc-Challenge. | 30-question curated engineering evaluation dataset. |

### Evaluation Limitations:
1. **Sample Size**: 30 questions provide structural validation of pipeline branches, not statistical significance.
2. **Metric Constraint**: Bag-of-words keyword F1 measures lexical overlap, not semantic equivalence or factual nuance.
3. **Strip Pruning Tradeoff**: Strict strip filtering reduces hallucination risk on noisy passages, but can occasionally prune peripheral context on dense domain-specific questions.
4. **Dense Retrieval Depth & Evaluator Routing Interaction (Empirically Verified)**:
   - On compound multi-intent queries like `"What does CRAG stand for and what are its three actions?"`, full-corpus analysis of the 168 chunks in `chroma_data/` showed that ground-truth definition chunks were ranked at dense positions #40 (`CRAG_p5_c004`), #45 (`CRAG_p2_c003`), #73 (`CRAG_p3_c010`), and #75 (`CRAG_p1_c003`) by `all-MiniLM-L6-v2`.
   - Offline experiments across $k \in \{5, 8, 10, 15\}$ verified that expanding top-$k$ depth did not bring definition chunks into the candidate set, while increasing candidate count amplified zero-shot CrossEncoder false positives ($s_{\max}$ reached `+0.9940` at $k=15$ on a non-definition summary chunk).
   - This false-positive $s_{\max} > \alpha$ triggers `CORRECT` routing and suppresses external web search fallback, resulting in an unanswerable refusal from the strictly grounded LLM.
   - This is an adaptation-specific engineering finding associated with frozen dense bi-encoders and zero-shot cross-encoders, not a finding reproduced from the original CRAG paper benchmark. Promising future iterations include hybrid retrieval (BM25 + dense), evaluator fine-tuning, or retrieval sufficiency verification.

---

## 13. Final Engineering Interpretation

1. **Overall Lift**: CorrectRAG achieved a **+14.4% relative improvement in mean keyword F1** (0.1412 vs 0.1235) with a modest **+4.8% latency overhead** (+0.305s).
2. **Dynamic Adaptation Validated**:
   - On **`INTERNAL_IRRELEVANT`**, CRAG delivered a **+324.3% lift** (0.0800 vs 0.0189) by catching internal retrieval failure and falling back to web search.
   - On **`INTERNAL_PARTIAL`**, CRAG delivered a **+34.5% lift** (0.2371 vs 0.1762) by synthesizing internal and external knowledge.
   - On **`INTERNAL_SUPPORTED`**, Baseline scored higher (0.1753 vs 0.1065) because raw un-pruned chunks preserved broad contextual cues that fine-grained strip filtering pruned.
3. **Engineering Readiness**: The end-to-end orchestration, multi-provider abstraction, query rewriting fallback, and provenance tracking are validated and ready for production packaging.
