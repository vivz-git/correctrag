# CorrectRAG: Research Specification & Engineering Extraction

> **Source Paper**: "Corrective Retrieval Augmented Generation" (CRAG)  
> **Authors**: Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling (USTC / UCLA / Google DeepMind)  
> **arXiv Identifier**: `arXiv:2401.15884v3` [cs.CL] (7 Oct 2024, first published Jan 2024)  
> **Official Code**: `https://github.com/HuskyInSalt/CRAG`

---

## 1. Paper Information
- **Title**: Corrective Retrieval Augmented Generation
- **Authors & Affiliations**:
  - Shi-Qi Yan (National Engineering Research Center of Speech and Language Information Processing, University of Science and Technology of China)
  - Jia-Chen Gu (Department of Computer Science, University of California, Los Angeles)
  - Yun Zhu (Google DeepMind)
  - Zhen-Hua Ling (University of Science and Technology of China)
- **Key Contributions** *(Sec 1, p. 2)*:
  1. Systematic formulation of failure modes in RAG when retrieval returns inaccurate, low-quality, or noisy results.
  2. Proposal of **CRAG (Corrective RAG)**, a plug-and-play architectural framework incorporating:
     - A lightweight **Retrieval Evaluator** estimating document and strip relevance.
     - A three-way **Action Trigger** (`{Correct, Incorrect, Ambiguous}`).
     - A **Decompose-then-Recompose Knowledge Refinement** algorithm.
     - A **Web Search** mechanism for external knowledge supplementation.
  3. Empirical validation across short-form QA (PopQA), long-form biography generation (Biography), and closed-set reasoning (PubHealth, Arc-Challenge).

---

## 2. Problem Being Solved
- **LLM Hallucinations & Factual Vulnerability** *(Sec 1, p. 1; Sec 2, p. 2)*:
  - Generative LLMs rely on static parametric knowledge and struggle with factual errors on long-tail, specialized, or time-sensitive entities.
- **Retriever Fragility in Standard RAG** *(Sec 1, p. 1-2; Sec 3, p. 3)*:
  - Standard RAG couples retriever and generator unconditionally:
    $$\mathcal{P}(Y|X) = \mathcal{P}(D|X)\mathcal{P}(Y, D|X)$$
  - Standard RAG exhibits **low risk tolerance**: if the retriever fetches irrelevant, misleading, or obsolete documents, the generator incorporates them and generates confident hallucinations *(Sec 1, p. 1, Figure 1; Sec 3, p. 3)*.
- **Document-Level Noise & Granularity Mismatch** *(Sec 1, p. 2; Sec 4.4, p. 6)*:
  - Retrievers return whole documents/passages where only a minor subset of sentences is pertinent. Passing entire documents introduces non-essential context and distracts the generator.
- **Static Corpus Boundaries** *(Sec 1, p. 2; Sec 4.5, p. 6)*:
  - Static local corpora cannot resolve queries that exceed their indexed knowledge. Without an adaptive external retrieval mechanism, standard RAG fails silently.

---

## 3. Standard RAG Failure Mode
- **Indiscriminate Context Injection** *(Sec 1, p. 1-2; Sec 3, p. 3)*:
  - Standard RAG blindly prepends top-$K$ retrieved passages into the generator prompt without evaluating relevance, factuality, or coverage.
- **Documented Failure Modes in Paper** *(Sec 1, p. 1, Figure 1; Sec 5.6, p. 9)*:
  - *Distractor Passages / False Positives*: Retrievers retrieve passages that share high lexical or topical overlap but lack the specific answer, actively misleading the LLM (e.g., retrieving 1989 Batman movie production notes for a question about "Death of a Batman", leading the generator to hallucinate wrong screenwriter names).
  - *Retrieval Quality Collapse*: Degrading retriever recall severely degrades downstream generator accuracy in unaugmented RAG and Self-RAG *(Sec 5.6, p. 9, Figure 3)*.
  - *Context Distraction*: Dense, unrefined documents bury key facts inside irrelevant paragraphs *(Sec 2, p. 3; Sec 4.4, p. 6)*.

---

## 4. CRAG Core Idea
- **Confidence-Guided Action Routing** *(Sec 4.1, p. 3; Sec 4.3, p. 5)*:
  - Evaluate document relevance prior to generation using a dedicated evaluator.
  - Route the query dynamically based on confidence thresholds:
    - **CORRECT**: Internal knowledge is trustworthy $\to$ refine internal documents.
    - **INCORRECT**: Internal knowledge is unhelpful $\to$ discard internal documents and search the web.
    - **AMBIGUOUS**: Evaluator is uncertain $\to$ combine refined internal documents with external web search knowledge.
- **Decompose-then-Recompose Knowledge Refinement** *(Sec 4.4, p. 6)*:
  - Break coarse documents into atomic multi-sentence "knowledge strips", evaluate each strip independently, filter out irrelevant strips below a filter threshold, and recompose the most salient strips.
- **Search Query Rewriting** *(Sec 4.5, p. 6; Appendix A, p. 14)*:
  - Transform complex or verbose questions into concise keyword queries before calling external search engines.

---

## 5. System Architecture

```
                       +-------------------------+
                       |    Input Query (x)      |
                       +------------+------------+
                                    |
                                    v
                       +-------------------------+
                       |   Retriever R (Top-K)   |
                       +------------+------------+
                                    |
                           Retrieved Documents D
                                    |
                                    v
                       +-------------------------+
                       |  Retrieval Evaluator E  |
                       |  Scores: {score_1..k}   |
                       +------------+------------+
                                    |
                                    v
                       +-------------------------+
                       |     Action Trigger      |
                       | (Thresholds: alpha, beta)|
                       +----+-------+-------+----+
                            |       |       |
            +---------------+       |       +---------------+
            | (Correct)             | (Ambiguous)           | (Incorrect)
            v                       v                       v
  +-------------------+   +-------------------+   +-------------------+
  | Knowledge Refine  |   | Knowledge Refine  |   |  Query Rewriter   |
  |  (Decompose ->    |   |  (Internal: k_in) |   |    W(x) -> q      |
  |  Filter ->        |   +---------+---------+   +---------+---------+
  |  Recompose)       |             |                       |
  |     -> k_in       |             |                       v
  +---------+---------+             |             +-------------------+
            |                       |             |    Web Search     |
            |                       |             |  (External: k_ex) |
            |                       |             +---------+---------+
            |                       |                       |
            |             +---------+---------+             |
            |             |   Combine k_in    |             |
            |             |     + k_ex        |             |
            |             +---------+---------+             |
            |                       |                       |
            v                       v                       v
     k = Internal             k = Internal             k = External
      Knowledge                + External               Knowledge
            \                       |                      /
             \                      |                     /
              +-----------------> + - + <----------------+
                                    |
                                    v
                       +-------------------------+
                       |       Generator G       |
                       |      Output y=G(x, k)   |
                       +-------------------------+
```

### Frozen Production Stack & Component Status Matrix

| Component | Paper Implementation | Our Frozen Production Stack | Adaptation Label |
| :--- | :--- | :--- | :--- |
| **Retriever ($R$)** | Contriever ($K=10$) on Wikipedia dumps | **ChromaDB** with `all-MiniLM-L6-v2` dense embeddings | `[OUR ADAPTATION]` |
| **Evaluator ($E$)** | Supervised fine-tuned T5-large (0.77B), scores $\in [-1, 1]$ | **Calibrated Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with calibrated temperature scaling to $[-1, 1]$ | `[OUR ADAPTATION]` |
| **Action Trigger** | Hardcoded empirical thresholds $(\alpha, \beta)$ | **Configurable Decision Trigger** implementing exact 3-way branching logic with explicit configurable thresholds | `[OUR ADAPTATION]` |
| **Knowledge Refiner** | Strip decomposition + strip scoring + filter threshold ($\gamma$) + top-5 | **Regex sentence-window chunker** + cross-encoder strip scoring + filter threshold ($\gamma$) + top-5 recomposer | `[OUR ADAPTATION]` |
| **Query Rewriter ($W$)** | ChatGPT (GPT-3.5) few-shot prompt extracting $\le 3$ keywords | **Structured Few-Shot LLM Rewriter** adhering to the exact 3-keyword paper prompt | `[OUR ADAPTATION]` |
| **Web Search Engine** | Google Search API + `<p>` tag scraping | **Tavily Search API** (with fallback to `DuckDuckGo Search` for zero-key local execution) + paragraph refinement | `[OUR ADAPTATION]` |
| **Generator ($G$)** | LLaMA2-hf-7B / SelfRAG-LLaMA2-7B | **Standard OpenAI/Gemini-compatible LLM Client** (e.g. `gpt-4o-mini` / `gemini-1.5-flash`) | `[OUR ADAPTATION]` |

---

## 6. Retrieval Evaluator

### [PAPER] *(Sec 4.2, p. 4; Appendix B.3, p. 15)*
- **Architecture**: Fine-tuned **T5-large** (0.77B parameters).
- **Input Representation**: Pairwise concatenation of query and document `(x, d_i)` for each retrieved document $d_i \in D$ ($K=10$).
- **Scoring Range**: Evaluator produces continuous relevance scores $s_i \in [-1, 1]$, where positive target label is $+1$ and negative target label is $-1$.
- **Training Strategy**:
  - Supervised fine-tuning on PopQA training split (12.6k samples).
  - Positive samples: Wikipedia golden subject passages.
  - Negative samples: Randomly sampled topically-similar but irrelevant retrieval outputs.
- **Empirical Accuracy** *(Sec 5.5, p. 9, Table 4)*:
  - T5-large Evaluator: **84.3%** accuracy on PopQA document relevance classification.
  - ChatGPT direct prompt: 58.0%; ChatGPT-CoT: 62.4%; ChatGPT few-shot: 64.7%.

### [OUR ADAPTATION]
- **Calibrated Relevance Scoring**:
  - Rather than applying arbitrary mathematical normalization (e.g., naive min-max scaling or uncalibrated linear stretching which skews decision thresholds), we employ **calibrated relevance scoring**.
  - Using a frozen Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), raw cross-entropy logits $z$ are calibrated via temperature-scaled logistic transformation mapped to $[-1, 1]$:
    $$s = 2 \cdot \sigma\left(\frac{z}{T}\right) - 1, \quad \text{where } \sigma(u) = \frac{1}{1 + e^{-u}}$$
  - The temperature parameter $T$ and bias are calibrated so that $s > 0$ denotes probable relevance, $s < 0$ denotes probable irrelevance, and $|s| \to 1$ represents high model certainty.
- Takes `(query_x, doc_text)` $\to$ returns calibrated float $s \in [-1.0, 1.0]$.

### [NOT IMPLEMENTING]
- Custom fine-tuning of 0.77B T5 model checkpoints from scratch on A800 GPU clusters.

---

## 7. Confidence / Action Trigger

### [PAPER] *(Sec 4.3, p. 5; Appendix B.3, p. 15)*
- Given individual document relevance scores: $\{score_1, score_2, \dots, score_k\}$.
- Maximum document relevance score: $s_{max} = \max_{i \in [1, k]} (score_i)$.
- **Decision Logic**:
  1. **CORRECT**: If at least one document confidence score exceeds upper threshold $\alpha$:
     $$s_{max} > \alpha \implies \textbf{CORRECT}$$
  2. **INCORRECT**: If all retrieved document confidence scores fall below lower threshold $\beta$:
     $$s_{max} < \beta \implies \textbf{INCORRECT}$$
  3. **AMBIGUOUS**: If the maximum score falls between the thresholds:
     $$\beta \le s_{max} \le \alpha \implies \textbf{AMBIGUOUS}$$
- **Academic Benchmark Thresholds** *(Appendix B.3, p. 15)*:
  - PopQA: $(\alpha=0.59, \beta=-0.99)$
  - PubHealth: $(\alpha=0.50, \beta=-0.91)$
  - Arc-Challenge: $(\alpha=0.50, \beta=-0.91)$
  - Biography: $(\alpha=0.95, \beta=-0.91)$

### [OUR ADAPTATION]
- **No Universal Default Thresholds**:
  - The paper's threshold values $(\alpha=0.59, \beta=-0.99)$ were derived empirically on specific academic datasets using their specific fine-tuned T5-large evaluator.
  - In production engineering, these numbers are **not universal scientific constants**. Thresholds $(\alpha, \beta)$ are **domain- and model-dependent hyperparameters**.
  - The system exposes explicit configurable hyperparameters $(\alpha, \beta)$ in system configuration, allowing application engineers to calibrate thresholds against their validation domain.
  - Baseline starter parameters: $\alpha = 0.50, \beta = -0.50$ (subject to domain calibration).

### [NOT IMPLEMENTING]
- Hardcoding static, unconfigurable threshold constants into pipeline logic.

---

## 8. CORRECT Action

### [PAPER] *(Sec 4.3, p. 5; Sec 4.4, p. 6; Algorithm 1 Lines 3-5)*
- **Trigger Condition**: $s_{max} > \alpha$.
- **Semantic Meaning**: At least one internal retrieved document contains verified, high-confidence relevant information.
- **Workflow**:
  1. Raw unrefined documents $D$ are not fed directly to the generator.
  2. Decompose-then-recompose Knowledge Refinement is executed on $D$:
     $$\text{Internal\_Knowledge} = \text{Knowledge\_Refine}(x, D)$$
  3. Final reference knowledge is set to internal knowledge: $k = k_{in}$.
  4. External web search is bypassed to optimize latency and prevent external web noise.
  5. Generator $G$ receives $(x, k_{in})$.

### [OUR ADAPTATION]
- Fully reproduces the paper workflow: passes internal documents to the Knowledge Refiner, produces $k_{in}$, bypasses web search, and passes $(x, k_{in})$ to generator.

### [NOT IMPLEMENTING]
- None (faithfully reproduced).

---

## 9. INCORRECT Action

### [PAPER] *(Sec 4.3, p. 5; Sec 4.5, p. 6; Algorithm 1 Lines 6-8)*
- **Trigger Condition**: $s_{max} < \beta$.
- **Semantic Meaning**: Every document retrieved from the internal corpus is irrelevant, misleading, or noisy.
- **Workflow**:
  1. All internal retrieved documents $D$ are **completely discarded**.
  2. Query Rewriter $W$ rewrites input question $x$ into concise search engine keywords:
     $$q = W(x)$$
  3. External web search is executed using query $q$.
  4. Retrieved web content is segmented and filtered via Knowledge Refinement to extract $k_{ex}$:
     $$\text{External\_Knowledge} = \text{Web\_Search}(W(x))$$
  5. Final reference knowledge is set strictly to external knowledge: $k = k_{ex}$.
  6. Generator $G$ receives $(x, k_{ex})$.

### [OUR ADAPTATION]
- Fully reproduces the paper workflow: completely drops internal documents, triggers keyword query rewrite, queries search engine, refines web paragraphs via the evaluator, and routes $(x, k_{ex})$ to generator.

### [NOT IMPLEMENTING]
- Hardcoding Google Search API exclusively with strict Wikipedia-only URL filtering heuristics.

---

## 10. AMBIGUOUS Action

### [PAPER] *(Sec 4.3, p. 5-6; Algorithm 1 Lines 9-12)*
- **Trigger Condition**: $\beta \le s_{max} \le \alpha$.
- **Semantic Meaning**: The evaluator cannot confidently confirm or refute relevance; documents have intermediate quality or partial relevance.
- **Workflow**:
  1. Internal retrieved documents $D$ are refined into internal knowledge strips:
     $$\text{Internal\_Knowledge} = \text{Knowledge\_Refine}(x, D)$$
  2. Query Rewriter and external web search are executed to extract external knowledge strips:
     $$\text{External\_Knowledge} = \text{Web\_Search}(W(x))$$
  3. Internal and external knowledge are combined via ordered concatenation:
     $$k = \text{Internal\_Knowledge} + \text{External\_Knowledge} = k_{in} + k_{ex}$$
  4. Generator $G$ receives $(x, k_{in} + k_{ex})$.
- **Ablation Insight** *(Sec 4.3 Discussion, p. 5-6; Sec 5.4, p. 8 Table 2)*:
  - The ambiguous action acts as a soft fallback mechanism that insulates the pipeline against evaluator misclassification and boundary noise.

### [OUR ADAPTATION]
- Parallel/sequential execution of internal refinement and web search refinement, concatenating $k_{in}$ and $k_{ex}$ cleanly into prompt context $k$.

### [NOT IMPLEMENTING]
- None (faithfully reproduced).

---

## 11. Knowledge Refinement

### [PAPER] *(Sec 4.4, p. 6; Appendix B.3, p. 15)*
- **Decompose-then-Recompose Pipeline**:
  1. **Decomposition (Strip Segmentation)**:
     - If retrieved passage is 1–2 sentences, treat as single strip.
     - Otherwise, split document into fine-grained "knowledge strips" (each consisting of a few sentences representing an independent factual statement).
  2. **Strip Scoring**:
     - The Retrieval Evaluator $E$ calculates relevance score for each strip:
       $$score_{strip} = E(x, strip)$$
  3. **Strip Filtering**:
     - Strips with $score_{strip} < \gamma$ are discarded. (Paper parameter: $\gamma = -0.5$).
  4. **Selection & Recomposition**:
     - Rank valid strips by score and retain top-$k_{strip}$ strips (Paper parameter: $k_{strip} = 5$).
     - Concatenate retained strips in order to form $k_{in}$ (or $k_{ex}$).
- **Ablation Impact** *(Sec 5.4, p. 8 Table 3)*:
  - Removing refinement caused severe performance drop on PopQA (accuracy fell from 54.9 to 49.8 / 59.8 to 54.2).

### [OUR ADAPTATION]
- Sentence-boundary segmenter chunking documents into sliding/tumbling windows of 1–3 sentences.
- Evaluator scores each strip, drops strips below configurable strip threshold $\gamma$ (calibrated against evaluator), retains top-5 strips, and recomposes them into clean context.

### [NOT IMPLEMENTING]
- Passing raw un-chunked document blocks directly into LLM prompts.

---

## 12. Web Search / External Knowledge

### [PAPER] *(Sec 4.5, p. 6; Appendix B.3, p. 15)*
- **Search Provider**: Google Search API generating URL links.
- **Source Prioritization**: Preferentially adds authoritative and regulated domains (e.g. Wikipedia).
- **Web Content Processing**:
  - Navigates URLs and transcribes HTML content.
  - Leverages `<p>` tags as natural segment boundaries.
  - Evaluates paragraphs using the Retrieval Evaluator $E$.
  - Filters and selects top-$k_{ext} = 5$ paragraphs to construct $k_{ex}$.

### [OUR ADAPTATION]
- **Frozen Search Stack**: Tavily Search API with automated structured passage extraction, with fallback to DuckDuckGo Search for offline/keyless local runs.
- Web paragraphs are scored by the calibrated Cross-Encoder, filtered by threshold $\gamma$, and top-5 strips are selected for $k_{ex}$.

### [NOT IMPLEMENTING]
- Heavy browser automation (e.g., Selenium/Playwright) for runtime scraping during live generation requests when clean structured text is available via search API.

---

## 13. Query Rewriting

### [PAPER] *(Sec 4.5, p. 6; Appendix A, p. 14 Table 7)*
- **Objective**: Transform complex/dialogue user queries into concise keyword queries tailored for search engine indexing.
- **Prompt Formulation** *(Appendix A, Table 7, p. 14)*:
  ```text
  Extract at most three keywords separated by comma from the following dialogues and questions as queries for the web search, including topic background within dialogues and main intent within questions.

  question: What is Henry Feilden’s occupation?
  query: Henry Feilden, occupation

  question: In what city was Billy Carlson born?
  query: city, Billy Carlson, born

  question: What is the religion of John Gwynn?
  query: religion of John Gwynn

  question: What sport does Kiribati men’s national basketball team play?
  query: sport, Kiribati men’s national basketball team play

  question: [question]
  query:
  ```
- **Ablation Impact** *(Sec 5.4, p. 8 Table 3)*:
  - Removing rewriting dropped PopQA accuracy from 54.9 to 51.7 / 59.8 to 56.2.

### [OUR ADAPTATION]
- Structured Few-Shot LLM Rewriter enforcing the exact few-shot exemplar prompt and parsing the resulting $\le 3$ comma-separated keywords into search queries.

### [NOT IMPLEMENTING]
- Unmodified conversational question forwarding to search engine endpoints.

---

## 14. Final Generation

### [PAPER] *(Sec 3, p. 3; Sec 4.1, p. 3; Algorithm 1 Line 14; Appendix B.3, p. 15)*
- **Mechanism**: Generator $G$ generates output response $y$ conditioned on question $x$ and consolidated knowledge context $k$:
  $$y = G(x, k)$$
  Where:
  $$k = \begin{cases} 
  k_{in} & \text{if Confidence} = [\text{CORRECT}] \\ 
  k_{ex} & \text{if Confidence} = [\text{INCORRECT}] \\ 
  k_{in} + k_{ex} & \text{if Confidence} = [\text{AMBIGUOUS}] 
  \end{cases}$$
- **Evaluated LLMs in Paper**: LLaMA2-hf-7B, Alpaca-7B/13B, LLaMA2-13B, SelfRAG-LLaMA2-7B.
- **Plug-and-Play Contract**: CRAG modifies the reference context, making it compatible with any standard generative LLM without specialized token tuning.

### [OUR ADAPTATION]
- Generator component wrapping standard LLM API (`gpt-4o-mini`, `gemini-1.5-flash`, or local Ollama) providing structured streaming and metadata provenance logging.

### [NOT IMPLEMENTING]
- Fine-tuning special reflection token decoders (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`).

---

## 15. Algorithm 1 Translated into Plain Engineering Steps

Below is the step-by-step translation of **Algorithm 1 (CRAG Inference)** from Section 4.1, Page 5 of the paper into explicit engineering pseudocode:

```python
def crag_pipeline(
    query_x: str,
    retriever_R,
    evaluator_E,
    rewriter_W,
    web_search_engine,
    generator_G,
    alpha: float,            # Upper confidence threshold (configurable hyperparameter)
    beta: float,             # Lower confidence threshold (configurable hyperparameter)
    gamma: float,            # Strip filter threshold (configurable hyperparameter)
    top_k_strips: int = 5    # Max knowledge strips to retain
) -> CRAGResult:
    """
    Plain engineering translation of Paper Algorithm 1.
    """
    # 1. Retrieve initial top-K documents from internal corpus
    #    D = {d_1, d_2, ..., d_k}
    raw_documents = retriever_R.retrieve(query=query_x, top_k=10)
    
    # 2. Score relevance of each retrieved document pair (x, d_i)
    doc_scores = []
    for doc in raw_documents:
        score_i = evaluator_E.evaluate_pair(query=query_x, document=doc.text)
        doc_scores.append(score_i)
    
    # 3. Calculate maximum confidence score across retrieved documents
    max_confidence = max(doc_scores) if doc_scores else -1.0
    
    # 4. Determine Action Trigger based on thresholds alpha and beta
    if max_confidence > alpha:
        action = "CORRECT"
    elif max_confidence < beta:
        action = "INCORRECT"
    else:
        action = "AMBIGUOUS"
        
    # 5. Execute Action Branches
    k_internal = ""
    k_external = ""
    
    if action == "CORRECT":
        # Internal Knowledge Refinement only
        k_internal = refine_knowledge(
            query=query_x,
            documents=raw_documents,
            evaluator=evaluator_E,
            filter_threshold=gamma,
            top_k=top_k_strips
        )
        final_knowledge = k_internal
        
    elif action == "INCORRECT":
        # Discard internal documents; execute Query Rewriting + Web Search Refinement
        search_query = rewriter_W.rewrite(query=query_x)
        web_paragraphs = web_search_engine.search_and_extract_paragraphs(
            query=search_query,
            top_k_urls=5
        )
        k_external = refine_web_knowledge(
            query=query_x,
            paragraphs=web_paragraphs,
            evaluator=evaluator_E,
            filter_threshold=gamma,
            top_k=top_k_strips
        )
        final_knowledge = k_external
        
    elif action == "AMBIGUOUS":
        # Dual-source: Refine internal documents AND perform Web Search
        k_internal = refine_knowledge(
            query=query_x,
            documents=raw_documents,
            evaluator=evaluator_E,
            filter_threshold=gamma,
            top_k=top_k_strips
        )
        search_query = rewriter_W.rewrite(query=query_x)
        web_paragraphs = web_search_engine.search_and_extract_paragraphs(
            query=search_query,
            top_k_urls=5
        )
        k_external = refine_web_knowledge(
            query=query_x,
            paragraphs=web_paragraphs,
            evaluator=evaluator_E,
            filter_threshold=gamma,
            top_k=top_k_strips
        )
        # Combine both knowledge sources via ordered concatenation
        final_knowledge = f"{k_internal}\n\n{k_external}".strip()
        
    # 6. Final Generation
    response = generator_G.generate(
        query=query_x,
        knowledge_context=final_knowledge
    )
    
    return CRAGResult(
        response=response,
        action_triggered=action,
        confidence_score=max_confidence,
        doc_scores=doc_scores,
        knowledge_used=final_knowledge
    )
```

---

## 16. Paper's Experimental Setup
- **Evaluation Datasets** *(Sec 5.1, p. 6; Appendix B.1, p. 15)*:
  1. **PopQA**: Short-form open-domain QA on 1,399 long-tail entities (<100 monthly Wikipedia views). Metric: **Accuracy**.
  2. **Biography**: Long-form biography generation for entities. Metric: **FactScore** (atomic fact precision).
  3. **PubHealth**: Biomedical claim verification (True/False). Metric: **Accuracy**.
  4. **Arc-Challenge**: Multiple-choice science questions. Metric: **Accuracy**.
- **Retriever Setup** *(Sec 4.2, p. 4; Appendix B.3, p. 15)*:
  - Dense retriever: **Contriever** (Izacard et al., 2022) retrieving top-10 documents from Wikipedia corpus dumps.
- **Hardware & Compute Resources** *(Appendix B.2, p. 15)*:
  - NVIDIA A800 80GB GPUs.
  - LLaMA-2 7B inference required >40GB GPU memory.
  - T5-large fine-tuning required significantly lower compute overhead than 7B LLM tuning.
- **Performance Highlights** *(Sec 5.3, p. 7 Table 1, p. 8)*:
  - CRAG outperformed standard RAG across all datasets:
    - PopQA: $+7.0\%$ (SelfRAG-LLaMA2-7b) / $+4.4\%$ (LLaMA2-hf-7b)
    - Biography FactScore: $+14.9\%$ (SelfRAG-LLaMA2-7b) / $+2.8\%$ (LLaMA2-hf-7b)
    - PubHealth: $+36.6\%$ (SelfRAG-LLaMA2-7b) / $+10.6\%$ (LLaMA2-hf-7b)
    - Arc-Challenge: $+15.4\%$ (SelfRAG-LLaMA2-7b) / $+10.3\%$ (LLaMA2-hf-7b)

---

## 17. Paper-Specific Implementation Details
- **Evaluator Parameter Size**: T5-large has **0.77B** parameters *(Sec 4.2, p. 4)*.
- **Training Labels**: Positive = $+1$, Negative = $-1$ *(Appendix B.3, p. 15)*.
- **Scoring Range**: Evaluator outputs relevance scores in the continuous interval $[-1, 1]$ *(Appendix B.3, p. 15)*.
- **Thresholds**:
  - PopQA Action Trigger: $\alpha = 0.59, \beta = -0.99$.
  - Strip Filtering Threshold: $\gamma = -0.5$ *(Appendix B.3, p. 15)*.
  - Top Strips Retained: $k_{strip} = 5$ for internal, $k_{ext} = 5$ for external *(Appendix B.3, p. 15)*.
- **Web Search Engine**: Google Search API *(Sec 4.5 footnote 3, p. 6)*.
- **HTML Parsing**: Segmented by HTML `<p>` tags directly rather than sentence-splitting *(Appendix B.3, p. 15)*.
- **Query Rewriting Format**: GPT-3.5 Turbo few-shot prompt outputting maximum 3 comma-separated keywords *(Appendix A, p. 14 Table 7)*.

---

## 18. Components We MUST Reproduce
To faithfully implement the core algorithmic architecture of CRAG, our implementation MUST reproduce:
1. **Initial Vector/Document Retrieval**: Fetching top-$K$ candidate documents from a corpus.
2. **Relevance Evaluator Module**: Scoring each retrieved document individually with a calibrated score $s_i \in [-1, 1]$.
3. **Three-Way Action Trigger**:
   - $\max(s_i) > \alpha \implies \textbf{CORRECT}$
   - $\max(s_i) < \beta \implies \textbf{INCORRECT}$
   - $\beta \le \max(s_i) \le \alpha \implies \textbf{AMBIGUOUS}$
4. **Knowledge Decomposition**: Splitting documents into atomic multi-sentence strips.
5. **Knowledge Filtering & Ranking**: Filtering strips below threshold $\gamma$ and picking top-5 strips.
6. **Knowledge Recomposition**: Concatenating selected strips into cleaned internal knowledge $k_{in}$.
7. **Query Rewriter**: Translating queries into $\le 3$ keywords using the few-shot template.
8. **Web Search Augmentation**: Querying external web APIs on INCORRECT and AMBIGUOUS paths.
9. **Dual Knowledge Merging**: Combining $k_{in} + k_{ex}$ cleanly for the AMBIGUOUS state.
10. **Refined Final Generation**: Feeding the filtered knowledge $k$ to the LLM generator.

---

## 19. Components We Will Intentionally Simplify
1. **Evaluator Backing Model**:
   - *[PAPER]*: Supervised fine-tuned T5-large checkpoint (0.77B).
   - *[OUR ADAPTATION]*: Frozen Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with calibrated temperature scaling to $[-1, 1]$.
   - *Rationale*: Eliminates requirement for heavy custom GPU model fine-tuning infrastructure while preserving identical pairwise scoring contracts and filtering mechanics.
2. **Web Search Infrastructure**:
   - *[PAPER]*: Direct Google Search API + raw HTML scraping and `<p>` tag parsing.
   - *[OUR ADAPTATION]*: Tavily Search API with automated text extraction (with fallback to DuckDuckGo Search).
   - *Rationale*: Production search APIs handle anti-bot measures, JavaScript execution, and HTML boilerplate cleaning out of the box.
3. **Threshold Calibration Architecture**:
   - *[PAPER]*: Hardcoded per-dataset empirical thresholds (PopQA: $0.59, -0.99$, PubHealth: $0.5, -0.91$).
   - *[OUR ADAPTATION]*: Centralized, configurable threshold parameters $(\alpha, \beta, \gamma)$ in application configuration, acknowledging that thresholds must be calibrated per domain/evaluator.

---

## 20. Components We Will NOT Reproduce
1. **Self-RAG Reflection Token Architecture** *(Sec 5.2, p. 7; Appendix B.3, p. 15)*:
   - Special token vocabulary (`[Retrieve]`, `[IsRel]`, `[Critique]`, `[IsSup]`) and fine-tuning on LLaMA-2 7B.
   - *Reason*: CRAG operates at the context routing level and is designed to plug into any standard generator.
2. **Academic Benchmark Evaluation Suite**:
   - Running FactScore on 1,399 PopQA long-tail entities and full Biography dataset on GPU clusters.
   - *Reason*: This project is a production software system, not an academic re-publication harness.
3. **Legacy OpenAI Endpoints**:
   - Deprecated `gpt-3.5-turbo` legacy client calls.

---

## 21. Known Deviations from the Paper

| Item | Research Paper (arXiv:2401.15884v3) | Production CorrectRAG Adaptation | Justification |
| :--- | :--- | :--- | :--- |
| **Evaluator Checkpoint** | Fine-tuned T5-large (0.77B) on PopQA | Calibrated Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | Eliminates GPU fine-tuning requirements; enables lightweight, portable deployment |
| **Evaluator Score Calibration** | Raw fine-tuned T5 output logits $\in [-1, 1]$ | Calibrated logistic transformation $s = 2\sigma(z/T) - 1$ | Prevents arbitrary threshold mismatches; ensures consistent confidence semantics |
| **Search Engine** | Google Search API + Wikipedia preference filter | Tavily Search API (with DuckDuckGo fallback) | Modern production search reliability and rate-limit resilience |
| **Corpus Storage** | Static Wikipedia passage dump files | ChromaDB vector store with sentence-transformers | Production indexing, persistence, and querying |
| **Web Parser** | Raw HTML `<p>` tag regex split | Structured search content parser & sentence-window chunker | Handles dynamic web content cleanly without brittle HTML parsing |
| **Threshold Presets** | Dataset-specific empirical constants | Configurable hyperparameter configuration | Acknowledges that thresholds are domain- and model-dependent hyperparameters |

---

## 22. Verification Plan

The verification strategy is strictly decoupled into independent unit tests for isolated component behavior and integration tests for pipeline orchestration.

### 1. Isolated Retrieval Evaluator Unit Tests
*Tests the evaluator scoring contract in isolation without invoking routing logic.*
- **Relevance Discrimination**:
  - Pass a known highly relevant pair: `(query="Capital of France", doc="Paris is the capital of France.")` $\to$ assert calibrated score $s > 0$.
  - Pass a known irrelevant pair: `(query="Capital of France", doc="The recipe requires two cups of flour.")` $\to$ assert calibrated score $s < 0$.
- **Monotonicity**:
  - Assert $s(\text{query}, \text{relevant\_doc}) > s(\text{query}, \text{irrelevant\_doc})$.
- **Score Bounds Contract**:
  - Assert that for all evaluated pairs, $-1.0 \le s \le 1.0$.

### 2. Isolated Action Trigger & Routing Unit Tests
*Tests the 3-way decision branching in pure isolation using synthetic arrays of float scores against configurable thresholds $(\alpha = 0.50, \beta = -0.50)$ without executing neural inference.*
- **CORRECT Routing Test**:
  - Input synthetic scores: `[0.85, 0.20, -0.40]` $\to \max = 0.85 > \alpha \implies$ assert action is `CORRECT`.
- **INCORRECT Routing Test**:
  - Input synthetic scores: `[-0.60, -0.75, -0.90]` $\to \max = -0.60 < \beta \implies$ assert action is `INCORRECT`.
- **AMBIGUOUS Routing Test**:
  - Input synthetic scores: `[0.10, -0.20, -0.30]` $\to \max = 0.10 \in [\beta, \alpha] \implies$ assert action is `AMBIGUOUS`.
- **Edge Case / Empty Scores Test**:
  - Input empty score list: `[]` $\to$ assert safe fallback to `INCORRECT` (triggers web search).

### 3. Knowledge Refinement Unit Tests (Decompose-Filter-Recompose)
*Tests strip splitting, scoring filter, and recomposition.*
- **Decomposition**:
  - Input a multi-paragraph text $\to$ assert segmented into fine-grained strips of 1–3 sentences.
- **Filter Threshold ($\gamma$)**:
  - Mock strip scores: `[0.8, -0.7, 0.6, -0.9, 0.4]`. With $\gamma = -0.5$, assert strips with scores $-0.7$ and $-0.9$ are excluded.
- **Top-$K$ Selection & Recomposition**:
  - Assert top-5 ranked surviving strips are concatenated into the output knowledge string $k_{in}$ in deterministic sequence.

### 4. Query Rewriter Unit Tests
*Tests prompt formatting and keyword extraction.*
- Input: `"What is Henry Feilden’s occupation?"`
- Assert output adheres to format: comma-separated string containing $\le 3$ keywords (e.g. `"Henry Feilden, occupation"`).

### 5. Web Search Adapter Unit Tests
*Tests search API querying and paragraph extraction with mocked search responses.*
- Verify search query dispatch, URL parsing, snippet extraction, and fallback to DuckDuckGo if primary provider key is absent.

### 6. Action Path Integration Tests
*Tests end-to-end pipeline execution and knowledge routing for all three branches.*
- **Path 1 (CORRECT Flow)**:
  - Inject relevant internal documents $\to$ verify `CORRECT` action triggered $\to$ verify web search is **not** called $\to$ verify internal knowledge refinement executed $\to$ verify generator receives $k_{in}$.
- **Path 2 (INCORRECT Flow)**:
  - Inject irrelevant internal documents $\to$ verify `INCORRECT` action triggered $\to$ verify internal docs discarded $\to$ verify query rewriting and web search executed $\to$ verify generator receives $k_{ex}$.
- **Path 3 (AMBIGUOUS Flow)**:
  - Inject borderline/partially relevant documents $\to$ verify `AMBIGUOUS` action triggered $\to$ verify both internal refinement and web search executed $\to$ verify generator receives combined $k_{in} + k_{ex}$.

### 7. Provenance & Metadata Tracing
- For every query, verify that the returned result contains a complete execution audit object:
  - `action`: `CORRECT` | `INCORRECT` | `AMBIGUOUS`
  - `doc_relevance_scores`: List of calibrated float scores
  - `selected_strips`: List of extracted knowledge strips
  - `web_queries_generated`: Search queries generated if web search was invoked
  - `final_generation`: Clean generated answer string
