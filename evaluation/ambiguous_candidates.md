# CorrectRAG AMBIGUOUS Action Candidate Analysis

This document records the local, offline relevance evaluation of all 30 questions from [`evaluation/dataset.json`](dataset.json) against the local ChromaDB vector store (`CRAG.pdf` 168 chunks) using the frozen CrossEncoder relevance evaluator (`ms-marco-MiniLM-L-6-v2`) and the production `ActionRouter` thresholds ($\alpha = 0.5, \beta = -0.2$).

Zero external API calls (Gemini, Groq, or Tavily) were made during this analysis.

---

## 1. Complete 30-Question Evaluation Summary

$$\text{Decision Rule: } \text{Action} = \begin{cases} \text{CORRECT}, & s_{\max} > 0.5 \\ \text{INCORRECT}, & s_{\max} < -0.2 \\ \text{AMBIGUOUS}, & -0.2 \le s_{\max} \le 0.5 \end{cases}$$

| ID | Dataset Category | Question Text | $s_{\max}$ | Action |
|---|---|---|:---:|:---:|
| **`q001`** | `INTERNAL_SUPPORTED` | What does CRAG stand for? | `+0.9621` | `CORRECT` |
| **`q002`** | `INTERNAL_SUPPORTED` | What are the three actions defined by CRAG's action trigger? | `+0.8576` | `CORRECT` |
| **`q003`** | `INTERNAL_SUPPORTED` | What retrieval evaluator model does the CRAG paper use? | `+0.9958` | `CORRECT` |
| **`q004`** | `INTERNAL_SUPPORTED` | What notation does the CRAG paper use for the maximum relevance score across retrieved documents? | `+0.8124` | `CORRECT` |
| **`q005`** | `INTERNAL_SUPPORTED` | What web search engine does the CRAG paper use for external knowledge retrieval? | `+0.9929` | `CORRECT` |
| **`q006`** | `INTERNAL_SUPPORTED` | Which datasets were used to evaluate CRAG in the paper? | `+0.9804` | `CORRECT` |
| **`q007`** | `INTERNAL_SUPPORTED` | What is knowledge refinement in the context of CRAG? | `+0.9871` | `CORRECT` |
| **`q008`** | `INTERNAL_SUPPORTED` | What does the INCORRECT action trigger in CRAG cause the system to do? | `+0.9224` | `CORRECT` |
| **`q009`** | `INTERNAL_SUPPORTED` | Who are the authors of the CRAG paper? | `-0.6648` | `INCORRECT` |
| **`q010`** | `INTERNAL_SUPPORTED` | What is the AMBIGUOUS action in CRAG and when is it triggered? | `+0.3957` | **`AMBIGUOUS`** |
| **`q011`** | `INTERNAL_IRRELEVANT` | What is the capital city of France? | `-1.0000` | `INCORRECT` |
| **`q012`** | `INTERNAL_IRRELEVANT` | In what year was the Python programming language first released? | `-0.9999` | `INCORRECT` |
| **`q013`** | `INTERNAL_IRRELEVANT` | What is the boiling point of water at standard atmospheric pressure? | `-1.0000` | `INCORRECT` |
| **`q014`** | `INTERNAL_IRRELEVANT` | Who wrote the play Romeo and Juliet? | `-0.9966` | `INCORRECT` |
| **`q015`** | `INTERNAL_IRRELEVANT` | What is the chemical symbol for gold? | `-1.0000` | `INCORRECT` |
| **`q016`** | `INTERNAL_IRRELEVANT` | How many planets are in the solar system? | `-1.0000` | `INCORRECT` |
| **`q017`** | `INTERNAL_IRRELEVANT` | What is the current population of Tokyo? | `-1.0000` | `INCORRECT` |
| **`q018`** | `INTERNAL_IRRELEVANT` | Who painted the Mona Lisa? | `-1.0000` | `INCORRECT` |
| **`q019`** | `INTERNAL_IRRELEVANT` | What programming language is TensorFlow primarily written in? | `-0.9999` | `INCORRECT` |
| **`q020`** | `INTERNAL_IRRELEVANT` | What is the largest ocean on Earth? | `-1.0000` | `INCORRECT` |
| **`q021`** | `INTERNAL_PARTIAL` | What are the main limitations of standard Retrieval Augmented Generation? | `+0.9780` | `CORRECT` |
| **`q022`** | `INTERNAL_PARTIAL` | What is the T5 model and how is it typically used in NLP tasks? | `-0.9032` | `INCORRECT` |
| **`q023`** | `INTERNAL_PARTIAL` | How does BM25 differ from dense vector retrieval? | `-0.9958` | `INCORRECT` |
| **`q024`** | `INTERNAL_PARTIAL` | What is hallucination in large language models? | `+0.9965` | `CORRECT` |
| **`q025`** | `INTERNAL_PARTIAL` | What is ChatGPT and how does it relate to RAG systems? | `+0.3099` | **`AMBIGUOUS`** |
| **`q026`** | `INTERNAL_PARTIAL` | How is factual accuracy typically measured in question answering systems? | `-0.8738` | `INCORRECT` |
| **`q027`** | `INTERNAL_PARTIAL` | What are transformer attention mechanisms and why are they important for language models? | `-0.9731` | `INCORRECT` |
| **`q028`** | `INTERNAL_PARTIAL` | What is the role of fine-tuning in adapting pretrained language models for specific tasks? | `-0.6593` | `INCORRECT` |
| **`q029`** | `INTERNAL_PARTIAL` | How does query rewriting improve web search performance in RAG systems? | `+0.9795` | `CORRECT` |
| **`q030`** | `INTERNAL_PARTIAL` | What is the difference between open-domain and closed-domain question answering? | `-0.9342` | `INCORRECT` |

---

## 2. Top 5 Closest Candidates to the Ambiguous Interval $[-0.2, 0.5]$

| Rank | Question ID | Distance to Interval | $s_{\max}$ | Action | Category | Question Text |
|:---:|:---:|:---:|:---:|:---:|---|---|
| **1** | **`q010`** | **0.0000** | **`+0.3957`** | **`AMBIGUOUS`** | `INTERNAL_SUPPORTED` | *What is the AMBIGUOUS action in CRAG and when is it triggered?* |
| **2** | **`q025`** | **0.0000** | **`+0.3099`** | **`AMBIGUOUS`** | `INTERNAL_PARTIAL` | *What is ChatGPT and how does it relate to RAG systems?* |
| 3 | **`q004`** | 0.3124 | `+0.8124` | `CORRECT` | `INTERNAL_SUPPORTED` | *What notation does the CRAG paper use for the maximum relevance score across retrieved documents?* |
| 4 | **`q002`** | 0.3576 | `+0.8576` | `CORRECT` | `INTERNAL_SUPPORTED` | *What are the three actions defined by CRAG's action trigger?* |
| 5 | **`q008`** | 0.4224 | `+0.9224` | `CORRECT` | `INTERNAL_SUPPORTED` | *What does the INCORRECT action trigger in CRAG cause the system to do?* |

---

## 3. Recommended Questions for Live AMBIGUOUS Pilot

We recommend selecting **`q010`** and **`q025`** to directly exercise the `AMBIGUOUS` branch.

### Candidate 1: `q010`
- **Question**: *"What is the AMBIGUOUS action in CRAG and when is it triggered?"*
- **$s_{\max}$**: **`+0.3957`** (falls cleanly between $-0.2$ and $0.5$)
- **Per-Chunk Scores**: `[-0.7800, -0.7315, +0.3957, -0.0893, -0.6149]`
- **Why it is an ideal candidate**:
  - The internal vector retriever finds chunks describing the CRAG action framework, but the text gives a moderate relevance match (+0.3957) rather than a decisive >0.5 score.
  - This activates the AMBIGUOUS path: the pipeline will refine the internal strips from `CRAG.pdf` AND execute a web search to supplement with external explanations, combining both contexts into the final generation prompt.

### Candidate 2: `q025`
- **Question**: *"What is ChatGPT and how does it relate to RAG systems?"*
- **$s_{\max}$**: **`+0.3099`** (falls cleanly between $-0.2$ and $0.5$)
- **Per-Chunk Scores**: `[-0.9919, -0.9907, -0.9879, -0.9240, +0.3099]`
- **Why it is an ideal candidate**:
  - `CRAG.pdf` mentions ChatGPT in its related work and baselines (providing moderate internal relevance of +0.3099), but lacks a comprehensive general definition.
  - This perfectly matches the theoretical premise of `AMBIGUOUS`: internal documents contain partial domain clues, but external web search provides the missing foundational definition, exercising combined internal + external refinement.
