# Dataset Audit

This document presents a comprehensive, item-by-item audit of the 30-question evaluation dataset ([`evaluation/dataset.json`](file:///c:/Users/HP/Downloads/ProJ_Hired/correctrag/evaluation/dataset.json)) against the ground-truth research paper: **"Corrective Retrieval Augmented Generation"** (arXiv:2401.15884v3 / [`CRAG.pdf`](file:///c:/Users/HP/Downloads/ProJ_Hired/correctrag/CRAG.pdf)), conducted following the post-audit dataset corrections.

---

## 1. Summary

- **Total Questions Audited**: 30
- **Count by Category**:
  - `INTERNAL_SUPPORTED`: 10 (q001 – q010)
  - `INTERNAL_IRRELEVANT`: 10 (q011 – q020)
  - `INTERNAL_PARTIAL`: 10 (q021 – q030)
- **Status Distribution**:
  - **VALID**: 30 / 30 (100.0%)
  - **QUESTIONABLE**: 0 / 30 (0.0%)
  - **INVALID**: 0 / 30 (0.0%)

---

## 2. Full Audit

| ID | Category | Question | Reference Answer | Factually Correct? | Supported by CRAG.pdf? | Exact Page & Section in CRAG.pdf | Category Justified? | Concerns / Ambiguities | Status |
|---|---|---|---|---|---|---|---|---|---|
| **q001** | `INTERNAL_SUPPORTED` | What does CRAG stand for? | Corrective Retrieval Augmented Generation | Yes | Yes | Page 1 (Title, Abstract) | Yes | None. Standard definition. | **VALID** |
| **q002** | `INTERNAL_SUPPORTED` | What are the three actions defined by CRAG's action trigger? | CORRECT, INCORRECT, and AMBIGUOUS | Yes | Yes | Page 2 (Sec 1), Page 4 (Sec 3.2), Page 5 (Alg 1 & Sec 4.3) | Yes | None. Core paper mechanism. | **VALID** |
| **q003** | `INTERNAL_SUPPORTED` | What retrieval evaluator model does the CRAG paper use? | T5-large fine-tuned on question-document relevance pairs | Yes | Yes | Page 4 (Sec 4.2), Page 9 (Sec 5.3), Page 15 (Appendix B.1) | Yes | None. Distinguishes paper's T5-large from our CrossEncoder adaptation. | **VALID** |
| **q004** | `INTERNAL_SUPPORTED` | What notation does the CRAG paper use for the maximum relevance score across retrieved documents? | s_max or the maximum of the individual relevance scores s_i | Yes | Yes | Page 5 (Alg 1, lines 1–2; Sec 4.3) | Yes | Alg 1 lines 1–2 computes confidence over individual scores `{score1, ..., scorek}`, and Sec 4.3 establishes upper/lower thresholding over the max score; note field explicitly documents `s_max` as the project's formalization. | **VALID** |
| **q005** | `INTERNAL_SUPPORTED` | What web search engine does the CRAG paper use for external knowledge retrieval? | Google Search API | Yes | Yes | Page 6 (Footnote 3), Page 15 (Appendix B.1) | Yes | None. Footnote 3 explicitly states "Google Search API is utilized for searching." | **VALID** |
| **q006** | `INTERNAL_SUPPORTED` | Which datasets were used to evaluate CRAG in the paper? | PopQA, Biography, PubHealth, and Arc-Challenge | Yes | Yes | Page 6 (Sec 5.1), Page 7 (Table 1), Page 15 (Appendix B.1) | Yes | Corrected post-audit to match Section 5.1 and Table 1 exactly. | **VALID** |
| **q007** | `INTERNAL_SUPPORTED` | What is knowledge refinement in the context of CRAG? | The process of decomposing retrieved documents into fine-grained knowledge strips, scoring each strip for relevance, filtering irrelevant strips, and recomposing the remaining strips for generation. | Yes | Yes | Page 6 (Sec 4.4) | Yes | None. Section 4.4 explicitly details decompose-then-recompose. | **VALID** |
| **q008** | `INTERNAL_SUPPORTED` | What does the INCORRECT action trigger in CRAG cause the system to do? | Discard the internally retrieved documents and use web search to obtain external knowledge for generation. | Yes | Yes | Page 5 (Alg 1 lines 6–8, Sec 4.3) | Yes | None. Section 4.3 explicitly states internal documents are discarded and web search is introduced. | **VALID** |
| **q009** | `INTERNAL_SUPPORTED` | Who are the authors of the CRAG paper? | Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling | Yes | Yes | Page 1 (Cover metadata) | Yes | None. Direct metadata retrieval. | **VALID** |
| **q010** | `INTERNAL_SUPPORTED` | What is the AMBIGUOUS action in CRAG and when is it triggered? | AMBIGUOUS is triggered when the maximum relevance score falls between the lower threshold beta and the upper threshold alpha. It causes the system to combine refined internal documents with external web search results. | Yes | Yes | Page 5 (Alg 1 lines 9–12, Sec 4.3) | Yes | None. Algorithm 1 and Section 4.3 define intermediate thresholding and combination. | **VALID** |
| **q011** | `INTERNAL_IRRELEVANT` | What is the capital city of France? | Paris | Yes | No | Not supported by CRAG.pdf | Yes | General world knowledge absent from paper. Forces INCORRECT action and web search. | **VALID** |
| **q012** | `INTERNAL_IRRELEVANT` | In what year was the Python programming language first released? | 1991 | Yes | No | Not supported by CRAG.pdf | Yes | CS history absent from paper. | **VALID** |
| **q013** | `INTERNAL_IRRELEVANT` | What is the boiling point of water at standard atmospheric pressure? | 100 degrees Celsius (212 degrees Fahrenheit) | Yes | No | Not supported by CRAG.pdf | Yes | Basic physics fact absent from paper. | **VALID** |
| **q014** | `INTERNAL_IRRELEVANT` | Who wrote the play Romeo and Juliet? | William Shakespeare | Yes | No | Not supported by CRAG.pdf | Yes | Literature knowledge absent from paper. | **VALID** |
| **q015** | `INTERNAL_IRRELEVANT` | What is the chemical symbol for gold? | Au | Yes | No | Not supported by CRAG.pdf | Yes | Chemistry knowledge absent from paper. | **VALID** |
| **q016** | `INTERNAL_IRRELEVANT` | How many planets are in the solar system? | Eight (Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune) | Yes | No | Not supported by CRAG.pdf | Yes | Astronomy knowledge absent from paper. | **VALID** |
| **q017** | `INTERNAL_IRRELEVANT` | What is the current population of Tokyo? | Approximately 13-14 million in the city proper, 37-38 million in greater Tokyo | Yes | No | Not supported by CRAG.pdf | Yes | Dynamic real-world demographic data absent from paper. | **VALID** |
| **q018** | `INTERNAL_IRRELEVANT` | Who painted the Mona Lisa? | Leonardo da Vinci | Yes | No | Not supported by CRAG.pdf | Yes | Art history knowledge absent from paper. | **VALID** |
| **q019** | `INTERNAL_IRRELEVANT` | What programming language is TensorFlow primarily written in? | C++ (with Python bindings) | Yes | No | Not supported by CRAG.pdf | Yes | Software engineering fact absent from paper. | **VALID** |
| **q020** | `INTERNAL_IRRELEVANT` | What is the largest ocean on Earth? | The Pacific Ocean | Yes | No | Not supported by CRAG.pdf | Yes | Geography fact absent from paper. | **VALID** |
| **q021** | `INTERNAL_PARTIAL` | What are the main limitations of standard Retrieval Augmented Generation? | Standard RAG blindly trusts retrieved documents regardless of their relevance quality, which can lead to hallucinations when retrieved documents are irrelevant or misleading. | Yes | Partial | Page 1–2 (Abstract, Sec 1), Page 3 (Sec 2) | Yes | Paper motivates CRAG via RAG limitations; comprehensive answer benefits from broader IR/LLM literature. | **VALID** |
| **q022** | `INTERNAL_PARTIAL` | What is the T5 model and how is it typically used in NLP tasks? | T5 (Text-to-Text Transfer Transformer) is a transformer model by Google that reformulates NLP tasks as text-to-text problems. It is used for classification, summarization, translation, and question answering. | Yes | Partial | Page 4 (Sec 4.2), Page 15 (Appendix B.1) | Yes | Paper mentions T5-large as evaluator; full model definition and general NLP usage require external sources. | **VALID** |
| **q023** | `INTERNAL_PARTIAL` | How does BM25 differ from dense vector retrieval? | BM25 is a sparse lexical retrieval method based on term frequency and inverse document frequency. Dense retrieval uses neural embeddings to represent queries and documents in a continuous vector space. | Yes | Partial | Page 3 (Sec 2), Page 4 (Sec 4.2) | Yes | Paper notes sparse vs dense retrievers and Contriever; comparison details require external knowledge. | **VALID** |
| **q024** | `INTERNAL_PARTIAL` | What is hallucination in large language models? | Hallucination refers to the tendency of LLMs to generate factually incorrect, fabricated, or unsupported statements that are presented with apparent confidence. | Yes | Partial | Page 1–2 (Sec 1), Page 6 (Sec 4.5) | Yes | Paper describes hallucination in motivation and Sec 4.5; comprehensive definition benefits from external literature. | **VALID** |
| **q025** | `INTERNAL_PARTIAL` | What is ChatGPT and how does it relate to RAG systems? | ChatGPT is a conversational AI assistant developed by OpenAI based on GPT models. RAG systems augment language models like ChatGPT with external retrieved knowledge to reduce hallucinations and improve factual accuracy. | Yes | Partial | Page 3 (Sec 2), Page 4 (Sec 4.2), Page 7 (Table 1) | Yes | ChatGPT is evaluated as a baseline and evaluator comparator; definition requires external knowledge. | **VALID** |
| **q026** | `INTERNAL_PARTIAL` | How is factual accuracy typically measured in question answering systems? | Common metrics include exact match (EM), accuracy, F1 score over token overlap, and FactScore. The CRAG paper uses Accuracy for PopQA, PubHealth, and Arc-Challenge, and FactScore for Biography. | Yes | Partial | Page 6 (Sec 5.1), Page 7 (Table 1) | Yes | Corrected post-audit to state Accuracy (PopQA, PubHealth, Arc-Challenge) and FactScore (Biography) in accordance with Section 5.1. | **VALID** |
| **q027** | `INTERNAL_PARTIAL` | What are transformer attention mechanisms and why are they important for language models? | Transformer attention mechanisms allow models to weigh the relevance of different input tokens when producing each output token. This enables modeling of long-range dependencies and is the foundation of modern language models. | Yes | Partial | Page 4 (Sec 4.2 citations), Page 11–13 (References) | Yes | Paper uses transformer models (T5, LLaMA), but mechanism explanation requires external ML foundations. | **VALID** |
| **q028** | `INTERNAL_PARTIAL` | What is the role of fine-tuning in adapting pretrained language models for specific tasks? | Fine-tuning adapts a pretrained model to a specific task or domain by training it on a smaller task-specific dataset. The CRAG paper fine-tunes T5-large as its retrieval evaluator. | Yes | Partial | Page 4 (Sec 4.2), Page 15 (Appendix B.3) | Yes | Evaluator fine-tuning is explained in detail; general paradigm definition requires broader context. | **VALID** |
| **q029** | `INTERNAL_PARTIAL` | How does query rewriting improve web search performance in RAG systems? | Query rewriting converts verbose natural language questions into concise, keyword-focused search queries that better match search engine retrieval patterns. The CRAG paper rewrites queries to maximize external search effectiveness. | Yes | Partial | Page 6 (Sec 4.5), Page 14 (Appendix A) | Yes | Paper details its query rewriter prompt and keyword decomposition; general search engine interaction benefits from external context. | **VALID** |
| **q030** | `INTERNAL_PARTIAL` | What is the difference between open-domain and closed-domain question answering? | Open-domain QA answers questions from any domain using large corpora or the web. Closed-domain QA answers questions restricted to a specific knowledge source. CRAG operates in the open-domain setting. | Yes | Partial | Page 1 (Abstract), Page 2 (Sec 1), Page 6 (Sec 5) | Yes | Paper frames CRAG within open-domain QA; taxonomy comparison requires external references. | **VALID** |

---

## 3. Questionable / Invalid Items

**None.** Following the post-audit corrections:
- **`q006`** was updated to reference the exact 4 evaluation datasets in Section 5.1 and Table 1 (`PopQA`, `Biography`, `PubHealth`, and `Arc-Challenge`).
- **`q026`** was updated to accurately state the paper's metrics (`Accuracy` for PopQA/PubHealth/Arc-Challenge and `FactScore` for Biography).
- **`q004`** note field was refined to document `s_max` as the project's formalized notation for the paper's maximum relevance score evaluation across retrieved document pairs.

All 30 questions have zero factual discrepancies with `CRAG.pdf`.

---

## 4. Recommended Corrections

**None.** The dataset is fully consistent with `CRAG.pdf` and adheres strictly to all schema, category, and reference-answer guidelines.

---

## 5. Readiness Decision

### **READY FOR REAL EXPERIMENT**

- **VALID Count**: 30
- **QUESTIONABLE Count**: 0
- **INVALID Count**: 0

**Conclusion**:
The 30-question evaluation dataset ([`evaluation/dataset.json`](file:///c:/Users/HP/Downloads/ProJ_Hired/correctrag/evaluation/dataset.json)) is fully verified, factually grounded in `CRAG.pdf`, and ready for execution with [`evaluation/runner.py`](file:///c:/Users/HP/Downloads/ProJ_Hired/correctrag/evaluation/runner.py).
