# Before vs CorrectRAG Evaluation Summary

This document presents a head-to-head, controlled evaluation comparing **Plain RAG** (standard retrieve-then-generate) and **CorrectRAG** (two-stage relevance evaluation + corrective routing + web fallback) across 12 pre-defined questions.

---

## Controlled Evaluation Setup

- **Evaluation Date**: September 2, 2026
- **LLM Provider**: Groq (`openai/gpt-oss-120b`)
- **Embedding Model**: Gemini (`gemini-embedding-2`)
- **Knowledge Base**: `CRAG.pdf` (168 chunks, pure Python `InMemoryVectorStore`)
- **Web Search**: Tavily Search API
- **Top-K**: 5 chunks

---

## Results Summary Table

| ID | Category | Plain RAG Result | CorrectRAG Result | Plain Action | CRAG Action | Web Used | Outcome / Improvement |
|:---|:---|:---|:---|:---:|:---:|:---:|:---|
| **q01** | `clearly_supported` | Answered (p3 triggers) | Safe Refusal | N/A | CORRECT | No | Plain RAG retrieved p3 definition; CRAG dense top-5 missed p3 and safely refused. |
| **q02** | `clearly_supported` | Answered (p2) | Answered (p2, p10) | N/A | CORRECT | No | Equivalent: Both correctly explained decompose-then-recompose knowledge refinement. |
| **q03** | `clearly_supported` | Answered (p6) | Answered (p6) | N/A | CORRECT | No | Equivalent: Both accurately identified query rewriter functionality. |
| **q04** | `clearly_supported` | Answered (p2, p8) | Answered (p2, p8) | N/A | CORRECT | No | Equivalent: Both listed the 4 evaluation benchmark datasets. |
| **q05** | `weak_or_incomplete` | Safe Refusal | Safe Refusal | N/A | CORRECT | No | Equivalent: Both correctly refused to invent missing training hyperparameters. |
| **q06** | `weak_or_incomplete` | Safe Refusal | Safe Refusal | N/A | INCORRECT | Yes | CRAG Judge correctly identified missing mobile memory benchmarks and attempted web search. |
| **q07** | `weak_or_incomplete` | Safe Refusal | Safe Refusal | N/A | CORRECT | No | Equivalent: Both refused to claim multimodal support absent from the paper. |
| **q08** | `outside_document` | Safe Refusal | **Answered via Web** | N/A | INCORRECT | Yes | **CRAG Improvement**: Judge triggered web search; accurately answered 2023 World Cup winner & venue. |
| **q09** | `outside_document` | Safe Refusal | Safe Refusal | N/A | INCORRECT | Yes | CRAG Judge correctly recognized out-of-domain query and attempted web search. |
| **q10** | `outside_document` | Safe Refusal | **Answered via Web** | N/A | INCORRECT | Yes | **CRAG Improvement**: Judge triggered web search; accurately provided Australian capital & population. |
| **q11** | `ambiguous_or_tricky` | Safe Refusal | Safe Refusal | N/A | CORRECT | No | Equivalent: Both refused compound query lacking GraphRAG comparative text. |
| **q12** | `ambiguous_or_tricky` | Answered (p8, p15) | Answered (p15) | N/A | CORRECT | No | Equivalent: Both accurately differentiated Self-RAG from CRAG. |

---

## Quantitative Breakdown

| Metric | Plain RAG | CorrectRAG | Notes |
|:---|:---:|:---:|:---|
| **Total Questions** | 12 | 12 | Fixed test set |
| **Grounded Answers** | 12 / 12 | 12 / 12 | Both systems remained strictly grounded (no unsupported claims or hallucinations) |
| **Unsupported Claims** | 0 / 12 | 0 / 12 | Neither system fabricated answers when context was absent |
| **Off-Topic Answers** | 0 / 12 | 0 / 12 | Both systems stayed strictly focused on the user query |
| **Successful External Web Corrections** | 0 (N/A) | 2 / 3 out-of-domain | CorrectRAG answered out-of-domain questions (q08, q10) where Plain RAG failed |
| **Judge Invocations** | 0 (N/A) | 4 / 12 queries | Stage 2 LLM Judge was called only on borderline/out-of-domain queries |

---

## Qualitative Analysis & Key Insights

1. **Where CorrectRAG Provided Concrete Value**:
   - On out-of-domain questions (**q08**: 2023 Cricket World Cup, **q10**: Australian capital & population), the Stage 2 LLM Judge identified that retrieved internal paper chunks were irrelevant (`judge_decision: INCORRECT`), triggered query rewriting and Tavily web search, and synthesized fully grounded answers. Plain RAG could only return an unanswerable refusal.

2. **Where Plain RAG and CorrectRAG Were Equivalent**:
   - On core, supported questions within `CRAG.pdf` (**q02**, **q03**, **q04**, **q12**), both systems extracted the correct facts and provided grounded answers.
   - On questions where neither the document nor web snippets contained sufficient evidence (**q05**, **q07**, **q09**, **q11**), both systems avoided hallucinating and safely refused to answer.

3. **Known Dense Retrieval Interaction (q01)**:
   - On **q01**, dense retrieval with top-5 retrieved broad introductory text that met the pre-filter similarity threshold (`s_max > 0.5`), routing to `CORRECT`. Knowledge refinement then filtered out sentences lacking the exact trigger definitions, leading the grounded LLM to refuse. Plain RAG happened to include p3 chunk in its raw prompt window. This reinforces our documented architectural recommendation for hybrid BM25 + dense retrieval on exact keyword queries.
