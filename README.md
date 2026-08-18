# CorrectRAG

CorrectRAG is a production-oriented implementation of the Corrective Retrieval Augmented Generation framework designed to mitigate hallucinations in retrieval-augmented language models. By evaluating retrieved document quality prior to generation, the system dynamically routes queries across three confidence states—refining high-confidence internal documents into clean knowledge strips, discarding low-confidence results in favor of rewritten external web searches, or combining both when retrieval relevance is ambiguous.

## 📄 Research Paper

- **Paper Title**: Corrective Retrieval Augmented Generation
- **Authors**: Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling (USTC / UCLA / Google DeepMind)
- **Reference**: [arXiv:2401.15884v3](https://arxiv.org/abs/2401.15884) [cs.CL]
- **Document Reference**: [CRAG.pdf](CRAG.pdf) & [paper_spec.md](paper_spec.md)

## 📌 Project Status

**Current Status**: `Scaffolding`

The project repository and basic service structure are established. No RAG retrieval, evaluator inference, web search, or database logic is active at this stage.

## 🏗️ High-Level Planned Architecture

1. **Retrieval Layer**: Retrieves initial top-$K$ candidate passages from local vector storage for a given query.
2. **Retrieval Evaluator**: Computes calibrated relevance scores for retrieved passages.
3. **Action Trigger**: Routes execution into one of three deterministic actions based on confidence thresholds:
   - **CORRECT**: Refines internal documents (Decompose $\to$ Filter $\to$ Recompose) into internal knowledge strips ($k_{in}$).
   - **INCORRECT**: Drops internal documents, rewrites query into keywords, searches the web, and refines web content into external knowledge strips ($k_{ex}$).
   - **AMBIGUOUS**: Synthesizes both internal knowledge strips and external web knowledge strips ($k_{in} + k_{ex}$).
4. **Knowledge Refinement**: Segments coarse documents into fine-grained atomic strips, filters out irrelevant noise, and recomposes salient content.
5. **Generator Layer**: Passes the refined reference knowledge context to a generative LLM to produce a grounded response with execution provenance.

## ⚠️ Important Note on Adaptation

This project is a **production-oriented engineering adaptation** of the CRAG methodology, not an exact reproduction of the paper's academic research environment. It adapts the paper's conceptual mechanisms (relevance evaluation, confidence routing, knowledge refinement, keyword query rewriting, and web augmentation) to standard production components rather than running supervised fine-tuning of 0.77B T5 models or legacy benchmark harnesses on A800 GPU clusters.
