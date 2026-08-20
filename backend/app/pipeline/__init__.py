"""
Pipeline Package for CorrectRAG.

Provides the CRAG orchestration pipeline that connects all individual
components into the full corrective retrieval-augmented generation workflow
described in Algorithm 1 of the CRAG paper (arXiv:2401.15884v3).

Exports:
    CRAGPipeline  — Main CRAG inference pipeline
    CRAGResult    — Structured output of the CRAG pipeline
"""

from app.pipeline.crag_pipeline import CRAGPipeline, CRAGResult

__all__ = [
    "CRAGPipeline",
    "CRAGResult",
]
