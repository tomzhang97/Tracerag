"""
TraceRAG: Vector-Native, Revision-Aware, and Layout-Linked Multimodal Retrieval
for Technical Documents.

A high-precision retrieval system designed for large, multimodal technical PDFs
(CAD drawings, P&IDs, schematics, manuals) with exact, auditable, spatially precise,
and version-correct evidence grounding.
"""

__version__ = "0.1.0"
__author__ = "TraceRAG Team"

from tracerag.common.types import (
    BBox,
    VectorObject,
    PatchGrid,
    RegionEvidence,
    Claim,
    CertifiedClaim,
)

__all__ = [
    "BBox",
    "VectorObject",
    "PatchGrid",
    "RegionEvidence",
    "Claim",
    "CertifiedClaim",
]
