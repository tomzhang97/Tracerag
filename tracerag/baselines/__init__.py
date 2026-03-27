"""
Baseline Wrappers for TraceRAG.

Provides standard interfaces for comparison models used in the paper:

OCR baselines (classical floor):
    TesseractBaseline       — Tesseract with optional rotation retries
    PaddleOCRBaseline       — PP-OCRv5

OCR + text RAG baseline ("what if I just OCR and do RAG?"):
    OCRTextRAGBaseline      — OCR → BM25/FAISS → shared LLM reader

Visual retrieval baselines (main direct comparators):
    ColPaliBaseline         — ColPali v1.3 with MaxSim late interaction
    ColQwen2Baseline        — ColQwen2 v1.0 with MaxSim late interaction

End-to-end VLM baseline ("what if I just give pages to a VLM?"):
    Qwen2VLBaseline         — Qwen2.5-VL-7B-Instruct (or Qwen2-VL-7B-Instruct)

Diff baselines (for the visual-diff / revision task):
    AbsDiffBaseline         — Absolute pixel difference
    SiameseDiffBaseline     — Feature-level diff (pixel fallback)
    OCRTextDiffBaseline     — difflib over OCR text

Shared answerer:
    All retrieval-stack baselines (OCR+RAG, ColPali, ColQwen2, TraceRAG)
    should use Qwen2.5-7B-Instruct as the downstream reader so that
    retrieval / grounding is the only variable.
    Qwen2.5-VL is kept as a SEPARATE end-to-end VLM baseline.

Runner:
    BaselineRunner          — Unified multi-baseline evaluation orchestrator
"""

from tracerag.baselines.ocr import BaseOCR, TesseractBaseline, PaddleOCRBaseline
from tracerag.baselines.vision import BaseVisionRetriever, ColPaliBaseline, ColQwen2Baseline
from tracerag.baselines.text_rag import OCRTextRAGBaseline
from tracerag.baselines.vlm import Qwen2VLBaseline
from tracerag.baselines.diff import AbsDiffBaseline, SiameseDiffBaseline, OCRTextDiffBaseline
from tracerag.baselines.runner import BaselineRunner

__all__ = [
    # OCR
    "BaseOCR",
    "TesseractBaseline",
    "PaddleOCRBaseline",
    # Vision retrieval
    "BaseVisionRetriever",
    "ColPaliBaseline",
    "ColQwen2Baseline",
    # OCR + text RAG
    "OCRTextRAGBaseline",
    # End-to-end VLM
    "Qwen2VLBaseline",
    # Diff
    "AbsDiffBaseline",
    "SiameseDiffBaseline",
    "OCRTextDiffBaseline",
    # Runner
    "BaselineRunner",
]
