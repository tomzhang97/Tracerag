"""
OCR + Text RAG Baseline.

Pipeline:
    1. Run OCR (Tesseract or PaddleOCR) on every page image.
    2. Index the OCR text with BM25 or dense (FAISS) retrieval.
    3. Answer queries with a shared LLM reader (default: Qwen2.5-7B-Instruct).

This is the "what if I just OCR the docs and do normal RAG?" baseline.
The shared answerer must be the same model used across ColPali and TraceRAG
to isolate the retrieval contribution.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OCR Text RAG Baseline
# ---------------------------------------------------------------------------

class OCRTextRAGBaseline:
    """
    End-to-end OCR + retrieval + reader baseline.

    Args:
        ocr:          Instance of TesseractBaseline or PaddleOCRBaseline.
                      If None, a TesseractBaseline is created with defaults.
        index_type:   "bm25" or "faiss".
        top_k:        Number of pages to feed to the reader.
        llm_config:   Config dict for LLMAnswerer (provider, model_name, …).
                      Defaults to Qwen2.5-7B-Instruct via a local vLLM endpoint.
    """

    def __init__(
        self,
        ocr=None,
        index_type: str = "bm25",
        top_k: int = 5,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        self.top_k = top_k

        # OCR engine
        if ocr is None:
            from tracerag.baselines.ocr import TesseractBaseline
            ocr = TesseractBaseline()
        self.ocr = ocr

        # Text index
        self._index_type = index_type
        self._page_records: List[Dict[str, Any]] = []  # {page_idx, page_id, text}
        self._index_built = False

        # LLM reader
        if llm_config is None:
            llm_config = _default_qwen_config()
        from tracerag.retrieval.llm_answerer import LLMAnswerer
        self.answerer = LLMAnswerer(llm_config)

        # BM25 / FAISS state
        self._bm25 = None
        self._faiss_index = None
        self._faiss_encoder = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_pages(
        self,
        pages: List[Tuple[str, Image.Image]],
    ) -> None:
        """
        OCR all pages and build the retrieval index.

        Args:
            pages: List of (page_id, PIL image) tuples.
        """
        logger.info(f"OCR-ingesting {len(pages)} pages with {self.ocr.__class__.__name__} ...")
        self._page_records = []
        for idx, (page_id, image) in enumerate(pages):
            text = self.ocr.page_text(image)
            self._page_records.append({"page_idx": idx, "page_id": page_id, "text": text})
            logger.debug(f"  [{idx+1}/{len(pages)}] {page_id}: {len(text)} chars")

        self._build_index()

    def _build_index(self) -> None:
        texts = [r["text"] for r in self._page_records]
        if self._index_type == "bm25":
            self._build_bm25(texts)
        else:
            self._build_faiss(texts)
        self._index_built = True
        logger.info(f"Text index built ({self._index_type}, {len(texts)} pages).")

    def _build_bm25(self, texts: List[str]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 is required: pip install rank-bm25")
        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def _build_faiss(self, texts: List[str]) -> None:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "faiss-cpu and sentence-transformers are required for dense indexing."
            )
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        embs = encoder.encode(texts, show_progress_bar=False).astype(np.float32)
        faiss.normalize_L2(embs)
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)
        self._faiss_index = index
        self._faiss_encoder = encoder

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Return top-k page records sorted by relevance."""
        if not self._index_built:
            raise RuntimeError("Call ingest_pages() before retrieve().")
        if self._index_type == "bm25":
            return self._search_bm25(query)
        return self._search_faiss(query)

    def _search_bm25(self, query: str) -> List[Dict[str, Any]]:
        scores = self._bm25.get_scores(query.lower().split())
        top_idxs = np.argsort(scores)[::-1][: self.top_k]
        return [self._page_records[i] for i in top_idxs]

    def _search_faiss(self, query: str) -> List[Dict[str, Any]]:
        import faiss

        q_emb = self._faiss_encoder.encode([query]).astype(np.float32)
        faiss.normalize_L2(q_emb)
        _, idxs = self._faiss_index.search(q_emb, self.top_k)
        return [
            self._page_records[i] for i in idxs[0] if 0 <= i < len(self._page_records)
        ]

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def answer(self, query: str) -> Dict[str, Any]:
        """
        Retrieve top-k pages and generate an answer with the shared LLM reader.

        Returns dict with keys: answer, page_ids, context_text.
        """
        top_pages = self.retrieve(query)
        context_parts = []
        for rec in top_pages:
            context_parts.append(f"[Page {rec['page_id']}]\n{rec['text']}")
        context = "\n\n".join(context_parts)

        # Build a lightweight evidence list compatible with LLMAnswerer
        from tracerag.common.types import RegionEvidence

        fake_evidences = []
        evidence_texts: Dict[str, str] = {}
        for i, rec in enumerate(top_pages):
            ev = RegionEvidence(
                object_id=f"ocr_page_{rec['page_id']}_{i}",
                page_id=rec["page_id"],
                doc_id="",
                version_id="",
                obj_type="text_block",
                bbox=(0.0, 0.0, 1.0, 1.0),
                score=1.0 / (i + 1),
            )
            fake_evidences.append(ev)
            evidence_texts[ev.object_id] = rec["text"]

        result = self.answerer.generate_answer_with_claims(
            query=query,
            evidences=fake_evidences,
            evidence_texts=evidence_texts,
            query_type="attribute",
        )

        return {
            "answer": result.get("answer", ""),
            "claims": result.get("claims", []),
            "page_ids": [r["page_id"] for r in top_pages],
            "context_text": context,
        }


# ---------------------------------------------------------------------------
# Default LLM config helper
# ---------------------------------------------------------------------------

def _default_qwen_config() -> Dict[str, Any]:
    """
    Default config for Qwen2.5-7B-Instruct as shared answerer.

    Users should override base_url / api_key to point at their vLLM / Ollama
    instance, or set provider="openai_compatible" with a local endpoint.

    For Alibaba Cloud DashScope, set:
        provider: openai_compatible
        base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
        api_key: <DASHSCOPE_API_KEY>
    """
    return {
        "provider": "openai_compatible",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
        "temperature": 0.0,
        "max_tokens": 1024,
        "claim_extraction": {"enabled": True, "max_claims_per_answer": 5},
    }
