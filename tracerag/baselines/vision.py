"""
Visual Retrieval Baselines: ColPali v1.3 and ColQwen2 v1.0.

Both models use late-interaction (MaxSim) scoring over patch-level embeddings,
following the ColPali paper.  Each baseline exposes:

    encode_queries(queries)  -> List[Tensor]   shape [L, d] per query
    encode_pages(images)     -> List[Tensor]   shape [H*W+1, d] per page
    score(query_emb, page_embs) -> List[float]

A shared `LLMAnswerer` can be plugged in via the `answerer` argument so that
retrieval baselines use the exact same downstream reader as TraceRAG.
"""
from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseVisionRetriever(abc.ABC):
    """Minimal interface for patch-embedding visual retrievers."""

    @abc.abstractmethod
    def encode_queries(self, queries: List[str]) -> List[Any]:
        """Encode text queries → list of embedding tensors."""

    @abc.abstractmethod
    def encode_pages(self, images: List[Image.Image]) -> List[Any]:
        """Encode page images → list of embedding tensors."""

    def score(
        self,
        query_emb: Any,
        page_embs: List[Any],
    ) -> List[float]:
        """
        MaxSim late-interaction score between one query embedding and
        each page embedding.  Returns one float per page.

        Default implementation works for any tensor pair [L, d] / [N, d].
        """
        scores: List[float] = []
        q = torch.tensor(query_emb) if not isinstance(query_emb, torch.Tensor) else query_emb
        q = q.float()
        for p in page_embs:
            p_t = torch.tensor(p) if not isinstance(p, torch.Tensor) else p
            p_t = p_t.float()
            # [L, d] x [N, d]^T → [L, N] → max over N → sum over L
            sim = torch.matmul(q, p_t.T)          # [L, N]
            scores.append(sim.max(dim=1).values.sum().item())
        return scores

    def retrieve(
        self,
        query: str,
        images: List[Image.Image],
        top_k: int = 5,
    ) -> List[int]:
        """
        Retrieve top-k page indices for *query* from *images*.

        Returns sorted list of page indices (most relevant first).
        """
        q_embs = self.encode_queries([query])
        p_embs = self.encode_pages(images)
        page_scores = self.score(q_embs[0], p_embs)
        ranked = sorted(range(len(page_scores)), key=lambda i: page_scores[i], reverse=True)
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Shared loader helpers
# ---------------------------------------------------------------------------

def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_colpali_model(model_name: str, device: str):
    """
    Load a ColPali-style model via colpali-engine if available,
    with a transformers fallback.
    """
    try:
        from colpali_engine.models import ColPali, ColPaliProcessor  # type: ignore

        logger.info(f"Loading {model_name} via colpali-engine ...")
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
        ).eval()
        processor = ColPaliProcessor.from_pretrained(model_name)
        return model, processor, "colpali"
    except ImportError:
        logger.warning("colpali-engine not installed; falling back to transformers.")
    except Exception as exc:
        logger.warning(f"colpali-engine load failed ({exc}); falling back to transformers.")

    try:
        from transformers import AutoModel, AutoProcessor  # type: ignore

        logger.info(f"Loading {model_name} via transformers ...")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        return model, processor, "transformers"
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {model_name}: {exc}. "
            "Install colpali-engine: pip install colpali-engine"
        ) from exc


def _load_colqwen2_model(model_name: str, device: str):
    """
    Load a ColQwen2-style model via colpali-engine if available,
    with a transformers fallback.
    """
    try:
        from colpali_engine.models import ColQwen2, ColQwen2Processor  # type: ignore

        logger.info(f"Loading {model_name} via colpali-engine ...")
        model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map=device,
        ).eval()
        processor = ColQwen2Processor.from_pretrained(model_name)
        return model, processor, "colpali"
    except ImportError:
        logger.warning("colpali-engine not installed; falling back to transformers.")
    except Exception as exc:
        logger.warning(f"colpali-engine load failed ({exc}); falling back to transformers.")

    try:
        from transformers import AutoModel, AutoProcessor  # type: ignore

        logger.info(f"Loading {model_name} via transformers ...")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        return model, processor, "transformers"
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {model_name}: {exc}. "
            "Install colpali-engine: pip install colpali-engine"
        ) from exc


# ---------------------------------------------------------------------------
# ColPali v1.3
# ---------------------------------------------------------------------------

class ColPaliBaseline(BaseVisionRetriever):
    """
    ColPali v1.3 visual retrieval baseline.

    Args:
        model_name:  HuggingFace model ID (default: vidore/colpali-v1.3).
        device:      "cuda" or "cpu" (auto-detected by default).
        batch_size:  Page-encoding batch size.
        answerer:    Optional LLMAnswerer for downstream answer generation.
    """

    def __init__(
        self,
        model_name: str = "vidore/colpali-v1.3",
        device: Optional[str] = None,
        batch_size: int = 4,
        answerer=None,
    ):
        self.model_name = model_name
        self.device = device or _get_device()
        self.batch_size = batch_size
        self.answerer = answerer
        self.model, self.processor, self._backend = _load_colpali_model(
            model_name, self.device
        )

    @torch.no_grad()
    def encode_queries(self, queries: List[str]) -> List[torch.Tensor]:
        if self._backend == "colpali":
            inputs = self.processor.process_queries(queries).to(self.device)
            embs = self.model(**inputs)  # [B, L, d]
            return [embs[i].cpu() for i in range(len(queries))]
        else:
            # Generic transformers path: tokenise + forward
            inputs = self.processor(
                text=queries, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            out = self.model(**inputs).last_hidden_state  # [B, L, d]
            return [out[i].cpu() for i in range(len(queries))]

    @torch.no_grad()
    def encode_pages(self, images: List[Image.Image]) -> List[torch.Tensor]:
        embs: List[torch.Tensor] = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            if self._backend == "colpali":
                inputs = self.processor.process_images(batch).to(self.device)
                out = self.model(**inputs)  # [B, N, d]
            else:
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                out = self.model(**inputs).last_hidden_state  # [B, N, d]
            for j in range(len(batch)):
                embs.append(out[j].cpu())
        return embs


# ---------------------------------------------------------------------------
# ColQwen2 v1.0
# ---------------------------------------------------------------------------

class ColQwen2Baseline(BaseVisionRetriever):
    """
    ColQwen2 v1.0 visual retrieval baseline.

    Args:
        model_name:  HuggingFace model ID (default: vidore/colqwen2-v1.0).
        device:      "cuda" or "cpu" (auto-detected by default).
        batch_size:  Page-encoding batch size.
        answerer:    Optional LLMAnswerer for downstream answer generation.
    """

    def __init__(
        self,
        model_name: str = "vidore/colqwen2-v1.0",
        device: Optional[str] = None,
        batch_size: int = 4,
        answerer=None,
    ):
        self.model_name = model_name
        self.device = device or _get_device()
        self.batch_size = batch_size
        self.answerer = answerer
        self.model, self.processor, self._backend = _load_colqwen2_model(
            model_name, self.device
        )

    @torch.no_grad()
    def encode_queries(self, queries: List[str]) -> List[torch.Tensor]:
        if self._backend == "colpali":
            inputs = self.processor.process_queries(queries).to(self.device)
            embs = self.model(**inputs)
            return [embs[i].cpu() for i in range(len(queries))]
        else:
            inputs = self.processor(
                text=queries, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            out = self.model(**inputs).last_hidden_state
            return [out[i].cpu() for i in range(len(queries))]

    @torch.no_grad()
    def encode_pages(self, images: List[Image.Image]) -> List[torch.Tensor]:
        embs: List[torch.Tensor] = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            if self._backend == "colpali":
                inputs = self.processor.process_images(batch).to(self.device)
                out = self.model(**inputs)
            else:
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                out = self.model(**inputs).last_hidden_state
            for j in range(len(batch)):
                embs.append(out[j].cpu())
        return embs
