"""
End-to-end VLM Baseline: Qwen2.5-VL / Qwen2-VL.

This baseline answers "what if I just give the pages to a strong VLM?"
It does NOT use a separate retriever — it passes the top-k (or all) page
images directly to the VLM together with the query.

Supported models:
    vidore/Qwen2.5-VL-7B-Instruct   (recommended, strong on doc understanding)
    Qwen/Qwen2-VL-7B-Instruct       (older, still competitive on DocVQA)

Usage:
    baseline = Qwen2VLBaseline()
    result = baseline.answer(query="What is the torque spec?", images=[page1, page2])
    print(result["answer"])
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


class Qwen2VLBaseline:
    """
    End-to-end Qwen2.5-VL (or Qwen2-VL) multimodal baseline.

    Args:
        model_name:   HuggingFace model ID.
        device:       "cuda" or "cpu" (auto-detected).
        max_new_tokens: Maximum tokens to generate.
        max_pages:    Hard cap on pages sent per call (keeps VRAM finite).
        min_pixels:   Lower bound on image resolution fed to the VLM.
        max_pixels:   Upper bound (use 1280*28*28 for 7B models on 24 GB GPU).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        max_pages: int = 8,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.max_pages = max_pages
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.model, self.processor = self._load()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load(self):
        """Load Qwen2.5-VL via transformers (qwen-vl-utils optional but recommended)."""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor  # type: ignore

            logger.info(f"Loading {self.model_name} (Qwen2.5-VL) ...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
            ).eval()
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
            return model, processor
        except (ImportError, AttributeError):
            pass

        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor  # type: ignore

            logger.info(f"Loading {self.model_name} (Qwen2-VL) ...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
            ).eval()
            processor = AutoProcessor.from_pretrained(self.model_name)
            return model, processor
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {self.model_name}. "
                "Install: pip install transformers>=4.45.0 qwen-vl-utils"
            ) from exc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def answer(
        self,
        query: str,
        images: List[Image.Image],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Answer *query* given the provided page *images*.

        Args:
            query:         Natural-language question.
            images:        Page images (PIL), in reading order.
                           Truncated to max_pages if necessary.
            system_prompt: Optional system-level instruction.

        Returns:
            dict with keys: answer (str), pages_used (int).
        """
        images = images[: self.max_pages]

        # Build conversation in Qwen2-VL chat format
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": query})

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        # Apply chat template
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        except ImportError:
            # Fallback: use processor directly with PIL images
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=images if images else None,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        # Strip prompt tokens from output
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return {"answer": response.strip(), "pages_used": len(images)}

    # ------------------------------------------------------------------
    # Convenience: answer with page selection via text search
    # ------------------------------------------------------------------

    def answer_with_retrieval(
        self,
        query: str,
        pages: List[tuple],  # List of (page_id, PIL image)
        ocr_texts: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Optionally pre-select the most relevant pages via BM25 over OCR text,
        then call the VLM.

        Args:
            query:     Query string.
            pages:     List of (page_id, PIL image).
            ocr_texts: Pre-computed OCR text per page (same order as pages).
                       If None, all pages (up to max_pages) are used.
            top_k:     Pages to keep after BM25 pre-filtering.

        Returns:
            Same dict as answer(), plus "page_ids" key.
        """
        if ocr_texts and len(ocr_texts) == len(pages):
            try:
                from rank_bm25 import BM25Okapi
                import numpy as np

                tokenized = [t.lower().split() for t in ocr_texts]
                bm25 = BM25Okapi(tokenized)
                scores = bm25.get_scores(query.lower().split())
                top_idxs = sorted(
                    range(len(scores)), key=lambda i: scores[i], reverse=True
                )[:top_k]
                selected = [pages[i] for i in top_idxs]
            except ImportError:
                selected = pages[:top_k]
        else:
            selected = pages[:top_k]

        page_ids = [pid for pid, _ in selected]
        imgs = [img for _, img in selected]
        result = self.answer(query, imgs)
        result["page_ids"] = page_ids
        return result
