"""
Diff Baselines: AbsDiff, Siamese Diff, and OCR-Text Diff.

These are used for the visual-diff / revision task.

AbsDiff:         Pixel-level absolute difference (cheapest baseline).
SiameseDiffBaseline: Feature-level diff via a Siamese CNN (currently a stub
                 using raw pixel diff as fallback).
OCRTextDiffBaseline: Diff OCR text between two page versions.  Returns changed
                 line regions identified by difflib.
"""
from __future__ import annotations

import difflib
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AbsDiff
# ---------------------------------------------------------------------------

class AbsDiffBaseline:
    """
    Absolute pixel-difference baseline.

    Resizes both images to the same size before diffing so page-size
    mismatches (e.g. different scan resolutions) do not cause false positives.

    Args:
        threshold:  Pixel-difference threshold (0-255) to flag a change.
        min_area:   Minimum connected-region area (pixels²) to report.
    """

    def __init__(self, threshold: int = 30, min_area: int = 50):
        self.threshold = threshold
        self.min_area = min_area

    def compute_diff(
        self, img1: Image.Image, img2: Image.Image
    ) -> np.ndarray:
        """
        Returns a boolean mask (H, W) where True = changed pixel.
        Both images are converted to grayscale and resized to img1's size.
        """
        w, h = img1.size
        a1 = np.array(img1.convert("L")).astype(np.float32)
        a2 = np.array(img2.convert("L").resize((w, h), Image.LANCZOS)).astype(np.float32)
        return np.abs(a1 - a2) > self.threshold

    def changed_bboxes(
        self, img1: Image.Image, img2: Image.Image
    ) -> List[Tuple[int, int, int, int]]:
        """
        Return list of (x_min, y_min, x_max, y_max) bounding boxes for
        connected changed regions above min_area.
        """
        try:
            from scipy import ndimage  # type: ignore

            mask = self.compute_diff(img1, img2).astype(np.uint8)
            labeled, n = ndimage.label(mask)
            boxes: List[Tuple[int, int, int, int]] = []
            for region_id in range(1, n + 1):
                ys, xs = np.where(labeled == region_id)
                if ys.size < self.min_area:
                    continue
                boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
            return boxes
        except ImportError:
            # Fallback: single global bbox if any change exists
            mask = self.compute_diff(img1, img2)
            if mask.any():
                ys, xs = np.where(mask)
                return [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))]
            return []


# ---------------------------------------------------------------------------
# Siamese Diff (stub – pixel diff fallback)
# ---------------------------------------------------------------------------

class SiameseDiffBaseline:
    """
    Feature-level diff via a Siamese CNN.

    Currently uses a raw pixel-diff fallback.  To use a real Siamese network,
    provide a model_path pointing to a torchscript or HuggingFace checkpoint.
    """

    def __init__(self, model_path: Optional[str] = None, threshold: int = 50):
        self.model_path = model_path
        self.threshold = threshold
        if model_path:
            logger.warning(
                "SiameseDiffBaseline: model_path provided but custom model loading "
                "is not yet implemented; falling back to pixel diff."
            )

    def compute_diff(self, img1: Image.Image, img2: Image.Image) -> np.ndarray:
        """Returns a boolean change mask (H, W)."""
        w, h = img1.size
        a1 = np.array(img1.convert("L")).astype(np.float32)
        a2 = np.array(img2.convert("L").resize((w, h), Image.LANCZOS)).astype(np.float32)
        return np.abs(a1 - a2) > self.threshold


# ---------------------------------------------------------------------------
# OCR-Text Diff
# ---------------------------------------------------------------------------

class OCRTextDiffBaseline:
    """
    OCR-text diff baseline for the revision / visual-diff task.

    Runs OCR on both page versions and uses difflib to identify changed lines.
    The output includes:
      - changed_lines: list of dicts per changed block
      - unified_diff: unified-diff string
      - has_changes: True if any textual changes were found

    Args:
        ocr:           An instance of TesseractBaseline or PaddleOCRBaseline.
                       Defaults to TesseractBaseline (no rotations).
        context_lines: Lines of context to include around each change.
    """

    def __init__(self, ocr=None, context_lines: int = 2):
        if ocr is None:
            from tracerag.baselines.ocr import TesseractBaseline
            ocr = TesseractBaseline(try_rotations=False)
        self.ocr = ocr
        self.context_lines = context_lines

    def diff_pages(
        self,
        img_v1: Image.Image,
        img_v2: Image.Image,
    ) -> Dict[str, Any]:
        """
        Diff two page images and return a structured diff report.

        Returns:
            {
                "has_changes": bool,
                "text_v1": str,
                "text_v2": str,
                "changed_lines": [{"tag": str, "a_text": str, "b_text": str, ...}],
                "unified_diff": str,
                "added_count": int,
                "removed_count": int,
            }
        """
        text_v1 = self.ocr.page_text(img_v1)
        text_v2 = self.ocr.page_text(img_v2)
        return self._compute_text_diff(text_v1, text_v2)

    def diff_texts(self, text_v1: str, text_v2: str) -> Dict[str, Any]:
        """Diff two already-extracted text strings."""
        return self._compute_text_diff(text_v1, text_v2)

    def _compute_text_diff(self, text_v1: str, text_v2: str) -> Dict[str, Any]:
        lines_v1 = text_v1.splitlines()
        lines_v2 = text_v2.splitlines()

        matcher = difflib.SequenceMatcher(None, lines_v1, lines_v2, autojunk=False)
        opcodes = matcher.get_opcodes()

        changed_lines: List[Dict[str, Any]] = []
        added_count = 0
        removed_count = 0

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == "equal":
                continue
            a_lines = lines_v1[i1:i2]
            b_lines = lines_v2[j1:j2]
            changed_lines.append({
                "tag": tag,           # "replace" | "insert" | "delete"
                "a_text": "\n".join(a_lines),
                "b_text": "\n".join(b_lines),
                "a_line_range": (i1, i2),
                "b_line_range": (j1, j2),
            })
            if tag in ("delete", "replace"):
                removed_count += len(a_lines)
            if tag in ("insert", "replace"):
                added_count += len(b_lines)

        unified = "\n".join(
            difflib.unified_diff(
                lines_v1, lines_v2,
                fromfile="version_1", tofile="version_2",
                n=self.context_lines,
            )
        )

        return {
            "has_changes": len(changed_lines) > 0,
            "text_v1": text_v1,
            "text_v2": text_v2,
            "changed_lines": changed_lines,
            "unified_diff": unified,
            "added_count": added_count,
            "removed_count": removed_count,
        }

    def batch_diff(
        self,
        page_pairs: List[Tuple[Image.Image, Image.Image]],
    ) -> List[Dict[str, Any]]:
        """Diff a list of (img_v1, img_v2) pairs."""
        return [self.diff_pages(a, b) for a, b in page_pairs]
