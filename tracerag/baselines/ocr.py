"""
OCR Baselines: Tesseract and PaddleOCR (PP-OCRv5).

Both return a uniform list of dicts:
    {"text": str, "bbox": [x_min, y_min, x_max, y_max], "conf": float}

Bboxes are in pixel coordinates of the input image.
"""
from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseOCR(abc.ABC):
    @abc.abstractmethod
    def extract_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Return list of {text, bbox, conf} dicts for *image*."""

    def extract_text_batch(self, images: List[Image.Image]) -> List[List[Dict[str, Any]]]:
        """Default: iterate sequentially. Subclasses may override for batching."""
        return [self.extract_text(img) for img in images]

    def page_text(self, image: Image.Image) -> str:
        """Convenience: full OCR text for one page (top-to-bottom reading order)."""
        items = self.extract_text(image)
        # Sort by vertical centre then horizontal centre (simple reading-order)
        items = sorted(items, key=lambda d: (
            (d["bbox"][1] + d["bbox"][3]) / 2,
            (d["bbox"][0] + d["bbox"][2]) / 2,
        ))
        return "\n".join(d["text"] for d in items if d["text"].strip())


# ---------------------------------------------------------------------------
# Tesseract
# ---------------------------------------------------------------------------

class TesseractBaseline(BaseOCR):
    """
    Tesseract OCR baseline.

    Runs tesseract with HOCR / TSV output via pytesseract.
    Optionally retries with multiple orientations (0 / 90 / 180 / 270 degrees)
    and keeps the rotation that produces the best mean confidence.

    Args:
        lang:           Tesseract language string, e.g. "eng" or "eng+deu".
        config:         Extra tesseract config flags, e.g. "--psm 6".
        dpi:            DPI hint fed to tesseract (default 300).
        try_rotations:  If True, try all four 90° rotations and pick the best.
        conf_threshold: Minimum confidence (0–100) to keep a word.
    """

    def __init__(
        self,
        lang: str = "eng",
        config: str = "--psm 6",
        dpi: int = 300,
        try_rotations: bool = True,
        conf_threshold: float = 30.0,
    ):
        self.lang = lang
        self.config = config
        self.dpi = dpi
        self.try_rotations = try_rotations
        self.conf_threshold = conf_threshold
        self._pytesseract = self._import_pytesseract()

    @staticmethod
    def _import_pytesseract():
        try:
            import pytesseract
            return pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract is required for TesseractBaseline. "
                "Install it with: pip install pytesseract\n"
                "Also ensure the Tesseract binary is on your PATH: "
                "https://github.com/UB-Mannheim/tesseract/wiki"
            )

    def _run_one(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Run tesseract on a single PIL image, return raw word records."""
        pt = self._pytesseract
        data = pt.image_to_data(
            image,
            lang=self.lang,
            config=self.config,
            output_type=pt.Output.DICT,
        )
        results: List[Dict[str, Any]] = []
        n = len(data["text"])
        for i in range(n):
            text = (data["text"][i] or "").strip()
            conf = float(data["conf"][i] if data["conf"][i] != -1 else 0)
            if not text or conf < self.conf_threshold:
                continue
            x, y, w, h = (
                data["left"][i], data["top"][i],
                data["width"][i], data["height"][i],
            )
            results.append({
                "text": text,
                "bbox": [x, y, x + w, y + h],
                "conf": conf / 100.0,  # normalise to [0, 1]
            })
        return results

    def _mean_conf(self, items: List[Dict[str, Any]]) -> float:
        if not items:
            return 0.0
        return sum(d["conf"] for d in items) / len(items)

    def extract_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        if not self.try_rotations:
            return self._run_one(image)

        best_items: List[Dict[str, Any]] = []
        best_conf = -1.0
        for angle in (0, 90, 180, 270):
            rotated = image.rotate(angle, expand=True) if angle != 0 else image
            items = self._run_one(rotated)
            conf = self._mean_conf(items)
            if conf > best_conf:
                best_conf = conf
                # Un-rotate bboxes back to original image coordinates
                if angle != 0:
                    w, h = image.size
                    rw, rh = rotated.size
                    items = _rotate_bboxes_back(items, angle, w, h, rw, rh)
                best_items = items
        return best_items


def _rotate_bboxes_back(
    items: List[Dict[str, Any]],
    angle: int,
    orig_w: int,
    orig_h: int,
    rot_w: int,
    rot_h: int,
) -> List[Dict[str, Any]]:
    """Transform bboxes from rotated-image space back to original-image space."""
    out = []
    for d in items:
        x0, y0, x1, y1 = d["bbox"]
        if angle == 90:
            # rotated(x,y) → orig: (y, rot_w - x)
            nx0 = y0
            ny0 = rot_w - x1
            nx1 = y1
            ny1 = rot_w - x0
        elif angle == 180:
            nx0 = rot_w - x1
            ny0 = rot_h - y1
            nx1 = rot_w - x0
            ny1 = rot_h - y0
        elif angle == 270:
            # rotated(x,y) → orig: (rot_h - y, x)
            nx0 = rot_h - y1
            ny0 = x0
            nx1 = rot_h - y0
            ny1 = x1
        else:
            nx0, ny0, nx1, ny1 = x0, y0, x1, y1
        out.append({"text": d["text"], "bbox": [nx0, ny0, nx1, ny1], "conf": d["conf"]})
    return out


# ---------------------------------------------------------------------------
# PaddleOCR (PP-OCRv5)
# ---------------------------------------------------------------------------

class PaddleOCRBaseline(BaseOCR):
    """
    PaddleOCR / PP-OCRv5 baseline.

    Uses the paddleocr Python package.  The model is loaded once in __init__
    and reused across all calls.

    Args:
        lang:           Language code, e.g. "en", "ch", "fr".
        use_angle_cls:  Enable orientation classifier (recommended for scans).
        use_gpu:        Use GPU if available.
        version:        PP-OCR version string passed to PaddleOCR constructor.
        conf_threshold: Minimum confidence to keep a word box.
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        version: str = "PP-OCRv5",
        conf_threshold: float = 0.3,
    ):
        self.conf_threshold = conf_threshold
        self._ocr = self._load_model(lang, use_angle_cls, use_gpu, version)

    @staticmethod
    def _load_model(lang, use_angle_cls, use_gpu, version):
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError:
            raise ImportError(
                "paddleocr is required for PaddleOCRBaseline. "
                "Install it with: pip install paddleocr paddlepaddle"
            )
        logger.info(f"Loading PaddleOCR ({version}, lang={lang}) ...")
        return PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            use_gpu=use_gpu,
            ocr_version=version,
            show_log=False,
        )

    def extract_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        img_array = np.array(image.convert("RGB"))
        result = self._ocr.ocr(img_array, cls=True)
        items: List[Dict[str, Any]] = []
        if result is None:
            return items
        # PaddleOCR returns a list-of-pages; we pass one image so result[0] is the page
        page_result = result[0] if isinstance(result[0], list) else result
        if page_result is None:
            return items
        for line in page_result:
            if line is None:
                continue
            # Each line: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]], (text, conf)
            quad, (text, conf) = line
            text = (text or "").strip()
            if not text or conf < self.conf_threshold:
                continue
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            items.append({
                "text": text,
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "conf": float(conf),
            })
        return items
