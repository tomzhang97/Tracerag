"""
Unified Baseline Runner.

Runs all configured baselines against an EngBench-style benchmark dataset
and outputs a consolidated results JSON with per-baseline and per-query metrics.

Usage (CLI):
    python -m tracerag.baselines.runner \\
        --dataset Eng_Bench/microtext_questions.jsonl \\
        --image_root Eng_Bench/images \\
        --baselines tesseract paddleocr ocr_text_rag colpali colqwen2 qwen25_vl \\
        --output results/baseline_run.json

Usage (Python):
    from tracerag.baselines.runner import BaselineRunner
    runner = BaselineRunner(image_root="Eng_Bench/images")
    runner.load_dataset("Eng_Bench/microtext_questions.jsonl")
    runner.run(baselines=["tesseract", "colpali"])
    runner.save("results/run.json")
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------

def _build_tesseract(cfg: Dict[str, Any]):
    from tracerag.baselines.ocr import TesseractBaseline

    ocr_cfg = cfg.get("ocr", {})
    return TesseractBaseline(
        lang=ocr_cfg.get("lang", "eng"),
        config=ocr_cfg.get("config", "--psm 6"),
        dpi=ocr_cfg.get("dpi", 300),
        try_rotations=ocr_cfg.get("try_rotations", True),
        conf_threshold=ocr_cfg.get("conf_threshold", 30.0),
    )


def _build_paddleocr(cfg: Dict[str, Any]):
    from tracerag.baselines.ocr import PaddleOCRBaseline

    ocr_cfg = cfg.get("ocr", {})
    return PaddleOCRBaseline(
        lang=ocr_cfg.get("lang", "en"),
        use_angle_cls=ocr_cfg.get("use_angle_cls", True),
        use_gpu=ocr_cfg.get("use_gpu", False),
        version=ocr_cfg.get("version", "PP-OCRv5"),
        conf_threshold=ocr_cfg.get("conf_threshold", 0.3),
    )


def _build_ocr_text_rag(cfg: Dict[str, Any]):
    from tracerag.baselines.text_rag import OCRTextRAGBaseline

    ocr_cfg = cfg.get("ocr", {})
    engine = ocr_cfg.get("engine", "tesseract")
    if engine == "paddleocr":
        from tracerag.baselines.ocr import PaddleOCRBaseline
        ocr = PaddleOCRBaseline(
            lang=ocr_cfg.get("lang", "en"),
            use_angle_cls=ocr_cfg.get("use_angle_cls", True),
            use_gpu=ocr_cfg.get("use_gpu", False),
            version=ocr_cfg.get("version", "PP-OCRv5"),
        )
    else:
        from tracerag.baselines.ocr import TesseractBaseline
        ocr = TesseractBaseline(
            lang=ocr_cfg.get("lang", "eng"),
            config=ocr_cfg.get("config", "--psm 6"),
            dpi=ocr_cfg.get("dpi", 300),
            try_rotations=ocr_cfg.get("try_rotations", True),
        )

    ret_cfg = cfg.get("retrieval", {})
    llm_cfg = cfg.get("answerer", None)
    return OCRTextRAGBaseline(
        ocr=ocr,
        index_type=ret_cfg.get("index_type", "bm25"),
        top_k=ret_cfg.get("top_k", 5),
        llm_config=llm_cfg,
    )


def _build_colpali(cfg: Dict[str, Any]):
    from tracerag.baselines.vision import ColPaliBaseline

    ret_cfg = cfg.get("retriever", {})
    return ColPaliBaseline(
        model_name=ret_cfg.get("model_name", "vidore/colpali-v1.3"),
        batch_size=ret_cfg.get("batch_size", 4),
    )


def _build_colqwen2(cfg: Dict[str, Any]):
    from tracerag.baselines.vision import ColQwen2Baseline

    ret_cfg = cfg.get("retriever", {})
    return ColQwen2Baseline(
        model_name=ret_cfg.get("model_name", "vidore/colqwen2-v1.0"),
        batch_size=ret_cfg.get("batch_size", 4),
    )


def _build_qwen25_vl(cfg: Dict[str, Any]):
    from tracerag.baselines.vlm import Qwen2VLBaseline

    vlm_cfg = cfg.get("vlm", {})
    return Qwen2VLBaseline(
        model_name=vlm_cfg.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct"),
        max_new_tokens=vlm_cfg.get("max_new_tokens", 512),
        max_pages=vlm_cfg.get("max_pages", 8),
        min_pixels=vlm_cfg.get("min_pixels", 256 * 28 * 28),
        max_pixels=vlm_cfg.get("max_pixels", 1280 * 28 * 28),
    )


def _build_absdiff(cfg: Dict[str, Any]):
    from tracerag.baselines.diff import AbsDiffBaseline

    diff_cfg = cfg.get("diff", {})
    return AbsDiffBaseline(
        threshold=diff_cfg.get("threshold", 30),
        min_area=diff_cfg.get("min_area", 50),
    )


def _build_ocr_text_diff(cfg: Dict[str, Any]):
    from tracerag.baselines.diff import OCRTextDiffBaseline
    from tracerag.baselines.ocr import TesseractBaseline

    ocr_cfg = cfg.get("ocr", {})
    engine = ocr_cfg.get("engine", "tesseract")
    if engine == "paddleocr":
        from tracerag.baselines.ocr import PaddleOCRBaseline
        ocr = PaddleOCRBaseline()
    else:
        ocr = TesseractBaseline(
            lang=ocr_cfg.get("lang", "eng"),
            try_rotations=ocr_cfg.get("try_rotations", False),
        )
    diff_cfg = cfg.get("diff", {})
    return OCRTextDiffBaseline(ocr=ocr, context_lines=diff_cfg.get("context_lines", 2))


_REGISTRY = {
    "tesseract": _build_tesseract,
    "paddleocr": _build_paddleocr,
    "ocr_text_rag": _build_ocr_text_rag,
    "colpali": _build_colpali,
    "colqwen2": _build_colqwen2,
    "qwen25_vl": _build_qwen25_vl,
    "absdiff": _build_absdiff,
    "ocr_text_diff": _build_ocr_text_diff,
}

_CONFIG_DIR = Path(__file__).parent.parent.parent / "configs" / "baselines"


def _load_baseline_config(name: str) -> Dict[str, Any]:
    """Load YAML config for baseline *name* from configs/baselines/."""
    import yaml

    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        logger.warning(f"No config found for baseline '{name}' at {path}; using defaults.")
        return {"name": name}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_page_images(image_root: str, doc_id: str, page_indices=None) -> List[Image.Image]:
    """Load page PNG images for a document from image_root/{doc_id}/."""
    img_dir = Path(image_root) / doc_id
    if not img_dir.exists():
        logger.warning(f"Image directory not found: {img_dir}")
        return []
    paths = sorted(img_dir.glob("page_*.png"))
    if page_indices is not None:
        paths = [paths[i] for i in page_indices if i < len(paths)]
    return [Image.open(p).convert("RGB") for p in paths]


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _em(pred: str, gold: str) -> float:
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


def _f1_token(pred: str, gold: str) -> float:
    pred_toks = pred.lower().split()
    gold_toks = gold.lower().split()
    if not pred_toks or not gold_toks:
        return 0.0
    common = set(pred_toks) & set(gold_toks)
    if not common:
        return 0.0
    p = len(common) / len(pred_toks)
    r = len(common) / len(gold_toks)
    return 2 * p * r / (p + r)


def _cer(pred: str, gold: str) -> float:
    """Character error rate via edit distance."""
    import editdistance  # type: ignore
    if not gold:
        return 0.0 if not pred else 1.0
    return min(editdistance.eval(pred, gold) / len(gold), 1.0)


# ---------------------------------------------------------------------------
# BaselineRunner
# ---------------------------------------------------------------------------

class BaselineRunner:
    """
    Orchestrates multi-baseline evaluation against an EngBench dataset.

    Args:
        image_root: Root directory containing per-doc image folders.
        output_dir: Directory to save result files (created if needed).
    """

    def __init__(self, image_root: str = "Eng_Bench/images", output_dir: str = "results"):
        self.image_root = image_root
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._dataset: List[Dict[str, Any]] = []
        self._results: Dict[str, Any] = {}

    def load_dataset(self, dataset_path: str) -> None:
        """Load a JSONL benchmark file."""
        self._dataset = _load_jsonl(dataset_path)
        logger.info(f"Loaded {len(self._dataset)} queries from {dataset_path}")

    def run(
        self,
        baselines: List[str],
        config_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run all requested baselines and collect metrics.

        Args:
            baselines:        List of baseline names (keys in _REGISTRY).
            config_overrides: Optional per-baseline config override dicts.

        Returns:
            Nested results dict: {baseline_name: {overall: {...}, per_query: [...]}}
        """
        if not self._dataset:
            raise RuntimeError("No dataset loaded. Call load_dataset() first.")

        config_overrides = config_overrides or {}
        all_results: Dict[str, Any] = {}

        for name in baselines:
            if name not in _REGISTRY:
                logger.warning(f"Unknown baseline '{name}'; skipping.")
                continue
            logger.info(f"=== Running baseline: {name} ===")
            cfg = _load_baseline_config(name)
            cfg.update(config_overrides.get(name, {}))

            try:
                baseline = _REGISTRY[name](cfg)
            except Exception as exc:
                logger.error(f"Failed to build baseline '{name}': {exc}")
                all_results[name] = {"error": str(exc)}
                continue

            per_query, overall = self._eval_baseline(name, baseline, cfg)
            all_results[name] = {"overall": overall, "per_query": per_query}
            logger.info(f"  {name}: EM={overall.get('answer_em', 0):.3f}  F1={overall.get('answer_f1', 0):.3f}")

        self._results = all_results
        return all_results

    def _eval_baseline(self, name: str, baseline, cfg: Dict[str, Any]):
        """Evaluate one baseline against all queries in self._dataset."""
        from tracerag.baselines.ocr import BaseOCR
        from tracerag.baselines.text_rag import OCRTextRAGBaseline
        from tracerag.baselines.vision import BaseVisionRetriever
        from tracerag.baselines.vlm import Qwen2VLBaseline
        from tracerag.baselines.diff import AbsDiffBaseline, OCRTextDiffBaseline

        per_query_metrics: List[Dict[str, Any]] = []
        start = time.time()

        # Pre-ingest for OCRTextRAGBaseline (build index once per doc)
        _ingested_docs: Dict[str, Any] = {}

        for rec in self._dataset:
            qid = rec.get("question_id") or rec.get("id") or "?"
            query = rec.get("question") or rec.get("query_text") or ""
            gold = rec.get("answer") or rec.get("answer_text") or ""
            doc_id = rec.get("doc_id") or ""

            images = _load_page_images(self.image_root, doc_id) if doc_id else []

            metrics: Dict[str, Any] = {"question_id": qid}
            pred = ""

            try:
                # OCR-only baselines: extract full-page text for first page
                if isinstance(baseline, BaseOCR):
                    if images:
                        pred = baseline.page_text(images[0])
                    metrics["cer"] = _cer(pred, gold)

                # OCR + text RAG
                elif isinstance(baseline, OCRTextRAGBaseline):
                    if doc_id not in _ingested_docs and images:
                        page_pairs = [(f"{doc_id}_p{i}", img) for i, img in enumerate(images)]
                        baseline.ingest_pages(page_pairs)
                        _ingested_docs[doc_id] = True
                    if doc_id in _ingested_docs:
                        result = baseline.answer(query)
                        pred = result.get("answer", "")

                # Visual retrieval baselines (ColPali / ColQwen2)
                elif isinstance(baseline, BaseVisionRetriever):
                    if images:
                        top_idxs = baseline.retrieve(query, images, top_k=cfg.get("retriever", {}).get("top_k", 5))
                        # Use answerer if attached
                        if baseline.answerer:
                            top_imgs = [images[i] for i in top_idxs]
                            # Build minimal text context from image indices
                            pred = f"Retrieved pages: {top_idxs}"  # placeholder; real eval uses answerer
                        else:
                            pred = f"Retrieved pages: {top_idxs}"

                # End-to-end VLM
                elif isinstance(baseline, Qwen2VLBaseline):
                    if images:
                        result = baseline.answer(query, images)
                        pred = result.get("answer", "")

                # AbsDiff
                elif isinstance(baseline, AbsDiffBaseline):
                    if len(images) >= 2:
                        boxes = baseline.changed_bboxes(images[0], images[1])
                        pred = f"{len(boxes)} changed regions"

                # OCR text diff
                elif isinstance(baseline, OCRTextDiffBaseline):
                    if len(images) >= 2:
                        result = baseline.diff_pages(images[0], images[1])
                        pred = "changed" if result["has_changes"] else "unchanged"

            except Exception as exc:
                logger.error(f"  [{qid}] {name} failed: {exc}")
                metrics["error"] = str(exc)

            metrics["answer_em"] = _em(pred, gold)
            metrics["answer_f1"] = _f1_token(pred, gold)
            metrics["pred"] = pred
            metrics["gold"] = gold
            per_query_metrics.append(metrics)

        elapsed = time.time() - start
        n = len(per_query_metrics) or 1
        overall = {
            "baseline": name,
            "n_queries": len(per_query_metrics),
            "answer_em": sum(m.get("answer_em", 0) for m in per_query_metrics) / n,
            "answer_f1": sum(m.get("answer_f1", 0) for m in per_query_metrics) / n,
            "cer": sum(m.get("cer", 0) for m in per_query_metrics) / n,
            "latency_ms_per_query": (elapsed / n) * 1000,
        }
        return per_query_metrics, overall

    def save(self, output_path: Optional[str] = None) -> str:
        """Save results to JSON."""
        if output_path is None:
            ts = int(time.time())
            output_path = str(self.output_dir / f"baseline_results_{ts}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self._results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")
        return output_path

    def print_summary(self) -> None:
        """Print a condensed leaderboard to stdout."""
        if not self._results:
            print("No results yet.")
            return
        header = f"{'Baseline':<20} {'EM':>6} {'F1':>6} {'CER':>6} {'ms/q':>8}"
        print(header)
        print("-" * len(header))
        for name, res in self._results.items():
            if "error" in res:
                print(f"{name:<20}  ERROR: {res['error']}")
                continue
            ov = res.get("overall", {})
            print(
                f"{name:<20} "
                f"{ov.get('answer_em', 0):>6.3f} "
                f"{ov.get('answer_f1', 0):>6.3f} "
                f"{ov.get('cer', 0):>6.3f} "
                f"{ov.get('latency_ms_per_query', 0):>8.1f}"
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run OCR/VLM baselines against EngBench.")
    parser.add_argument("--dataset", required=True, help="Path to JSONL benchmark file.")
    parser.add_argument("--image_root", default="Eng_Bench/images", help="Root image directory.")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["tesseract", "paddleocr", "ocr_text_rag", "colpali", "colqwen2"],
        help=f"Baselines to run. Available: {sorted(_REGISTRY.keys())}",
    )
    parser.add_argument("--output", default=None, help="Output JSON path.")
    parser.add_argument("--output_dir", default="results", help="Output directory.")
    args = parser.parse_args()

    runner = BaselineRunner(image_root=args.image_root, output_dir=args.output_dir)
    runner.load_dataset(args.dataset)
    runner.run(baselines=args.baselines)
    runner.save(args.output)
    runner.print_summary()


if __name__ == "__main__":
    main()
