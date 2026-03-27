"""
Prepare a downloaded DocVQA dataset for TraceRAG.

This command:
1. Reads DocVQA questions
2. Resolves document images and OCR files
3. Builds TraceRAG structural + visual artifacts for each referenced document
4. Builds a TraceRAG text index
5. Writes a TraceRAG-ready benchmark JSONL for `tracerag-eval --benchmark docvqa`
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import typer
from PIL import Image
from loguru import logger

from tracerag.common.config import load_config, setup_logging
from tracerag.common.io import ensure_dir, get_page_id
from tracerag.common.types import PatchGrid, VectorObject
from tracerag.eval.docvqa_eval import load_docvqa_questions
from tracerag.retrieval.text_index import TextIndex
from tracerag.structural.index import PdfSpatialIndex
from tracerag.visual.encoder import VisualPageEncoder


app = typer.Typer()
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
OCR_EXTENSIONS = (".json", ".jsonl", ".txt")


def _sanitize_doc_id(value: str) -> str:
    stem = Path(str(value or "")).stem
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "document"


def _load_json_or_jsonl(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def _resolve_file(
    root_dir: Path,
    search_dir: Path | None,
    raw_value: str | None,
    allowed_suffixes: Sequence[str],
    cache: Dict[str, Path | None],
) -> Path | None:
    if not raw_value:
        return None
    if raw_value in cache:
        return cache[raw_value]

    raw_path = Path(raw_value)
    candidates: List[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    if search_dir is not None:
        candidates.append(search_dir / raw_path)
        candidates.append(search_dir / raw_path.name)
    candidates.append(root_dir / raw_path)
    candidates.append(root_dir / raw_path.name)

    for candidate in candidates:
        if candidate.exists():
            cache[raw_value] = candidate
            return candidate

    stem = raw_path.stem or raw_path.name
    for base_dir in (search_dir, root_dir):
        if base_dir is None or not base_dir.exists():
            continue
        for suffix in allowed_suffixes:
            matches = list(base_dir.rglob(f"{stem}{suffix}"))
            if matches:
                cache[raw_value] = matches[0]
                return matches[0]

    cache[raw_value] = None
    return None


def _polygon_to_bbox(points: Any) -> tuple[float, float, float, float] | None:
    if points is None:
        return None
    if isinstance(points, dict):
        if {"x0", "y0", "x1", "y1"} <= set(points):
            return (float(points["x0"]), float(points["y0"]), float(points["x1"]), float(points["y1"]))
        if {"left", "top", "width", "height"} <= set(points):
            left = float(points["left"])
            top = float(points["top"])
            width = float(points["width"])
            height = float(points["height"])
            return (left, top, left + width, top + height)
    if isinstance(points, (list, tuple)):
        if len(points) == 4 and all(isinstance(item, (int, float)) for item in points):
            return tuple(float(item) for item in points)  # type: ignore[return-value]
        if len(points) >= 8:
            xs = [float(points[idx]) for idx in range(0, len(points), 2)]
            ys = [float(points[idx]) for idx in range(1, len(points), 2)]
            return (min(xs), min(ys), max(xs), max(ys))
        if points and isinstance(points[0], dict):
            xs = [float(point.get("x", 0.0)) for point in points]
            ys = [float(point.get("y", 0.0)) for point in points]
            return (min(xs), min(ys), max(xs), max(ys))
        if points and isinstance(points[0], (list, tuple)) and len(points[0]) >= 2:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            return (min(xs), min(ys), max(xs), max(ys))
    return None


def _hash_object_id(doc_id: str, version_id: str, tag: str, index: int, text: str) -> str:
    payload = f"{doc_id}|{version_id}|{tag}|{index}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()[:16]


def _line_object(doc_id: str, version_id: str, page_id: str, index: int, text: str, bbox) -> VectorObject:
    return VectorObject(
        object_id=_hash_object_id(doc_id, version_id, "line", index, text),
        doc_id=doc_id,
        version_id=version_id,
        page_id=page_id,
        bbox=bbox,
        obj_type="text_block",
        text=text,
        coord_space="image",
        meta={"source": "docvqa_ocr", "level": "line"},
    )


def _word_object(doc_id: str, version_id: str, page_id: str, index: int, text: str, bbox, confidence: float) -> VectorObject:
    return VectorObject(
        object_id=_hash_object_id(doc_id, version_id, "word", index, text),
        doc_id=doc_id,
        version_id=version_id,
        page_id=page_id,
        bbox=bbox,
        obj_type="text_block",
        text=text,
        coord_space="image",
        meta={"source": "docvqa_ocr", "level": "word", "confidence": confidence},
    )


def _parse_ocr_records(ocr_path: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    data = _load_json_or_jsonl(ocr_path)
    words: List[Dict[str, Any]] = []
    lines: List[Dict[str, Any]] = []

    def append_word(text: str, bbox, confidence: float = 1.0):
        normalized_text = str(text or "").strip()
        if normalized_text and bbox is not None:
            words.append({"text": normalized_text, "bbox": bbox, "confidence": float(confidence)})

    def append_line(text: str, bbox):
        normalized_text = str(text or "").strip()
        if normalized_text and bbox is not None:
            lines.append({"text": normalized_text, "bbox": bbox})

    if isinstance(data, dict):
        if isinstance(data.get("recognitionResults"), list):
            for page in data["recognitionResults"]:
                for line in page.get("lines", []):
                    line_bbox = _polygon_to_bbox(line.get("boundingBox"))
                    append_line(line.get("text", ""), line_bbox)
                    for word in line.get("words", []):
                        append_word(word.get("text", ""), _polygon_to_bbox(word.get("boundingBox")), word.get("confidence", 1.0))
        elif isinstance(data.get("analyzeResult"), dict):
            analyze = data["analyzeResult"]
            pages = analyze.get("readResults") or analyze.get("pages") or []
            for page in pages:
                for line in page.get("lines", []):
                    line_bbox = _polygon_to_bbox(line.get("boundingBox") or line.get("polygon"))
                    append_line(line.get("text", "") or " ".join(word.get("content", "") for word in line.get("words", [])), line_bbox)
                    for word in line.get("words", []):
                        append_word(
                            word.get("text", "") or word.get("content", ""),
                            _polygon_to_bbox(word.get("boundingBox") or word.get("polygon")),
                            word.get("confidence", 1.0),
                        )
                for word in page.get("words", []):
                    append_word(
                        word.get("text", "") or word.get("content", ""),
                        _polygon_to_bbox(word.get("boundingBox") or word.get("polygon")),
                        word.get("confidence", 1.0),
                    )
        elif isinstance(data.get("words"), list):
            for word in data["words"]:
                append_word(
                    word.get("text", "") or word.get("word", ""),
                    _polygon_to_bbox(word.get("bbox") or word.get("box") or word.get("polygon")),
                    word.get("confidence", 1.0),
                )
        elif isinstance(data.get("lines"), list):
            for line in data["lines"]:
                append_line(line.get("text", ""), _polygon_to_bbox(line.get("bbox") or line.get("box") or line.get("polygon")))
        elif "text" in data and isinstance(data["text"], str):
            append_line(data["text"], (0.0, 0.0, 1.0, 1.0))
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("word") or item.get("content")
            bbox = _polygon_to_bbox(item.get("bbox") or item.get("box") or item.get("boundingBox") or item.get("polygon"))
            if text and bbox is not None:
                append_word(text, bbox, item.get("confidence", 1.0))
    elif isinstance(data, str):
        append_line(data, (0.0, 0.0, 1.0, 1.0))

    return lines, words


def _load_ocr_objects(doc_id: str, version_id: str, page_id: str, ocr_path: Path, image_size: tuple[int, int]) -> List[VectorObject]:
    if ocr_path.suffix.lower() == ".txt":
        text = ocr_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        width, height = image_size
        return [_line_object(doc_id, version_id, page_id, 0, text, (0.0, 0.0, float(width), float(height)))]

    lines, words = _parse_ocr_records(ocr_path)
    objects: List[VectorObject] = []
    for index, line in enumerate(lines):
        objects.append(_line_object(doc_id, version_id, page_id, index, line["text"], line["bbox"]))
    for index, word in enumerate(words):
        objects.append(_word_object(doc_id, version_id, page_id, index, word["text"], word["bbox"], word["confidence"]))
    return objects


def _save_patch_grid(patch_grid: PatchGrid, output_dir: Path):
    output_path = output_dir / f"{patch_grid.page_id}.npz"
    ensure_dir(str(output_dir))
    np.savez_compressed(
        output_path,
        doc_id=patch_grid.doc_id,
        version_id=patch_grid.version_id,
        page_id=patch_grid.page_id,
        H=patch_grid.H,
        W=patch_grid.W,
        embeddings=patch_grid.embeddings,
        patch_boxes=patch_grid.patch_boxes,
    )


@app.command()
def main(
    questions_path: str = typer.Option(..., help="Path to downloaded DocVQA questions JSON/JSONL"),
    images_dir: str = typer.Option(..., help="Directory containing document images"),
    output_index_root: str = typer.Option(..., help="Output TraceRAG index root"),
    output_benchmark_path: str = typer.Option(..., help="Output TraceRAG-ready DocVQA benchmark JSONL"),
    dataset_root: str = typer.Option(".", help="Dataset root for resolving relative paths"),
    ocr_dir: str = typer.Option(None, help="Optional OCR directory; if omitted, TraceRAG searches under dataset_root"),
    config_path: str = typer.Option(None, help="Optional TraceRAG config path"),
    split: str = typer.Option("val", help="Split label to record in benchmark metadata"),
    max_questions: int = typer.Option(None, help="Optional cap for a smoke-run subset"),
):
    config = load_config(config_path)
    setup_logging(config)

    dataset_root_path = Path(dataset_root).resolve()
    images_dir_path = Path(images_dir).resolve()
    ocr_dir_path = Path(ocr_dir).resolve() if ocr_dir else None
    output_index = Path(output_index_root).resolve()
    output_benchmark = Path(output_benchmark_path).resolve()

    raw_data = _load_json_or_jsonl(Path(questions_path).resolve())
    questions = load_docvqa_questions(raw_data)
    if max_questions:
        questions = questions[:max_questions]

    ensure_dir(str(output_index))
    ensure_dir(str(output_benchmark.parent))

    image_cache: Dict[str, Path | None] = {}
    ocr_cache: Dict[str, Path | None] = {}
    manifest: Dict[str, str] = {}
    all_objects: List[VectorObject] = []
    prepared_rows: List[Dict[str, Any]] = []
    processed_docs: set[str] = set()
    encoder = VisualPageEncoder(config.get("visual", {}))

    for question in questions:
        raw_item = question.metadata
        image_ref = (
            raw_item.get("image")
            or raw_item.get("image_path")
            or raw_item.get("file_name")
            or raw_item.get("filename")
            or raw_item.get("document")
            or question.doc_id
        )
        image_path = _resolve_file(dataset_root_path, images_dir_path, str(image_ref), IMAGE_EXTENSIONS, image_cache)
        if image_path is None:
            logger.warning(f"Skipping {question.question_id}: could not resolve image for '{image_ref}'")
            continue

        doc_id = question.doc_id or _sanitize_doc_id(image_path.name)
        version_id = question.version_id or "v1"
        page_id = get_page_id(doc_id, version_id, 0)
        ocr_ref = raw_item.get("ocr_path") or raw_item.get("ocr_file") or raw_item.get("ocr")
        ocr_path = _resolve_file(
            dataset_root_path,
            ocr_dir_path,
            str(ocr_ref or image_path.stem),
            OCR_EXTENSIONS,
            ocr_cache,
        )

        if doc_id not in processed_docs:
            with Image.open(image_path) as image:
                width, height = image.size
                if ocr_path is not None:
                    objects = _load_ocr_objects(doc_id, version_id, page_id, ocr_path, (width, height))
                else:
                    logger.warning(f"No OCR found for {image_path.name}; creating a single image object fallback")
                    objects = [
                        VectorObject(
                            object_id=_hash_object_id(doc_id, version_id, "image", 0, image_path.name),
                            doc_id=doc_id,
                            version_id=version_id,
                            page_id=page_id,
                            bbox=(0.0, 0.0, float(width), float(height)),
                            obj_type="image",
                            text=image_path.stem,
                            coord_space="image",
                            meta={"source": "docvqa_image_fallback"},
                        )
                    ]

                spatial_index = PdfSpatialIndex()
                for obj in objects:
                    spatial_index.add_object(obj)
                all_objects.extend(objects)

                struct_dir = output_index / "structural" / doc_id / version_id
                visual_dir = output_index / "visual" / doc_id / version_id
                ensure_dir(str(struct_dir))
                ensure_dir(str(visual_dir))

                with open(struct_dir / "objects.pkl", "wb") as f:
                    pickle.dump(objects, f)
                with open(struct_dir / "spatial_index.pkl", "wb") as f:
                    pickle.dump(spatial_index, f)

                patch_grid = encoder.encode_page(image.copy(), doc_id, version_id, 0, float(width), float(height))
                _save_patch_grid(patch_grid, visual_dir)

            manifest[doc_id] = str(image_path)
            processed_docs.add(doc_id)

        prepared_rows.append(
            {
                "question_id": question.question_id,
                "query": question.query,
                "answers": question.answers,
                "doc_id": doc_id,
                "version_id": version_id,
                "page_id": page_id,
                "split": split,
                "metadata": {
                    "image_path": str(image_path),
                    "ocr_path": str(ocr_path) if ocr_path else "",
                },
            }
        )

    if not prepared_rows:
        logger.error("No DocVQA questions were prepared; check your paths.")
        raise typer.Exit(code=1)

    text_index = TextIndex(index_type=config.get("retrieval", {}).get("text_index_type", "bm25"))
    text_index.add_objects_batch(all_objects)
    text_index.build()
    with open(output_index / "text_index.pkl", "wb") as f:
        pickle.dump(text_index, f)

    with open(output_index / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    with open(output_benchmark, "w", encoding="utf-8") as f:
        for row in prepared_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(
        "Prepared DocVQA benchmark: {} questions, {} documents, index_root={}".format(
            len(prepared_rows),
            len(processed_docs),
            output_index,
        )
    )


if __name__ == "__main__":
    app()
