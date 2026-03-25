"""
PDF text and primitive extraction logic.
Directly extracts text, primitives, and basic tables from a PDF using fitz/pdfplumber.
"""

import fitz  # PyMuPDF
import pdfplumber
from typing import List, Dict, Any, Optional
from loguru import logger
from pathlib import Path
import hashlib

from tracerag.common.types import VectorObject
from tracerag.common.io import get_page_id


class PdfExtractor:
    """Extract structural elements (text blocks, vectors, tables, images) from PDF."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.min_text_length = self.config.get("min_text_length", 1)
        self.extract_tables = self.config.get("extract_tables", True)
        self.extract_drawings = self.config.get("extract_drawings", True)
        self.extract_images = self.config.get("extract_images", False)
        # Table specific configs
        self.min_rows = self.config.get("min_rows", 2)
        self.min_cols = self.config.get("min_cols", 2)

    def extract_document(
        self,
        pdf_path: str,
        doc_id: str,
        version_id: str
    ) -> List[VectorObject]:
        """Extract all objects from the entire PDF document."""
        logger.info(f"Extracting PDF: {pdf_path} (doc_id={doc_id}, version_id={version_id})")

        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        objects: List[VectorObject] = []
        
        def _generate_id(obj_type: str, bbox: tuple, text: str = "") -> str:
            val = f"{doc_id}|{version_id}|{page_id}|{obj_type}|{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}|{text}"
            return hashlib.sha256(val.encode('utf-8')).hexdigest()[:16]

        # First pass: use PyMuPDF for text, drawings, and images
        doc = fitz.open(pdf_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_id = get_page_id(doc_id, version_id, page_idx)

            logger.debug(f"Processing page {page_idx} ({page_id})")

            # Text
            text_objects = self._extract_text_blocks(page, doc_id, version_id, page_id, _generate_id)
            objects.extend(text_objects)

            # Drawings
            if self.extract_drawings:
                drawing_objects = self._extract_drawings(page, doc_id, version_id, page_id, _generate_id)
                objects.extend(drawing_objects)

            # Images
            if self.extract_images:
                image_objects = self._extract_images(page, doc_id, version_id, page_id, _generate_id)
                objects.extend(image_objects)
        
        doc.close()

        # Second pass: use pdfplumber for tables
        if self.extract_tables:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_idx in range(len(pdf.pages)):
                        page_id = get_page_id(doc_id, version_id, page_idx)
                        
                        def _generate_id_plumber(obj_type: str, bbox: tuple, text: str = "") -> str:
                            val = f"{doc_id}|{version_id}|{page_id}|{obj_type}|{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}|{text}"
                            return hashlib.sha256(val.encode('utf-8')).hexdigest()[:16]
                            
                        table_objects = self._extract_tables(pdf.pages[page_idx], doc_id, version_id, page_id, _generate_id_plumber, page_idx)
                        objects.extend(table_objects)
            except Exception as e:
                logger.warning(f"Failed to extract tables using pdfplumber: {e}")

        logger.info(f"Extracted {len(objects)} total objects")
        
        # Canonical sort order: top to bottom, left to right, then by ID
        objects.sort(key=lambda obj: (obj.bbox[1], obj.bbox[0], obj.object_id))
        return objects

    def _extract_text_blocks(self, page: fitz.Page, doc_id: str, version_id: str, page_id: str, generate_id_fn) -> List[VectorObject]:
        objects = []
        text_dict = page.get_text("dict")

        for block_idx, block in enumerate(text_dict.get("blocks", [])):
            if "lines" not in block:
                continue

            bbox = tuple(block["bbox"])
            text_parts = []
            for line in block["lines"]:
                for span in line["spans"]:
                    text_parts.append(span["text"])

            text = " ".join(text_parts).strip()
            if len(text) < self.min_text_length:
                continue

            style = {}
            if block["lines"] and block["lines"][0]["spans"]:
                first_span = block["lines"][0]["spans"][0]
                style = {
                    "font": first_span.get("font", ""),
                    "size": first_span.get("size", 0),
                    "color": first_span.get("color", 0),
                    "flags": first_span.get("flags", 0),
                }

            obj = VectorObject(
                object_id=generate_id_fn("text_block", bbox, text),
                doc_id=doc_id,
                version_id=version_id,
                page_id=page_id,
                bbox=bbox,
                obj_type="text_block",
                text=text,
                layer=None,
                style=style,
                meta={"block_no": block["number"]}
            )
            objects.append(obj)
        return objects

    def _extract_drawings(self, page: fitz.Page, doc_id: str, version_id: str, page_id: str, generate_id_fn) -> List[VectorObject]:
        objects = []
        drawings = page.get_drawings()

        for draw_idx, draw in enumerate(drawings):
            bbox = tuple(draw["rect"])
            style = {
                "color": draw.get("color"),
                "fill": draw.get("fill"),
                "width": draw.get("width", 0),
                "closePath": draw.get("closePath", False),
                "type": draw.get("type", "unknown"),
            }

            obj = VectorObject(
                object_id=generate_id_fn("path_group", bbox),
                doc_id=doc_id,
                version_id=version_id,
                page_id=page_id,
                bbox=bbox,
                obj_type="path_group",
                text=None,
                layer=None,
                style=style,
                meta={"seqno": draw.get("seqno", 0)}
            )
            objects.append(obj)
        return objects

    def _extract_images(self, page: fitz.Page, doc_id: str, version_id: str, page_id: str, generate_id_fn) -> List[VectorObject]:
        objects = []
        image_list = page.get_images(full=True)

        for img_idx, img in enumerate(image_list):
            xref = img[0]
            bbox_list = page.get_image_bbox(xref)
            bbox = tuple(bbox_list) if bbox_list else (0, 0, 0, 0)

            obj = VectorObject(
                object_id=generate_id_fn("image", bbox, str(xref)),
                doc_id=doc_id,
                version_id=version_id,
                page_id=page_id,
                bbox=bbox,
                obj_type="image",
                text=None,
                layer=None,
                style={},
                meta={"xref": xref}
            )
            objects.append(obj)
        return objects

    def _extract_tables(self, page, doc_id: str, version_id: str, page_id: str, generate_id_fn, page_num: int) -> List[VectorObject]:
        objects = []
        try:
            tables = page.find_tables()
            for table_idx, table in enumerate(tables):
                table_data = table.extract()
                if not table_data or len(table_data) < self.min_rows:
                    continue
                if len(table_data[0]) < self.min_cols:
                    continue

                for cell_idx, cell in enumerate(table.cells):
                    cell_bbox = tuple(cell)
                    obj = VectorObject(
                        object_id=generate_id_fn("table_cell", cell_bbox, f"table_{table_idx}_cell_{cell_idx}"),
                        doc_id=doc_id,
                        version_id=version_id,
                        page_id=page_id,
                        bbox=cell_bbox,
                        obj_type="table_cell",
                        text="",  # Extracted explicitly if needed
                        layer=None,
                        style={},
                        meta={
                            "table_id": f"table_{table_idx}",
                            "cell_id": cell_idx,
                        }
                    )
                    objects.append(obj)
        except Exception as e:
            logger.warning(f"Failed to extract tables from page {page_num}: {e}")
            
        return objects
