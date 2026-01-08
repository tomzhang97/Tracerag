"""
PDF structural parser: extracts text, tables, and vector graphics from PDF files.

This module implements the "Structural Stream" (Stream S) that extracts the
PDF's construction primitives: text blocks, vector paths, tables, and images.
Unlike OCR-based approaches, this directly accesses the PDF's vector DOM.
"""

import fitz  # PyMuPDF
from rtree import index as rtree_index
from typing import List, Tuple, Dict, Any, Optional
from loguru import logger
from pathlib import Path

from tracerag.common.types import VectorObject, BBox
from tracerag.common.utils import get_page_id


class PdfSpatialIndex:
    """
    R-tree spatial index over VectorObjects for fast geometric queries.

    Enables efficient "what objects intersect this region?" queries needed for
    the Snapper alignment layer.
    """

    def __init__(self):
        """Initialize empty spatial index."""
        self.page_indexes: Dict[str, rtree_index.Index] = {}
        self.objects: Dict[str, VectorObject] = {}
        self._next_rtree_id = 0
        self._rtree_id_to_obj_id: Dict[int, str] = {}

    def add_object(self, obj: VectorObject):
        """
        Add VectorObject to spatial index.

        Args:
            obj: VectorObject to index
        """
        self.objects[obj.object_id] = obj

        # Get or create index for this page
        if obj.page_id not in self.page_indexes:
            self.page_indexes[obj.page_id] = rtree_index.Index()

        idx = self.page_indexes[obj.page_id]
        xmin, ymin, xmax, ymax = obj.bbox

        # R-tree requires integer IDs, so we maintain a mapping
        rtree_id = self._next_rtree_id
        self._next_rtree_id += 1
        self._rtree_id_to_obj_id[rtree_id] = obj.object_id

        idx.insert(rtree_id, (xmin, ymin, xmax, ymax))

    def query(self, page_id: str, bbox: BBox) -> List[VectorObject]:
        """
        Query objects that intersect with given bounding box on a page.

        Args:
            page_id: Page to query
            bbox: Bounding box to search

        Returns:
            List of VectorObjects that intersect bbox
        """
        idx = self.page_indexes.get(page_id)
        if idx is None:
            return []

        xmin, ymin, xmax, ymax = bbox
        rtree_ids = list(idx.intersection((xmin, ymin, xmax, ymax)))

        # Convert rtree IDs back to object IDs
        obj_ids = [self._rtree_id_to_obj_id[rid] for rid in rtree_ids]
        return [self.objects[oid] for oid in obj_ids]

    def get_object(self, object_id: str) -> Optional[VectorObject]:
        """
        Get object by ID.

        Args:
            object_id: Object ID

        Returns:
            VectorObject or None if not found
        """
        return self.objects.get(object_id)

    def get_page_objects(self, page_id: str) -> List[VectorObject]:
        """
        Get all objects on a page.

        Args:
            page_id: Page ID

        Returns:
            List of VectorObjects on page
        """
        return [obj for obj in self.objects.values() if obj.page_id == page_id]


class PdfStructuralParser:
    """
    Parse PDF to extract structural elements (text blocks, vector graphics, tables).

    This is the core of the "Structural Stream" that provides the deterministic,
    vector-native representation of the document.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize parser.

        Args:
            config: Configuration dictionary (from structural section of config)
        """
        self.config = config or {}
        self.min_text_length = self.config.get("min_text_length", 1)
        self.extract_tables = self.config.get("extract_tables", True)
        self.extract_drawings = self.config.get("extract_drawings", True)
        self.extract_images = self.config.get("extract_images", False)

    def parse_document(
        self,
        pdf_path: str,
        doc_id: str,
        version_id: str
    ) -> Tuple[List[VectorObject], PdfSpatialIndex]:
        """
        Parse entire PDF document.

        Args:
            pdf_path: Path to PDF file
            doc_id: Document identifier
            version_id: Version/revision identifier

        Returns:
            Tuple of (list of VectorObjects, PdfSpatialIndex)
        """
        logger.info(f"Parsing PDF: {pdf_path} (doc_id={doc_id}, version_id={version_id})")

        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        objects: List[VectorObject] = []
        spatial_index = PdfSpatialIndex()
        object_id_counter = 0

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_id = get_page_id(doc_id, version_id, page_idx)

            logger.debug(f"Processing page {page_idx} ({page_id})")

            # Extract text blocks
            text_objects = self._extract_text_blocks(
                page, doc_id, version_id, page_id, object_id_counter
            )
            object_id_counter += len(text_objects)
            objects.extend(text_objects)

            # Extract vector drawings
            if self.extract_drawings:
                drawing_objects = self._extract_drawings(
                    page, doc_id, version_id, page_id, object_id_counter
                )
                object_id_counter += len(drawing_objects)
                objects.extend(drawing_objects)

            # Extract tables (placeholder - full implementation would use dedicated table parser)
            if self.extract_tables:
                table_objects = self._extract_tables(
                    page, doc_id, version_id, page_id, object_id_counter
                )
                object_id_counter += len(table_objects)
                objects.extend(table_objects)

            # Extract images if requested
            if self.extract_images:
                image_objects = self._extract_images(
                    page, doc_id, version_id, page_id, object_id_counter
                )
                object_id_counter += len(image_objects)
                objects.extend(image_objects)

        # Build spatial index
        for obj in objects:
            spatial_index.add_object(obj)

        logger.info(f"Extracted {len(objects)} objects from {len(doc)} pages")
        doc.close()

        return objects, spatial_index

    def _extract_text_blocks(
        self,
        page: fitz.Page,
        doc_id: str,
        version_id: str,
        page_id: str,
        start_id: int
    ) -> List[VectorObject]:
        """
        Extract text blocks from page.

        Args:
            page: PyMuPDF Page object
            doc_id: Document ID
            version_id: Version ID
            page_id: Page ID
            start_id: Starting object ID counter

        Returns:
            List of VectorObjects representing text blocks
        """
        objects = []
        text_dict = page.get_text("dict")

        for block_idx, block in enumerate(text_dict.get("blocks", [])):
            if "lines" not in block:
                # Image block, skip
                continue

            bbox = tuple(block["bbox"])

            # Extract text from all lines and spans
            text_parts = []
            for line in block["lines"]:
                for span in line["spans"]:
                    text_parts.append(span["text"])

            text = " ".join(text_parts).strip()

            if len(text) < self.min_text_length:
                continue

            # Extract style information from first span
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
                object_id=f"obj_{start_id + block_idx}",
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

    def _extract_drawings(
        self,
        page: fitz.Page,
        doc_id: str,
        version_id: str,
        page_id: str,
        start_id: int
    ) -> List[VectorObject]:
        """
        Extract vector drawings (paths) from page.

        Args:
            page: PyMuPDF Page object
            doc_id: Document ID
            version_id: Version ID
            page_id: Page ID
            start_id: Starting object ID counter

        Returns:
            List of VectorObjects representing vector paths
        """
        objects = []
        drawings = page.get_drawings()

        for draw_idx, draw in enumerate(drawings):
            bbox = tuple(draw["rect"])

            # Extract style information
            style = {
                "color": draw.get("color"),
                "fill": draw.get("fill"),
                "width": draw.get("width", 0),
                "closePath": draw.get("closePath", False),
                "type": draw.get("type", "unknown"),
            }

            obj = VectorObject(
                object_id=f"obj_{start_id + draw_idx}",
                doc_id=doc_id,
                version_id=version_id,
                page_id=page_id,
                bbox=bbox,
                obj_type="path_group",
                text=None,
                layer=None,  # Could extract from PDF layers if available
                style=style,
                meta={"seqno": draw.get("seqno", 0)}
            )
            objects.append(obj)

        return objects

    def _extract_tables(
        self,
        page: fitz.Page,
        doc_id: str,
        version_id: str,
        page_id: str,
        start_id: int
    ) -> List[VectorObject]:
        """
        Extract tables from page.

        This is a placeholder. Full implementation would use pdfplumber or
        dedicated table detection algorithms.

        Args:
            page: PyMuPDF Page object
            doc_id: Document ID
            version_id: Version ID
            page_id: Page ID
            start_id: Starting object ID counter

        Returns:
            List of VectorObjects representing table cells
        """
        # Placeholder: would integrate with pdfplumber or table detection
        return []

    def _extract_images(
        self,
        page: fitz.Page,
        doc_id: str,
        version_id: str,
        page_id: str,
        start_id: int
    ) -> List[VectorObject]:
        """
        Extract images from page.

        Args:
            page: PyMuPDF Page object
            doc_id: Document ID
            version_id: Version ID
            page_id: Page ID
            start_id: Starting object ID counter

        Returns:
            List of VectorObjects representing images
        """
        objects = []
        image_list = page.get_images(full=True)

        for img_idx, img in enumerate(image_list):
            xref = img[0]
            bbox_list = page.get_image_bbox(xref)

            if bbox_list:
                bbox = tuple(bbox_list)
            else:
                # Fallback if bbox not available
                bbox = (0, 0, 0, 0)

            obj = VectorObject(
                object_id=f"obj_{start_id + img_idx}",
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
