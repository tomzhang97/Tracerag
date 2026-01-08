"""
Table detection and extraction from PDF.

Provides utilities for detecting and parsing tables using pdfplumber or custom algorithms.
"""

import pdfplumber
from typing import List, Dict, Any, Tuple
from loguru import logger

from tracerag.common.types import VectorObject, BBox
from tracerag.common.utils import get_page_id


class TableExtractor:
    """Extract tables from PDF using pdfplumber."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize table extractor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.min_rows = self.config.get("min_rows", 2)
        self.min_cols = self.config.get("min_cols", 2)

    def extract_tables_from_page(
        self,
        pdf_path: str,
        page_num: int,
        doc_id: str,
        version_id: str,
        start_id: int
    ) -> List[VectorObject]:
        """
        Extract tables from a single page.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            doc_id: Document ID
            version_id: Version ID
            start_id: Starting object ID counter

        Returns:
            List of VectorObjects representing table cells
        """
        objects = []
        page_id = get_page_id(doc_id, version_id, page_num)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num >= len(pdf.pages):
                    return objects

                page = pdf.pages[page_num]
                tables = page.find_tables()

                for table_idx, table in enumerate(tables):
                    # Get table bounding box
                    table_bbox = table.bbox  # (x0, top, x1, bottom)

                    # Extract table data
                    table_data = table.extract()

                    if not table_data or len(table_data) < self.min_rows:
                        continue

                    # Check column count
                    if table_data and len(table_data[0]) < self.min_cols:
                        continue

                    # Create cell objects
                    cells = table.cells

                    for cell_idx, cell in enumerate(cells):
                        cell_bbox = tuple(cell)  # (x0, top, x1, bottom)

                        # Try to get cell text
                        cell_text = ""
                        # Note: pdfplumber cell text extraction can be tricky
                        # This is a simplified version

                        obj = VectorObject(
                            object_id=f"obj_{start_id + cell_idx}",
                            doc_id=doc_id,
                            version_id=version_id,
                            page_id=page_id,
                            bbox=cell_bbox,
                            obj_type="table_cell",
                            text=cell_text,
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
