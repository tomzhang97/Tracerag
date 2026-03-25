"""
PDF rendering utilities.
Renders PDF pages as high-resolution images for visual encoding.
"""

import fitz  # PyMuPDF
from PIL import Image

def render_page(pdf_path: str, page_idx: int, dpi: int = 300) -> Image.Image:
    """
    Render PDF page as high-resolution image.

    Args:
        pdf_path: Path to PDF file
        page_idx: Page index (0-indexed)
        dpi: Target DPI for rendering

    Returns:
        PIL Image
    """
    doc = fitz.open(pdf_path)
    if page_idx >= len(doc):
        doc.close()
        raise ValueError(f"Page index {page_idx} out of range for PDF with {len(doc)} pages")

    page = doc[page_idx]

    # Render at specified DPI
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)

    # Convert to PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    doc.close()
    return img
