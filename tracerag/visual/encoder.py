"""
Visual page encoder using Vision-Language Models (VLMs).

Implements the "Visual Stream" that encodes pages as patch grids using
ColPali-style late interaction retrieval.
"""

import fitz  # PyMuPDF
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List
from loguru import logger
from pathlib import Path

from tracerag.common.types import PatchGrid
from tracerag.common.utils import get_page_id


class VisualPageEncoder:
    """
    Encode PDF pages as patch grids using Vision-Language Models.

    Uses ColPali-style architecture: Vision Transformer (ViT) + language model
    for late interaction retrieval.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize visual encoder.

        Args:
            config: Configuration dictionary (from visual section)
        """
        self.config = config
        self.model_name = config.get("model_name", "vidore/colpali-v1.2")
        self.patch_H = config.get("patch_H", 32)
        self.patch_W = config.get("patch_W", 32)
        self.dpi = config.get("dpi", 300)
        self.max_image_size = config.get("max_image_size", 1024)
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config.get("batch_size", 4)

        logger.info(f"Initializing visual encoder: {self.model_name} on {self.device}")
        self.model = self._load_model()

    def _load_model(self):
        """
        Load ColPali or compatible VLM.

        Returns:
            Loaded model
        """
        try:
            # Try to import ColPali components
            # Note: This assumes ColPali is installed or integrated
            # Placeholder for actual ColPali integration
            from transformers import AutoModel, AutoProcessor

            # Load model and processor
            processor = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            model = model.to(self.device)
            model.eval()

            self.processor = processor
            return model

        except Exception as e:
            logger.warning(f"Failed to load ColPali model: {e}")
            logger.warning("Falling back to mock encoder (for development)")
            return self._create_mock_model()

    def _create_mock_model(self):
        """Create a mock model for testing without actual VLM."""
        class MockModel:
            def __init__(self, device):
                self.device = device

            def encode_image(self, image):
                # Return random embeddings
                return torch.randn(32, 32, 128)

        return MockModel(self.device)

    def render_page(self, pdf_path: str, page_idx: int) -> Image.Image:
        """
        Render PDF page as high-resolution image.

        Args:
            pdf_path: Path to PDF file
            page_idx: Page index (0-indexed)

        Returns:
            PIL Image
        """
        doc = fitz.open(pdf_path)
        if page_idx >= len(doc):
            raise ValueError(f"Page index {page_idx} out of range for PDF with {len(doc)} pages")

        page = doc[page_idx]

        # Render at specified DPI
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        doc.close()
        return img

    def encode_page(
        self,
        image: Image.Image,
        doc_id: str,
        version_id: str,
        page_num: int
    ) -> PatchGrid:
        """
        Encode page image as patch grid with embeddings.

        Args:
            image: Page image
            doc_id: Document ID
            version_id: Version ID
            page_num: Page number

        Returns:
            PatchGrid with embeddings
        """
        page_id = get_page_id(doc_id, version_id, page_num)

        # Store original size
        w_orig, h_orig = image.size

        # Resize to model input size
        img_resized = image.resize((self.max_image_size, self.max_image_size))

        # Encode with model
        with torch.no_grad():
            if hasattr(self.model, 'encode_image'):
                # Mock model
                embeddings_tensor = self.model.encode_image(img_resized)
            else:
                # Real model (placeholder - actual implementation depends on ColPali API)
                # This would use the processor and model to get patch embeddings
                inputs = self.processor(images=img_resized, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                # Extract patch embeddings (shape depends on model architecture)
                embeddings_tensor = outputs.last_hidden_state  # Placeholder

        # Convert to numpy
        embeddings = embeddings_tensor.cpu().numpy()

        # Ensure correct shape [H, W, d]
        if len(embeddings.shape) == 2:
            # Reshape to grid if needed
            embeddings = embeddings.reshape(self.patch_H, self.patch_W, -1)
        elif len(embeddings.shape) == 3:
            # Already in correct shape
            pass
        else:
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

        H, W, d = embeddings.shape

        # Calculate patch bounding boxes in original page coordinates
        patch_boxes = np.zeros((H, W, 4), dtype=np.float32)
        patch_w = w_orig / W
        patch_h = h_orig / H

        for i in range(H):
            for j in range(W):
                x0 = j * patch_w
                y0 = i * patch_h
                x1 = (j + 1) * patch_w
                y1 = (i + 1) * patch_h
                patch_boxes[i, j] = [x0, y0, x1, y1]

        return PatchGrid(
            doc_id=doc_id,
            version_id=version_id,
            page_id=page_id,
            H=H,
            W=W,
            embeddings=embeddings,
            patch_boxes=patch_boxes
        )

    def encode_document(
        self,
        pdf_path: str,
        doc_id: str,
        version_id: str,
        output_dir: Optional[str] = None
    ) -> List[PatchGrid]:
        """
        Encode entire PDF document.

        Args:
            pdf_path: Path to PDF file
            doc_id: Document ID
            version_id: Version ID
            output_dir: Optional directory to save patch grids

        Returns:
            List of PatchGrids (one per page)
        """
        logger.info(f"Encoding document: {pdf_path}")

        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        doc.close()

        patch_grids = []

        for page_idx in range(num_pages):
            logger.debug(f"Encoding page {page_idx}/{num_pages}")

            # Render and encode
            image = self.render_page(pdf_path, page_idx)
            patch_grid = self.encode_page(image, doc_id, version_id, page_idx)
            patch_grids.append(patch_grid)

            # Optionally save to disk
            if output_dir:
                self._save_patch_grid(patch_grid, output_dir)

        logger.info(f"Encoded {num_pages} pages")
        return patch_grids

    def _save_patch_grid(self, patch_grid: PatchGrid, output_dir: str):
        """
        Save patch grid to disk.

        Args:
            patch_grid: PatchGrid to save
            output_dir: Output directory
        """
        output_path = Path(output_dir) / f"{patch_grid.page_id}.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_path,
            doc_id=patch_grid.doc_id,
            version_id=patch_grid.version_id,
            page_id=patch_grid.page_id,
            H=patch_grid.H,
            W=patch_grid.W,
            embeddings=patch_grid.embeddings,
            patch_boxes=patch_grid.patch_boxes
        )

    @staticmethod
    def load_patch_grid(file_path: str) -> PatchGrid:
        """
        Load patch grid from disk.

        Args:
            file_path: Path to saved .npz file

        Returns:
            Loaded PatchGrid
        """
        data = np.load(file_path)
        return PatchGrid(
            doc_id=str(data["doc_id"]),
            version_id=str(data["version_id"]),
            page_id=str(data["page_id"]),
            H=int(data["H"]),
            W=int(data["W"]),
            embeddings=data["embeddings"],
            patch_boxes=data["patch_boxes"]
        )
