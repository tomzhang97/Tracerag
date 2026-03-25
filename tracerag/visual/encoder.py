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
from tracerag.common.io import get_page_id
from tracerag.visual.render import render_page


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
        self.batch_size = config.get("batch_size", 4)
        
        # Handle device setting: "auto", "cuda", or "cpu"
        device_config = config.get("device", "auto")
        if device_config == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_config
        
        logger.info(f"Initializing visual encoder: {self.model_name} on {self.device}")
        self.processor = None
        self.model = self._load_model()

    def _load_model(self):
        """
        Load ColPali or compatible VLM.

        Loading priority:
        1. colpali-engine package (correct loader for colpali-v1.2-merged)
        2. transformers AutoModel (generic fallback)
        3. Mock encoder (development fallback)
        """
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # --- Strategy 1: colpali-engine (official, correct key mapping) ---
        try:
            from colpali_engine.models import ColPali, ColPaliProcessor
            logger.info(f"Loading ColPali via colpali-engine: {self.model_name}")
            model = ColPali.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device,  # Use device_map instead of .to() to avoid meta tensor error
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(self.model_name)
            self._loader = "colpali_engine"
            logger.info("ColPali loaded successfully via colpali-engine")
            return model
        except ImportError:
            logger.debug("colpali-engine not installed, trying AutoModel fallback")
        except Exception as e:
            logger.warning(f"colpali-engine load failed: {e}")

        # --- Strategy 2: transformers AutoModel (generic, no vlm.* prefix issue) ---
        try:
            from transformers import AutoModel, AutoProcessor
            logger.info(f"Loading model via AutoModel: {self.model_name}")
            model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map=self.device,
                trust_remote_code=True,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._loader = "auto_model"
            logger.info("Model loaded successfully via AutoModel")
            return model
        except Exception as e:
            logger.warning(f"AutoModel load failed: {e}")

        # --- Strategy 3: Mock encoder ---
        logger.warning("All model loading strategies failed. Using mock encoder.")
        self._loader = "mock"
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



    def encode_page(
        self,
        image: Image.Image,
        doc_id: str,
        version_id: str,
        page_num: int,
        w_pdf: float,
        h_pdf: float
    ) -> PatchGrid:
        """
        Encode page image as patch grid with embeddings.

        Args:
            image: Page image
            doc_id: Document ID
            version_id: Version ID
            page_num: Page number
            w_pdf: Original PDF point width
            h_pdf: Original PDF point height

        Returns:
            PatchGrid with embeddings
        """
        page_id = get_page_id(doc_id, version_id, page_num)

        # Store original size (for internal tracking if needed)
        w_orig, h_orig = image.size

        # Resize to model input size
        img_resized = image.resize((self.max_image_size, self.max_image_size))

        # Encode with model
        with torch.no_grad():
            if hasattr(self.model, 'encode_image'):
                # Mock model
                embeddings_tensor = self.model.encode_image(img_resized)

            elif self._loader == "colpali_engine":
                # colpali-engine: use process_images + forward
                batch = self.processor.process_images([img_resized]).to(self.device)
                outputs = self.model(**batch)
                embeddings_tensor = outputs

            else:
                # AutoModel / PaliGemma path
                # PaliGemma requires a text prefix with <image> token
                dummy_text = "<image>"  # Required prefix for PaliGemma
                inputs = self.processor(
                    text=dummy_text,
                    images=img_resized,
                    return_tensors="pt",
                    padding="longest",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                # Extract patch embeddings
                if hasattr(outputs, "last_hidden_state"):
                    embeddings_tensor = outputs.last_hidden_state
                else:
                    embeddings_tensor = outputs[0]

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

        # Calculate patch bounding boxes in PDF point coordinates
        patch_boxes = np.zeros((H, W, 4), dtype=np.float32)
        patch_w = w_pdf / W
        patch_h = h_pdf / H

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

        patch_grids = []

        for page_idx in range(num_pages):
            page = doc[page_idx]
            w_pdf, h_pdf = page.rect.width, page.rect.height
            logger.debug(f"Encoding page {page_idx}/{num_pages} ({w_pdf:.1f}x{h_pdf:.1f} points)")

            # Render and encode
            image = render_page(pdf_path, page_idx, dpi=self.dpi)
            patch_grid = self.encode_page(image, doc_id, version_id, page_idx, w_pdf, h_pdf)
            patch_grids.append(patch_grid)

            # Optionally save to disk
            if output_dir:
                self._save_patch_grid(patch_grid, output_dir)
                
        doc.close()

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
