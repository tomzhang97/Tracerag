"""
Image Handler for TraceRAG.
Converts PNG/JPG etc. into structural objects via OCR.
"""
from typing import List
from PIL import Image
from tracerag.structural.handlers import BaseHandler
from tracerag.common.types import VectorObject
from tracerag.common.io import get_page_id
import hashlib

class ImageHandler(BaseHandler):
    def process(self, file_path: str, doc_id: str, version_id: str) -> List[VectorObject]:
        # In a real system, we'd run OCR here (Tesseract/PaddleOCR)
        # For now, we return a single 'image' object or mock OCR results
        img = Image.open(file_path)
        w, h = img.size
        
        obj_id = hashlib.sha256(f"{doc_id}_{version_id}_img_0".encode()).hexdigest()[:16]
        
        return [
            VectorObject(
                object_id=obj_id,
                doc_id=doc_id,
                version_id=version_id,
                page_id=get_page_id(doc_id, version_id, 0),
                bbox=(0, 0, w, h),
                obj_type="image",
                text=f"Image from {file_path}",
                coord_space="ImageSpace"
            )
        ]
