"""
Text Handler for TraceRAG.
Converts TXT/MD into structural blocks.
"""
from typing import List
from tracerag.structural.handlers import BaseHandler
from tracerag.common.types import VectorObject
from tracerag.common.io import get_page_id
import hashlib

class TextHandler(BaseHandler):
    def process(self, file_path: str, doc_id: str, version_id: str) -> List[VectorObject]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # For simplicity, treat the whole file as one block for now
        obj_id = hashlib.sha256(f"{doc_id}_{version_id}_txt_0".encode()).hexdigest()[:16]
        
        return [
            VectorObject(
                object_id=obj_id,
                doc_id=doc_id,
                version_id=version_id,
                page_id=get_page_id(doc_id, version_id, 0),
                bbox=(0, 0, 1000, 1000), # Virtual bbox for text
                obj_type="text_block",
                text=content,
                coord_space="PDFSpace"
            )
        ]
