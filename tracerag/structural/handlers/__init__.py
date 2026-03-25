"""
Unified Document Handlers for TraceRAG.
Processes non-PDF formats into TraceRAG structural objects.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from tracerag.common.types import VectorObject

class BaseHandler(ABC):
    @abstractmethod
    def process(self, file_path: str, doc_id: str, version_id: str) -> List[VectorObject]:
        """Convert input file into a list of VectorObjects."""
        pass
