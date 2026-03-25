"""
Vision Retrieval Baselines: ColPali and ColQwen2.
"""
import abc
from typing import List, Dict, Any
from PIL import Image

class BaseVisionRetriever(abc.ABC):
    @abc.abstractmethod
    def encode_queries(self, queries: List[str]):
        pass
        
    @abc.abstractmethod
    def encode_pages(self, images: List[Image.Image]):
        pass

class ColPaliBaseline(BaseVisionRetriever):
    def __init__(self, model_name: str = "vidore/colpali-v1.3"):
        self.model_name = model_name
        
    def encode_queries(self, queries: List[str]):
        return []

    def encode_pages(self, images: List[Image.Image]):
        return []

class ColQwen2Baseline(BaseVisionRetriever):
    def __init__(self, model_name: str = "vidore/colqwen2-v1.0"):
        self.model_name = model_name

    def encode_queries(self, queries: List[str]):
        return []

    def encode_pages(self, images: List[Image.Image]):
        return []
