"""
OCR Baselines: Tesseract and PaddleOCR (PP-OCRv5).
"""
import abc
from typing import List, Dict, Any
from PIL import Image

class BaseOCR(abc.ABC):
    @abc.abstractmethod
    def extract_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Returns list of boxes with text, coords, and confidences."""
        pass

class TesseractBaseline(BaseOCR):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # pytesseract.pytesseract.tesseract_cmd = ...

    def extract_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Mock extraction using Tesseract."""
        # import pytesseract
        # data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        return [{"text": "mock_tesseract", "bbox": [0,0,10,10], "conf": 0.9}]

class PaddleOCRBaseline(BaseOCR):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        # from paddleocr import PaddleOCR
        # self.ocr = PaddleOCR(use_angle_cls=True, lang='en', version='PP-OCRv5')

    def extract_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Mock extraction using PaddleOCR."""
        # result = self.ocr.ocr(img_array, cls=True)
        return [{"text": "mock_paddle", "bbox": [0,0,10,10], "conf": 0.9}]
