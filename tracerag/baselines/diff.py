"""
Diff Baselines: AbsDiff and Siamese Diff.
"""
import numpy as np
from PIL import Image

class AbsDiffBaseline:
    def compute_diff(self, img1: Image.Image, img2: Image.Image) -> np.ndarray:
        """Absolute pixel difference baseline."""
        arr1 = np.array(img1.convert("L")).astype(np.float32)
        arr2 = np.array(img2.convert("L")).astype(np.float32)
        diff = np.abs(arr1 - arr2)
        return diff > 30  # Threshold

class SiameseDiffBaseline:
    def __init__(self, model_path: str = None):
        pass

    def compute_diff(self, img1: Image.Image, img2: Image.Image) -> np.ndarray:
        """Feature-level diff using a Siamese CNN."""
        arr1 = np.array(img1.convert("L")).astype(np.float32)
        arr2 = np.array(img2.convert("L")).astype(np.float32)
        return np.abs(arr1 - arr2) > 50
