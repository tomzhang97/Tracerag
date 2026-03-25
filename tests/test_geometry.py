import pytest
import numpy as np
from tracerag.common.geometry import CoordinateTransform

class MockRect:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.width = x1 - x0
        self.height = y1 - y0

class MockPage:
    def __init__(self, rect, rotation=0):
        self.rect = rect
        self.cropbox = rect
        self.mediabox = rect
        self.rotation = rotation

def test_coordinate_transform_round_trip():
    rect = MockRect(0, 0, 595, 842)  # A4 roughly
    page = MockPage(rect, rotation=0)
    transform = CoordinateTransform(page, dpi=72)
    
    original_pdf_bbox = (100.0, 150.0, 200.0, 250.0)
    
    # PDF -> Image -> PDF
    img_bbox = transform.pdf_to_image(original_pdf_bbox)
    reverted_pdf_bbox = transform.image_to_pdf(img_bbox)
    
    np.testing.assert_almost_equal(original_pdf_bbox, reverted_pdf_bbox, decimal=2)

def test_patch_transform_round_trip():
    rect = MockRect(0, 0, 600, 800)
    page = MockPage(rect, rotation=90)
    transform = CoordinateTransform(page, dpi=144) 
    
    original_pdf_bbox = (50.0, 50.0, 150.0, 150.0)
    patch_H, patch_W = 16, 16
    
    # PDF -> Patch -> PDF
    patch_bbox = transform.pdf_to_patch(original_pdf_bbox, patch_H, patch_W)
    reverted_pdf_bbox = transform.patch_to_pdf(patch_bbox, patch_H, patch_W)
    
    np.testing.assert_almost_equal(original_pdf_bbox, reverted_pdf_bbox, decimal=2)

def test_image_to_patch_round_trip():
    rect = MockRect(0, 0, 600, 800)
    page = MockPage(rect)
    transform = CoordinateTransform(page, dpi=72)
    
    original_img_bbox = (100.0, 100.0, 200.0, 200.0)
    patch_H, patch_W = 16, 16
    
    # Image -> Patch -> Image
    patch_bbox = transform.image_to_patch(original_img_bbox, patch_H, patch_W, transform.render_width, transform.render_height)
    reverted_img_bbox = transform.patch_to_image(patch_bbox, patch_H, patch_W, transform.render_width, transform.render_height)
    
    np.testing.assert_almost_equal(original_img_bbox, reverted_img_bbox, decimal=2)
