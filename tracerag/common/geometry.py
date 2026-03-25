"""
Geometry and coordinate transformations for TraceRAG.

Handles conversions between PDF Space (points, bottom-left origin),
Image Space (pixels, top-left origin), and Patch Space.
Also provides common bounding box operations.
"""

import fitz
import numpy as np
from typing import Tuple, List

# Type aliases for coordinates
BBox = Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
Polygon = List[Tuple[float, float]]

# Bounding Box Math Functions
def bbox_iou(box1: BBox, box2: BBox) -> float:
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

def bbox_intersection(box1: BBox, box2: BBox) -> float:
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)

def bbox_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

def bbox_center(box: BBox) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

def bbox_distance(box1: BBox, box2: BBox) -> float:
    c1, c2 = bbox_center(box1), bbox_center(box2)
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5

def normalize_bbox(bbox: BBox, page_width: float, page_height: float) -> BBox:
    return (bbox[0] / page_width, bbox[1] / page_height, bbox[2] / page_width, bbox[3] / page_height)

def denormalize_bbox(bbox: BBox, page_width: float, page_height: float) -> BBox:
    return (bbox[0] * page_width, bbox[1] * page_height, bbox[2] * page_width, bbox[3] * page_height)


class CoordinateTransform:
    """
    Handles coordinate transformations between PDF space and image/patch space.
    """

    def __init__(self, page: fitz.Page, dpi: int = 300):
        self.page = page
        self.dpi = dpi
        self.scale = dpi / 72.0
        
        self.mediabox = page.mediabox
        self.cropbox = page.cropbox if page.cropbox else page.mediabox
        self.rotation = page.rotation

        self.transform_matrix = self._compute_transform_matrix()
        
        self.render_width = int(page.rect.width * self.scale)
        self.render_height = int(page.rect.height * self.scale)

    def _compute_transform_matrix(self) -> np.ndarray:
        crop_x0, crop_y0 = self.cropbox.x0, self.cropbox.y0
        crop_width, crop_height = self.cropbox.width, self.cropbox.height

        flip_matrix = np.array([
            [1, 0, -crop_x0],
            [0, -1, crop_y0 + crop_height],
            [0, 0, 1]
        ])

        rotation_rad = np.radians(self.rotation)
        cos_r, sin_r = np.cos(rotation_rad), np.sin(rotation_rad)
        
        if self.rotation == 0:
            rotation_matrix = np.eye(3)
        elif self.rotation == 90:
            rotation_matrix = np.array([[0, 1, 0], [-1, 0, crop_width], [0, 0, 1]])
        elif self.rotation == 180:
            rotation_matrix = np.array([[-1, 0, crop_width], [0, -1, crop_height], [0, 0, 1]])
        elif self.rotation == 270:
            rotation_matrix = np.array([[0, -1, crop_height], [1, 0, 0], [0, 0, 1]])
        else:
            rotation_matrix = np.array([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]])

        scale_matrix = np.array([[self.scale, 0, 0], [0, self.scale, 0], [0, 0, 1]])
        return scale_matrix @ rotation_matrix @ flip_matrix

    def pdf_to_image(self, pdf_bbox: BBox) -> BBox:
        x0, y0, x1, y1 = pdf_bbox
        corners = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]]).T
        transformed = self.transform_matrix @ corners
        x_coords, y_coords = transformed[0, :], transformed[1, :]
        return (float(np.min(x_coords)), float(np.min(y_coords)), float(np.max(x_coords)), float(np.max(y_coords)))

    def image_to_patch(self, image_bbox: BBox, patch_H: int, patch_W: int, image_width: int, image_height: int) -> BBox:
        x0, y0, x1, y1 = image_bbox
        patch_width, patch_height = image_width / patch_W, image_height / patch_H
        return (x0 / patch_width, y0 / patch_height, x1 / patch_width, y1 / patch_height)

    def patch_to_image(self, patch_bbox: BBox, patch_H: int, patch_W: int, image_width: int, image_height: int) -> BBox:
        p_x0, p_y0, p_x1, p_y1 = patch_bbox
        patch_width, patch_height = image_width / patch_W, image_height / patch_H
        return (p_x0 * patch_width, p_y0 * patch_height, p_x1 * patch_width, p_y1 * patch_height)

    def image_to_pdf(self, image_bbox: BBox) -> BBox:
        inv_matrix = np.linalg.inv(self.transform_matrix)
        x0, y0, x1, y1 = image_bbox
        corners = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]]).T
        transformed = inv_matrix @ corners
        x_coords, y_coords = transformed[0, :], transformed[1, :]
        return (float(np.min(x_coords)), float(np.min(y_coords)), float(np.max(x_coords)), float(np.max(y_coords)))

    def pdf_to_patch(self, pdf_bbox: BBox, patch_H: int, patch_W: int) -> BBox:
        image_bbox = self.pdf_to_image(pdf_bbox)
        return self.image_to_patch(image_bbox, patch_H, patch_W, self.render_width, self.render_height)

    def patch_to_pdf(self, patch_bbox: BBox, patch_H: int, patch_W: int) -> BBox:
        image_bbox = self.patch_to_image(patch_bbox, patch_H, patch_W, self.render_width, self.render_height)
        return self.image_to_pdf(image_bbox)

def compute_patch_boxes_with_transform(page: fitz.Page, patch_H: int, patch_W: int, image_width: int, image_height: int, dpi: int = 300) -> np.ndarray:
    transformer = CoordinateTransform(page, dpi)
    patch_boxes = np.zeros((patch_H, patch_W, 4), dtype=np.float32)
    patch_width_px, patch_height_px = image_width / patch_W, image_height / patch_H
    
    for i in range(patch_H):
        for j in range(patch_W):
            img_x0, img_y0 = j * patch_width_px, i * patch_height_px
            img_x1, img_y1 = (j + 1) * patch_width_px, (i + 1) * patch_height_px
            patch_boxes[i, j] = (img_x0, img_y0, img_x1, img_y1)
            
    return patch_boxes
