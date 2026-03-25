"""
Snapper: Vector-Native Alignment Layer

The core innovation of TraceRAG. Maps probabilistic VLM patch attention
to deterministic PDF vector objects, ensuring every piece of evidence
corresponds to actual PDF primitives.
"""

import numpy as np
from typing import Dict, List, Optional
from loguru import logger
from dataclasses import dataclass

from tracerag.common.types import BBox, RegionEvidence, VectorObject
from tracerag.common.geometry import bbox_intersection, bbox_area, bbox_iou
from tracerag.structural.index import PdfSpatialIndex
from tracerag.common.debug import DebugExporter

@dataclass
class SnapperConfig:
    top_k_patches: int = 256
    min_obj_score: float = 0.01
    overlap_method: str = "iou"  # "iou", "weighted_iou", "coverage", "coverage_weighted"
    bbox_expansion: float = 0.0
    text_boost: float = 10.0
    generate_hashes: bool = False
    debug_export_dir: Optional[str] = None

@dataclass
class PatchRelevanceMap:
    page_id: str
    grid_h: int
    grid_w: int
    scores: np.ndarray        # [H, W] matrix
    patch_boxes: np.ndarray   # [H, W, 4] bounding boxes for patches

@dataclass
class ParsedPageObjects:
    page_id: str
    objects: List[VectorObject]
    spatial_index: PdfSpatialIndex

def _compute_overlap_weight(patch_bbox: BBox, obj_bbox: BBox, method: str) -> float:
    if method == "iou":
        return bbox_iou(patch_bbox, obj_bbox)
    elif method == "weighted_iou":
        intersection = bbox_intersection(patch_bbox, obj_bbox)
        obj_area = bbox_area(obj_bbox)
        if obj_area == 0: return 0.0
        coverage = intersection / obj_area
        iou = bbox_iou(patch_bbox, obj_bbox)
        return 0.7 * coverage + 0.3 * iou
    elif method == "coverage":
        intersection = bbox_intersection(patch_bbox, obj_bbox)
        obj_area = bbox_area(obj_bbox)
        if obj_area == 0: return 0.0
        return intersection / obj_area
    elif method == "coverage_weighted":
        intersection = bbox_intersection(patch_bbox, obj_bbox)
        obj_area = bbox_area(obj_bbox)
        patch_area = bbox_area(patch_bbox)
        if obj_area == 0 or patch_area == 0: return 0.0
        obj_coverage = intersection / obj_area
        patch_coverage = intersection / patch_area
        return 0.8 * obj_coverage + 0.2 * patch_coverage
    else:
        raise ValueError(f"Unknown overlap method: {method}")

def _expand_bbox(bbox: BBox, expansion: float) -> BBox:
    return (
        bbox[0] - expansion,
        bbox[1] - expansion,
        bbox[2] + expansion,
        bbox[3] + expansion
    )

def snap_page_relevance_to_objects(
    relevance_map: PatchRelevanceMap,
    page_objects: ParsedPageObjects,
    config: SnapperConfig
) -> List[RegionEvidence]:
    """
    Align patch attention to vector objects for a single page.
    """
    H, W = relevance_map.grid_h, relevance_map.grid_w
    assert relevance_map.scores.shape == (H, W), "Score dimensions mismatch"

    # 1. Select top-k patches by score
    flat_scores = relevance_map.scores.reshape(-1)
    flat_indices = flat_scores.argsort()[::-1][:config.top_k_patches]

    top_patches = []
    for idx in flat_indices:
        i, j = divmod(idx, W)
        score = float(relevance_map.scores[i, j])
        if score <= 0: continue
        bbox = tuple(relevance_map.patch_boxes[i, j])
        top_patches.append((i, j, bbox, score))

    # 2. Query spatial index
    object_scores: Dict[str, float] = {}
    for i, j, patch_bbox, patch_score in top_patches:
        if config.bbox_expansion > 0:
            patch_bbox = _expand_bbox(patch_bbox, config.bbox_expansion)
        
        candidates = page_objects.spatial_index.query(page_objects.page_id, patch_bbox)
        for obj in candidates:
            weight = _compute_overlap_weight(patch_bbox, obj.bbox, config.overlap_method)
            if weight <= 0: continue
            
            # Apply text boost if object has content
            boost = config.text_boost if (obj.text and len(obj.text.strip()) > 0) else 1.0
            
            contribution = patch_score * weight * boost
            object_scores[obj.object_id] = object_scores.get(obj.object_id, 0.0) + contribution

    # 3. Filter and rank
    ranked_objects = sorted(object_scores.items(), key=lambda kv: kv[1], reverse=True)
    evidences: List[RegionEvidence] = []
    
    for obj_id, score in ranked_objects:
        if score < config.min_obj_score: break
        obj = page_objects.spatial_index.get_object(obj_id)
        if obj is None: continue
        
        # Determine extraction method based on object type
        if obj.obj_type in ("text_block", "table_cell"): method = "vector_text"
        elif obj.obj_type in ("path_group", "symbol"): method = "vector_symbol"
        elif obj.obj_type == "image": method = "raster_image"
        else: method = "unknown"
        
        # Hashing only if requested
        evidence_hash = ""
        if config.generate_hashes:
            from tracerag.alignment.hashing import hash_region
            evidence_hash = hash_region(obj)
            
        evidences.append(RegionEvidence(
            doc_id=obj.doc_id, version_id=obj.version_id, page_id=obj.page_id,
            object_id=obj.object_id, bbox=obj.bbox, obj_type=obj.obj_type,
            extraction_method=method, score=score, hash=evidence_hash
        ))

    logger.debug(f"Generated {len(evidences)} RegionEvidences for page {relevance_map.page_id}")
    
    if config.debug_export_dir:
        exporter = DebugExporter(config.debug_export_dir)
        exporter.save_patch_heatmap(relevance_map.page_id, relevance_map.scores)
        exporter.save_snapped_objects_trace(relevance_map.page_id, evidences)

    return evidences
