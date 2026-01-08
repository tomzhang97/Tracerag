"""
Snapper: Vector-Native Alignment Layer

The core innovation of TraceRAG. Maps probabilistic VLM patch attention
to deterministic PDF vector objects, ensuring every piece of evidence
corresponds to actual PDF primitives.

Algorithm:
1. For each high-attention patch from VLM
2. Query spatial index to find candidate vector objects intersecting patch
3. Compute weighted relevance score based on IoU and attention density
4. Filter and rank objects
5. Return RegionEvidence anchored to exact vector objects
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger

from tracerag.common.types import BBox, RegionEvidence, VectorObject, PatchGrid
from tracerag.common.utils import bbox_intersection, bbox_area, bbox_iou
from tracerag.structural.parser import PdfSpatialIndex
from tracerag.alignment.hashing import hash_region


class Snapper:
    """
    Vector-Native Alignment Layer.

    Bridges the gap between fuzzy VLM attention and precise PDF geometry.
    """

    def __init__(
        self,
        spatial_index: PdfSpatialIndex,
        config: Optional[Dict] = None
    ):
        """
        Initialize Snapper.

        Args:
            spatial_index: PdfSpatialIndex for querying vector objects
            config: Configuration dictionary (from snapper section)
        """
        self.spatial_index = spatial_index
        self.config = config or {}

        self.top_k_patches = self.config.get("top_k_patches", 256)
        self.min_obj_score = self.config.get("min_obj_score", 0.01)
        self.overlap_method = self.config.get("overlap_method", "iou")
        self.bbox_expansion = self.config.get("bbox_expansion", 0.0)

    def snap_page(
        self,
        patch_grid: PatchGrid,
        patch_scores: np.ndarray
    ) -> List[RegionEvidence]:
        """
        Align patch attention to vector objects for a single page.

        Args:
            patch_grid: PatchGrid with embeddings and bbox info
            patch_scores: Patch relevance scores [H, W] from visual scorer

        Returns:
            List of RegionEvidence objects, ranked by score
        """
        H, W = patch_scores.shape
        assert (H, W) == (patch_grid.H, patch_grid.W), "Patch score dimensions mismatch"

        # Step 1: Select top-k patches by score
        flat_scores = patch_scores.reshape(-1)
        flat_indices = flat_scores.argsort()[::-1][:self.top_k_patches]

        # Convert flat indices to (i, j) coordinates
        top_patches = []
        for idx in flat_indices:
            i, j = divmod(idx, W)
            score = float(patch_scores[i, j])
            if score <= 0:
                continue
            bbox = tuple(patch_grid.patch_boxes[i, j])
            top_patches.append((i, j, bbox, score))

        logger.debug(f"Selected {len(top_patches)} top patches for page {patch_grid.page_id}")

        # Step 2: Query spatial index for each patch
        object_scores: Dict[str, float] = {}

        for i, j, patch_bbox, patch_score in top_patches:
            # Expand bbox slightly if configured
            if self.bbox_expansion > 0:
                patch_bbox = self._expand_bbox(patch_bbox, self.bbox_expansion)

            # Query candidates
            candidates = self.spatial_index.query(patch_grid.page_id, patch_bbox)

            # Score each candidate
            for obj in candidates:
                weight = self._compute_overlap_weight(patch_bbox, obj.bbox)
                if weight <= 0:
                    continue

                # Accumulate weighted score
                contribution = patch_score * weight
                object_scores[obj.object_id] = object_scores.get(obj.object_id, 0.0) + contribution

        # Step 3: Filter and rank objects
        ranked_objects = sorted(object_scores.items(), key=lambda kv: kv[1], reverse=True)

        evidences: List[RegionEvidence] = []
        for obj_id, score in ranked_objects:
            if score < self.min_obj_score:
                break

            obj = self.spatial_index.get_object(obj_id)
            if obj is None:
                logger.warning(f"Object {obj_id} not found in index")
                continue

            evidence = self._create_region_evidence(obj, score)
            evidences.append(evidence)

        logger.debug(f"Generated {len(evidences)} region evidences for page {patch_grid.page_id}")

        return evidences

    def snap_pages_batch(
        self,
        patch_grids: List[PatchGrid],
        patch_scores_list: List[np.ndarray]
    ) -> List[List[RegionEvidence]]:
        """
        Align patch attention to vector objects for multiple pages.

        Args:
            patch_grids: List of PatchGrids
            patch_scores_list: List of patch scores arrays

        Returns:
            List of evidence lists (one per page)
        """
        assert len(patch_grids) == len(patch_scores_list)

        results = []
        for patch_grid, patch_scores in zip(patch_grids, patch_scores_list):
            evidences = self.snap_page(patch_grid, patch_scores)
            results.append(evidences)

        return results

    def _compute_overlap_weight(self, patch_bbox: BBox, obj_bbox: BBox) -> float:
        """
        Compute overlap weight between patch and object bbox.

        Uses configured method (IoU or weighted IoU).

        Args:
            patch_bbox: Patch bounding box
            obj_bbox: Object bounding box

        Returns:
            Overlap weight [0, 1]
        """
        if self.overlap_method == "iou":
            return bbox_iou(patch_bbox, obj_bbox)

        elif self.overlap_method == "weighted_iou":
            # Weighted by how much of the object is covered
            intersection = bbox_intersection(patch_bbox, obj_bbox)
            obj_area = bbox_area(obj_bbox)

            if obj_area == 0:
                return 0.0

            # Fraction of object covered by patch
            coverage = intersection / obj_area

            # Also consider IoU for balance
            iou = bbox_iou(patch_bbox, obj_bbox)

            # Weighted combination
            return 0.7 * coverage + 0.3 * iou

        else:
            raise ValueError(f"Unknown overlap method: {self.overlap_method}")

    def _expand_bbox(self, bbox: BBox, expansion: float) -> BBox:
        """
        Expand bounding box by a margin.

        Args:
            bbox: Original bbox
            expansion: Pixels to expand in each direction

        Returns:
            Expanded bbox
        """
        return (
            bbox[0] - expansion,
            bbox[1] - expansion,
            bbox[2] + expansion,
            bbox[3] + expansion
        )

    def _create_region_evidence(self, obj: VectorObject, score: float) -> RegionEvidence:
        """
        Create RegionEvidence from VectorObject and score.

        Args:
            obj: VectorObject
            score: Relevance score

        Returns:
            RegionEvidence
        """
        # Determine extraction method based on object type
        if obj.obj_type in ("text_block", "table_cell"):
            method = "vector_text"
        elif obj.obj_type in ("path_group", "symbol"):
            method = "vector_symbol"
        elif obj.obj_type == "image":
            method = "raster_image"
        else:
            method = "unknown"

        # Generate tamper-evident hash
        evidence_hash = hash_region(obj)

        return RegionEvidence(
            doc_id=obj.doc_id,
            version_id=obj.version_id,
            page_id=obj.page_id,
            object_id=obj.object_id,
            bbox=obj.bbox,
            obj_type=obj.obj_type,
            extraction_method=method,
            score=score,
            hash=evidence_hash
        )
