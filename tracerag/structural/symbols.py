"""
Symbol grouping and recognition for CAD-derived PDFs.

Groups low-level vector primitives (lines, arcs, circles) into higher-level
symbols (valves, switches, components) based on geometric proximity and closure.
"""

from typing import List, Set, Dict, Any
from collections import defaultdict
import numpy as np

from tracerag.common.types import VectorObject, BBox
from tracerag.common.utils import bbox_distance, bbox_iou


class VectorGrouper:
    """
    Group vector primitives into logical symbols/components.

    Uses geometric heuristics:
    1. Proximity: paths close together likely belong to same symbol
    2. Closure: closed paths or nearly-closed groups
    3. Layer grouping: paths on same CAD layer
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize grouper.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.proximity_threshold = self.config.get("proximity_threshold", 5.0)
        self.layer_aware = self.config.get("layer_aware", True)

    def group_paths(self, path_objects: List[VectorObject]) -> List[List[VectorObject]]:
        """
        Group vector path objects into symbols.

        Args:
            path_objects: List of VectorObjects with obj_type='path_group'

        Returns:
            List of groups, where each group is a list of VectorObjects
        """
        if not path_objects:
            return []

        # Build adjacency graph based on proximity
        n = len(path_objects)
        adjacency = defaultdict(set)

        for i in range(n):
            for j in range(i + 1, n):
                if self._should_group(path_objects[i], path_objects[j]):
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        # Find connected components (groups)
        visited = set()
        groups = []

        for i in range(n):
            if i in visited:
                continue

            # BFS to find all connected paths
            group_indices = set()
            queue = [i]
            group_indices.add(i)
            visited.add(i)

            while queue:
                current = queue.pop(0)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        group_indices.add(neighbor)
                        queue.append(neighbor)

            # Convert indices to objects
            group = [path_objects[idx] for idx in group_indices]
            groups.append(group)

        return groups

    def _should_group(self, obj1: VectorObject, obj2: VectorObject) -> bool:
        """
        Determine if two objects should be grouped together.

        Args:
            obj1: First object
            obj2: Second object

        Returns:
            True if objects should be grouped
        """
        # Check layer if layer-aware
        if self.layer_aware:
            if obj1.layer and obj2.layer and obj1.layer != obj2.layer:
                return False

        # Check proximity
        distance = bbox_distance(obj1.bbox, obj2.bbox)
        if distance <= self.proximity_threshold:
            return True

        # Check if bounding boxes overlap
        if bbox_iou(obj1.bbox, obj2.bbox) > 0:
            return True

        return False

    def create_symbol_object(
        self,
        group: List[VectorObject],
        symbol_id: str
    ) -> VectorObject:
        """
        Create a composite symbol object from a group of paths.

        Args:
            group: List of VectorObjects to combine
            symbol_id: ID for the new symbol object

        Returns:
            New VectorObject representing the symbol
        """
        if not group:
            raise ValueError("Cannot create symbol from empty group")

        # Compute bounding box that encompasses all paths
        all_bboxes = [obj.bbox for obj in group]
        min_x = min(bbox[0] for bbox in all_bboxes)
        min_y = min(bbox[1] for bbox in all_bboxes)
        max_x = max(bbox[2] for bbox in all_bboxes)
        max_y = max(bbox[3] for bbox in all_bboxes)

        combined_bbox = (min_x, min_y, max_x, max_y)

        # Merge metadata
        combined_meta = {
            "component_count": len(group),
            "component_ids": [obj.object_id for obj in group],
        }

        # Use first object's attributes as base
        base = group[0]

        symbol = VectorObject(
            object_id=symbol_id,
            doc_id=base.doc_id,
            version_id=base.version_id,
            page_id=base.page_id,
            bbox=combined_bbox,
            obj_type="symbol",
            text=None,
            layer=base.layer,
            style=base.style.copy(),
            meta=combined_meta
        )

        return symbol
