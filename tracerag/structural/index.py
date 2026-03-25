"""
R-tree spatial indexing for VectorObjects.
"""
from typing import Dict, List, Optional
from rtree import index as rtree_index

from tracerag.common.types import VectorObject, BBox


class PdfSpatialIndex:
    """
    R-tree spatial index over VectorObjects for fast geometric queries.

    Enables efficient "what objects intersect this region?" queries needed for
    the Snapper alignment layer.
    """

    def __init__(self):
        """Initialize empty spatial index."""
        self.page_indexes: Dict[str, rtree_index.Index] = {}
        self.objects: Dict[str, VectorObject] = {}
        self._next_rtree_id = 0
        self._rtree_id_to_obj_id: Dict[int, str] = {}

    def add_object(self, obj: VectorObject):
        """Add VectorObject to spatial index."""
        self.objects[obj.object_id] = obj

        if obj.page_id not in self.page_indexes:
            self.page_indexes[obj.page_id] = rtree_index.Index()

        idx = self.page_indexes[obj.page_id]
        xmin, ymin, xmax, ymax = obj.bbox

        rtree_id = self._next_rtree_id
        self._next_rtree_id += 1
        self._rtree_id_to_obj_id[rtree_id] = obj.object_id

        idx.insert(rtree_id, (xmin, ymin, xmax, ymax))

    def query(self, page_id: str, bbox: BBox) -> List[VectorObject]:
        """Query objects that intersect with given bounding box on a page."""
        idx = self.page_indexes.get(page_id)
        if idx is None:
            return []

        xmin, ymin, xmax, ymax = bbox
        rtree_ids = list(idx.intersection((xmin, ymin, xmax, ymax)))

        # Convert rtree IDs back to object IDs
        obj_ids = [self._rtree_id_to_obj_id[rid] for rid in rtree_ids]
        return [self.objects[oid] for oid in obj_ids]

    def get_object(self, object_id: str) -> Optional[VectorObject]:
        """Get object by ID."""
        return self.objects.get(object_id)

    def get_page_objects(self, page_id: str) -> List[VectorObject]:
        """Get all objects on a page."""
        return [obj for obj in self.objects.values() if obj.page_id == page_id]

    def build_index(self, index_name: str, objects: List[VectorObject]):
        """Convenience: add a batch of objects at once."""
        for obj in objects:
            self.add_object(obj)
