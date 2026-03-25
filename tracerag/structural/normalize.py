"""
Normalization logic for VectorObjects.
Normalizes extracted coordinates to a consistent standard coordinate space if required.
"""

from typing import List
from tracerag.common.types import VectorObject
from tracerag.common.geometry import normalize_bbox, PageSpace


def normalize_objects(objects: List[VectorObject], page_width: float = 72.0 * 8.5, page_height: float = 72.0 * 11.0) -> List[VectorObject]:
    """
    Normalizes the bounding boxes of a list of VectorObjects.
    Given that PDF points are from top-left, this can ensure 
    that all objects are normalized to standard PageSpace (e.g. [0,1]).

    Args:
        objects: List of extracted VectorObjects
        page_width: Width of the page (in points)
        page_height: Height of the page (in points)

    Returns:
        List of objects with normalized coordinates.
    """
    normalized_objects = []
    
    # Assume source space is ImageSpace/PDF space and we normalize to PageSpace [0, 1]
    for obj in objects:
        new_bbox = normalize_bbox(obj.bbox, page_width, page_height)
        
        normalized_obj = VectorObject(
            object_id=obj.object_id,
            doc_id=obj.doc_id,
            version_id=obj.version_id,
            page_id=obj.page_id,
            bbox=new_bbox,
            obj_type=obj.obj_type,
            text=obj.text,
            layer=obj.layer,
            style=obj.style,
            meta=obj.meta
        )
        normalized_objects.append(normalized_obj)
        
    return normalized_objects
