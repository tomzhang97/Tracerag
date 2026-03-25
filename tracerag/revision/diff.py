"""
Diffing logic for TraceRAG.
Detects added, removed, moved, and modified objects between versions.
"""

from typing import List, Dict, Tuple, Any
from tracerag.common.types import VectorObject
from tracerag.common.geometry import bbox_iou


def infer_change_type(old_obj: VectorObject, new_obj: VectorObject) -> str:
    """
    Infer what type of change occurred.

    Args:
        old_obj: Old object
        new_obj: New object

    Returns:
        Change type string ("text_modified", "moved", "style_modified", "unchanged")
    """
    # Check text changes
    if old_obj.text != new_obj.text:
        return "text_modified"

    # Check bbox changes
    iou = bbox_iou(old_obj.bbox, new_obj.bbox)
    if iou < 0.9:
        return "moved"

    # Check style changes
    if old_obj.style != new_obj.style:
        return "style_modified"

    return "unchanged"


def compute_version_diff(
    old_objects: List[VectorObject],
    new_objects: List[VectorObject],
    alignments: List[Tuple[VectorObject, VectorObject, float]]
) -> Dict[str, List[VectorObject]]:
    """
    Categorize objects as added, removed, moved, modified, or unchanged
    based on the provided alignments.

    Args:
        old_objects: Objects in the old version
        new_objects: Objects in the new version
        alignments: List of (old_obj, new_obj, score) alignments

    Returns:
        Dictionary mapping change type to list of affected objects from
        the newer version (or older version for removals).
    """
    aligned_old = {old.object_id for old, _, _ in alignments}
    aligned_new = {new.object_id for _, new, _ in alignments}

    # Removals: Old objects that have no match in the new version
    removed = [obj for obj in old_objects if obj.object_id not in aligned_old]

    # Additions: New objects that have no match in the old version
    added = [obj for obj in new_objects if obj.object_id not in aligned_new]

    modified = []
    moved = []
    unchanged = []

    for old_obj, new_obj, _ in alignments:
        change_type = infer_change_type(old_obj, new_obj)
        if change_type == "text_modified" or change_type == "style_modified":
            # For simplicity, returning the new state of the modified object
            modified.append(new_obj)
        elif change_type == "moved":
            moved.append(new_obj)
        else:
            unchanged.append(new_obj)

    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "moved": moved,
        "unchanged": unchanged
    }
