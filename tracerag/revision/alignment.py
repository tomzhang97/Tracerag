"""
Alignment logic for matching document components across versions.
Extracts heuristic matchings from the old STLG module to a standalone, deterministic function.
"""

from typing import List, Tuple, Optional
from difflib import SequenceMatcher

from tracerag.common.types import VectorObject
from tracerag.common.geometry import bbox_iou


def compute_alignment_score(obj1: VectorObject, obj2: VectorObject) -> float:
    """
    Compute alignment score between two objects.

    Args:
        obj1: First object (from older version)
        obj2: Second object (from newer version)

    Returns:
        Alignment score [0, 1]
    """
    # Must be same object type
    if obj1.obj_type != obj2.obj_type:
        return 0.0

    # Bbox IoU
    iou = bbox_iou(obj1.bbox, obj2.bbox)

    # Text similarity if both have text
    if obj1.text and obj2.text:
        text_sim = SequenceMatcher(None, obj1.text, obj2.text).ratio()
        return 0.5 * iou + 0.5 * text_sim
    else:
        return iou


def auto_align_objects(
    old_objects: List[VectorObject],
    new_objects: List[VectorObject],
    bbox_iou_threshold: float = 0.5
) -> List[Tuple[VectorObject, VectorObject, float]]:
    """
    Automatically align objects between two versions.

    Uses heuristics:
    1. Text similarity for text blocks
    2. Bbox IoU for vector objects

    Args:
        old_objects: Objects in the old version
        new_objects: Objects in the new version
        bbox_iou_threshold: Threshold for alignment

    Returns:
        List of aligned tuples (old_object, new_object, score)
    """
    alignments = []

    for old_obj in old_objects:
        best_match = None
        best_score = 0.0

        for new_obj in new_objects:
            score = compute_alignment_score(old_obj, new_obj)

            if score > best_score:
                best_score = score
                best_match = new_obj

        # Threshold for alignment
        if best_score > bbox_iou_threshold and best_match is not None:
            alignments.append((old_obj, best_match, best_score))

    return alignments


def match_entities(
    entities_v1: List[str],
    entities_v2: List[str]
) -> List[Tuple[str, str]]:
    """
    Match entity labels across two versions using simple longest common substring.

    Args:
        entities_v1: Entity labels from version 1
        entities_v2: Entity labels from version 2

    Returns:
        List of (entity_1, entity_2) pairs representing matches
    """
    def _longest_common_substring(s1: str, s2: str) -> str:
        m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
        longest, x_longest = 0, 0
        for x in range(1, 1 + len(s1)):
            for y in range(1, 1 + len(s2)):
                if s1[x - 1] == s2[y - 1]:
                    m[x][y] = m[x - 1][y - 1] + 1
                    if m[x][y] > longest:
                        longest = m[x][y]
                        x_longest = x
                else:
                    m[x][y] = 0
        return s1[x_longest - longest: x_longest]

    matches = []
    
    for e1 in entities_v1:
        if e1 in entities_v2:
            matches.append((e1, e1))
            continue

        best_match = None
        best_score = 0.0

        for e2 in entities_v2:
            label1 = e1.lower()
            label2 = e2.lower()
            common = _longest_common_substring(label1, label2)
            similarity = 2 * len(common) / (len(label1) + len(label2))
            
            if similarity > best_score and similarity > 0.7:
                best_score = similarity
                best_match = e2

        if best_match:
            matches.append((e1, best_match))

    return matches
