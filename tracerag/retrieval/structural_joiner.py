from __future__ import annotations

from tracerag.common.types import VectorObject
from tracerag.retrieval.layout_clusterer import LayoutAtom


def _height(obj: VectorObject) -> float:
    return max(obj.bbox[3] - obj.bbox[1], 1.0)


def _width(obj: VectorObject) -> float:
    return max(obj.bbox[2] - obj.bbox[0], 1.0)


def _center_y(obj: VectorObject) -> float:
    return (obj.bbox[1] + obj.bbox[3]) / 2.0


def is_same_row(left: VectorObject, right: VectorObject) -> bool:
    threshold = max(_height(left), _height(right)) * 0.8
    return abs(_center_y(left) - _center_y(right)) <= threshold


def is_key_value_pair(left: VectorObject, right: VectorObject) -> bool:
    if not is_same_row(left, right):
        return False
    horizontal_gap = max(right.bbox[0] - left.bbox[2], left.bbox[0] - right.bbox[2], 0.0)
    return horizontal_gap <= max(_width(left), _width(right)) * 3.0


def structure_link_score(left: VectorObject, right: VectorObject) -> float:
    if left.page_id != right.page_id:
        return 0.0
    if is_same_row(left, right):
        return 1.0
    if is_key_value_pair(left, right):
        return 0.9
    return 0.0


def _atom_height(atom: LayoutAtom) -> float:
    return max(atom.bbox[3] - atom.bbox[1], 1.0)


def _atom_width(atom: LayoutAtom) -> float:
    return max(atom.bbox[2] - atom.bbox[0], 1.0)


def _atom_center(atom: LayoutAtom) -> tuple[float, float]:
    return ((atom.bbox[0] + atom.bbox[2]) / 2.0, (atom.bbox[1] + atom.bbox[3]) / 2.0)


def same_row_confidence(left: LayoutAtom, right: LayoutAtom) -> float:
    _, cy1 = _atom_center(left)
    _, cy2 = _atom_center(right)
    threshold = max(_atom_height(left), _atom_height(right))
    delta = abs(cy1 - cy2)
    if delta > threshold * 1.2:
        return 0.0
    return max(0.0, 1.0 - (delta / max(threshold, 1.0)))


def same_column_confidence(left: LayoutAtom, right: LayoutAtom) -> float:
    cx1, _ = _atom_center(left)
    cx2, _ = _atom_center(right)
    threshold = max(_atom_width(left), _atom_width(right))
    delta = abs(cx1 - cx2)
    if delta > threshold * 1.5:
        return 0.0
    return max(0.0, 1.0 - (delta / max(threshold * 1.5, 1.0)))


def local_block_adjacency(left: LayoutAtom, right: LayoutAtom) -> float:
    cx1, cy1 = _atom_center(left)
    cx2, cy2 = _atom_center(right)
    dx = abs(cx1 - cx2)
    dy = abs(cy1 - cy2)
    threshold_x = max(_atom_width(left), _atom_width(right)) * 4.0
    threshold_y = max(_atom_height(left), _atom_height(right)) * 3.0
    if dx > threshold_x or dy > threshold_y:
        return 0.0
    score_x = max(0.0, 1.0 - (dx / max(threshold_x, 1.0)))
    score_y = max(0.0, 1.0 - (dy / max(threshold_y, 1.0)))
    return (score_x + score_y) / 2.0


def soft_structure_score(left: LayoutAtom, right: LayoutAtom, same_cluster: bool) -> float:
    row = same_row_confidence(left, right)
    column = same_column_confidence(left, right)
    adjacency = local_block_adjacency(left, right)
    cluster = 0.6 if same_cluster else 0.0
    return max(row, 0.85 * column, 0.75 * adjacency, cluster)
