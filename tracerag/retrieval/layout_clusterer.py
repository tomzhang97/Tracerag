from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class LayoutAtom:
    atom_id: str
    page_id: str
    text: str
    bbox: BBox
    source_type: str
    source_confidence: float = 1.0


@dataclass
class LocalCluster:
    cluster_id: str
    page_id: str
    atom_ids: List[str]
    bbox: BBox
    text: str
    stats: Dict[str, float] = field(default_factory=dict)


def _width(bbox: BBox) -> float:
    return max(bbox[2] - bbox[0], 1.0)


def _height(bbox: BBox) -> float:
    return max(bbox[3] - bbox[1], 1.0)


def _center(bbox: BBox) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _expanded_bbox(atom: LayoutAtom) -> BBox:
    width = _width(atom.bbox)
    height = _height(atom.bbox)
    margin_x = max(width * 1.25, 24.0)
    margin_y = max(height * 1.00, 12.0)
    return (
        atom.bbox[0] - margin_x,
        atom.bbox[1] - margin_y,
        atom.bbox[2] + margin_x,
        atom.bbox[3] + margin_y,
    )


def _intersects(left: BBox, right: BBox) -> bool:
    return not (
        left[2] < right[0]
        or right[2] < left[0]
        or left[3] < right[1]
        or right[3] < left[1]
    )


def _same_band(left: LayoutAtom, right: LayoutAtom) -> bool:
    _, cy1 = _center(left.bbox)
    _, cy2 = _center(right.bbox)
    return abs(cy1 - cy2) <= max(_height(left.bbox), _height(right.bbox)) * 1.1


def _nearby(left: LayoutAtom, right: LayoutAtom) -> bool:
    if _intersects(_expanded_bbox(left), _expanded_bbox(right)):
        return True
    if _same_band(left, right):
        cx1, _ = _center(left.bbox)
        cx2, _ = _center(right.bbox)
        return abs(cx1 - cx2) <= max(_width(left.bbox), _width(right.bbox)) * 4.0
    return False


def _union_bbox(atom_ids: Sequence[str], atoms_by_id: Dict[str, LayoutAtom]) -> BBox:
    boxes = [atoms_by_id[atom_id].bbox for atom_id in atom_ids]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def cluster_atoms(atoms: Sequence[LayoutAtom], seed_atom_ids: Iterable[str] | None = None) -> List[LocalCluster]:
    atoms = [atom for atom in atoms if atom.text]
    if not atoms:
        return []

    atoms_by_id = {atom.atom_id: atom for atom in atoms}
    adjacency: Dict[str, List[str]] = {atom.atom_id: [] for atom in atoms}
    for idx, atom in enumerate(atoms):
        for other in atoms[idx + 1 :]:
            if atom.page_id != other.page_id:
                continue
            if not _nearby(atom, other):
                continue
            adjacency[atom.atom_id].append(other.atom_id)
            adjacency[other.atom_id].append(atom.atom_id)

    components: List[List[str]] = []
    visited = set()
    for atom in atoms:
        if atom.atom_id in visited:
            continue
        stack = [atom.atom_id]
        component: List[str] = []
        visited.add(atom.atom_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        components.append(component)

    seed_set = set(seed_atom_ids or ())
    clusters: List[LocalCluster] = []
    for idx, component in enumerate(components):
        if seed_set and not seed_set.intersection(component):
            continue
        component.sort(key=lambda atom_id: (atoms_by_id[atom_id].bbox[1], atoms_by_id[atom_id].bbox[0]))
        bbox = _union_bbox(component, atoms_by_id)
        clusters.append(
            LocalCluster(
                cluster_id=f"{atoms_by_id[component[0]].page_id}_cluster_{idx}",
                page_id=atoms_by_id[component[0]].page_id,
                atom_ids=component,
                bbox=bbox,
                text=" ".join(atoms_by_id[atom_id].text for atom_id in component if atoms_by_id[atom_id].text).strip(),
                stats={"num_atoms": float(len(component))},
            )
        )

    clusters.sort(key=lambda cluster: (-cluster.stats.get("num_atoms", 0.0), cluster.bbox[1], cluster.bbox[0]))
    return clusters
