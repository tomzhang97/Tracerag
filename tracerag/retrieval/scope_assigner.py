from __future__ import annotations

import re

from tracerag.retrieval.layout_clusterer import LayoutAtom, LocalCluster


_NUMBERED_HEADING_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*|[一二三四五六七八九十]+[、.])")


def _is_heading_like(atom: LayoutAtom) -> bool:
    text = atom.text.strip()
    if not text or len(text) > 40:
        return False
    return (
        text.endswith(":")
        or text.endswith("：")
        or len(text) <= 16
        or bool(_NUMBERED_HEADING_RE.match(text))
    )


def _match_score(text: str, query_signals) -> float:
    lowered = text.lower()
    score = 0.0

    for term in query_signals.entity_terms:
        if (term.lower() in lowered) if term.isascii() else (term in text):
            score = max(score, 0.24)
            break

    for term in query_signals.attribute_aliases:
        if (term.lower() in lowered) if term.isascii() else (term in text):
            score = max(score, 0.18)
            break

    for code in query_signals.codes:
        if code in lowered:
            score = max(score, 0.28)
            break

    return score


def _top_band_atoms(cluster: LocalCluster, atoms_by_id) -> list[LayoutAtom]:
    cluster_height = max(cluster.bbox[3] - cluster.bbox[1], 1.0)
    band_bottom = cluster.bbox[1] + min(max(cluster_height * 0.22, 16.0), 48.0)
    atoms = [
        atoms_by_id[atom_id]
        for atom_id in cluster.atom_ids
        if atoms_by_id[atom_id].bbox[1] <= band_bottom
    ]
    atoms.sort(key=lambda atom: (atom.bbox[1], atom.bbox[0]))
    return atoms


def _nearest_previous_heading(cluster: LocalCluster, atoms_by_id) -> LayoutAtom | None:
    cluster_top = cluster.bbox[1]
    cluster_height = max(cluster.bbox[3] - cluster.bbox[1], 1.0)
    best_atom = None
    best_distance = None

    for atom in atoms_by_id.values():
        if atom.page_id != cluster.page_id:
            continue
        if atom.atom_id in cluster.atom_ids:
            continue
        if not _is_heading_like(atom):
            continue
        if atom.bbox[3] > cluster_top:
            continue

        distance = cluster_top - atom.bbox[3]
        if distance > max(180.0, cluster_height * 2.5):
            continue
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_atom = atom

    return best_atom


def score_cluster_scope(cluster: LocalCluster, atoms_by_id, query_signals) -> float:
    best = 0.0
    top_band = _top_band_atoms(cluster, atoms_by_id)

    for atom in top_band:
        if not _is_heading_like(atom):
            continue
        best = max(best, _match_score(atom.text, query_signals))

    previous_heading = _nearest_previous_heading(cluster, atoms_by_id)
    if previous_heading is not None:
        best = max(best, _match_score(previous_heading.text, query_signals) * 0.9)

    return min(best, 0.35)
