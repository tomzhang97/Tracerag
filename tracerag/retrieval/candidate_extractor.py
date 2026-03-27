from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from tracerag.common.types import VectorObject
from tracerag.retrieval.answer_validator import validate_answer
from tracerag.retrieval.layout_clusterer import LayoutAtom, LocalCluster, cluster_atoms


@dataclass(frozen=True)
class CandidateSets:
    atoms_by_id: Dict[str, LayoutAtom]
    entity_atom_ids: Tuple[str, ...]
    attribute_atom_ids: Tuple[str, ...]
    value_atom_ids: Tuple[str, ...]
    clusters: Tuple[LocalCluster, ...]


def _contains_term(text: str, term: str) -> bool:
    if term.isascii():
        return term.lower() in text.lower()
    return term in text


def is_attribute_text(text: str, query_signals) -> bool:
    attribute_terms = query_signals.attribute_aliases or query_signals.cn_segments
    if not attribute_terms:
        return False

    matched_non_unit = False
    matched_unit_only = False
    normalized_units = {unit.lower() for unit in query_signals.units}

    for term in attribute_terms:
        if not _contains_term(text, term):
            continue
        lowered = term.lower()
        if lowered in normalized_units:
            matched_unit_only = True
        else:
            matched_non_unit = True

    if matched_non_unit:
        return True

    if matched_unit_only and not validate_answer(
        text,
        query_signals.answer_type,
        query_signals.units,
        query_signals.regex_hints,
        query_signals.normalization_rules,
    ).is_valid:
        normalized_text = text.strip().lower()
        return normalized_text in normalized_units

    return False


def build_layout_atoms(page_objects: Sequence[VectorObject]) -> List[LayoutAtom]:
    atoms: List[LayoutAtom] = []
    for obj in page_objects:
        text = (obj.text or "").strip()
        if not text:
            continue
        atoms.append(
            LayoutAtom(
                atom_id=obj.object_id,
                page_id=obj.page_id,
                text=text,
                bbox=obj.bbox,
                source_type=obj.obj_type,
                source_confidence=1.0,
            )
        )
    return atoms


def find_entity_atoms(atoms: Sequence[LayoutAtom], query_signals) -> Tuple[str, ...]:
    hits = []
    for atom in atoms:
        if any(_contains_term(atom.text, term) for term in query_signals.entity_terms):
            hits.append(atom.atom_id)
    return tuple(dict.fromkeys(hits))


def find_attribute_atoms(atoms: Sequence[LayoutAtom], query_signals) -> Tuple[str, ...]:
    hits = []
    for atom in atoms:
        if is_attribute_text(atom.text, query_signals):
            hits.append(atom.atom_id)
    return tuple(dict.fromkeys(hits))


def find_value_atoms(atoms: Sequence[LayoutAtom], query_signals) -> Tuple[str, ...]:
    hits = []
    for atom in atoms:
        if validate_answer(
            atom.text,
            query_signals.answer_type,
            query_signals.units,
            query_signals.regex_hints,
            query_signals.normalization_rules,
        ).is_valid:
            hits.append(atom.atom_id)
    return tuple(dict.fromkeys(hits))


def collect_candidate_sets(page_objects: Sequence[VectorObject], query_signals) -> CandidateSets:
    atoms = build_layout_atoms(page_objects)
    atoms_by_id = {atom.atom_id: atom for atom in atoms}
    entity_atom_ids = find_entity_atoms(atoms, query_signals)
    attribute_atom_ids = find_attribute_atoms(atoms, query_signals)
    value_atom_ids = find_value_atoms(atoms, query_signals)
    seed_atom_ids = entity_atom_ids + attribute_atom_ids + value_atom_ids
    clusters = tuple(cluster_atoms(atoms, seed_atom_ids=seed_atom_ids))
    return CandidateSets(
        atoms_by_id=atoms_by_id,
        entity_atom_ids=entity_atom_ids,
        attribute_atom_ids=attribute_atom_ids,
        value_atom_ids=value_atom_ids,
        clusters=clusters,
    )
