from __future__ import annotations

from tracerag.retrieval.answer_validator import validate_answer
from tracerag.retrieval.query_decomposer import load_attribute_registry
from tracerag.retrieval.structural_joiner import soft_structure_score


def _contains(text: str, term: str) -> bool:
    return term.lower() in text.lower() if term.isascii() else term in text


def _matching_registry_keys(text: str) -> set[str]:
    matches: set[str] = set()
    for key, entry in load_attribute_registry().items():
        aliases = entry.get("aliases", ()) or ()
        for alias in aliases:
            if _contains(text, str(alias)):
                matches.add(str(key))
                break
    return matches


def contradiction_penalty(
    cluster_atom_ids,
    atoms_by_id,
    entity_atom_id: str,
    attribute_atom_id: str,
    value_atom_id: str,
    query_signals,
) -> float:
    if not query_signals.answer_type:
        return 0.0

    entity_atom = atoms_by_id[entity_atom_id]
    attribute_atom = atoms_by_id[attribute_atom_id]
    value_atom = atoms_by_id[value_atom_id]
    attribute_terms = query_signals.attribute_aliases or query_signals.cn_segments

    penalty = 0.0
    competing_values = []
    competing_attributes = []
    conflicting_attributes = []
    target_attribute = query_signals.canonical_attribute

    for atom_id in cluster_atom_ids:
        if atom_id == value_atom_id:
            continue
        atom = atoms_by_id[atom_id]
        if validate_answer(
            atom.text,
            query_signals.answer_type,
            query_signals.units,
            query_signals.regex_hints,
            query_signals.normalization_rules,
        ).is_valid:
            competing_values.append(atom)

    for atom_id in cluster_atom_ids:
        if atom_id == attribute_atom_id:
            continue
        atom = atoms_by_id[atom_id]
        if any(_contains(atom.text, term) for term in attribute_terms):
            competing_attributes.append(atom)
            continue

        registry_keys = _matching_registry_keys(atom.text)
        if registry_keys and target_attribute and target_attribute not in registry_keys:
            conflicting_attributes.append(atom)

    # Penalize clusters with many competing values.
    penalty += min(len(competing_values) * 0.06, 0.18)

    # Penalize if another attribute is closer to the chosen value than the selected attribute.
    selected_link = soft_structure_score(attribute_atom, value_atom, same_cluster=True)
    for other_attr in competing_attributes:
        other_link = soft_structure_score(other_attr, value_atom, same_cluster=True)
        if other_link > selected_link + 0.10:
            penalty += 0.18
            break

    # Penalize semantically conflicting attribute aliases in the same neighborhood.
    for conflicting_attr in conflicting_attributes:
        conflict_link = soft_structure_score(conflicting_attr, value_atom, same_cluster=True)
        if conflict_link > selected_link + 0.08:
            penalty += 0.14
            break

    # Penalize if another value is more tightly bound to the selected attribute than the chosen one.
    for other_value in competing_values:
        other_link = soft_structure_score(attribute_atom, other_value, same_cluster=True)
        if other_link > selected_link + 0.10:
            penalty += 0.16
            break

    # Penalize if another value is tightly bound to a conflicting attribute nearby.
    for conflicting_attr in conflicting_attributes:
        for other_value in competing_values:
            attr_value_link = soft_structure_score(conflicting_attr, other_value, same_cluster=True)
            if attr_value_link > selected_link + 0.10:
                penalty += 0.12
                break
        if penalty >= 0.45:
            break

    # Penalize weak entity-value assembly even if attribute-value is strong.
    entity_value_link = soft_structure_score(entity_atom, value_atom, same_cluster=True)
    if entity_value_link < 0.18:
        penalty += 0.14
    elif entity_value_link < 0.25:
        penalty += 0.10

    return min(penalty, 0.45)
