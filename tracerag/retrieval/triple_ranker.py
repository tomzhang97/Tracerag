from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from tracerag.retrieval.answer_validator import ValidationResult, validate_answer
from tracerag.retrieval.candidate_extractor import CandidateSets, is_attribute_text
from tracerag.retrieval.contradiction_scorer import contradiction_penalty
from tracerag.retrieval.scope_assigner import score_cluster_scope
from tracerag.retrieval.structural_joiner import soft_structure_score


@dataclass
class CandidateTriple:
    entity_atom_id: str
    attribute_atom_id: str
    value_atom_id: str
    cluster_id: str
    features: Dict[str, object]
    final_score: float


def _is_ascii_term(term: str) -> bool:
    return term.isascii()


def _contains(text: str, term: str) -> bool:
    return term.lower() in text.lower() if _is_ascii_term(term) else term in text


def _entity_score(text: str, query_signals) -> float:
    if any(_contains(text, term) for term in query_signals.entity_terms):
        return 1.0
    return 0.0


def _attribute_score(text: str, query_signals) -> float:
    if is_attribute_text(text, query_signals):
        return 1.0
    return 0.0


def _value_score(text: str, query_signals) -> ValidationResult:
    return validate_answer(
        text,
        query_signals.answer_type,
        query_signals.units,
        query_signals.regex_hints,
        query_signals.normalization_rules,
    )


def rank_candidate_triples(candidate_sets: CandidateSets, query_signals) -> List[CandidateTriple]:
    triples: List[CandidateTriple] = []
    atoms_by_id = candidate_sets.atoms_by_id

    for cluster in candidate_sets.clusters:
        cluster_atom_ids = set(cluster.atom_ids)
        entity_ids = [atom_id for atom_id in candidate_sets.entity_atom_ids if atom_id in cluster_atom_ids]
        attribute_ids = [atom_id for atom_id in candidate_sets.attribute_atom_ids if atom_id in cluster_atom_ids]
        value_ids = [atom_id for atom_id in candidate_sets.value_atom_ids if atom_id in cluster_atom_ids]
        if not entity_ids or not attribute_ids or not value_ids:
            continue

        scope_score = score_cluster_scope(cluster, atoms_by_id, query_signals)
        cluster_coherence = min(len(cluster.atom_ids) / 6.0, 1.0)

        for entity_atom_id in entity_ids:
            entity_atom = atoms_by_id[entity_atom_id]
            for attribute_atom_id in attribute_ids:
                attribute_atom = atoms_by_id[attribute_atom_id]
                for value_atom_id in value_ids:
                    value_atom = atoms_by_id[value_atom_id]
                    entity_score = _entity_score(entity_atom.text, query_signals)
                    attribute_score = _attribute_score(attribute_atom.text, query_signals)
                    value_validation = _value_score(value_atom.text, query_signals)
                    value_score = value_validation.confidence if value_validation.is_valid else 0.0
                    if not (entity_score and attribute_score and value_score):
                        continue

                    entity_attribute_link = soft_structure_score(entity_atom, attribute_atom, same_cluster=True)
                    attribute_value_link = soft_structure_score(attribute_atom, value_atom, same_cluster=True)
                    entity_value_link = soft_structure_score(entity_atom, value_atom, same_cluster=True)
                    local_structure_score = max(
                        min(entity_attribute_link, attribute_value_link),
                        0.75 * min(entity_attribute_link, entity_value_link),
                    )
                    contradiction = contradiction_penalty(
                        cluster.atom_ids,
                        atoms_by_id,
                        entity_atom_id=entity_atom_id,
                        attribute_atom_id=attribute_atom_id,
                        value_atom_id=value_atom_id,
                        query_signals=query_signals,
                    )

                    final_score = (
                        (0.24 * entity_score)
                        + (0.22 * attribute_score)
                        + (0.22 * value_score)
                        + (0.20 * local_structure_score)
                        + (0.06 * cluster_coherence)
                        + (0.06 * scope_score)
                        - contradiction
                    )

                    triples.append(
                        CandidateTriple(
                            entity_atom_id=entity_atom_id,
                            attribute_atom_id=attribute_atom_id,
                            value_atom_id=value_atom_id,
                            cluster_id=cluster.cluster_id,
                            features={
                                "entity_match": entity_score,
                                "attribute_match": attribute_score,
                                "value_match": value_score,
                                "local_structure_score": local_structure_score,
                                "scope_score": scope_score,
                                "cluster_coherence": cluster_coherence,
                                "contradiction_penalty": contradiction,
                                "normalized_value": value_validation.normalized_value or "",
                                "match_reason": value_validation.match_reason,
                            },
                            final_score=final_score,
                        )
                    )

    triples.sort(key=lambda triple: triple.final_score, reverse=True)
    return triples
