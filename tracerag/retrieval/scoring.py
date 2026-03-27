"""
Shared scoring helpers for TraceRAG retrieval routes.

This module keeps visual evidence, symbolic bonuses, and document priors
as separate components so later stages do not mutate an already-boosted
score distribution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from tracerag.common.types import RegionEvidence, VectorObject
from tracerag.retrieval.candidate_extractor import collect_candidate_sets, is_attribute_text
from tracerag.retrieval.answer_validator import ValidationResult, validate_answer
from tracerag.retrieval.query_decomposer import decompose_query
from tracerag.retrieval.structural_joiner import soft_structure_score
from tracerag.retrieval.triple_ranker import rank_candidate_triples


CN_STOP_WORDS = {
    "有哪些",
    "什么是",
    "列出",
    "多少",
    "怎么",
    "如何",
    "一个",
    "这些",
    "这个",
}

SUMMARY_KEYWORDS = (
    "合计",
    "总计",
    "总重",
    "总净重",
    "总毛重",
    "共计",
    "total",
    "sum",
    "subtotal",
)

DOC_TYPE_KEYWORDS = {
    "certificate": ("证书", "certificate", "certification"),
    "report": ("报告", "report", "检测"),
    "manual": ("说明书", "manual", "手册"),
    "drawing": ("图纸", "drawing"),
    "spec": ("规范", "spec", "specification"),
}


@dataclass(frozen=True)
class QuerySignals:
    raw_query: str
    codes: Tuple[str, ...]
    cn_segments: Tuple[str, ...]
    cn_bigrams: Tuple[str, ...]
    entity_terms: Tuple[str, ...] = ()
    canonical_attribute: str = ""
    attribute_aliases: Tuple[str, ...] = ()
    answer_type: str = ""
    units: Tuple[str, ...] = ()
    regex_hints: Tuple[str, ...] = ()
    normalization_rules: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteWeights:
    visual: float
    symbolic: float
    scope: float
    document: float


@dataclass
class DocumentProfile:
    doc_id: str
    abs_path: str = ""
    folder_name: str = ""
    ancestor_folders: Tuple[str, ...] = ()
    file_name: str = ""
    file_stem: str = ""
    title_text: str = ""
    doc_type: str = ""
    path_text: str = ""

    @property
    def identity_text(self) -> str:
        return " ".join(
            part
            for part in (
                self.folder_name,
                " ".join(self.ancestor_folders),
                self.file_name,
                self.file_stem,
                self.title_text,
                self.doc_type,
            )
            if part
        )


def _dedupe_preserve_order(values: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def extract_query_signals(query: str) -> QuerySignals:
    """Extract stable symbolic terms used across scoring routes."""
    codes = _dedupe_preserve_order(
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_.]{1,}", query)
    )

    cn_segments = _dedupe_preserve_order(
        segment
        for segment in re.findall(r"[\u4e00-\u9fff]{2,}", query)
        if segment not in CN_STOP_WORDS
    )

    cn_chars = re.findall(r"[\u4e00-\u9fff]", query)
    cn_bigrams = _dedupe_preserve_order(
        cn_chars[idx] + cn_chars[idx + 1]
        for idx in range(len(cn_chars) - 1)
        if (cn_chars[idx] + cn_chars[idx + 1]) not in CN_STOP_WORDS
    )

    decomposition = decompose_query(query, cn_segments, codes)

    return QuerySignals(
        raw_query=query,
        codes=codes,
        cn_segments=cn_segments,
        cn_bigrams=cn_bigrams,
        entity_terms=decomposition.entity_terms,
        canonical_attribute=decomposition.canonical_attribute,
        attribute_aliases=decomposition.attribute_aliases,
        answer_type=decomposition.answer_type,
        units=decomposition.units,
        regex_hints=decomposition.regex_hints,
        normalization_rules=decomposition.normalization_rules,
    )


def _page_number_from_id(page_id: str) -> int:
    match = re.search(r"_p(\d+)$", page_id)
    return int(match.group(1)) if match else 0


def _infer_doc_type(text: str) -> str:
    lowered = text.lower()
    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return doc_type
    return ""


def infer_target_doc_type(query: str) -> str:
    lowered = query.lower()
    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return doc_type
    return ""


def _extract_title_text_from_objects(text_objects: Sequence[VectorObject]) -> str:
    if not text_objects:
        return ""

    ordered_objects = list(text_objects)
    ordered_objects.sort(key=lambda obj: (_page_number_from_id(obj.page_id), obj.bbox[1], obj.bbox[0]))
    first_page = _page_number_from_id(ordered_objects[0].page_id)
    first_page_objects = [obj for obj in ordered_objects if _page_number_from_id(obj.page_id) == first_page]
    first_page_objects.sort(key=lambda obj: (obj.bbox[1], obj.bbox[0]))

    title_lines: List[str] = []
    for obj in first_page_objects[:8]:
        text = (obj.text or "").strip()
        if not text:
            continue
        title_lines.append(text)
        if len(" ".join(title_lines)) >= 180:
            break

    return " ".join(title_lines)[:180]


def _extract_title_text(doc_id: str, spatial_index) -> str:
    text_objects = [
        obj
        for obj in spatial_index.objects.values()
        if obj.doc_id == doc_id and obj.text and obj.text.strip()
    ]
    return _extract_title_text_from_objects(text_objects)


def build_document_profiles(
    doc_ids: Iterable[str],
    manifest: Dict[str, str],
    spatial_index,
) -> Dict[str, DocumentProfile]:
    """Build lightweight document metadata shared by all routes."""
    profiles: Dict[str, DocumentProfile] = {}
    text_objects_by_doc: Dict[str, List[VectorObject]] = {}
    for obj in spatial_index.objects.values():
        if obj.text and obj.text.strip():
            text_objects_by_doc.setdefault(obj.doc_id, []).append(obj)

    for doc_id in dict.fromkeys(doc_ids):
        abs_path = manifest.get(doc_id, "")
        path = Path(abs_path) if abs_path else None
        folder_name = path.parent.name.lower() if path else ""
        ancestor_folders = ()
        if path:
            ancestor_names = []
            for parent in list(path.parents)[1:4]:
                if parent.name:
                    ancestor_names.append(parent.name.lower())
            ancestor_folders = tuple(ancestor_names)
        file_name = path.name.lower() if path else ""
        file_stem = path.stem.lower() if path else doc_id.lower()
        title_text = _extract_title_text_from_objects(text_objects_by_doc.get(doc_id, ())).lower()
        path_text = " ".join(
            part for part in (folder_name, " ".join(ancestor_folders), file_name, file_stem, title_text) if part
        )
        doc_type = _infer_doc_type(path_text)

        profiles[doc_id] = DocumentProfile(
            doc_id=doc_id,
            abs_path=abs_path,
            folder_name=folder_name,
            ancestor_folders=ancestor_folders,
            file_name=file_name,
            file_stem=file_stem,
            title_text=title_text,
            doc_type=doc_type,
            path_text=path_text,
        )

    return profiles


def _score_term_matches(term: str, profile: DocumentProfile) -> float:
    score = 0.0
    if term in profile.folder_name:
        score += 0.24
    for depth, folder in enumerate(profile.ancestor_folders, start=1):
        if term in folder:
            score += max(0.18 - (depth - 1) * 0.04, 0.06)
    if term in profile.file_stem:
        score += 0.18
    if term in profile.file_name:
        score += 0.08
    if term in profile.title_text:
        score += 0.20
    return score


def compute_document_prior(profile: DocumentProfile, query_signals: QuerySignals) -> float:
    """
    Compute a soft document prior from stable metadata.

    This is intentionally capped and mild so it cannot drown stronger
    object-level evidence.
    """
    prior = 0.0

    code_score = 0.0
    for code in query_signals.codes:
        code_score = max(code_score, _score_term_matches(code, profile))
    prior += min(code_score, 0.45)

    attribute_score = 0.0
    for segment in query_signals.cn_segments:
        attribute_score = max(attribute_score, _score_term_matches(segment, profile))
    prior += min(attribute_score, 0.30)

    if profile.doc_type and any(keyword in query_signals.raw_query.lower() for keyword in DOC_TYPE_KEYWORDS.get(profile.doc_type, ())):
        prior += 0.12

    return min(prior, 1.0)


def compute_document_priors(
    doc_ids: Iterable[str],
    manifest: Dict[str, str],
    spatial_index,
    query_signals: QuerySignals,
) -> Tuple[Dict[str, DocumentProfile], Dict[str, float]]:
    profiles = build_document_profiles(doc_ids, manifest, spatial_index)
    priors = {
        doc_id: compute_document_prior(profile, query_signals)
        for doc_id, profile in profiles.items()
    }
    return profiles, priors


def text_signal_matches(text: str, query_signals: QuerySignals) -> Dict[str, object]:
    """Return the symbolic matches carried by a text block."""
    lowered = text.lower()
    entity_match = any(code in lowered for code in query_signals.codes)
    if not entity_match:
        entity_match = any(term in text for term in query_signals.entity_terms if not term.isascii())

    attribute_match = is_attribute_text(text, query_signals)
    summary_match = any(keyword in lowered for keyword in SUMMARY_KEYWORDS)
    value_validation = validate_answer(
        text,
        query_signals.answer_type,
        query_signals.units,
        query_signals.regex_hints,
        query_signals.normalization_rules,
    )

    bigram_hits = 0
    for bigram in query_signals.cn_bigrams:
        if bigram in text:
            bigram_hits += 1

    return {
        "entity_match": entity_match,
        "attribute_match": attribute_match,
        "summary_match": summary_match,
        "value_match": value_validation.is_valid,
        "value_validation": value_validation,
        "bigram_bonus": min(bigram_hits * 0.02, 0.08),
    }


def _combine_symbolic_features(
    entity_match: float,
    attribute_match: float,
    value_match: float,
    local_structure_score: float,
    bigram_bonus: float,
    ) -> float:
    return min(
        (0.32 * entity_match)
        + (0.24 * attribute_match)
        + (0.18 * value_match)
        + (0.18 * local_structure_score)
        + bigram_bonus,
        1.0,
    )


def _atom_trace_payload(atom) -> Dict[str, Any]:
    return {
        "object_id": atom.atom_id,
        "text": atom.text,
        "bbox": list(atom.bbox),
    }


def _build_evidence_trace(candidate_sets, triple) -> Dict[str, Any]:
    entity_atom = candidate_sets.atoms_by_id[triple.entity_atom_id]
    attribute_atom = candidate_sets.atoms_by_id[triple.attribute_atom_id]
    value_atom = candidate_sets.atoms_by_id[triple.value_atom_id]
    return {
        "cluster_id": triple.cluster_id,
        "entity_span": _atom_trace_payload(entity_atom),
        "attribute_span": _atom_trace_payload(attribute_atom),
        "value_span": _atom_trace_payload(value_atom),
        "normalized_value": str(triple.features.get("normalized_value", "") or ""),
        "match_reason": str(triple.features.get("match_reason", "") or ""),
        "contradiction_penalty": float(triple.features.get("contradiction_penalty", 0.0) or 0.0),
    }


def build_top_candidate_traces(evidences: Sequence[RegionEvidence], limit: int = 10) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    for evidence in sorted(evidences, key=lambda item: item.score, reverse=True)[:limit]:
        trace = evidence.trace or {}
        traces.append(
            {
                "object_id": evidence.object_id,
                "page_id": evidence.page_id,
                "doc_id": evidence.doc_id,
                "cluster_id": trace.get("cluster_id", evidence.cluster_id),
                "entity_span": trace.get("entity_span"),
                "attribute_span": trace.get("attribute_span"),
                "value_span": trace.get("value_span"),
                "normalized_value": trace.get("normalized_value", evidence.normalized_value),
                "entity_match": evidence.entity_match,
                "attribute_match": evidence.attribute_match,
                "value_match": evidence.value_match,
                "local_structure_score": evidence.local_structure_score,
                "scope_score": evidence.scope_score,
                "contradiction_penalties": {
                    "total": trace.get("contradiction_penalty", evidence.contradiction_penalty),
                },
                "doc_prior": evidence.doc_prior,
                "final_score": evidence.score,
                "validator_confidence": evidence.validator_confidence,
                "match_reason": trace.get("match_reason", evidence.match_reason),
            }
        )
    return traces


def compute_symbolic_bonus(
    evidence: RegionEvidence,
    spatial_index,
    query_signals: QuerySignals,
    page_objects_cache: Dict[str, Sequence[VectorObject]] | None = None,
) -> Tuple[float, bool]:
    """Compute bounded symbolic features, then combine them into a symbolic score."""
    obj = spatial_index.get_object(evidence.object_id)
    if obj is None:
        return (0.0, False)

    page_objects = (
        page_objects_cache.get(evidence.page_id, ())
        if page_objects_cache is not None
        else spatial_index.get_page_objects(evidence.page_id)
    )
    matches = text_signal_matches(obj.text or "", query_signals)
    value_validation = matches["value_validation"] if isinstance(matches["value_validation"], ValidationResult) else ValidationResult(False)
    entity_match = 1.0 if matches["entity_match"] else 0.0
    attribute_match = 1.0 if matches["attribute_match"] else 0.0
    value_match = value_validation.confidence if value_validation.is_valid else 0.0
    local_structure_score = 0.0
    scope_score = 0.0
    bigram_bonus = float(matches["bigram_bonus"])

    evidence.normalized_value = value_validation.normalized_value or ""
    evidence.validator_confidence = value_validation.confidence
    evidence.match_reason = value_validation.match_reason if value_validation.is_valid else ""

    candidate_sets = collect_candidate_sets(page_objects, query_signals)
    triples = rank_candidate_triples(candidate_sets, query_signals)
    for triple in triples:
        if evidence.object_id not in (triple.entity_atom_id, triple.attribute_atom_id, triple.value_atom_id):
            continue
        entity_match = max(entity_match, triple.features.get("entity_match", 0.0))
        attribute_match = max(attribute_match, triple.features.get("attribute_match", 0.0))
        value_match = max(value_match, triple.features.get("value_match", 0.0))
        local_structure_score = max(local_structure_score, triple.features.get("local_structure_score", 0.0))
        scope_score = max(scope_score, triple.features.get("scope_score", 0.0))
        evidence.cluster_id = triple.cluster_id
        evidence.normalized_value = str(triple.features.get("normalized_value", evidence.normalized_value) or "")
        evidence.validator_confidence = max(
            evidence.validator_confidence,
            float(triple.features.get("value_match", 0.0) or 0.0),
        )
        evidence.match_reason = str(triple.features.get("match_reason", evidence.match_reason) or "")
        evidence.contradiction_penalty = float(triple.features.get("contradiction_penalty", 0.0) or 0.0)
        evidence.trace = _build_evidence_trace(candidate_sets, triple)
        break

    if local_structure_score == 0.0 and evidence.object_id in candidate_sets.atoms_by_id:
        atom = candidate_sets.atoms_by_id[evidence.object_id]
        for cluster in candidate_sets.clusters:
            if evidence.object_id not in cluster.atom_ids:
                continue
            for other_atom_id in cluster.atom_ids:
                if other_atom_id == evidence.object_id:
                    continue
                other_atom = candidate_sets.atoms_by_id[other_atom_id]
                other_matches = text_signal_matches(other_atom.text, query_signals)
                if entity_match and other_matches["attribute_match"]:
                    local_structure_score = max(local_structure_score, soft_structure_score(atom, other_atom, same_cluster=True))
                if attribute_match and other_matches["entity_match"]:
                    local_structure_score = max(local_structure_score, soft_structure_score(atom, other_atom, same_cluster=True))
            if local_structure_score > 0.0:
                break

    if matches["summary_match"]:
        scope_score = max(scope_score, 0.2)

    symbolic_bonus = _combine_symbolic_features(
        entity_match=entity_match,
        attribute_match=attribute_match,
        value_match=value_match,
        local_structure_score=local_structure_score,
        bigram_bonus=bigram_bonus,
    )
    if evidence.extraction_method == "keyword_fallback" and (entity_match or attribute_match):
        symbolic_bonus = min(symbolic_bonus + 0.08, 1.0)

    evidence.entity_match = entity_match
    evidence.attribute_match = attribute_match
    evidence.value_match = value_match
    evidence.local_structure_score = local_structure_score
    evidence.scope_score = scope_score
    return (symbolic_bonus, bool(entity_match))


def expand_keyword_fallback_candidates(
    candidate_page_ids: Sequence[str],
    evidences: List[RegionEvidence],
    spatial_index,
    query_signals: QuerySignals,
    doc_priors: Dict[str, float],
) -> List[RegionEvidence]:
    """
    Expand recall when symbolic identifiers are missing from snapped evidence.

    Added candidates enter the same score contract as every other evidence:
    zero visual score, bounded symbolic bonus, and a soft inherited doc prior.
    """
    if not query_signals.codes and not query_signals.cn_segments:
        return evidences

    expanded = list(evidences)
    existing_ids = {evidence.object_id for evidence in evidences}

    for page_id in candidate_page_ids:
        for obj in spatial_index.get_page_objects(page_id):
            if obj.object_id in existing_ids or not obj.text:
                continue
            matches = text_signal_matches(obj.text, query_signals)
            if not matches["entity_match"] and not matches["attribute_match"]:
                continue

            value_validation = matches["value_validation"]

            expanded.append(
                RegionEvidence(
                    doc_id=obj.doc_id,
                    version_id=obj.version_id,
                    page_id=obj.page_id,
                    object_id=obj.object_id,
                    bbox=obj.bbox,
                    obj_type=obj.obj_type,
                    extraction_method="keyword_fallback",
                    score=0.0,
                    visual_score=0.0,
                    symbolic_bonus=0.0,
                    doc_prior=doc_priors.get(obj.doc_id, 0.0),
                    normalized_value=value_validation.normalized_value or "",
                    validator_confidence=value_validation.confidence,
                    match_reason=value_validation.match_reason if value_validation.is_valid else "",
                    hash="",
                )
            )
            existing_ids.add(obj.object_id)

    return expanded


def apply_route_scores(evidences: Sequence[RegionEvidence], weights: RouteWeights) -> None:
    """Combine frozen visual score, symbolic bonus, scope score, and document prior."""
    max_visual = max((evidence.visual_score for evidence in evidences), default=0.0)
    visual_anchor = max_visual if max_visual > 0 else 1.0

    for evidence in evidences:
        normalized_visual = evidence.visual_score / visual_anchor
        evidence.score = (
            (weights.visual * normalized_visual)
            + (weights.symbolic * evidence.symbolic_bonus)
            + (weights.scope * evidence.scope_score)
            + (weights.document * evidence.doc_prior)
        )


def summarize_document_support(
    evidences: Sequence[RegionEvidence],
) -> Dict[str, Dict[str, float]]:
    """Aggregate routed evidence into per-document support features."""
    support: Dict[str, Dict[str, float]] = {}
    for evidence in evidences:
        entry = support.setdefault(
            evidence.doc_id,
            {
                "visual_score": 0.0,
                "symbolic_bonus": 0.0,
                "scope_score": 0.0,
                "score": 0.0,
                "doc_prior": evidence.doc_prior,
            },
        )
        entry["visual_score"] = max(entry["visual_score"], evidence.visual_score)
        entry["symbolic_bonus"] = max(entry["symbolic_bonus"], evidence.symbolic_bonus)
        entry["scope_score"] = max(entry["scope_score"], evidence.scope_score)
        entry["score"] = max(entry["score"], evidence.score)
        entry["doc_prior"] = max(entry["doc_prior"], evidence.doc_prior)
    return support
