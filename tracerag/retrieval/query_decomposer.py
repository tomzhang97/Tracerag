from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml


QUERY_NOISE = (
    "是多少",
    "是什么",
    "有哪些",
    "多少",
    "什么",
    "请问",
    "一下",
    "的",
    "是",
    "有",
)


@dataclass(frozen=True)
class QueryDecomposition:
    entity_terms: Tuple[str, ...]
    canonical_attribute: str = ""
    attribute_aliases: Tuple[str, ...] = ()
    answer_type: str = ""
    units: Tuple[str, ...] = ()
    regex_hints: Tuple[str, ...] = ()
    normalization_rules: Tuple[str, ...] = ()


def _dedupe(values: Iterable[str]) -> Tuple[str, ...]:
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


@lru_cache(maxsize=1)
def load_attribute_registry() -> Dict[str, Dict[str, object]]:
    registry_path = Path(__file__).resolve().parent.parent / "config" / "attribute_registry.yaml"
    with open(registry_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _contains_alias(text: str, alias: str) -> bool:
    if alias.isascii():
        return alias.lower() in text.lower()
    return alias in text


def _match_attribute(query: str) -> Tuple[str, Dict[str, object]]:
    registry = load_attribute_registry()
    best_key = ""
    best_entry: Dict[str, object] = {}
    best_len = 0

    for key, entry in registry.items():
        aliases = entry.get("aliases", ()) or ()
        for alias in aliases:
            if _contains_alias(query, alias) and len(alias) > best_len:
                best_key = key
                best_entry = entry
                best_len = len(alias)

    return best_key, best_entry


def _strip_noise(text: str) -> str:
    cleaned = text
    for token in QUERY_NOISE:
        cleaned = cleaned.replace(token, "")
    return cleaned


def decompose_query(query: str, cn_segments: Iterable[str], codes: Iterable[str]) -> QueryDecomposition:
    canonical_attribute, entry = _match_attribute(query)
    aliases = tuple(entry.get("aliases", ()) or ())
    answer_type = str(entry.get("answer_type", "") or "")
    units = tuple(entry.get("allowed_units", entry.get("units", ())) or ())
    regex_hints = tuple(entry.get("regex_hints", ()) or ())
    normalization_rules = tuple(entry.get("normalization_rules", ()) or ())

    remainder = query
    for code in codes:
        remainder = re.sub(re.escape(code), " ", remainder, flags=re.IGNORECASE)
    for alias in aliases:
        remainder = re.sub(re.escape(alias), " ", remainder, flags=re.IGNORECASE)
    remainder = _strip_noise(remainder)

    entity_terms = list(codes)
    entity_terms.extend(segment for segment in cn_segments if segment and segment not in aliases)
    entity_terms.extend(
        segment
        for segment in re.findall(r"[\u4e00-\u9fff]{2,}", remainder)
        if segment and segment not in aliases
    )

    return QueryDecomposition(
        entity_terms=_dedupe(entity_terms),
        canonical_attribute=canonical_attribute,
        attribute_aliases=_dedupe(aliases),
        answer_type=answer_type,
        units=_dedupe(units),
        regex_hints=_dedupe(regex_hints),
        normalization_rules=_dedupe(normalization_rules),
    )
