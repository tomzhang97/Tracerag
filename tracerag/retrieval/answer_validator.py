from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence


NUMBER_RE = re.compile(r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")
IDENTIFIER_RE = re.compile(r"\b[A-Z0-9]{2,}(?:[-_/][A-Z0-9]{1,})+\b", re.IGNORECASE)


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    normalized_value: str | None = None
    confidence: float = 0.0
    match_reason: str = "no_match"
    raw_match: str | None = None


def _invalid(reason: str = "no_match") -> ValidationResult:
    return ValidationResult(is_valid=False, confidence=0.0, match_reason=reason)


def _normalize_unit(unit: str) -> str:
    unit = unicodedata.normalize("NFKC", unit or "").strip()
    unit = unit.replace("° C", "°C").replace("℃", "°C")
    return unit.lower()


def _format_number(number_text: str) -> str:
    number = unicodedata.normalize("NFKC", number_text or "").replace(",", "").strip()
    if not number:
        return ""
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number


def _apply_text_rules(text: str, rules: Sequence[str]) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    for rule in rules:
        if rule == "normalize_fullwidth":
            normalized = unicodedata.normalize("NFKC", normalized)
        elif rule == "strip_spaces":
            normalized = normalized.replace(" ", "")
        elif rule == "collapse_spaces":
            normalized = re.sub(r"\s+", " ", normalized).strip()
        elif rule == "strip_commas":
            normalized = normalized.replace(",", "")
        elif rule == "lowercase":
            normalized = normalized.lower()
        elif rule == "uppercase":
            normalized = normalized.upper()
        elif rule == "normalize_unit_spacing":
            normalized = re.sub(r"\s+", " ", normalized).strip()
            normalized = re.sub(r"\s+(?=[A-Za-z°/%])", " ", normalized)
        elif rule == "normalize_range_dash":
            normalized = re.sub(r"\s*(?:to|TO|~|～|至|—|–|-)\s*", "-", normalized)
        elif rule == "normalize_phi_prefix":
            normalized = normalized.replace("φ", "Φ").replace("Φ ", "Φ")
        elif rule == "iso_date":
            normalized = normalized.replace("年", "-").replace("月", "-").replace("日", "")
    return normalized.strip()


def _best_regex_match(text: str, regex_hints: Sequence[str]) -> str | None:
    best_match = None
    best_length = -1
    for hint in regex_hints:
        try:
            match = re.search(hint, text)
        except re.error:
            continue
        if not match:
            continue
        matched = match.group(0).strip()
        if len(matched) > best_length:
            best_match = matched
            best_length = len(matched)
    return best_match


def _candidate_numbers(text: str) -> list[tuple[str, int, int]]:
    candidates: list[tuple[str, int, int]] = []
    for match in NUMBER_RE.finditer(text):
        candidates.append((match.group(0), match.start(), match.end()))
    return candidates


def _validate_numeric_with_unit(
    text: str,
    units: Sequence[str],
    regex_hints: Sequence[str],
    rules: Sequence[str],
) -> ValidationResult:
    match = _best_regex_match(text, regex_hints)
    allowed_units = tuple(_normalize_unit(unit) for unit in units if unit)

    if match:
        phi_match = re.search(r"[Φφ]\s*([-+]?\d+(?:\.\d+)?)", match)
        if phi_match:
            number = _format_number(phi_match.group(1))
            normalized = f"{number} {allowed_units[0]}".strip() if allowed_units else number
            normalized = _apply_text_rules(normalized, rules)
            return ValidationResult(True, normalized, 0.99, "matched_phi_numeric_with_unit", match)

        value_match = re.search(r"([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*([A-Za-z°/%]+|℃)", match)
        if value_match:
            number = _format_number(value_match.group(1))
            unit = _normalize_unit(value_match.group(2))
            if allowed_units and unit not in allowed_units:
                return _invalid("unit_not_allowed")
            normalized = _apply_text_rules(f"{number} {unit}", rules)
            return ValidationResult(True, normalized, 0.98, "matched_numeric_with_allowed_unit", match)

    phi_match = re.search(r"[Φφ]\s*([-+]?\d+(?:\.\d+)?)", text)
    if phi_match:
        number = _format_number(phi_match.group(1))
        normalized = f"{number} {allowed_units[0]}".strip() if allowed_units else number
        normalized = _apply_text_rules(normalized, rules)
        return ValidationResult(True, normalized, 0.97, "matched_phi_numeric_with_unit", phi_match.group(0).strip())

    for number_text, _, end in _candidate_numbers(text):
        tail = text[end:].lstrip()
        unit_match = re.match(r"([A-Za-z°/%]+|℃)", tail)
        if not unit_match:
            continue
        unit = _normalize_unit(unit_match.group(1))
        if allowed_units and unit not in allowed_units:
            continue
        normalized = _apply_text_rules(f"{_format_number(number_text)} {unit}", rules)
        raw = f"{number_text}{tail[: len(unit_match.group(1))]}"
        return ValidationResult(True, normalized, 0.95, "matched_numeric_unit_suffix", raw.strip())

    return _invalid("no_numeric_with_unit_match")


def _validate_numeric_plain(text: str, regex_hints: Sequence[str], rules: Sequence[str]) -> ValidationResult:
    match = _best_regex_match(text, regex_hints)
    if match:
        normalized = _apply_text_rules(_format_number(match), rules)
        return ValidationResult(True, normalized, 0.93, "matched_numeric_plain_hint", match)

    for number_text, _, end in _candidate_numbers(text):
        tail = text[end:].lstrip()
        if tail and re.match(r"[A-Za-z\u4e00-\u9fff°/%]", tail[0]):
            continue
        normalized = _apply_text_rules(_format_number(number_text), rules)
        return ValidationResult(True, normalized, 0.88, "matched_numeric_plain", number_text)

    return _invalid("no_numeric_plain_match")


def _validate_range_with_unit(
    text: str,
    units: Sequence[str],
    regex_hints: Sequence[str],
    rules: Sequence[str],
) -> ValidationResult:
    match = _best_regex_match(text, regex_hints)
    candidate_text = match or text
    separator_pattern = r"(?:to|TO|~|～|至|[-\u2013\u2014])"
    range_match = re.search(
        rf"([-+]?\d+(?:,\d{{3}})*(?:\.\d+)?)\s*{separator_pattern}\s*([-+]?\d+(?:,\d{{3}})*(?:\.\d+)?)\s*([A-Za-z°/%]+|℃)",
        candidate_text,
    )
    if not range_match:
        return _invalid("no_range_with_unit_match")

    allowed_units = tuple(_normalize_unit(unit) for unit in units if unit)
    unit = _normalize_unit(range_match.group(3))
    if allowed_units and unit not in allowed_units:
        return _invalid("unit_not_allowed")

    start_value = _format_number(range_match.group(1))
    end_value = _format_number(range_match.group(2))
    normalized = _apply_text_rules(f"{start_value}-{end_value} {unit}", rules)
    confidence = 0.97 if match else 0.92
    reason = "matched_range_with_unit_hint" if match else "matched_range_with_unit"
    return ValidationResult(True, normalized, confidence, reason, range_match.group(0).strip())


def _validate_identifier(text: str, regex_hints: Sequence[str], rules: Sequence[str], answer_type: str) -> ValidationResult:
    match = _best_regex_match(text, regex_hints)
    if not match:
        regex_match = IDENTIFIER_RE.search(unicodedata.normalize("NFKC", text or ""))
        match = regex_match.group(0) if regex_match else None
    if not match:
        return _invalid(f"no_{answer_type}_match")

    normalized = _apply_text_rules(match, rules)
    confidence = 0.97 if regex_hints else 0.9
    reason = f"matched_{answer_type}_pattern"
    return ValidationResult(True, normalized, confidence, reason, match)


def _parse_date(date_text: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", date_text or "").strip()
    normalized = normalized.replace("年", "-").replace("月", "-").replace("日", "")
    normalized = normalized.replace(".", "-").replace("/", "-")
    parts = [part for part in normalized.split("-") if part]
    if len(parts) != 3:
        return None
    try:
        year, month, day = (int(part) for part in parts)
        return datetime(year=year, month=month, day=day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _validate_date(text: str, regex_hints: Sequence[str], rules: Sequence[str]) -> ValidationResult:
    match = _best_regex_match(text, regex_hints)
    if not match:
        match = _best_regex_match(
            text,
            (
                r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}",
                r"\d{4}年\d{1,2}月\d{1,2}日",
            ),
        )
    if not match:
        return _invalid("no_date_match")

    iso_value = _parse_date(match)
    if not iso_value:
        return _invalid("invalid_date")
    normalized = _apply_text_rules(iso_value, rules)
    confidence = 0.98 if regex_hints else 0.94
    return ValidationResult(True, normalized, confidence, "matched_date", match)


def _validate_enum(text: str, regex_hints: Sequence[str], rules: Sequence[str]) -> ValidationResult:
    normalized_text = unicodedata.normalize("NFKC", text or "")
    for hint in regex_hints:
        try:
            match = re.search(hint, normalized_text)
        except re.error:
            continue
        if not match:
            continue
        matched = match.group(0).strip()
        normalized = _apply_text_rules(matched, rules)
        return ValidationResult(True, normalized, 0.9, "matched_enum_choice", matched)
    return _invalid("no_enum_match")


def validate_answer(
    text: str,
    answer_type: str,
    units: Iterable[str] = (),
    regex_hints: Iterable[str] = (),
    normalization_rules: Iterable[str] = (),
) -> ValidationResult:
    if not text or not answer_type:
        return _invalid("missing_text_or_answer_type")

    answer_types = tuple(part.strip() for part in answer_type.split("/") if part.strip())
    rules = tuple(normalization_rules or ())
    allowed_units = tuple(units or ())
    hint_list = tuple(regex_hints or ())
    normalized_text = _apply_text_rules(text, ("normalize_fullwidth",))

    for candidate_type in answer_types:
        if candidate_type == "numeric_with_unit":
            result = _validate_numeric_with_unit(normalized_text, allowed_units, hint_list, rules)
        elif candidate_type == "numeric_plain":
            result = _validate_numeric_plain(normalized_text, hint_list, rules)
        elif candidate_type == "range_with_unit":
            result = _validate_range_with_unit(normalized_text, allowed_units, hint_list, rules)
        elif candidate_type in {"identifier", "doc_id", "certificate_id"}:
            result = _validate_identifier(normalized_text, hint_list, rules, candidate_type)
        elif candidate_type == "date":
            result = _validate_date(normalized_text, hint_list, rules)
        elif candidate_type == "enum":
            result = _validate_enum(normalized_text, hint_list, rules)
        else:
            result = _invalid(f"unsupported_answer_type:{candidate_type}")

        if result.is_valid:
            return result

    return _invalid("no_match")


def extract_validated_value(
    text: str,
    answer_type: str,
    units: Iterable[str] = (),
    regex_hints: Iterable[str] = (),
    normalization_rules: Iterable[str] = (),
) -> str | None:
    result = validate_answer(
        text=text,
        answer_type=answer_type,
        units=units,
        regex_hints=regex_hints,
        normalization_rules=normalization_rules,
    )
    if result.is_valid:
        return result.normalized_value
    return None
