"""
Document-level aggregation mode.

This route ranks whole documents using the same scoring contract as
standard retrieval:
- frozen visual support when available
- bounded symbolic support
- soft document priors from metadata
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from tracerag.common.types import RegionEvidence, VectorObject
from tracerag.retrieval.scoring import (
    QuerySignals,
    RouteWeights,
    build_document_profiles,
    compute_document_priors,
    DOC_TYPE_KEYWORDS,
    infer_target_doc_type,
    summarize_document_support,
)


def classify_aggregation_scope(query: str) -> str:
    """
    Classify whether a list query targets documents or in-document objects.
    """
    q = query.lower()
    doc_keywords = [
        "证书",
        "报告",
        "说明书",
        "检测",
        "文件",
        "手册",
        "图纸",
        "certificate",
        "report",
        "manual",
    ]
    if any(keyword in q for keyword in doc_keywords):
        return "document_list"
    return "object_list"


class DocumentAggregationProcessor:
    """Handle retrieval and deduplication for document-level collections."""

    def __init__(self, manifest: Dict[str, str], spatial_index):
        self.manifest = manifest
        self.spatial_index = spatial_index

    def retrieve_candidate_documents(
        self,
        all_evidences: List[RegionEvidence],
        query_signals: QuerySignals,
        route_weights: RouteWeights,
    ) -> List[Dict[str, Any]]:
        """
        Rank documents using the common score contract.

        Document mode is still document-first, but it can use routed
        evidence support as a reranker instead of bypassing visual scoring.
        """
        support = summarize_document_support(all_evidences)
        doc_ids = list(dict.fromkeys(list(self.manifest.keys()) + list(support.keys())))
        profiles, doc_priors = compute_document_priors(
            doc_ids=doc_ids,
            manifest=self.manifest,
            spatial_index=self.spatial_index,
            query_signals=query_signals,
        )

        max_visual = max((entry["visual_score"] for entry in support.values()), default=0.0)
        visual_anchor = max_visual if max_visual > 0 else 1.0
        target_doc_type = infer_target_doc_type(query_signals.raw_query)

        ranked_docs: List[Dict[str, Any]] = []
        for doc_id in doc_ids:
            profile = profiles.get(doc_id) or build_document_profiles([doc_id], self.manifest, self.spatial_index)[doc_id]
            support_entry = support.get(
                doc_id,
                {
                    "visual_score": 0.0,
                    "symbolic_bonus": 0.0,
                    "scope_score": 0.0,
                    "score": 0.0,
                    "doc_prior": doc_priors.get(doc_id, 0.0),
                },
            )
            normalized_visual = support_entry["visual_score"] / visual_anchor
            doc_score = (
                (route_weights.visual * normalized_visual)
                + (route_weights.symbolic * support_entry["symbolic_bonus"])
                + (route_weights.scope * support_entry["scope_score"])
                + (route_weights.document * doc_priors.get(doc_id, 0.0))
            )

            type_alignment = self._doc_type_alignment(profile.doc_type, target_doc_type)
            if target_doc_type:
                doc_score += 0.35 * type_alignment
                if profile.doc_type and profile.doc_type != target_doc_type:
                    doc_score -= 0.10

                # Suppress clearly mismatched docs unless they have unusually strong evidence.
                if (
                    profile.doc_type
                    and profile.doc_type != target_doc_type
                    and support_entry["symbolic_bonus"] < 0.20
                    and doc_priors.get(doc_id, 0.0) < 0.30
                ):
                    continue

                if not self._matches_target_type(profile, target_doc_type):
                    continue

            if doc_score <= 0:
                continue

            ranked_docs.append(
                {
                    "doc_id": doc_id,
                    "score": doc_score,
                    "profile": profile,
                    "doc_prior": doc_priors.get(doc_id, 0.0),
                    "visual_score": support_entry["visual_score"],
                    "symbolic_bonus": support_entry["symbolic_bonus"],
                    "scope_score": support_entry["scope_score"],
                    "type_alignment": type_alignment,
                }
            )

        ranked_docs.sort(key=lambda item: item["score"], reverse=True)
        return self._filter_to_collection_facet(
            ranked_docs=ranked_docs,
            profiles=profiles,
            doc_priors=doc_priors,
            support=support,
            visual_anchor=visual_anchor,
            route_weights=route_weights,
            target_doc_type=target_doc_type,
            query_signals=query_signals,
        )

    def _doc_type_alignment(self, profile_doc_type: str, target_doc_type: str) -> float:
        if not target_doc_type:
            return 0.0
        if profile_doc_type == target_doc_type:
            return 1.0
        if profile_doc_type:
            return 0.05
        return 0.35

    def _matches_target_type(self, profile, target_doc_type: str) -> bool:
        if not target_doc_type:
            return True
        if profile.doc_type == target_doc_type:
            return True
        keywords = DOC_TYPE_KEYWORDS.get(target_doc_type, ())
        haystacks = [profile.folder_name, profile.file_stem, profile.title_text, profile.path_text]
        for haystack in haystacks:
            lowered = (haystack or "").lower()
            if any(keyword.lower() in lowered for keyword in keywords):
                return True
        return False

    def _collection_key(self, candidate: Dict[str, Any]) -> str:
        profile = candidate["profile"]
        if profile.abs_path:
            try:
                return str(Path(profile.abs_path).parent).lower()
            except Exception:
                pass
        for folder in (profile.folder_name, *profile.ancestor_folders):
            if folder and any(token in folder for token in ("证书", "certificate", "certification")):
                return folder
        return profile.folder_name or (profile.ancestor_folders[0] if profile.ancestor_folders else "")

    def _filter_to_collection_facet(
        self,
        ranked_docs: List[Dict[str, Any]],
        profiles: Dict[str, Any],
        doc_priors: Dict[str, float],
        support: Dict[str, Dict[str, float]],
        visual_anchor: float,
        route_weights: RouteWeights,
        target_doc_type: str,
        query_signals: QuerySignals,
    ) -> List[Dict[str, Any]]:
        if not ranked_docs or not target_doc_type:
            return ranked_docs

        bucket_scores: Dict[str, float] = {}
        bucket_counts: Dict[str, int] = {}
        for candidate in ranked_docs:
            product_key = self._product_anchor(candidate["profile"], query_signals)
            if not product_key:
                continue
            bucket_scores[product_key] = bucket_scores.get(product_key, 0.0) + candidate["score"]
            bucket_counts[product_key] = bucket_counts.get(product_key, 0) + 1

        if not bucket_scores:
            return ranked_docs

        dominant_product = max(
            bucket_scores,
            key=lambda key: (bucket_counts.get(key, 0), bucket_scores[key]),
        )

        filtered_map = {
            candidate["doc_id"]: candidate
            for candidate in ranked_docs
            if self._same_product_root(candidate["profile"], dominant_product, query_signals)
            and self._matches_target_type(candidate["profile"], target_doc_type)
        }

        # Once we identify the dominant product root, bring in every matching
        # certificate-like document under that product tree, even without visual support.
        for doc_id, profile in profiles.items():
            if not self._same_product_root(profile, dominant_product, query_signals):
                continue
            if not self._matches_target_type(profile, target_doc_type):
                continue
            if doc_id in filtered_map:
                continue

            support_entry = support.get(
                doc_id,
                {
                    "visual_score": 0.0,
                    "symbolic_bonus": 0.0,
                    "score": 0.0,
                    "doc_prior": doc_priors.get(doc_id, 0.0),
                },
            )
            normalized_visual = support_entry["visual_score"] / visual_anchor
            doc_score = (
                (route_weights.visual * normalized_visual)
                + (route_weights.symbolic * support_entry["symbolic_bonus"])
                + (route_weights.document * doc_priors.get(doc_id, 0.0))
                + (0.35 * self._doc_type_alignment(profile.doc_type, target_doc_type))
            )
            if doc_score <= 0:
                continue

            filtered_map[doc_id] = {
                "doc_id": doc_id,
                "score": doc_score,
                "profile": profile,
                "doc_prior": doc_priors.get(doc_id, 0.0),
                "visual_score": support_entry["visual_score"],
                "symbolic_bonus": support_entry["symbolic_bonus"],
                "type_alignment": self._doc_type_alignment(profile.doc_type, target_doc_type),
            }

        filtered = sorted(
            filtered_map.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        filtered = [
            candidate
            for candidate in filtered
            if self._same_product_root(candidate["profile"], dominant_product, query_signals)
            and self._matches_target_type(candidate["profile"], target_doc_type)
        ]
        if len(filtered) >= 3:
            return filtered
        return ranked_docs

    def _product_anchor(self, profile, query_signals: QuerySignals) -> str:
        if profile.abs_path:
            abs_path = self._normalize_path(profile.abs_path)
            path = Path(profile.abs_path)
            for parent in reversed(path.parents):
                folder_name = parent.name.lower()
                if any(code in folder_name for code in query_signals.codes):
                    return self._normalize_path(str(parent))
            return self._normalize_path(str(path.parent))

        folders = list(profile.ancestor_folders) + [profile.folder_name]
        for code in query_signals.codes:
            for folder in folders:
                if folder and code in folder:
                    return folder
        return profile.ancestor_folders[0] if profile.ancestor_folders else profile.folder_name

    def _same_product_root(self, profile, dominant_root: str, query_signals: QuerySignals) -> bool:
        profile_root = self._product_anchor(profile, query_signals)
        if not profile_root or not dominant_root:
            return False
        if profile.abs_path and dominant_root.startswith("/"):
            normalized_path = self._normalize_path(profile.abs_path)
            root = dominant_root.rstrip("/")
            return normalized_path == root or normalized_path.startswith(root + "/")
        return profile_root == dominant_root

    def _normalize_path(self, path_str: str) -> str:
        return path_str.replace("\\", "/").rstrip("/").lower()

    def extract_document_identity(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract a stable document identity for display and deduplication.
        """
        doc_id = candidate["doc_id"]
        profile = candidate["profile"]
        short_title = self._short_title(profile)
        identity = {
            "doc_id": doc_id,
            "filename": profile.file_name or doc_id,
            "folder": profile.folder_name or "unknown",
            "extracted_title": short_title,
            "certificate_number": "",
            "score": candidate["score"],
            "doc_prior": candidate["doc_prior"],
            "visual_score": candidate["visual_score"],
            "symbolic_bonus": candidate["symbolic_bonus"],
        }

        page_zero_id = f"{doc_id}_v1_p0"
        try:
            page_zero_objects = self.spatial_index.get_page_objects(page_zero_id)
            for obj in page_zero_objects[:10]:
                text = obj.text or ""
                match = re.search(r"(证书编号|报告编号|编号|No\.|NO\.)[:：\s]*([A-Za-z0-9\-_]+)", text)
                if match:
                    identity["certificate_number"] = match.group(2)
                    break
        except Exception:
            pass

        return identity

    def _short_title(self, profile) -> str:
        stem = (profile.file_stem or "").strip()
        title = (profile.title_text or "").strip()

        for raw in (stem, title):
            if not raw:
                continue
            cleaned = re.sub(r"_task-[A-Za-z0-9]+$", "", raw)
            cleaned = re.sub(r"\bversion\s+[a-z0-9]+\b", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\b(shanghai mitsubishi|maintenance instruction|adjustment instruction)\b", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_")
            if not cleaned:
                continue
            if any(token in cleaned.lower() for token in ("<figure>", "<img", "image from ")):
                continue
            if len(cleaned) <= 80:
                return cleaned
            first_chunk = re.split(r"[，,:：;；]", cleaned, maxsplit=1)[0].strip()
            if 2 <= len(first_chunk) <= 80:
                return first_chunk

        return profile.file_stem or profile.title_text or profile.doc_id

    def aggregate_documents(self, identities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate documents by title and explicit certificate/report number.
        """
        unique_docs: List[Dict[str, Any]] = []
        seen_titles: List[str] = []
        seen_numbers = set()
        seen_base_keys = set()

        for identity in sorted(identities, key=lambda item: item["score"], reverse=True):
            title = identity["extracted_title"]
            number = identity["certificate_number"]
            norm_title = re.sub(r"[\s\n\(\)（）_]+", "", title.lower())
            base_key = self._canonical_document_key(identity)

            is_duplicate = False
            if number and number in seen_numbers:
                is_duplicate = True
            if base_key and base_key in seen_base_keys:
                is_duplicate = True

            if not is_duplicate:
                for seen in seen_titles:
                    if difflib.SequenceMatcher(None, norm_title, seen).ratio() > 0.85:
                        is_duplicate = True
                        break

            if is_duplicate:
                continue

            seen_titles.append(norm_title)
            if number:
                seen_numbers.add(number)
            if base_key:
                seen_base_keys.add(base_key)
            unique_docs.append(identity)

        return unique_docs

    def _canonical_document_key(self, identity: Dict[str, Any]) -> str:
        filename = (identity.get("filename") or identity.get("doc_id") or "").lower()
        filename = re.sub(r"\.[a-z0-9]+$", "", filename)
        filename = re.sub(r"_task-[a-z0-9]+$", "", filename)
        filename = re.sub(r"[\s_\-()（）]+", "", filename)
        return filename

    def process(
        self,
        query: str,
        all_evidences: List[RegionEvidence],
        query_signals: QuerySignals,
        route_weights: RouteWeights,
    ) -> List[RegionEvidence]:
        """Produce synthetic RegionEvidence items representing documents."""
        logger.info(f"Triggering document-level aggregation for: {query}")

        candidates = self.retrieve_candidate_documents(all_evidences, query_signals, route_weights)
        logger.info(f"Found {len(candidates)} candidate documents using shared score components.")

        identities = [self.extract_document_identity(candidate) for candidate in candidates[:50]]
        unique_docs = self.aggregate_documents(identities)
        logger.info(f"Deduplicated down to {len(unique_docs)} unique document identities.")

        evidences: List[RegionEvidence] = []
        for rank, doc in enumerate(unique_docs):
            display_text = f"[{doc['folder']}] {doc['extracted_title']}"
            if doc["certificate_number"]:
                display_text += f" (No: {doc['certificate_number']})"

            synthetic_object_id = f"doc_agg_{doc['doc_id']}"
            page_id = f"{doc['doc_id']}_v1_p0"
            synthetic_evidence = RegionEvidence(
                doc_id=doc["doc_id"],
                version_id="v1",
                page_id=page_id,
                object_id=synthetic_object_id,
                bbox=(0.0, 0.0, 0.0, 0.0),
                obj_type="document_identity",
                extraction_method="document_aggregation",
                score=max(doc["score"] - (rank * 1e-6), 0.0),
                visual_score=doc["visual_score"],
                symbolic_bonus=doc["symbolic_bonus"],
                doc_prior=doc["doc_prior"],
                hash="",
            )

            if self.spatial_index.get_object(synthetic_object_id) is None:
                dummy_obj = VectorObject(
                    object_id=synthetic_object_id,
                    doc_id=doc["doc_id"],
                    version_id="v1",
                    page_id=page_id,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    obj_type="document_identity",
                    text=display_text,
                    meta={
                        "synthetic": True,
                        "folder": doc["folder"],
                        "title": doc["extracted_title"],
                        "certificate_number": doc["certificate_number"],
                    },
                )
                self.spatial_index.add_object(dummy_obj)

            evidences.append(synthetic_evidence)

        return evidences
