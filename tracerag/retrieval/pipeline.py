"""
TraceRAG main retrieval pipeline.

Orchestrates the complete query flow:
1. Load candidate pages
2. Score patches
3. Snap to vector objects
4. Compose stable score components
5. Route and assemble an evidence pack
6. Optionally synthesize an answer
"""

from __future__ import annotations

import time
import re
from typing import Any, Dict, List, Optional

from loguru import logger

from tracerag.alignment.snapper import (
    ParsedPageObjects,
    PatchRelevanceMap,
    SnapperConfig,
    snap_page_relevance_to_objects,
)
from tracerag.common.types import CertifiedClaim, Claim, PatchGrid, RegionEvidence
from tracerag.retrieval.candidate_filter import CandidateFilter
from tracerag.retrieval.evidence_pack import EvidencePack
from tracerag.retrieval.llm_answerer import LLMAnswerer
from tracerag.retrieval.scoring import (
    RouteWeights,
    apply_route_scores,
    build_top_candidate_traces,
    compute_document_priors,
    compute_symbolic_bonus,
    expand_keyword_fallback_candidates,
    extract_query_signals,
)
from tracerag.structural.index import PdfSpatialIndex
from tracerag.visual.scorer import VisualScorer


class PatchGridStore:
    """Simple in-memory store for PatchGrids."""

    def __init__(self):
        self.grids: Dict[str, PatchGrid] = {}

    def add(self, patch_grid: PatchGrid):
        self.grids[patch_grid.page_id] = patch_grid

    def get(self, page_id: str) -> Optional[PatchGrid]:
        return self.grids.get(page_id)

    def get_batch(self, page_ids: List[str]) -> List[PatchGrid]:
        return [self.grids[page_id] for page_id in page_ids if page_id in self.grids]


class TraceRAGSystem:
    """Main TraceRAG retrieval system."""

    def __init__(
        self,
        config: Dict[str, Any],
        candidate_filter: CandidateFilter,
        patch_grid_store: PatchGridStore,
        spatial_index: PdfSpatialIndex,
        visual_scorer: VisualScorer,
        llm_answerer: Optional[LLMAnswerer] = None,
        manifest: Optional[Dict[str, str]] = None,
    ):
        self.config = config
        self.candidate_filter = candidate_filter
        self.patch_grid_store = patch_grid_store
        self.spatial_index = spatial_index
        self.visual_scorer = visual_scorer
        self.llm_answerer = llm_answerer
        self.manifest = manifest or {}

        self.snapper_config = SnapperConfig(**config.get("snapper", {}))
        logger.info("TraceRAG system initialized")

    def answer(self, query: str) -> EvidencePack:
        """Execute the retrieval pipeline for a query."""
        from tracerag.retrieval.aggregation import AggregationProcessor
        from tracerag.retrieval.document_aggregation import (
            DocumentAggregationProcessor,
            classify_aggregation_scope,
        )

        start_time = time.time()
        logger.info(f"Processing query: {query}")

        agg_processor = AggregationProcessor()
        is_agg = agg_processor.is_aggregation_query(query)
        agg_scope = classify_aggregation_scope(query) if is_agg else None
        query_signals = extract_query_signals(query)

        # Stage 1: Candidate pages
        candidate_page_ids = self.candidate_filter.get_candidate_pages(query)
        logger.info(f"Retrieved {len(candidate_page_ids)} candidate pages")

        # Stage 2: Visual page scoring
        patch_grids = self.patch_grid_store.get_batch(candidate_page_ids)
        scored_pages = self.visual_scorer.score_pages_batch(query, patch_grids)

        # Stage 3: Snap page relevance to objects and freeze the visual score.
        all_evidences = []
        for page_id, _, patch_scores in scored_pages:
            patch_grid = self.patch_grid_store.get(page_id)
            if patch_grid is None:
                continue

            relevance_map = PatchRelevanceMap(
                page_id=page_id,
                grid_h=patch_grid.H,
                grid_w=patch_grid.W,
                scores=patch_scores,
                patch_boxes=patch_grid.patch_boxes,
            )
            page_objects = ParsedPageObjects(
                page_id=page_id,
                objects=self.spatial_index.get_page_objects(page_id),
                spatial_index=self.spatial_index,
            )

            evidences = snap_page_relevance_to_objects(relevance_map, page_objects, self.snapper_config)
            for evidence in evidences:
                evidence.visual_score = evidence.score
            all_evidences.extend(evidences)

        # Stage 3.5: Independent document and symbolic features.
        evidence_doc_ids = [evidence.doc_id for evidence in all_evidences]
        profile_doc_ids = evidence_doc_ids or list(self.manifest.keys())
        _, doc_priors = compute_document_priors(
            doc_ids=profile_doc_ids,
            manifest=self.manifest,
            spatial_index=self.spatial_index,
            query_signals=query_signals,
        )
        for evidence in all_evidences:
            evidence.doc_prior = doc_priors.get(evidence.doc_id, 0.0)

        page_objects_cache = {
            page_id: self.spatial_index.get_page_objects(page_id)
            for page_id in candidate_page_ids
        }
        matched_entity = False
        for evidence in all_evidences:
            evidence.symbolic_bonus, entity_match = compute_symbolic_bonus(
                evidence,
                self.spatial_index,
                query_signals,
                page_objects_cache=page_objects_cache,
            )
            matched_entity = matched_entity or entity_match

        # Stage 3.6: Expand recall candidates, but let the common scorer rank them.
        if (query_signals.codes or query_signals.cn_segments) and not matched_entity:
            logger.info("No snapped evidence carried direct identifier matches. Expanding fallback candidates.")
            all_evidences = expand_keyword_fallback_candidates(
                candidate_page_ids=candidate_page_ids,
                evidences=all_evidences,
                spatial_index=self.spatial_index,
                query_signals=query_signals,
                doc_priors=doc_priors,
            )
            for evidence in all_evidences:
                if evidence.extraction_method != "keyword_fallback" or evidence.symbolic_bonus > 0:
                    continue
                evidence.symbolic_bonus, _ = compute_symbolic_bonus(
                    evidence,
                    self.spatial_index,
                    query_signals,
                    page_objects_cache=page_objects_cache,
                )

        # Stage 3.7: Route-specific composition over stable components only.
        if is_agg and agg_scope == "document_list":
            route_weights = RouteWeights(visual=0.25, symbolic=0.25, scope=0.35, document=0.90)
            route_name = "document_aggregation"
        elif is_agg:
            route_weights = RouteWeights(visual=0.60, symbolic=0.70, scope=0.12, document=0.25)
            route_name = "object_aggregation"
        else:
            route_weights = RouteWeights(visual=1.00, symbolic=0.45, scope=0.10, document=0.15)
            route_name = "standard_qa"

        apply_route_scores(all_evidences, route_weights)
        ranked_trace_candidates = sorted(all_evidences, key=lambda item: item.score, reverse=True)
        logger.info(
            "Route: {} (visual={}, symbolic={}, scope={}, document={})".format(
                route_name,
                route_weights.visual,
                route_weights.symbolic,
                route_weights.scope,
                route_weights.document,
            )
        )
        for evidence in sorted(all_evidences, key=lambda item: item.score, reverse=True)[:3]:
            logger.info(
                "  Top evidence: {} | final={:.3f} (vis={:.3f}, sym={:.3f}, scope={:.3f}, doc={:.3f}, entity={:.2f}, attr={:.2f}, value={:.2f}, local={:.2f})".format(
                    evidence.object_id,
                    evidence.score,
                    evidence.visual_score,
                    evidence.symbolic_bonus,
                    evidence.scope_score,
                    evidence.doc_prior,
                    evidence.entity_match,
                    evidence.attribute_match,
                    evidence.value_match,
                    evidence.local_structure_score,
                )
            )

        # Stage 4: Route-specific assembly.
        if is_agg and agg_scope == "document_list":
            logger.info("Routing query to document-level aggregation mode")
            doc_agg_processor = DocumentAggregationProcessor(self.manifest, self.spatial_index)
            all_evidences = doc_agg_processor.process(
                query=query,
                all_evidences=all_evidences,
                query_signals=query_signals,
                route_weights=route_weights,
            )
        elif is_agg:
            logger.info("Routing query to object-level aggregation mode")
            all_evidences = agg_processor.process(query, all_evidences, self.spatial_index)
        else:
            ranked_evidences = sorted(all_evidences, key=lambda evidence: evidence.score, reverse=True)
            all_evidences = self._expand_summary_context(
                query=query,
                ranked_evidences=ranked_evidences,
                doc_priors=doc_priors,
                query_signals=query_signals,
            )

        pack = EvidencePack(query=query, evidences=all_evidences)
        pack.metadata["route_name"] = route_name
        pack.metadata["query_type"] = (
            "document_aggregation" if route_name == "document_aggregation"
            else "aggregation" if route_name == "object_aggregation"
            else "general"
        )
        pack.metadata["top_candidate_traces"] = build_top_candidate_traces(ranked_trace_candidates, limit=10)
        pack.metadata["trace_summary"] = {
            "num_ranked_candidates": len(ranked_trace_candidates),
            "num_traced_candidates": len(pack.metadata["top_candidate_traces"]),
            "max_contradiction_penalty": max(
                (trace["contradiction_penalties"]["total"] for trace in pack.metadata["top_candidate_traces"]),
                default=0.0,
            ),
        }

        # Stage 5: Optional answer synthesis.
        if self.llm_answerer:
            self._synthesize_answer(pack)

        pack.metadata["elapsed_time"] = time.time() - start_time
        pack.metadata["num_candidate_pages"] = len(candidate_page_ids)
        logger.info(f"Query answered in {pack.metadata['elapsed_time']:.2f}s")
        return pack

    def _synthesize_answer(self, pack: EvidencePack):
        """Populate the EvidencePack with an LLM-synthesized answer."""
        if pack.metadata.get("query_type") == "document_aggregation":
            self._synthesize_document_aggregation_answer(pack)
            return

        evidence_texts = {}
        for evidence in pack.evidences:
            obj = self.spatial_index.get_object(evidence.object_id)
            if obj and obj.text:
                evidence_texts[evidence.object_id] = obj.text

        if not evidence_texts:
            pack.answer = "No relevant text information found in the identified evidence regions."
            return

        llm_result = self.llm_answerer.generate_answer_with_claims(
            query=pack.query,
            evidences=pack.evidences,
            evidence_texts=evidence_texts,
            query_type=pack.metadata.get("query_type", "general"),
        )

        pack.answer = llm_result.get("answer", "")
        claims_data = llm_result.get("claims", [])
        pack.certified_claims = self.llm_answerer.map_claims_to_evidences(claims_data, pack.evidences)

    def _synthesize_document_aggregation_answer(self, pack: EvidencePack):
        """Deterministically enumerate document identities for list queries."""
        items = []
        claims = []

        for evidence in pack.evidences:
            obj = self.spatial_index.get_object(evidence.object_id)
            if obj is None:
                continue

            meta = obj.meta or {}
            title = (meta.get("title") or obj.text or evidence.object_id).strip()
            folder = (meta.get("folder") or "").strip()
            certificate_number = (meta.get("certificate_number") or "").strip()

            if obj.text and not meta.get("title"):
                title = self._parse_document_identity_text(obj.text)
                number_from_text = self._parse_certificate_number(obj.text)
                if number_from_text and not certificate_number:
                    certificate_number = number_from_text

            items.append(
                {
                    "title": title,
                    "folder": folder,
                    "certificate_number": certificate_number,
                    "evidence": evidence,
                }
            )

        if not items:
            pack.answer = "No relevant document identities found."
            pack.certified_claims = []
            return

        lines = [f"共找到{len(items)}份相关文档："]
        for index, item in enumerate(items, start=1):
            line = f"{index}. {item['title']}"
            lines.append(line)

            claims.append(
                CertifiedClaim(
                    claim=Claim(
                        text=item["title"],
                        value=item["certificate_number"] or None,
                        entity_id=item["title"],
                        claim_type="general",
                    ),
                    evidences=[item["evidence"]],
                    confidence=item["evidence"].score,
                    reasoning="Document aggregation evidence",
                )
            )

        pack.answer = "\n".join(lines)
        pack.certified_claims = claims

    def _parse_document_identity_text(self, text: str) -> str:
        cleaned = re.sub(r"^\[[^\]]+\]\s*", "", text).strip()
        cleaned = re.sub(r"\s*\(No:\s*[^)]+\)\s*$", "", cleaned).strip()
        return cleaned or text.strip()

    def _parse_certificate_number(self, text: str) -> str:
        match = re.search(r"\(No:\s*([^)]+)\)", text)
        if match:
            return match.group(1).strip()
        match = re.search(r"(证书编号|报告编号|编号)[:：\s]*([A-Za-z0-9\-_]+)", text)
        if match:
            return match.group(2).strip()
        return ""

    def _expand_summary_context(
        self,
        query: str,
        ranked_evidences,
        doc_priors,
        query_signals,
        limit: int = 50,
    ):
        """
        Expand standard QA evidence with page-local context for total/sum queries.

        This is an evidence-pack assembly step, not a score mutation step.
        It keeps the strongest ranked seeds, then adds sibling text objects from
        the top summary page so line items and the final total row reach the LLM
        together.
        """
        if not ranked_evidences:
            return []

        lowered = query.lower()
        summary_query = any(
            keyword in lowered
            for keyword in ("合计", "总计", "总毛重", "总净重", "total", "sum", "subtotal")
        )
        if not summary_query:
            return ranked_evidences[:limit]

        selected = []
        seen_ids = set()
        seeds = ranked_evidences[: min(12, limit)]
        for evidence in seeds:
            selected.append(evidence)
            seen_ids.add(evidence.object_id)

        page_order = []
        seen_pages = set()
        for evidence in seeds:
            if evidence.page_id in seen_pages:
                continue
            seen_pages.add(evidence.page_id)
            page_order.append(evidence.page_id)

        context_score = max(min((evidence.score for evidence in seeds), default=0.0) - 1e-4, 0.0)
        for page_id in page_order[:2]:
            page_objects = sorted(
                self.spatial_index.get_page_objects(page_id),
                key=lambda obj: (obj.bbox[1], obj.bbox[0], obj.object_id),
            )
            for obj in page_objects:
                if len(selected) >= limit:
                    return selected[:limit]
                if obj.object_id in seen_ids or not obj.text or not self._is_summary_context_object(obj.text, query_signals):
                    continue

                context_evidence = RegionEvidence(
                    doc_id=obj.doc_id,
                    version_id=obj.version_id,
                    page_id=obj.page_id,
                    object_id=obj.object_id,
                    bbox=obj.bbox,
                    obj_type=obj.obj_type,
                    extraction_method="page_context",
                    score=context_score,
                    visual_score=0.0,
                    symbolic_bonus=0.0,
                    doc_prior=doc_priors.get(obj.doc_id, 0.0),
                    hash="",
                )
                context_evidence.symbolic_bonus, _ = compute_symbolic_bonus(
                    context_evidence,
                    self.spatial_index,
                    query_signals,
                )
                selected.append(context_evidence)
                seen_ids.add(obj.object_id)
                context_score = max(context_score - 1e-6, 0.0)

        for evidence in ranked_evidences:
            if len(selected) >= limit:
                break
            if evidence.object_id in seen_ids:
                continue
            selected.append(evidence)
            seen_ids.add(evidence.object_id)

        return selected[:limit]

    def _is_summary_context_object(self, text: str, query_signals) -> bool:
        stripped = text.strip()
        if not stripped:
            return False

        lowered = stripped.lower()
        if any(keyword in lowered for keyword in ("合计", "总计", "总毛重", "总净重", "total", "sum", "subtotal", "kg", "重量", "毛重", "净重")):
            return True
        if any(code in lowered for code in query_signals.codes):
            return True
        if any(segment in stripped for segment in query_signals.cn_segments):
            return True
        if any(char.isdigit() for char in stripped):
            return len(stripped) <= 120
        return len(stripped) <= 60
