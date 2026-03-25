"""
TraceRAG main retrieval pipeline.

Orchestrates the complete query flow:
1. Load Candidate Pages
2. Score Patches (Visual Encoding & Interaction)
3. Snap To Objects (Vector-Native Alignment)
4. Assemble Evidence Pack
5. (Optional) Generate Answer Synthesis
"""

import time
from typing import Dict, Any, List, Optional
from loguru import logger
from tracerag.common.types import RegionEvidence

from tracerag.common.types import PatchGrid, VectorObject
from tracerag.alignment.snapper import SnapperConfig, PatchRelevanceMap, ParsedPageObjects, snap_page_relevance_to_objects
from tracerag.retrieval.candidate_filter import CandidateFilter
from tracerag.retrieval.evidence_pack import EvidencePack
from tracerag.retrieval.llm_answerer import LLMAnswerer
from tracerag.visual.scorer import VisualScorer
from tracerag.structural.index import PdfSpatialIndex


class PatchGridStore:
    """Simple in-memory store for PatchGrids."""
    def __init__(self):
        self.grids: Dict[str, PatchGrid] = {}
    def add(self, patch_grid: PatchGrid):
        self.grids[patch_grid.page_id] = patch_grid
    def get(self, page_id: str) -> Optional[PatchGrid]:
        return self.grids.get(page_id)
    def get_batch(self, page_ids: List[str]) -> List[PatchGrid]:
        return [self.grids[pid] for pid in page_ids if pid in self.grids]


class TraceRAGSystem:
    """Main TraceRAG retrieval system. Combines all components into a flattened pipeline."""

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
        """Main entry point: execute the full retrieval pipeline linearly."""
        start_time = time.time()
        logger.info(f"Processing query: {query}")

        # Stage 1: Load Candidate Pages
        candidate_page_ids = self.candidate_filter.get_candidate_pages(query)
        logger.info(f"Retrieved {len(candidate_page_ids)} candidate pages")

        # Stage 2: Score Patches
        patch_grids = self.patch_grid_store.get_batch(candidate_page_ids)
        scored_pages = self.visual_scorer.score_pages_batch(query, patch_grids)

        # Stage 3: Snap to Objects (Vector Alignment)
        all_evidences = []
        for page_id, _, patch_scores in scored_pages:
            patch_grid = self.patch_grid_store.get(page_id)
            if not patch_grid: continue

            # Construct inputs for pure function Snapper
            relevance_map = PatchRelevanceMap(
                page_id=page_id,
                grid_h=patch_grid.H,
                grid_w=patch_grid.W,
                scores=patch_scores,
                patch_boxes=patch_grid.patch_boxes
            )
            
            # Since we just have a global spatial index for now, we provide it.
            # In a distributed system we'd pull page-specific objects.
            page_objects = ParsedPageObjects(
                page_id=page_id,
                objects=self.spatial_index.get_page_objects(page_id),
                spatial_index=self.spatial_index
            )

            evidences = snap_page_relevance_to_objects(relevance_map, page_objects, self.snapper_config)
            all_evidences.extend(evidences)

        # ================================================================
        # Stage 3.5: Unified Evidence Re-Ranking
        # Single pass with non-compounding boosts, all relative to the
        # ORIGINAL max visual score (captured once, never recalculated).
        # ================================================================
        import re
        codes = re.findall(r'[A-Za-z0-9][A-Za-z0-9\-_.]{1,}', query)
        cn_stop_words = {'有哪些', '什么是', '列出', '多少', '怎么', '如何', '一个', '这张', '这些'}
        cn_chars = re.findall(r'[\u4e00-\u9fff]', query)
        cn_terms = [cn_chars[i] + cn_chars[i+1] for i in range(len(cn_chars) - 1)]
        cn_terms = [t for t in cn_terms if t not in cn_stop_words]
        cn_segments = re.findall(r'[\u4e00-\u9fff]{2,}', query)
        cn_segments = [s for s in cn_segments if s not in cn_stop_words]
        summary_keywords = ['合计', '总计', '总重', '总净重', '总毛重', '共计', 'total', 'sum', 'subtotal']
        
        code_matched = False
        
        if all_evidences:
            # Capture the original max score ONCE — all boosts are relative to this
            base_max = max((ev.score for ev in all_evidences), default=1.0)
            
            # --- Sub-pass A: Build page-level and doc-level context maps ---
            page_concepts = {}    # page_id -> set("entity", "attribute")
            doc_context = {}      # doc_id  -> set("entity", "attribute")
            
            for ev in all_evidences:
                # Doc-level: check folder path (only once per doc)
                if ev.doc_id not in doc_context:
                    doc_context[ev.doc_id] = set()
                    path = self.manifest.get(ev.doc_id, "").lower() if self.manifest else ""
                    if path:
                        for code in codes:
                            if code.lower() in path:
                                doc_context[ev.doc_id].add("entity")
                                break
                        for seg in cn_segments:
                            if seg in path:
                                doc_context[ev.doc_id].add("attribute")
                                break
                
                # Page-level: check text objects
                obj = self.spatial_index.get_object(ev.object_id)
                if not (obj and obj.text):
                    continue
                text = obj.text
                text_lower = text.lower()
                
                if ev.page_id not in page_concepts:
                    page_concepts[ev.page_id] = doc_context.get(ev.doc_id, set()).copy()
                
                for code in codes:
                    if code.lower() in text_lower:
                        page_concepts[ev.page_id].add("entity")
                        break
                for seg in cn_segments:
                    if seg in text or (len(seg) >= 2 and any(t in text for t in [seg[i:i+2] for i in range(len(seg)-1)])):
                        page_concepts[ev.page_id].add("attribute")
                        break
            
            # --- Sub-pass B: Apply boosts (all additive from base_max) ---
            for ev in all_evidences:
                obj = self.spatial_index.get_object(ev.object_id)
                if not (obj and obj.text):
                    continue
                text = obj.text
                text_lower = text.lower()
                bonus = 0.0
                
                # Tier 1: Entity match in THIS object's text (+4)
                for code in codes:
                    if code.lower() in text_lower:
                        bonus += 4.0
                        code_matched = True
                        break  # Only count entity once per object
                
                # Tier 2: Attribute match in THIS object's text (+3)
                for seg in cn_segments:
                    if seg in text or (len(seg) >= 2 and any(t in text for t in [seg[i:i+2] for i in range(len(seg)-1)])):
                        bonus += 3.0
                        break
                
                # Tier 3: Contextual Chinese terms (+0.5 each, capped at +2)
                cn_bonus = 0.0
                for cn in cn_terms:
                    if cn in text:
                        cn_bonus += 0.5
                cn_bonus = min(cn_bonus, 2.0)
                bonus += cn_bonus
                
                # Tier 4: Page-level coverage — entity + attribute on same page (+8)
                coverage = page_concepts.get(ev.page_id, set())
                if "entity" in coverage and "attribute" in coverage:
                    bonus += 8.0
                
                # Tier 5: Summary keyword — contains pre-computed total (+5)
                for kw in summary_keywords:
                    if kw in text_lower:
                        bonus += 5.0
                        break
                
                # Apply bonus (all relative to original base_max)
                if bonus > 0:
                    ev.score += base_max * bonus
                    if bonus >= 12.0:
                        logger.info(f"High-confidence hit: {ev.object_id} on {ev.page_id} (bonus: +{bonus:.1f}x)")
        
        # Stage 3.6: Keyword Fallback Injection
        # Only if NO evidence objects matched the entity codes at all

        if codes and not code_matched:
            logger.info(f"No visual evidence matched codes {codes}. Injecting keyword-matched objects.")
            existing_ids = set(ev.object_id for ev in all_evidences)
            max_score = max((ev.score for ev in all_evidences), default=5.0)
            
            for page_id in candidate_page_ids:
                page_objs = self.spatial_index.get_page_objects(page_id)
                for obj in page_objs:
                    if obj.object_id in existing_ids:
                        continue
                    if not obj.text:
                        continue
                    text_lower = obj.text.lower()
                    for code in codes:
                        if code.lower() in text_lower:
                            # Inject as high-priority evidence
                            injected = RegionEvidence(
                                doc_id=getattr(obj, 'doc_id', ''),
                                version_id=getattr(obj, 'version_id', ''),
                                page_id=obj.page_id,
                                object_id=obj.object_id,
                                bbox=obj.bbox,
                                obj_type=getattr(obj, 'obj_type', 'text_block'),
                                extraction_method='keyword_fallback',
                                score=max_score * 2.0,  # Boost above all visual evidence
                                hash=getattr(obj, 'hash', ''),
                            )
                            all_evidences.append(injected)
                            existing_ids.add(obj.object_id)
                            logger.info(f"Injected keyword evidence: {obj.object_id} ({obj.text[:60]}...)")
                            break

        # Stage 4: Assemble Evidence Pack
        from tracerag.retrieval.aggregation import AggregationProcessor
        from tracerag.retrieval.document_aggregation import classify_aggregation_scope, DocumentAggregationProcessor
        
        agg_processor = AggregationProcessor()
        
        if agg_processor.is_aggregation_query(query):
            scope = classify_aggregation_scope(query)
            if scope == "document_list":
                logger.info("Routing query to Document-Level Aggregation Mode")
                doc_agg_processor = DocumentAggregationProcessor(self.manifest, self.spatial_index)
                all_evidences = doc_agg_processor.process(query)
            else:
                logger.info("Routing query to Object-Level Aggregation Mode")
                # Collection Mode (High Recall + Deduplication)
                all_evidences = agg_processor.process(query, all_evidences, self.spatial_index)
        else:
            # Standard Precision Mode (Top-K)
            # Increase limit to 50 to accommodate longer lists (e.g., 21+ boxes)
            all_evidences = sorted(all_evidences, key=lambda e: e.score, reverse=True)[:50]
            
        pack = EvidencePack(query=query, evidences=all_evidences)
        
        # Stage 5: (Optional) Answer Synthesis
        if self.llm_answerer:
            self._synthesize_answer(pack)

        pack.metadata["elapsed_time"] = time.time() - start_time
        pack.metadata["num_candidate_pages"] = len(candidate_page_ids)
        logger.info(f"Query answered in {pack.metadata['elapsed_time']:.2f}s")

        return pack

    def _synthesize_answer(self, pack: EvidencePack):
        """Populate the EvidencePack with an LLM-synthesized answer."""
        evidence_texts = {}
        for ev in pack.evidences:
            obj = self.spatial_index.get_object(ev.object_id)
            if obj and obj.text:
                evidence_texts[ev.object_id] = obj.text

        if not evidence_texts:
            pack.answer = "No relevant text information found in the identified evidence regions."
            return

        llm_result = self.llm_answerer.generate_answer_with_claims(
            query=pack.query,
            evidences=pack.evidences,
            evidence_texts=evidence_texts,
            query_type="general" 
        )

        pack.answer = llm_result.get("answer", "")
        claims_data = llm_result.get("claims", [])
        pack.certified_claims = self.llm_answerer.map_claims_to_evidences(claims_data, pack.evidences)
