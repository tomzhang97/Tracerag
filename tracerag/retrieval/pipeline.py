"""
TraceRAG main retrieval pipeline.

Orchestrates the complete query flow:
1. Query classification
2. Text-based candidate retrieval
3. Visual scoring
4. Vector-native alignment (Snapper)
5. Graph expansion
6. Claim extraction and certification
"""

import time
from typing import Dict, Any, List, Optional
from loguru import logger

from tracerag.common.types import (
    PatchGrid, RegionEvidence, QueryResult, CertifiedClaim, Claim
)
from tracerag.retrieval.classifier import QueryClassifier
from tracerag.retrieval.text_index import TextIndex
from tracerag.retrieval.llm_answerer import LLMAnswerer
from tracerag.visual.encoder import VisualPageEncoder
from tracerag.visual.scorer import VisualScorer
from tracerag.alignment.snapper import Snapper
from tracerag.graph.stlg import STLayoutGraph
from tracerag.graph.ldg import LayoutDependencyGraph
from tracerag.structural.parser import PdfSpatialIndex


class PatchGridStore:
    """Simple in-memory store for PatchGrids."""

    def __init__(self):
        self.grids: Dict[str, PatchGrid] = {}

    def add(self, patch_grid: PatchGrid):
        """Add patch grid."""
        self.grids[patch_grid.page_id] = patch_grid

    def get(self, page_id: str) -> Optional[PatchGrid]:
        """Get patch grid by page ID."""
        return self.grids.get(page_id)

    def get_batch(self, page_ids: List[str]) -> List[PatchGrid]:
        """Get multiple patch grids."""
        return [self.grids[pid] for pid in page_ids if pid in self.grids]


class TraceRAGSystem:
    """
    Main TraceRAG retrieval system.

    Combines all components into a unified pipeline.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        text_index: TextIndex,
        patch_grid_store: PatchGridStore,
        spatial_index: PdfSpatialIndex,
        visual_encoder: VisualPageEncoder,
        stlg: Optional[STLayoutGraph] = None,
        ldg: Optional[LayoutDependencyGraph] = None,
    ):
        """
        Initialize TraceRAG system.

        Args:
            config: Configuration dictionary
            text_index: Text index for candidate retrieval
            patch_grid_store: Store for visual patch grids
            spatial_index: Spatial index for vector objects
            visual_encoder: Visual encoder for scoring
            stlg: Optional Spatio-Temporal Layout Graph
            ldg: Optional Layout Dependency Graph
        """
        self.config = config
        self.text_index = text_index
        self.patch_grid_store = patch_grid_store
        self.spatial_index = spatial_index
        self.visual_encoder = visual_encoder

        self.stlg = stlg
        self.ldg = ldg

        # Initialize components
        self.classifier = QueryClassifier(
            method=config.get("query_classifier", {}).get("method", "heuristic")
        )

        self.visual_scorer = VisualScorer(
            config=config.get("visual", {}),
            encoder=visual_encoder
        )

        self.snapper = Snapper(
            spatial_index=spatial_index,
            config=config.get("snapper", {})
        )

        # Initialize LLM answerer
        self.llm_answerer = LLMAnswerer(config.get("llm", {}))

        # Retrieval config
        self.top_k_pages = config.get("retrieval", {}).get("top_k_pages", 10)
        self.hybrid_alpha = config.get("retrieval", {}).get("hybrid_alpha", 0.5)

        logger.info("TraceRAG system initialized")

    def answer(
        self,
        query: str,
        query_type: Optional[str] = None,
        return_evidences: bool = True
    ) -> QueryResult:
        """
        Main entry point: answer a query with certified claims.

        Args:
            query: Query string
            query_type: Optional pre-classified query type
            return_evidences: Whether to return full evidence objects

        Returns:
            QueryResult with answer and certified claims
        """
        start_time = time.time()

        logger.info(f"Processing query: {query}")

        # Step 1: Classify query
        if query_type is None:
            query_type = self.classifier.classify(query)

        logger.info(f"Query type: {query_type}")

        # Step 2: Retrieve candidate pages
        candidate_page_ids = self._retrieve_candidate_pages(query, query_type)

        logger.info(f"Retrieved {len(candidate_page_ids)} candidate pages")

        # Step 3: Visual scoring
        patch_grids = self.patch_grid_store.get_batch(candidate_page_ids)
        scored_pages = self.visual_scorer.score_pages_batch(query, patch_grids)

        # Step 4: Vector-native alignment (Snapper)
        all_evidences: List[RegionEvidence] = []

        for page_id, page_score, patch_scores in scored_pages:
            patch_grid = self.patch_grid_store.get(page_id)
            if patch_grid is None:
                continue

            evidences = self.snapper.snap_page(patch_grid, patch_scores)
            all_evidences.extend(evidences)

        logger.info(f"Generated {len(all_evidences)} initial evidences")

        # Step 5: Graph expansion (if available)
        if self.stlg and query_type in ("revision", "diff"):
            all_evidences = self.stlg.expand_from_evidences(all_evidences, max_hops=2)
            logger.info(f"Expanded to {len(all_evidences)} evidences via STLG")

        # Step 6: Rank and filter evidences
        all_evidences = sorted(all_evidences, key=lambda e: e.score, reverse=True)
        top_evidences = all_evidences[:20]  # Keep top 20

        # Step 7: Compose answer and extract claims
        result = self._compose_answer_with_certificates(
            query, query_type, top_evidences
        )

        # Add metadata
        elapsed_time = time.time() - start_time
        result.metadata = {
            "elapsed_time": elapsed_time,
            "query_type": query_type,
            "num_candidate_pages": len(candidate_page_ids),
            "num_evidences": len(top_evidences),
        }

        logger.info(f"Query answered in {elapsed_time:.2f}s")

        return result

    def _retrieve_candidate_pages(
        self,
        query: str,
        query_type: str
    ) -> List[str]:
        """
        Retrieve candidate pages using hybrid text + visual retrieval.

        Args:
            query: Query string
            query_type: Query type

        Returns:
            List of page IDs
        """
        # For now, use text index
        # In full implementation, would combine with visual retrieval

        if query_type in ("locator", "attribute"):
            # Identifier-focused queries: use text index
            page_ids = self.text_index.get_page_ids(query, self.top_k_pages)
        else:
            # Broader queries: use text index but retrieve more
            page_ids = self.text_index.get_page_ids(query, self.top_k_pages * 2)

        return page_ids

    def _compose_answer_with_certificates(
        self,
        query: str,
        query_type: str,
        evidences: List[RegionEvidence]
    ) -> QueryResult:
        """
        Compose natural language answer and extract certified claims using LLM.

        Args:
            query: Query string
            query_type: Query type
            evidences: List of evidence regions

        Returns:
            QueryResult
        """
        # Extract text from evidences
        evidence_texts = {}
        for ev in evidences[:20]:  # Limit to top 20
            obj = self.spatial_index.get_object(ev.object_id)
            if obj and obj.text:
                evidence_texts[ev.object_id] = obj.text

        # Handle no evidence case
        if not evidence_texts:
            return QueryResult(
                query=query,
                query_type=query_type,
                answer="No relevant information found in the technical documents.",
                certified_claims=[],
                metadata={}
            )

        # Use LLM to generate answer and extract claims
        llm_result = self.llm_answerer.generate_answer_with_claims(
            query=query,
            evidences=evidences[:20],
            evidence_texts=evidence_texts,
            query_type=query_type
        )

        answer = llm_result.get("answer", "")
        claims_data = llm_result.get("claims", [])

        # Map claims to CertifiedClaim objects
        certified_claims = self.llm_answerer.map_claims_to_evidences(
            claims_data=claims_data,
            evidences=evidences[:20]
        )

        return QueryResult(
            query=query,
            query_type=query_type,
            answer=answer,
            certified_claims=certified_claims,
            metadata={}
        )

    def _extract_claims_simple(
        self,
        evidences: List[RegionEvidence]
    ) -> List[CertifiedClaim]:
        """
        Simple claim extraction (placeholder for LLM-based).

        Args:
            evidences: List of evidences

        Returns:
            List of certified claims
        """
        claims = []

        # Group evidences by page
        page_groups: Dict[str, List[RegionEvidence]] = {}
        for ev in evidences[:5]:  # Top 5
            if ev.page_id not in page_groups:
                page_groups[ev.page_id] = []
            page_groups[ev.page_id].append(ev)

        # Create one claim per page group
        for page_id, page_evidences in page_groups.items():
            obj = self.spatial_index.get_object(page_evidences[0].object_id)
            if not obj or not obj.text:
                continue

            claim = Claim(
                text=obj.text[:200],  # Truncate
                value=None,
                entity_id=None,
                claim_type="general"
            )

            certified_claim = CertifiedClaim(
                claim=claim,
                evidences=page_evidences,
                confidence=page_evidences[0].score,
                reasoning=f"Found in page {page_id}"
            )

            claims.append(certified_claim)

        return claims
