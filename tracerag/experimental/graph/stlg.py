"""
Spatio-Temporal Layout Graph (STLG)

Models document evolution across versions with explicit spatial and temporal edges.
Unlike VersionRAG (text-only), STLG tracks visual entities across revisions.

Node types:
- Document: Root node
- Version: Specific revision (with timestamp)
- Page: Individual pages
- Region: VectorObjects (text blocks, symbols, tables)
- Entity: Abstract concepts (components, parts) with multiple visual states

Edge types:
- contains: hierarchical containment (Doc->Version->Page->Region)
- mentions: Region->Entity (what concept does this region refer to)
- aligned_with: Region<->Region across versions (same entity, different revisions)
- evolved_from: Version->Version (temporal succession)
"""

import networkx as nx
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
from difflib import SequenceMatcher

from tracerag.common.types import VectorObject, RegionEvidence
from tracerag.common.geometry import bbox_iou


class STLayoutGraph:
    """
    Spatio-Temporal Layout Graph.

    Indexes documents with explicit version, spatial, and cross-version edges.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize STLG.

        Args:
            config: Configuration dictionary (from graph.stlg section)
        """
        self.G = nx.MultiDiGraph()
        self.config = config or {}

        # Entity matching config
        self.entity_match_method = self.config.get("entity_matching", {}).get("method", "fuzzy")
        self.entity_match_threshold = self.config.get("entity_matching", {}).get("threshold", 0.85)

        # Version alignment config
        self.visual_diff_threshold = self.config.get("version_alignment", {}).get(
            "visual_diff_threshold", 0.3
        )
        self.bbox_iou_threshold = self.config.get("version_alignment", {}).get(
            "bbox_iou_threshold", 0.5
        )

        # Indexes for fast lookup
        self.entity_index: Dict[str, str] = {}  # label -> entity_id
        self.region_to_entity: Dict[str, str] = {}  # object_id -> entity_id

    def add_document(self, doc_id: str, metadata: Optional[Dict] = None):
        """
        Add document node.

        Args:
            doc_id: Document ID
            metadata: Optional metadata dict
        """
        node_id = f"doc:{doc_id}"
        attrs = metadata or {}
        attrs["type"] = "Document"
        attrs["doc_id"] = doc_id
        self.G.add_node(node_id, **attrs)
        logger.debug(f"Added document node: {node_id}")

    def add_version(
        self,
        doc_id: str,
        version_id: str,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add version node and link to document.

        Args:
            doc_id: Parent document ID
            version_id: Version ID
            timestamp: Optional timestamp string
            metadata: Optional metadata dict
        """
        doc_node = f"doc:{doc_id}"
        ver_node = f"ver:{version_id}"

        attrs = metadata or {}
        attrs["type"] = "Version"
        attrs["version_id"] = version_id
        attrs["timestamp"] = timestamp

        self.G.add_node(ver_node, **attrs)
        self.G.add_edge(doc_node, ver_node, type="has_version")

        logger.debug(f"Added version node: {ver_node}")

    def add_page(
        self,
        version_id: str,
        page_id: str,
        page_num: int,
        metadata: Optional[Dict] = None
    ):
        """
        Add page node and link to version.

        Args:
            version_id: Parent version ID
            page_id: Page ID
            page_num: Page number
            metadata: Optional metadata dict
        """
        ver_node = f"ver:{version_id}"
        page_node = f"page:{page_id}"

        attrs = metadata or {}
        attrs["type"] = "Page"
        attrs["page_id"] = page_id
        attrs["page_num"] = page_num

        self.G.add_node(page_node, **attrs)
        self.G.add_edge(ver_node, page_node, type="contains")

        logger.debug(f"Added page node: {page_node}")

    def add_region(self, obj: VectorObject):
        """
        Add region node (VectorObject) and link to page.

        Args:
            obj: VectorObject to add
        """
        page_node = f"page:{obj.page_id}"
        region_node = f"region:{obj.object_id}"

        attrs = {
            "type": "Region",
            "obj": obj,
            "obj_type": obj.obj_type,
            "bbox": obj.bbox,
            "text": obj.text,
        }

        self.G.add_node(region_node, **attrs)
        self.G.add_edge(page_node, region_node, type="contains")

    def add_entity(self, entity_id: str, label: str, metadata: Optional[Dict] = None):
        """
        Add entity node (abstract concept).

        Args:
            entity_id: Entity ID
            label: Human-readable label (e.g., component name)
            metadata: Optional metadata dict
        """
        ent_node = f"ent:{entity_id}"

        attrs = metadata or {}
        attrs["type"] = "Entity"
        attrs["entity_id"] = entity_id
        attrs["label"] = label

        self.G.add_node(ent_node, **attrs)
        self.entity_index[label] = entity_id

        logger.debug(f"Added entity node: {ent_node} ({label})")

    def link_region_entity(self, object_id: str, entity_id: str):
        """
        Link region to entity (region mentions entity).

        Args:
            object_id: Object ID
            entity_id: Entity ID
        """
        region_node = f"region:{object_id}"
        ent_node = f"ent:{entity_id}"

        self.G.add_edge(region_node, ent_node, type="mentions")
        self.region_to_entity[object_id] = entity_id

    def align_regions(
        self,
        old_obj_id: str,
        new_obj_id: str,
        diff_meta: Optional[Dict] = None
    ):
        """
        Align regions across versions (same entity, different revisions).

        Args:
            old_obj_id: Object ID in old version
            new_obj_id: Object ID in new version
            diff_meta: Optional diff metadata (changes, visual diff score, etc.)
        """
        old_node = f"region:{old_obj_id}"
        new_node = f"region:{new_obj_id}"

        edge_attrs = diff_meta or {}
        edge_attrs["type"] = "aligned_with"

        self.G.add_edge(old_node, new_node, **edge_attrs)
        self.G.add_edge(new_node, old_node, **edge_attrs)  # Bidirectional

    def auto_align_versions(
        self,
        old_version_id: str,
        new_version_id: str
    ):
        """
        Automatically align regions between two versions.

        Uses heuristics:
        1. Text similarity for text blocks
        2. Bbox IoU for vector objects
        3. Layer matching for CAD objects

        Args:
            old_version_id: Old version ID
            new_version_id: New version ID
        """
        logger.info(f"Auto-aligning versions: {old_version_id} -> {new_version_id}")

        # Get all regions in each version
        old_regions = self._get_version_regions(old_version_id)
        new_regions = self._get_version_regions(new_version_id)

        logger.debug(f"Old version: {len(old_regions)} regions, New version: {len(new_regions)} regions")

        # Match regions
        alignments = []

        for old_obj in old_regions:
            best_match = None
            best_score = 0.0

            for new_obj in new_regions:
                score = self._compute_alignment_score(old_obj, new_obj)

                if score > best_score:
                    best_score = score
                    best_match = new_obj

            # Threshold for alignment
            if best_score > self.bbox_iou_threshold:
                diff_meta = {
                    "alignment_score": best_score,
                    "change_type": self._infer_change_type(old_obj, best_match)
                }
                alignments.append((old_obj.object_id, best_match.object_id, diff_meta))

        # Add alignments to graph
        for old_id, new_id, diff_meta in alignments:
            self.align_regions(old_id, new_id, diff_meta)

        logger.info(f"Created {len(alignments)} alignments")

    def _get_version_regions(self, version_id: str) -> List[VectorObject]:
        """Get all regions in a version."""
        ver_node = f"ver:{version_id}"
        regions = []

        # Traverse: Version -> Pages -> Regions
        for _, page_node in self.G.out_edges(ver_node):
            if self.G.nodes[page_node].get("type") != "Page":
                continue

            for _, region_node in self.G.out_edges(page_node):
                if self.G.nodes[region_node].get("type") != "Region":
                    continue

                obj = self.G.nodes[region_node].get("obj")
                if obj:
                    regions.append(obj)

        return regions

    def _compute_alignment_score(self, obj1: VectorObject, obj2: VectorObject) -> float:
        """
        Compute alignment score between two objects.

        Args:
            obj1: First object
            obj2: Second object

        Returns:
            Alignment score [0, 1]
        """
        # Must be same object type
        if obj1.obj_type != obj2.obj_type:
            return 0.0

        # Bbox IoU
        iou = bbox_iou(obj1.bbox, obj2.bbox)

        # Text similarity if both have text
        if obj1.text and obj2.text:
            text_sim = SequenceMatcher(None, obj1.text, obj2.text).ratio()
            return 0.5 * iou + 0.5 * text_sim
        else:
            return iou

    def _infer_change_type(self, old_obj: VectorObject, new_obj: VectorObject) -> str:
        """
        Infer what type of change occurred.

        Args:
            old_obj: Old object
            new_obj: New object

        Returns:
            Change type string
        """
        # Check text changes
        if old_obj.text != new_obj.text:
            return "text_modified"

        # Check bbox changes
        iou = bbox_iou(old_obj.bbox, new_obj.bbox)
        if iou < 0.9:
            return "moved"

        # Check style changes
        if old_obj.style != new_obj.style:
            return "style_modified"

        return "unchanged"

    def get_entity_states(self, entity_id: str) -> List[Tuple[str, VectorObject]]:
        """
        Get all visual states (regions) of an entity across versions.

        Args:
            entity_id: Entity ID

        Returns:
            List of (version_id, VectorObject) tuples
        """
        ent_node = f"ent:{entity_id}"
        states = []

        # Find all regions that mention this entity
        for region_node, _ in self.G.in_edges(ent_node):
            if not region_node.startswith("region:"):
                continue

            obj = self.G.nodes[region_node].get("obj")
            if obj:
                states.append((obj.version_id, obj))

        return states

    def expand_from_evidences(
        self,
        evidences: List[RegionEvidence],
        max_hops: int = 2
    ) -> List[RegionEvidence]:
        """
        Expand evidences by traversing graph (e.g., follow aligned_with edges).

        Args:
            evidences: Initial evidences
            max_hops: Maximum graph traversal depth

        Returns:
            Expanded list of evidences
        """
        expanded = list(evidences)
        visited = set(e.object_id for e in evidences)

        for evidence in evidences:
            region_node = f"region:{evidence.object_id}"

            if region_node not in self.G:
                continue

            # Traverse graph
            neighbors = self._traverse_neighbors(region_node, max_hops)

            for neighbor_node in neighbors:
                if not neighbor_node.startswith("region:"):
                    continue

                obj_id = neighbor_node.split(":", 1)[1]
                if obj_id in visited:
                    continue

                obj = self.G.nodes[neighbor_node].get("obj")
                if obj:
                    # Create new evidence (with lower score)
                    expanded_evidence = RegionEvidence(
                        doc_id=obj.doc_id,
                        version_id=obj.version_id,
                        page_id=obj.page_id,
                        object_id=obj.object_id,
                        bbox=obj.bbox,
                        obj_type=obj.obj_type,
                        extraction_method="graph_expansion",
                        score=evidence.score * 0.5,  # Decay score
                        hash=""  # Would recompute
                    )
                    expanded.append(expanded_evidence)
                    visited.add(obj_id)

        return expanded

    def _traverse_neighbors(self, start_node: str, max_hops: int) -> List[str]:
        """BFS traversal from start node."""
        visited = set()
        queue = [(start_node, 0)]
        visited.add(start_node)
        neighbors = []

        while queue:
            node, depth = queue.pop(0)

            if depth >= max_hops:
                continue

            # Out edges
            for _, neighbor in self.G.out_edges(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    neighbors.append(neighbor)
                    queue.append((neighbor, depth + 1))

            # In edges
            for neighbor, _ in self.G.in_edges(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    neighbors.append(neighbor)
                    queue.append((neighbor, depth + 1))

        return neighbors
