"""
Layout Dependency Graph (LDG)

Models explicit cross-page dependencies in technical manuals:
- Figure/Table references ("see Fig. 3", "refer to Table 2")
- Legend to symbol linkage
- Procedure steps to warning boxes
- Parameter tables to diagrams

Unlike STLG (which models versions), LDG models within-document structure.
Can be integrated into STLG or maintained separately.

Node types:
- Figure, Table, Legend, ProcedureStep, Warning, Parameter

Edge types:
- refers_to: Procedure -> Figure/Table/Section
- explained_by: Symbol -> Legend entry
- parameterized_by: Component -> Parameter table
"""

import re
import networkx as nx
from typing import Dict, Any, List, Optional, Set, Tuple
from loguru import logger

from tracerag.common.types import VectorObject, RegionEvidence


class LayoutDependencyGraph:
    """
    Layout Dependency Graph.

    Captures cross-page structural relationships in technical documents.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LDG.

        Args:
            config: Configuration dictionary (from graph.ldg section)
        """
        self.G = nx.DiGraph()
        self.config = config or {}

        # Reference detection config
        self.reference_detection_enabled = self.config.get("reference_detection", {}).get(
            "enabled", True
        )
        self.reference_patterns = self.config.get("reference_detection", {}).get(
            "patterns",
            [
                r"see\s+Fig\.?\s*(\d+)",
                r"Table\s+(\d+)",
                r"Section\s+([\d\.]+)",
                r"refer\s+to\s+(?:Fig\.?|Table)\s*(\d+)",
            ]
        )

        # Compile regex patterns
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.reference_patterns]

        # Indexes
        self.figure_index: Dict[str, str] = {}  # figure_num -> node_id
        self.table_index: Dict[str, str] = {}  # table_num -> node_id
        self.section_index: Dict[str, str] = {}  # section_num -> node_id

    def add_figure(
        self,
        figure_id: str,
        figure_num: str,
        page_id: str,
        bbox: Optional[Tuple] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add figure node.

        Args:
            figure_id: Unique figure ID
            figure_num: Figure number (e.g., "3", "4.2")
            page_id: Page where figure appears
            bbox: Optional bounding box
            metadata: Optional metadata
        """
        node_id = f"fig:{figure_id}"
        attrs = metadata or {}
        attrs.update({
            "type": "Figure",
            "figure_id": figure_id,
            "figure_num": figure_num,
            "page_id": page_id,
            "bbox": bbox,
        })

        self.G.add_node(node_id, **attrs)
        self.figure_index[figure_num] = node_id

        logger.debug(f"Added figure node: {node_id} (Fig. {figure_num})")

    def add_table(
        self,
        table_id: str,
        table_num: str,
        page_id: str,
        bbox: Optional[Tuple] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add table node.

        Args:
            table_id: Unique table ID
            table_num: Table number
            page_id: Page where table appears
            bbox: Optional bounding box
            metadata: Optional metadata
        """
        node_id = f"table:{table_id}"
        attrs = metadata or {}
        attrs.update({
            "type": "Table",
            "table_id": table_id,
            "table_num": table_num,
            "page_id": page_id,
            "bbox": bbox,
        })

        self.G.add_node(node_id, **attrs)
        self.table_index[table_num] = node_id

        logger.debug(f"Added table node: {node_id} (Table {table_num})")

    def add_procedure_step(
        self,
        step_id: str,
        step_num: str,
        page_id: str,
        text: str,
        bbox: Optional[Tuple] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add procedure step node.

        Args:
            step_id: Unique step ID
            step_num: Step number
            page_id: Page where step appears
            text: Step text content
            bbox: Optional bounding box
            metadata: Optional metadata
        """
        node_id = f"step:{step_id}"
        attrs = metadata or {}
        attrs.update({
            "type": "ProcedureStep",
            "step_id": step_id,
            "step_num": step_num,
            "page_id": page_id,
            "text": text,
            "bbox": bbox,
        })

        self.G.add_node(node_id, **attrs)

        # Auto-detect references in step text
        if self.reference_detection_enabled:
            self._detect_and_link_references(node_id, text)

    def add_warning(
        self,
        warning_id: str,
        page_id: str,
        text: str,
        bbox: Optional[Tuple] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add warning node.

        Args:
            warning_id: Unique warning ID
            page_id: Page where warning appears
            text: Warning text
            bbox: Optional bounding box
            metadata: Optional metadata
        """
        node_id = f"warn:{warning_id}"
        attrs = metadata or {}
        attrs.update({
            "type": "Warning",
            "warning_id": warning_id,
            "page_id": page_id,
            "text": text,
            "bbox": bbox,
        })

        self.G.add_node(node_id, **attrs)

    def link_refers_to(self, source_id: str, target_id: str):
        """
        Add 'refers_to' edge (e.g., procedure step -> figure).

        Args:
            source_id: Source node ID
            target_id: Target node ID
        """
        self.G.add_edge(source_id, target_id, type="refers_to")

    def link_explained_by(self, symbol_id: str, legend_id: str):
        """
        Add 'explained_by' edge (symbol -> legend).

        Args:
            symbol_id: Symbol node ID
            legend_id: Legend node ID
        """
        self.G.add_edge(symbol_id, legend_id, type="explained_by")

    def _detect_and_link_references(self, node_id: str, text: str):
        """
        Detect figure/table/section references in text and create links.

        Args:
            node_id: Source node ID
            text: Text to scan for references
        """
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)

            for match in matches:
                ref_num = match.strip()

                # Try to link to figure
                if ref_num in self.figure_index:
                    target_id = self.figure_index[ref_num]
                    self.link_refers_to(node_id, target_id)
                    logger.debug(f"Linked {node_id} -> Fig. {ref_num}")

                # Try to link to table
                elif ref_num in self.table_index:
                    target_id = self.table_index[ref_num]
                    self.link_refers_to(node_id, target_id)
                    logger.debug(f"Linked {node_id} -> Table {ref_num}")

                # Try to link to section
                elif ref_num in self.section_index:
                    target_id = self.section_index[ref_num]
                    self.link_refers_to(node_id, target_id)
                    logger.debug(f"Linked {node_id} -> Section {ref_num}")

    def expand_evidences_with_dependencies(
        self,
        evidences: List[RegionEvidence],
        stlg=None
    ) -> List[RegionEvidence]:
        """
        Expand evidences by following layout dependencies.

        For example:
        - If evidence is a procedure step, include referenced figures/tables
        - If evidence is a symbol, include legend
        - If evidence is a component, include parameter table

        Args:
            evidences: Initial evidences
            stlg: Optional STLG instance for mapping regions to LDG nodes

        Returns:
            Expanded evidences
        """
        expanded = list(evidences)
        visited_nodes = set()

        for evidence in evidences:
            # Map evidence to LDG node (placeholder - would need proper mapping)
            # For now, assume object_id maps to node_id
            candidate_nodes = self._find_ldg_nodes_for_evidence(evidence)

            for node_id in candidate_nodes:
                if node_id in visited_nodes:
                    continue

                visited_nodes.add(node_id)

                # Follow outgoing 'refers_to' edges
                for _, target_id in self.G.out_edges(node_id):
                    if self.G[node_id][target_id].get("type") == "refers_to":
                        # Get target node attrs
                        target_attrs = self.G.nodes[target_id]

                        # Convert to evidence (placeholder - would need full implementation)
                        # This would require mapping LDG nodes back to VectorObjects

        return expanded

    def _find_ldg_nodes_for_evidence(self, evidence: RegionEvidence) -> List[str]:
        """
        Find LDG nodes corresponding to an evidence region.

        Args:
            evidence: RegionEvidence

        Returns:
            List of LDG node IDs
        """
        # Placeholder: would implement proper mapping
        # Could use spatial overlap, page_id matching, etc.
        return []

    def get_dependencies(self, node_id: str, max_depth: int = 2) -> List[str]:
        """
        Get all dependencies of a node.

        Args:
            node_id: Source node ID
            max_depth: Maximum traversal depth

        Returns:
            List of dependent node IDs
        """
        dependencies = []
        visited = set()
        queue = [(node_id, 0)]

        while queue:
            current, depth = queue.pop(0)

            if current in visited or depth > max_depth:
                continue

            visited.add(current)

            if current != node_id:
                dependencies.append(current)

            # Follow outgoing edges
            for _, neighbor in self.G.out_edges(current):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))

        return dependencies

    def get_referring_nodes(self, node_id: str) -> List[str]:
        """
        Get all nodes that refer to this node.

        Args:
            node_id: Target node ID

        Returns:
            List of referring node IDs
        """
        referring = []
        for source, target in self.G.in_edges(node_id):
            if self.G[source][target].get("type") == "refers_to":
                referring.append(source)

        return referring
