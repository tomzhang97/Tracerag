"""
Core data models and types for TraceRAG.

Defines the fundamental data structures used throughout the system:
- VectorObject: Represents a structural element from PDF (text, table, vector graphic)
- PatchGrid: Stores visual embeddings for a page
- RegionEvidence: Links query results to exact PDF regions
- Claim/CertifiedClaim: Structured answers with evidence certificates
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

# Type aliases
BBox = Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max) in page coordinates
ObjectID = str
EntityID = str
PageID = str
VersionID = str
DocumentID = str


@dataclass
class VectorObject:
    """
    Represents a structural element extracted from a PDF.

    This is the fundamental unit of the structural stream. Each object corresponds
    to an actual PDF primitive (text block, table cell, vector path group) with
    precise coordinates.

    Attributes:
        object_id: Unique identifier for this object
        doc_id: Document identifier
        version_id: Version/revision identifier
        page_id: Page identifier (typically {doc_id}_{version_id}_p{page_num})
        bbox: Bounding box in page coordinates (x_min, y_min, x_max, y_max)
        obj_type: Type of object (text_block, table_cell, path_group, symbol, image)
        text: Extracted text content (if applicable)
        layer: CAD layer name (if from CAD-derived PDF)
        style: Visual styling attributes (color, stroke_width, fill, font_size, etc.)
        meta: Additional metadata (entity_id, figure_number, table_id, etc.)
    """
    object_id: ObjectID
    doc_id: DocumentID
    version_id: VersionID
    page_id: PageID
    bbox: BBox
    obj_type: str  # text_block, table_cell, path_group, symbol, image
    text: Optional[str] = None
    layer: Optional[str] = None
    style: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def area(self) -> float:
        """Calculate bounding box area."""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    def center(self) -> Tuple[float, float]:
        """Calculate center point of bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )


@dataclass
class PatchGrid:
    """
    Stores visual embeddings for a document page.

    Represents the output of the visual encoding stream (ColPali-style).
    Each page is divided into a grid of patches, each with an embedding vector.

    Attributes:
        doc_id: Document identifier
        version_id: Version identifier
        page_id: Page identifier
        H: Height of patch grid
        W: Width of patch grid
        embeddings: Patch embeddings, shape [H, W, d]
        patch_boxes: Bounding boxes for each patch in page coordinates, shape [H, W, 4]
    """
    doc_id: DocumentID
    version_id: VersionID
    page_id: PageID
    H: int  # Grid height
    W: int  # Grid width
    embeddings: np.ndarray  # [H, W, d]
    patch_boxes: np.ndarray  # [H, W, 4] - bbox for each patch in page coords

    def __post_init__(self):
        """Validate dimensions."""
        assert self.embeddings.shape[:2] == (self.H, self.W), \
            f"Embeddings shape {self.embeddings.shape} doesn't match grid {self.H}x{self.W}"
        assert self.patch_boxes.shape == (self.H, self.W, 4), \
            f"Patch boxes shape {self.patch_boxes.shape} doesn't match grid {self.H}x{self.W}"


@dataclass
class RegionEvidence:
    """
    Represents a piece of evidence from the PDF, grounded to exact vector objects.

    This is the output of the Snapper (vector-native alignment layer).
    Unlike pixel-based bounding boxes, this is anchored to actual PDF objects.

    Attributes:
        doc_id: Document identifier
        version_id: Version identifier
        page_id: Page identifier
        object_id: ID of the underlying VectorObject
        bbox: Bounding box in page coordinates
        obj_type: Type of object
        extraction_method: How this evidence was extracted (vector_text, table, vector_symbol)
        score: Relevance score from alignment
        hash: Tamper-evident fingerprint for verification
    """
    doc_id: DocumentID
    version_id: VersionID
    page_id: PageID
    object_id: ObjectID
    bbox: BBox
    obj_type: str
    extraction_method: str  # vector_text, table, vector_symbol
    score: float
    hash: str  # tamper-evident fingerprint

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "doc_id": self.doc_id,
            "version_id": self.version_id,
            "page_id": self.page_id,
            "object_id": self.object_id,
            "bbox": list(self.bbox),
            "obj_type": self.obj_type,
            "extraction_method": self.extraction_method,
            "score": self.score,
            "hash": self.hash,
        }


@dataclass
class Claim:
    """
    A single factual claim extracted from an answer.

    Attributes:
        text: Human-readable claim statement
        value: Structured value (e.g., "50 Nm" for torque), optional
        entity_id: Reference to entity this claim is about, optional
        claim_type: Type of claim (attribute, location, procedure_step, etc.)
    """
    text: str
    value: Optional[str] = None
    entity_id: Optional[EntityID] = None
    claim_type: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "text": self.text,
            "value": self.value,
            "entity_id": self.entity_id,
            "claim_type": self.claim_type,
        }


@dataclass
class CertifiedClaim:
    """
    A claim with attached evidence certificates.

    This is the key innovation for engineering-grade retrieval: every claim
    is backed by explicit, vector-native evidence regions with cryptographic hashes.

    Attributes:
        claim: The claim itself
        evidences: List of RegionEvidence objects supporting this claim
        confidence: Confidence score [0, 1]
        reasoning: Optional explanation of how evidence supports claim
    """
    claim: Claim
    evidences: List[RegionEvidence]
    confidence: float
    reasoning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "claim": self.claim.to_dict(),
            "evidences": [e.to_dict() for e in self.evidences],
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class QueryResult:
    """
    Complete result for a query, including answer and certified claims.

    Attributes:
        query: Original query string
        query_type: Classified query type (locator, attribute, procedural, revision, diff)
        answer: Natural language answer
        certified_claims: List of claims with evidence certificates
        metadata: Additional information (retrieval time, model versions, etc.)
    """
    query: str
    query_type: str
    answer: str
    certified_claims: List[CertifiedClaim]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        return {
            "query": self.query,
            "query_type": self.query_type,
            "answer": self.answer,
            "claims": [c.to_dict() for c in self.certified_claims],
            "metadata": self.metadata,
        }


# Node types for graphs
@dataclass
class GraphNode:
    """Base class for graph nodes in STLG/LDG."""
    node_id: str
    node_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentNode(GraphNode):
    """Represents a document in the graph."""
    def __init__(self, doc_id: DocumentID):
        super().__init__(
            node_id=f"doc:{doc_id}",
            node_type="Document",
            attributes={"doc_id": doc_id}
        )


@dataclass
class VersionNode(GraphNode):
    """Represents a version/revision in the graph."""
    def __init__(self, version_id: VersionID, timestamp: Optional[str] = None):
        super().__init__(
            node_id=f"ver:{version_id}",
            node_type="Version",
            attributes={"version_id": version_id, "timestamp": timestamp}
        )


@dataclass
class PageNode(GraphNode):
    """Represents a page in the graph."""
    def __init__(self, page_id: PageID, page_num: int):
        super().__init__(
            node_id=f"page:{page_id}",
            node_type="Page",
            attributes={"page_id": page_id, "page_num": page_num}
        )


@dataclass
class RegionNode(GraphNode):
    """Represents a region (VectorObject) in the graph."""
    def __init__(self, obj: VectorObject):
        super().__init__(
            node_id=f"region:{obj.object_id}",
            node_type="Region",
            attributes={"object": obj}
        )


@dataclass
class EntityNode(GraphNode):
    """Represents an abstract entity (component, part, concept) in the graph."""
    def __init__(self, entity_id: EntityID, label: str):
        super().__init__(
            node_id=f"ent:{entity_id}",
            node_type="Entity",
            attributes={"entity_id": entity_id, "label": label}
        )
