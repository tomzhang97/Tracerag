"""
Core data models and types for TraceRAG.

Defines the fundamental data structures used throughout the system.
Now hardened with Pydantic for strict schema validation across boundaries.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

# Type aliases
BBox = Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max) in page coordinates
ObjectID = str
EntityID = str
PageID = str
VersionID = str
DocumentID = str

class TraceRAGBaseModel(BaseModel):
    """Base schema for all TraceRAG domain models."""
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

class VectorObject(TraceRAGBaseModel):
    """
    Represents a structural element extracted from a PDF.
    Formerly ParsedObject.
    """
    object_id: ObjectID
    doc_id: DocumentID
    version_id: VersionID
    page_id: PageID
    coord_space: str = Field(default="pdf", description="Space of the bbox, usually pdf or image")
    bbox: BBox
    obj_type: str  # text_block, table_cell, path_group, symbol, image
    text: Optional[str] = None
    layer: Optional[str] = None
    style: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)

    @property
    def area(self) -> float:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )

# Use ParsedObject as an explicit alias for semantic clarity across layers
ParsedObject = VectorObject

class PageArtifact(TraceRAGBaseModel):
    """
    Complete extraction payload for a single PDF page.
    Sent between visual rendering and structural extraction to indexing.
    """
    page_id: PageID
    doc_id: DocumentID
    version_id: VersionID
    page_num: int
    dimensions: Tuple[float, float]  # w, h
    objects: List[ParsedObject] = Field(default_factory=list)
    image_path: Optional[str] = None

class PatchGrid(TraceRAGBaseModel):
    """Stores visual embeddings for a document page."""
    doc_id: DocumentID
    version_id: VersionID
    page_id: PageID
    H: int
    W: int
    embeddings: Any  # Actually np.ndarray [H, W, d]
    patch_boxes: Any # Actually np.ndarray [H, W, 4]

class RegionEvidence(TraceRAGBaseModel):
    """
    Represents a piece of evidence from the PDF, grounded to exact vector objects.
    Also known as SnappedEvidence.
    """
    doc_id: DocumentID
    version_id: VersionID
    page_id: PageID
    coord_space: str = Field(default="pdf", description="Space of the bbox")
    object_id: ObjectID
    bbox: BBox
    obj_type: str
    extraction_method: str
    score: float  # Final weighted combination (computed last)
    visual_score: float = 0.0     # Frozen after Snapper (Stage 3)
    symbolic_bonus: float = 0.0   # Computed in Stage 3.5 (entity, attribute, coverage)
    doc_prior: float = 0.0        # Computed in Stage 3.5 from manifest/folder context
    entity_match: float = 0.0
    attribute_match: float = 0.0
    value_match: float = 0.0
    local_structure_score: float = 0.0
    scope_score: float = 0.0
    normalized_value: str = ""
    validator_confidence: float = 0.0
    match_reason: str = ""
    contradiction_penalty: float = 0.0
    cluster_id: str = ""
    trace: Dict[str, Any] = Field(default_factory=dict)
    hash: str = ""

SnappedEvidence = RegionEvidence

class RevisionPair(TraceRAGBaseModel):
    """
    Two versions of a document aligned together.
    """
    base_doc_id: DocumentID
    base_version_id: VersionID
    new_doc_id: DocumentID
    new_version_id: VersionID
    aligned_pages: Dict[int, int]  # base page_num -> new page_num
    diff_summary: Optional[str] = None

class Claim(TraceRAGBaseModel):
    text: str
    value: Optional[str] = None
    entity_id: Optional[EntityID] = None
    claim_type: str = "general"

class CertifiedClaim(TraceRAGBaseModel):
    claim: Claim
    evidences: List[RegionEvidence]
    confidence: float
    reasoning: Optional[str] = None

class QueryResult(TraceRAGBaseModel):
    query: str
    query_type: str
    answer: str
    certified_claims: List[CertifiedClaim]
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Legacy node types for completeness
class GraphNode(TraceRAGBaseModel):
    node_id: str
    node_type: str
    attributes: Dict[str, Any] = Field(default_factory=dict)

class DocumentNode(GraphNode):
    pass
class VersionNode(GraphNode):
    pass
class PageNode(GraphNode):
    pass
class RegionNode(GraphNode):
    pass
class EntityNode(GraphNode):
    pass

class BenchmarkExample(TraceRAGBaseModel):
    """Single benchmark query with ground truth."""
    query_id: str
    query_text: str
    ground_truth_answer: str
    query_type: str
    relevant_pages: List[str]
    relevant_objects: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EvalResult(TraceRAGBaseModel):
    """Results of a benchmark evaluation run."""
    overall: Dict[str, Any]
    per_query: List[Dict[str, Any]]
