"""
Evidence Pack definition.
Represents the final set of retrieved evidences and optional LLM answers.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from tracerag.common.types import RegionEvidence, CertifiedClaim


class EvidencePack(BaseModel):
    """
    Standardized payload for retrieval results containing grounded evidences.
    Can be optionally populated with LLM answers downstream.
    """
    query: str
    evidences: List[RegionEvidence]
    answer: Optional[str] = None
    certified_claims: List[CertifiedClaim] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return self.model_dump()
