"""
Evidence hashing for tamper-evident certificates.

Provides cryptographic fingerprints of evidence regions to ensure
they cannot be tampered with after extraction.
"""

import hashlib
import json
from tracerag.common.types import VectorObject, RegionEvidence


def hash_region(obj: VectorObject) -> str:
    """
    Generate tamper-evident hash for a VectorObject.

    Creates a deterministic fingerprint that includes:
    - Document and version IDs
    - Exact bounding box coordinates
    - Object type
    - Text content (if any)
    - Style attributes

    Args:
        obj: VectorObject to hash

    Returns:
        SHA256 hex digest
    """
    # Create deterministic payload
    payload = {
        "doc_id": obj.doc_id,
        "version_id": obj.version_id,
        "page_id": obj.page_id,
        "bbox": obj.bbox,
        "obj_type": obj.obj_type,
        "text": obj.text or "",
        "style": sorted(obj.style.items()) if obj.style else [],
    }

    # Convert to canonical JSON string
    payload_str = json.dumps(payload, sort_keys=True, separators=(',', ':'))

    # Hash
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


def hash_evidence(evidence: RegionEvidence) -> str:
    """
    Generate hash for a RegionEvidence object.

    Args:
        evidence: RegionEvidence to hash

    Returns:
        SHA256 hex digest
    """
    payload = {
        "doc_id": evidence.doc_id,
        "version_id": evidence.version_id,
        "page_id": evidence.page_id,
        "object_id": evidence.object_id,
        "bbox": evidence.bbox,
        "obj_type": evidence.obj_type,
        "extraction_method": evidence.extraction_method,
    }

    payload_str = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


def verify_evidence(evidence: RegionEvidence, expected_hash: str) -> bool:
    """
    Verify that evidence hash matches expected value.

    Args:
        evidence: RegionEvidence to verify
        expected_hash: Expected hash value

    Returns:
        True if hash matches
    """
    actual_hash = hash_evidence(evidence)
    return actual_hash == expected_hash
