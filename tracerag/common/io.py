"""
IO and hashing utilities for TraceRAG.
"""

import os
import hashlib
from typing import Tuple

def hash_content(content: str) -> str:
    """Generate SHA256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def ensure_dir(path: str):
    """Ensure directory exists, create if not."""
    os.makedirs(path, exist_ok=True)

def get_page_id(doc_id: str, version_id: str, page_num: int) -> str:
    """Generate standard page ID."""
    return f"{doc_id}_{version_id}_p{page_num}"

def parse_page_id(page_id: str) -> Tuple[str, str, int]:
    """Parse page ID into (doc_id, version_id, page_num)."""
    parts = page_id.rsplit('_p', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid page_id format: {page_id}")
    prefix = parts[0]
    page_num = int(parts[1])
    doc_version_parts = prefix.split('_', 1)
    if len(doc_version_parts) != 2:
        raise ValueError(f"Invalid page_id format: {page_id}")
    doc_id, version_id = doc_version_parts
    return doc_id, version_id, page_num
