"""
Utility functions for TraceRAG.
"""

import hashlib
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
from loguru import logger
from tracerag.common.types import BBox


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, loads default config.

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Load default config
        default_config = Path(__file__).parent.parent / "config" / "defaults.yaml"
        config_path = str(default_config)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Expand environment variables in paths
    config = expand_env_vars(config)

    return config


def expand_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively expand ${VAR} references in config values.

    Args:
        config: Configuration dictionary

    Returns:
        Config with expanded variables
    """
    def expand(value):
        if isinstance(value, str):
            # Simple ${VAR} expansion
            while "${" in value:
                start = value.index("${")
                end = value.index("}", start)
                var_name = value[start+2:end]
                # Try to find in config first, then environment
                var_value = config.get(var_name, os.environ.get(var_name, ""))
                value = value[:start] + str(var_value) + value[end+1:]
            return value
        elif isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [expand(item) for item in value]
        else:
            return value

    return expand(config)


def setup_logging(config: Dict[str, Any]):
    """
    Set up logging based on configuration.

    Args:
        config: Configuration dictionary
    """
    log_config = config.get("logging", {})
    level = log_config.get("level", "INFO")
    log_file = log_config.get("file")
    log_format = log_config.get("format",
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}")

    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        lambda msg: print(msg, end=""),
        format=log_format,
        level=level,
        colorize=True
    )

    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(
            log_file,
            format=log_format,
            level=level,
            rotation="100 MB",
            retention="30 days"
        )


def bbox_iou(box1: BBox, box2: BBox) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1: First bounding box (x_min, y_min, x_max, y_max)
        box2: Second bounding box (x_min, y_min, x_max, y_max)

    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def bbox_intersection(box1: BBox, box2: BBox) -> float:
    """
    Calculate intersection area of two bounding boxes.

    Args:
        box1: First bounding box
        box2: Second bounding box

    Returns:
        Intersection area
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return (x2 - x1) * (y2 - y1)


def bbox_area(box: BBox) -> float:
    """Calculate area of bounding box."""
    return (box[2] - box[0]) * (box[3] - box[1])


def bbox_center(box: BBox) -> Tuple[float, float]:
    """Calculate center point of bounding box."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def bbox_distance(box1: BBox, box2: BBox) -> float:
    """
    Calculate Euclidean distance between centers of two bounding boxes.

    Args:
        box1: First bounding box
        box2: Second bounding box

    Returns:
        Distance between centers
    """
    c1 = bbox_center(box1)
    c2 = bbox_center(box2)
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5


def normalize_bbox(bbox: BBox, page_width: float, page_height: float) -> BBox:
    """
    Normalize bounding box to [0, 1] range.

    Args:
        bbox: Bounding box in page coordinates
        page_width: Width of page
        page_height: Height of page

    Returns:
        Normalized bounding box
    """
    return (
        bbox[0] / page_width,
        bbox[1] / page_height,
        bbox[2] / page_width,
        bbox[3] / page_height
    )


def denormalize_bbox(bbox: BBox, page_width: float, page_height: float) -> BBox:
    """
    Denormalize bounding box from [0, 1] to page coordinates.

    Args:
        bbox: Normalized bounding box
        page_width: Width of page
        page_height: Height of page

    Returns:
        Bounding box in page coordinates
    """
    return (
        bbox[0] * page_width,
        bbox[1] * page_height,
        bbox[2] * page_width,
        bbox[3] * page_height
    )


def hash_content(content: str) -> str:
    """
    Generate SHA256 hash of content.

    Args:
        content: Content to hash

    Returns:
        Hex digest of hash
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def ensure_dir(path: str):
    """
    Ensure directory exists, create if not.

    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_page_id(doc_id: str, version_id: str, page_num: int) -> str:
    """
    Generate standard page ID.

    Args:
        doc_id: Document ID
        version_id: Version ID
        page_num: Page number (0-indexed)

    Returns:
        Page ID string
    """
    return f"{doc_id}_{version_id}_p{page_num}"


def parse_page_id(page_id: str) -> Tuple[str, str, int]:
    """
    Parse page ID into components.

    Args:
        page_id: Page ID string (format: {doc_id}_{version_id}_p{page_num})

    Returns:
        Tuple of (doc_id, version_id, page_num)
    """
    parts = page_id.rsplit('_p', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid page_id format: {page_id}")

    prefix = parts[0]
    page_num = int(parts[1])

    # Split prefix into doc_id and version_id
    # Assuming format: {doc_id}_{version_id}
    doc_version_parts = prefix.split('_', 1)
    if len(doc_version_parts) != 2:
        raise ValueError(f"Invalid page_id format: {page_id}")

    doc_id, version_id = doc_version_parts
    return doc_id, version_id, page_num
