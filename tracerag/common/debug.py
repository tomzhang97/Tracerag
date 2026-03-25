"""
Debugging tools for visualizing TraceRAG stages.
Generates trace JSONs and visual overlays for pipeline stages.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from PIL import Image, ImageDraw

from tracerag.common.types import RegionEvidence

class DebugExporter:
    """Exports structured traces and visual maps for debugging run stages."""
    
    def __init__(self, export_dir: str):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
    def save_failure_trace(self, query_id: str, payload: Dict[str, Any]):
        """Save a complete JSON trace for a failed query."""
        path = self.export_dir / "traces" / f"{query_id}_failure.json"
        path.parent.mkdir(exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save failure trace: {e}")

    def save_patch_heatmap(self, page_id: str, scores: np.ndarray, out_name: str = "heatmap"):
        """Save a simple numpy array represention of patch scores."""
        path = self.export_dir / "heatmaps" / f"{page_id}_{out_name}.npy"
        path.parent.mkdir(exist_ok=True)
        np.save(path, scores)

    def save_snapped_objects_trace(self, page_id: str, evidences: List[RegionEvidence], out_name: str = "snapped"):
        """Save a JSON summary of snapped evidences."""
        path = self.export_dir / "overlays" / f"{page_id}_{out_name}.json"
        path.parent.mkdir(exist_ok=True)
        data = [e.model_dump() for e in evidences]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
            
    def export_revision_diff(self, pair_id: str, diff_summary: str):
        """Export a pure text diff summary."""
        path = self.export_dir / "diffs" / f"{pair_id}.txt"
        path.parent.mkdir(exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(diff_summary)
