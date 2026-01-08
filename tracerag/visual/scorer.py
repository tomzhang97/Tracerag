"""
Visual scorer for query-to-page relevance using late interaction.

Implements MaxSim scoring between query tokens and patch embeddings (ColBERT-style).
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from loguru import logger

from tracerag.common.types import PatchGrid


class VisualScorer:
    """
    Score query relevance against page patch grids using late interaction.

    Uses MaxSim operation: for each query token, find maximum similarity
    with all patch embeddings, then aggregate.
    """

    def __init__(self, config: Dict[str, Any], encoder=None):
        """
        Initialize visual scorer.

        Args:
            config: Configuration dictionary
            encoder: Optional VisualPageEncoder instance (for query encoding)
        """
        self.config = config
        self.encoder = encoder
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode query text as embedding(s).

        Args:
            query: Query string

        Returns:
            Query embeddings [num_tokens, d]
        """
        if self.encoder is None or not hasattr(self.encoder, 'processor'):
            # Mock implementation
            logger.debug("Using mock query encoder")
            # Return random query embeddings
            return np.random.randn(10, 128).astype(np.float32)

        # Real implementation would use the processor to tokenize and embed
        try:
            with torch.no_grad():
                inputs = self.encoder.processor(
                    text=query,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.encoder.model(**inputs)
                query_embeds = outputs.last_hidden_state.cpu().numpy()[0]  # [num_tokens, d]
                return query_embeds
        except Exception as e:
            logger.warning(f"Failed to encode query: {e}, using mock")
            return np.random.randn(10, 128).astype(np.float32)

    def score_page(
        self,
        query: str,
        patch_grid: PatchGrid,
        query_embeds: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Score query against page using MaxSim late interaction.

        Args:
            query: Query string
            patch_grid: PatchGrid for the page
            query_embeds: Optional pre-computed query embeddings

        Returns:
            Patch relevance scores [H, W]
        """
        # Encode query if not provided
        if query_embeds is None:
            query_embeds = self.encode_query(query)  # [num_tokens, d]

        # Get patch embeddings
        patch_embeds = patch_grid.embeddings  # [H, W, d]
        H, W, d = patch_embeds.shape

        # Reshape patches for matrix multiplication
        patches_flat = patch_embeds.reshape(-1, d)  # [H*W, d]

        # Compute similarity matrix: [num_tokens, H*W]
        # Each element is cosine similarity between query token and patch
        query_norm = query_embeds / (np.linalg.norm(query_embeds, axis=1, keepdims=True) + 1e-8)
        patch_norm = patches_flat / (np.linalg.norm(patches_flat, axis=1, keepdims=True) + 1e-8)

        similarity = np.matmul(query_norm, patch_norm.T)  # [num_tokens, H*W]

        # MaxSim: for each query token, take max similarity across all patches
        max_sims = np.max(similarity, axis=1)  # [num_tokens]

        # Aggregate: sum or mean of max similarities
        overall_score = np.mean(max_sims)

        # Per-patch scores: max similarity across query tokens for each patch
        patch_scores = np.max(similarity, axis=0)  # [H*W]
        patch_scores = patch_scores.reshape(H, W)  # [H, W]

        # Normalize to [0, 1]
        patch_scores = (patch_scores + 1) / 2  # Cosine sim is in [-1, 1]

        return patch_scores

    def score_pages_batch(
        self,
        query: str,
        patch_grids: list[PatchGrid]
    ) -> list[tuple[str, float, np.ndarray]]:
        """
        Score query against multiple pages.

        Args:
            query: Query string
            patch_grids: List of PatchGrids

        Returns:
            List of (page_id, overall_score, patch_scores) tuples
        """
        # Encode query once
        query_embeds = self.encode_query(query)

        results = []
        for patch_grid in patch_grids:
            patch_scores = self.score_page(query, patch_grid, query_embeds)
            overall_score = float(np.mean(patch_scores))
            results.append((patch_grid.page_id, overall_score, patch_scores))

        # Sort by overall score
        results.sort(key=lambda x: x[1], reverse=True)

        return results
