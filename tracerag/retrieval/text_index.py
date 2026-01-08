"""
Text-based indexing for fast retrieval of pages/objects by content.

Supports BM25 (sparse) and FAISS (dense) indexing.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from loguru import logger

from tracerag.common.types import VectorObject


class TextIndex:
    """
    Text index for fast retrieval based on text content.

    Supports two modes:
    - BM25: Sparse keyword-based retrieval (fast, good for identifiers)
    - FAISS: Dense semantic retrieval (slower, good for semantic queries)
    """

    def __init__(self, index_type: str = "bm25"):
        """
        Initialize text index.

        Args:
            index_type: 'bm25' or 'faiss'
        """
        self.index_type = index_type
        self.documents: List[Dict[str, Any]] = []  # List of {id, text, metadata}
        self.doc_id_to_idx: Dict[str, int] = {}

        if index_type == "bm25":
            self._init_bm25()
        elif index_type == "faiss":
            self._init_faiss()
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def _init_bm25(self):
        """Initialize BM25 index."""
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = None  # Will be built after adding documents
            self.BM25Okapi = BM25Okapi
        except ImportError:
            logger.error("rank_bm25 not installed. Install with: pip install rank-bm25")
            raise

    def _init_faiss(self):
        """Initialize FAISS index."""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            self.faiss_index = None  # Will be built after adding documents
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        except ImportError:
            logger.error("faiss-cpu or sentence-transformers not installed")
            raise

    def add_object(self, obj: VectorObject):
        """
        Add VectorObject to index.

        Args:
            obj: VectorObject with text content
        """
        if not obj.text:
            return

        doc_id = obj.object_id
        doc = {
            "id": doc_id,
            "text": obj.text,
            "page_id": obj.page_id,
            "doc_id": obj.doc_id,
            "version_id": obj.version_id,
            "obj_type": obj.obj_type,
            "bbox": obj.bbox,
        }

        idx = len(self.documents)
        self.documents.append(doc)
        self.doc_id_to_idx[doc_id] = idx

    def add_objects_batch(self, objects: List[VectorObject]):
        """
        Add multiple objects to index.

        Args:
            objects: List of VectorObjects
        """
        for obj in objects:
            self.add_object(obj)

    def build(self):
        """
        Build index from added documents.
        Call this after adding all documents.
        """
        logger.info(f"Building {self.index_type} index with {len(self.documents)} documents")

        if self.index_type == "bm25":
            self._build_bm25()
        elif self.index_type == "faiss":
            self._build_faiss()

    def _build_bm25(self):
        """Build BM25 index."""
        # Tokenize documents
        tokenized_docs = [doc["text"].lower().split() for doc in self.documents]
        self.bm25 = self.BM25Okapi(tokenized_docs)
        logger.info("BM25 index built")

    def _build_faiss(self):
        """Build FAISS index."""
        import faiss

        # Encode all documents
        texts = [doc["text"] for doc in self.documents]
        embeddings = self.encoder.encode(texts, show_progress_bar=True)

        # Build FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine with normalized vectors)

        # Normalize embeddings
        faiss.normalize_L2(embeddings)

        self.faiss_index.add(embeddings)
        logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search index for query.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of (doc_id, score, metadata) tuples, sorted by relevance
        """
        if self.index_type == "bm25":
            return self._search_bm25(query, top_k)
        elif self.index_type == "faiss":
            return self._search_faiss(query, top_k)

    def _search_bm25(self, query: str, top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search using BM25."""
        if self.bm25 is None:
            logger.warning("BM25 index not built. Call build() first.")
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            score = float(scores[idx])
            results.append((doc["id"], score, doc))

        return results

    def _search_faiss(self, query: str, top_k: int) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search using FAISS."""
        import faiss

        if self.faiss_index is None:
            logger.warning("FAISS index not built. Call build() first.")
            return []

        # Encode query
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            results.append((doc["id"], float(score), doc))

        return results

    def get_page_ids(self, query: str, top_k: int = 10) -> List[str]:
        """
        Get unique page IDs for top-k results.

        Args:
            query: Query string
            top_k: Number of pages to return

        Returns:
            List of page IDs
        """
        results = self.search(query, top_k * 3)  # Get more to ensure enough unique pages

        page_ids = []
        seen = set()

        for _, _, doc in results:
            page_id = doc["page_id"]
            if page_id not in seen:
                page_ids.append(page_id)
                seen.add(page_id)

            if len(page_ids) >= top_k:
                break

        return page_ids
