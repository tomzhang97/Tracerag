"""
Candidate filtering for retrieval pipeline.
Returns initial set of pages to consider before visual inspection.
"""

from typing import List, Dict, Any, Optional
from tracerag.retrieval.text_index import TextIndex


class CandidateFilter:
    """Filters candidate pages using fast retrieval methods (e.g. text/BM25)."""

    def __init__(self, text_index: TextIndex, config: Optional[Dict[str, Any]] = None):
        self.text_index = text_index
        self.config = config or {}
        self.top_k = self.config.get("top_k_pages", 10)

    def get_candidate_pages(self, query: str) -> List[str]:
        """
        Retrieve candidate pages based on query.

        Args:
            query: Query string

        Returns:
            List of page IDs
        """
        # A full system could branch on query type here, but for now we unify
        return self.text_index.get_page_ids(query, self.top_k)
