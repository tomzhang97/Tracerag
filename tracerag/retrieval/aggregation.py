"""
Collection Query Aggregation Module.

Dynamically switches retrieval from top-k precision to high-recall deduplication
when 'list', 'all', '有哪些', etc. are detected in the query.
"""

from typing import List, Set
from loguru import logger
import re
import difflib

from tracerag.common.types import RegionEvidence

class AggregationProcessor:
    """Processes evidence sets for aggregation/list queries."""
    
    def __init__(self):
        # Heuristics for triggering aggregation mode
        self.trigger_keywords = [
            "有哪些", "列出", "列表", "所有", "汇总", 
            "list", "all", "what are the", "which",
            "certifications", "证书"
        ]

    def is_aggregation_query(self, query: str) -> bool:
        """Heuristically determine if query requires collection mode."""
        q_lower = query.lower()
        if "有哪些" in q_lower or "列出所有" in q_lower or "list all" in q_lower:
            return True
            
        # Check if multiple triggering keywords are present
        matches = sum(1 for hw in self.trigger_keywords if hw in q_lower)
        return matches >= 1

    def clean_text(self, text: str) -> str:
        """Normalize string for deduplication (strip spaces, punctuation)."""
        if not text:
            return ""
        # Remove whitespace and common punctuation
        text = re.sub(r'[\s\n\r\t，、。；：！？()（）【】\[\]{}.,;:!?]+', '', text)
        return text.strip()

    def deduplicate(self, evidences: List[RegionEvidence], spatial_index) -> List[RegionEvidence]:
        """
        Deduplicate objects based on semantic string similarity.
        Instead of returning identical strings from 5 different pages, we keep 1.
        """
        unique_evidences = []
        seen_texts: Set[str] = set()
        
        for ev in evidences:
            obj = spatial_index.get_object(ev.object_id)
            if not obj or not obj.text:
                continue
                
            raw_text = obj.text
            norm_text = self.clean_text(raw_text)
            
            # Skip empty or very short garbage
            if len(norm_text) < 2:
                continue
                
            # Perform similarity check against already seen texts
            # Using difflib for simple string matching (similar to Levenshtein distance)
            is_duplicate = False
            for seen in seen_texts:
                # If similarity ratio is > 0.85, consider it a duplicate
                if difflib.SequenceMatcher(None, norm_text, seen).ratio() > 0.85:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                seen_texts.add(norm_text)
                unique_evidences.append(ev)
                
        return unique_evidences
        
    def process(self, query: str, all_evidences: List[RegionEvidence], spatial_index) -> List[RegionEvidence]:
        """
        Main entrypoint. Takes the high-recall expanded evidence pool,
        filters noise, deduplicates, and returns the aggregated set.
        """
        logger.info(f"Triggering Collection Query Mode (Aggregation) for: {query}")
        
        # 1. Gather all texts without strict top-20 visual score cutoff
        # Sort by visual score first so we prioritize the most relevant visuals 
        # before we start dropping duplicates.
        sorted_evidences = sorted(all_evidences, key=lambda e: e.score, reverse=True)
        
        # Take a high-recall pool (e.g., top 300 objects instead of 20)
        high_recall_pool = sorted_evidences[:300]
        
        # 2. Heuristic text filtering 
        # (Exclude layout noise, pure numbers, super long arbitrary text blocks)
        filtered_pool = []
        for ev in high_recall_pool:
            obj = spatial_index.get_object(ev.object_id)
            if not obj or not obj.text:
                continue
                
            # Filter non-informative objects
            # If looking for certificates/lists, usually they are short titles or table cells.
            if len(obj.text) > 200: 
                continue # Block of running text, not a discrete item
                
            filtered_pool.append(ev)
            
        logger.debug(f"Aggregation pool filtered from {len(high_recall_pool)} to {len(filtered_pool)} candidates.")
        
        # 3. Deduplicate 
        final_evidences = self.deduplicate(filtered_pool, spatial_index)
        
        logger.info(f"Aggregation mode completed: Outputting {len(final_evidences)} unique aggregated items.")
        return final_evidences
