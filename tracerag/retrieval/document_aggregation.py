"""
Document-Level Aggregation Mode.

Specialized retrieval mode for "what certificates exist" or "what reports exist".
Aggregates at the document level using folder names, filenames, and front-page title content
rather than arbitrary deep-page text blocks.
"""

from typing import List, Dict, Any, Tuple
from loguru import logger
import re
from pathlib import Path
import difflib

from tracerag.common.types import RegionEvidence

def classify_aggregation_scope(query: str) -> str:
    """
    Classify whether a list/aggregation query is asking for:
    - 'document_list': enumerating files/docs (certificates, reports, manuals)
    - 'object_list': enumerating items within docs (parameters, parts, warnings)
    """
    q = query.lower()
    
    # Document-level targets
    doc_keywords = ["证书", "报告", "说明书", "检测", "文件", "手册", "图纸", "certificate", "report", "manual"]
    if any(k in q for k in doc_keywords):
        return "document_list"
        
    return "object_list"


class DocumentAggregationProcessor:
    """Handles retrieval and deduplication for Document-Level collections."""

    def __init__(self, manifest: Dict[str, str], spatial_index):
        self.manifest = manifest
        self.spatial_index = spatial_index

    def retrieve_candidate_documents(self, query: str) -> List[Tuple[str, float]]:
        """
        Score documents based on fast metadata (folder, filename) and query overlap.
        Returns a list of (doc_id, score)
        """
        # Extract target entities from query (e.g. "LEHY-L-S", "型式试验")
        # Very simple heuristic extraction for scoring
        alphanumerics = re.findall(r'[A-Za-z0-9][A-Za-z0-9\-_.]{1,}', query)
        cn_chars = re.findall(r'[\u4e00-\u9fff]', query)
        cn_terms = [cn_chars[i] + cn_chars[i+1] for i in range(len(cn_chars) - 1)] if len(cn_chars) > 1 else cn_chars
        
        doc_scores = {}
        for doc_id, abs_path_str in self.manifest.items():
            path = Path(abs_path_str)
            folder_name = path.parent.name.lower()
            filename = path.stem.lower()
            
            score = 0.0
            
            # 1. Folder match
            for term in cn_terms + alphanumerics:
                if term.lower() in folder_name:
                    score += 5.0  # Strong prior for correct functional folder
                    
            # 2. Filename match
            for term in cn_terms + alphanumerics:
                if term.lower() in filename:
                    score += 5.0
                    
            if score > 0:
                doc_scores[doc_id] = score
                
        # Sort docs by score
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_docs

    def extract_document_identity(self, doc_id: str) -> Dict[str, Any]:
        """
        Extract the core identity of the document (Title, Certificate #) 
        from the first few objects/pages of the document in the spatial index.
        """
        abs_path = self.manifest.get(doc_id, "")
        path = Path(abs_path) if abs_path else None
        
        identity = {
            "doc_id": doc_id,
            "filename": path.name if path else doc_id,
            "folder": path.parent.name if path else "unknown",
            "extracted_title": "",
            "certificate_number": ""
        }
        
        # In a full system, you would grab page 0 objects and run regex:
        # For now, we fall back heavily to the filename which contains rich info in Engineering corpora
        
        # Use stem as the primary title unless we extract something better
        identity["extracted_title"] = path.stem if path else doc_id
        
        # Look for explicit certificate blocks on page 0
        p0_id = f"{doc_id}_v1_p0"
        try:
            p0_objs = self.spatial_index.get_page_objects(p0_id)
            for obj in p0_objs[:10]: # Check top blocks
                text = obj.text or ""
                # Heuristic certificate number extraction
                m = re.search(r'(证书编号|报告编号|编号|No\.|NO\.)[:：\s]*([A-Za-z0-9\-\_]+)', text)
                if m:
                    identity["certificate_number"] = m.group(2)
                    break
        except Exception:
            pass # page 0 might not be loaded if filtered out
            
        return identity

    def aggregate_documents(self, identities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate at the document identity level.
        If two PDFs have the same extract_title or cert number, collapse them.
        """
        unique_docs = []
        seen_titles = set()
        seen_numbers = set()
        
        for ident in identities:
            title = ident["extracted_title"]
            number = ident["certificate_number"]
            
            # Normalize title for fuzzy matching
            norm_title = re.sub(r'[\s\n\(\)（）_]+', '', title.lower())
            
            is_dup = False
            
            # Check number strict match
            if number and number in seen_numbers:
                is_dup = True
                
            # Check title fuzzy match
            if not is_dup:
                for seen in seen_titles:
                    if difflib.SequenceMatcher(None, norm_title, seen).ratio() > 0.85:
                        is_dup = True
                        break
            
            if not is_dup:
                seen_titles.add(norm_title)
                if number:
                    seen_numbers.add(number)
                unique_docs.append(ident)
        
        return unique_docs

    def process(self, query: str) -> List[RegionEvidence]:
        """Produce a fake/injected RegionEvidence list representing the documents."""
        logger.info(f"Triggering Document-Level Aggregation for: {query}")
        
        candidates = self.retrieve_candidate_documents(query)
        logger.info(f"Found {len(candidates)} candidate documents based on folder/filename priors.")
        
        # Extract identities for top candidates (e.g. top 50)
        identities = []
        for doc_id, score in candidates[:50]:
            ident = self.extract_document_identity(doc_id)
            identities.append(ident)
            
        # Deduplicate
        unique_docs = self.aggregate_documents(identities)
        logger.info(f"Deduplicated down to {len(unique_docs)} unique document identities.")
        
        # Convert to RegionEvidence so the pipeline can consume it natively
        evidences = []
        for i, doc in enumerate(unique_docs):
            # Create a synthetic evidence block describing the document
            display_text = f"【{doc['folder']}】 {doc['extracted_title']}"
            if doc['certificate_number']:
                display_text += f" (编号/No: {doc['certificate_number']})"
                
            Synthetic_Ev = RegionEvidence(
                doc_id=doc["doc_id"],
                version_id="v1",
                page_id=f"{doc['doc_id']}_v1_p0",
                object_id=f"doc_agg_{i}",
                bbox=(0,0,0,0),
                obj_type="document_identity",
                extraction_method="document_aggregation",
                score=100.0 - i, # Enforce synthetic ranking
                hash="",
            )
            
            # We must monkey-patch the spatial index to return a dummy VectorObject 
            # so `main.py` can print `obj.text` successfully!
            from tracerag.common.types import VectorObject
            dummy_obj = VectorObject(
                object_id=f"doc_agg_{i}",
                doc_id=doc["doc_id"],
                version_id="v1",
                page_id=Synthetic_Ev.page_id,
                bbox=(0,0,0,0),
                obj_type="document_identity",
                text=display_text,
                path_ops=[],
            )
            self.spatial_index.add_object(dummy_obj)
            
            evidences.append(Synthetic_Ev)
            
        return evidences
