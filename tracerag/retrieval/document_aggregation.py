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

    def compute_doc_prior(self, query: str) -> Dict[str, float]:
        """
        Compute a doc_prior score for every document based on folder/filename overlap
        with the query. Uses the same formula as the pipeline's scoring contract:
          +2.0 per matched alphanumeric code (first match only)
          +1.5 per matched Chinese segment (first match only)
          +2.0 for filename alphanumeric match (first match only)
          +1.5 for filename Chinese segment match (first match only)

        Returns a dict mapping doc_id -> doc_prior (0.0 for non-matching docs).
        """
        alphanumerics = re.findall(r'[A-Za-z0-9][A-Za-z0-9\-_.]{1,}', query)
        cn_stop_words = {'有哪些', '什么是', '列出', '多少', '怎么', '如何', '一个', '这张', '这些'}
        cn_chars = re.findall(r'[\u4e00-\u9fff]', query)
        cn_segments = re.findall(r'[\u4e00-\u9fff]{2,}', query)
        cn_segments = [s for s in cn_segments if s not in cn_stop_words]

        doc_prior_map: Dict[str, float] = {}
        for doc_id, abs_path_str in self.manifest.items():
            path = Path(abs_path_str)
            folder_name = path.parent.name.lower()
            filename = path.stem.lower()
            prior = 0.0

            # Folder signal
            for code in alphanumerics:
                if code.lower() in folder_name:
                    prior += 2.0
                    break
            for seg in cn_segments:
                if seg in folder_name:
                    prior += 1.5
                    break

            # Filename signal
            for code in alphanumerics:
                if code.lower() in filename:
                    prior += 2.0
                    break
            for seg in cn_segments:
                if seg in filename:
                    prior += 1.5
                    break

            doc_prior_map[doc_id] = prior

        return doc_prior_map

    def retrieve_candidate_documents(self, query: str) -> List[Tuple[str, float]]:
        """
        Rank documents by their doc_prior score (folder/filename overlap with query).
        Returns a list of (doc_id, doc_prior) sorted descending, non-zero only.
        """
        doc_prior_map = self.compute_doc_prior(query)
        ranked_docs = sorted(
            ((doc_id, score) for doc_id, score in doc_prior_map.items() if score > 0),
            key=lambda x: x[1],
            reverse=True,
        )
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
        """
        Produce a RegionEvidence list representing the unique documents matched.

        Scoring uses the doc_prior from retrieve_candidate_documents as the
        primary ranking signal.  Each evidence carries:
          base_visual_score = 0.0   (no visual scoring in document-list mode)
          symbolic_bonus    = 0.0   (no per-object text matching at this level)
          doc_prior         = score computed from folder/filename overlap
          score             = doc_prior (assembled once, consistent with pipeline contract)
        """
        logger.info(f"Triggering Document-Level Aggregation for: {query}")

        # Retrieve candidates ranked by doc_prior
        candidates = self.retrieve_candidate_documents(query)
        logger.info(f"Found {len(candidates)} candidate documents based on folder/filename priors.")

        # Build a doc_id -> doc_prior lookup for later injection
        doc_prior_map: Dict[str, float] = {doc_id: prior for doc_id, prior in candidates}

        # Extract identities for top candidates (top 50)
        identities = []
        for doc_id, prior in candidates[:50]:
            ident = self.extract_document_identity(doc_id)
            ident["doc_prior"] = prior
            identities.append(ident)

        # Deduplicate
        unique_docs = self.aggregate_documents(identities)
        logger.info(f"Deduplicated down to {len(unique_docs)} unique document identities.")

        # Convert to RegionEvidence so the pipeline can consume it natively
        from tracerag.common.types import VectorObject
        evidences = []
        for i, doc in enumerate(unique_docs):
            display_text = f"【{doc['folder']}】 {doc['extracted_title']}"
            if doc['certificate_number']:
                display_text += f" (编号/No: {doc['certificate_number']})"

            d_prior = doc.get("doc_prior", doc_prior_map.get(doc["doc_id"], 0.0))

            synthetic_ev = RegionEvidence(
                doc_id=doc["doc_id"],
                version_id="v1",
                page_id=f"{doc['doc_id']}_v1_p0",
                object_id=f"doc_agg_{i}",
                bbox=(0, 0, 0, 0),
                obj_type="document_identity",
                extraction_method="document_aggregation",
                score=d_prior,
                base_visual_score=0.0,
                symbolic_bonus=0.0,
                doc_prior=d_prior,
                hash="",
            )

            # Register a dummy VectorObject so downstream consumers can read obj.text
            dummy_obj = VectorObject(
                object_id=f"doc_agg_{i}",
                doc_id=doc["doc_id"],
                version_id="v1",
                page_id=synthetic_ev.page_id,
                bbox=(0, 0, 0, 0),
                obj_type="document_identity",
                text=display_text,
                path_ops=[],
            )
            self.spatial_index.add_object(dummy_obj)

            evidences.append(synthetic_ev)

        return evidences
