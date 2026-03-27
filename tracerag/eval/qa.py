"""
QA Evaluation Module.
Evaluates the quality of TraceRAG's answers against ground truth data.
"""

from typing import Dict, Any, List
from tracerag.common.types import BenchmarkExample
from tracerag.eval.metrics import canonical_route_name
from tracerag.retrieval.evidence_pack import EvidencePack

def _token_f1(prediction: str, ground_truth: str) -> float:
    """Simple token-level F1 score."""
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


class QAEvaluator:
    """Evaluates question answering performance."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def evaluate_question(
        self,
        question: BenchmarkExample,
        result: EvidencePack
    ) -> Dict[str, Any]:
        """
        Evaluate a single question.

        Args:
            question: BenchmarkExample with ground truth
            result: EvidencePack from the system

        Returns:
            Dictionary of metrics
        """
        answer_text = result.answer or ""
        metrics = {
            "query_id": question.query_id,
            "route_name": canonical_route_name(result.metadata.get("route_name"), fallback=question.query_type),
            "query_type": question.query_type,
            "top_candidate_traces": result.metadata.get("top_candidate_traces", []),
            "trace_summary": result.metadata.get("trace_summary", {}),
        }

        # Answer metrics
        if question.ground_truth_answer:
            metrics["answer_f1"] = _token_f1(answer_text, question.ground_truth_answer)
            metrics["answer_em"] = 1.0 if answer_text.strip().lower() == question.ground_truth_answer.strip().lower() else 0.0
            metrics["answer_correct"] = metrics["answer_em"] == 1.0
            
            # Subtask specific set metrics for Document Aggregation
            if question.query_type == "document_aggregation":
                pred_tokens = [t.strip() for t in answer_text.lower().replace(',', ' ').split()]
                gt_tokens = [t.strip() for t in question.ground_truth_answer.lower().replace(',', ' ').split()]
                
                common = set(pred_tokens) & set(gt_tokens)
                set_p = len(common) / len(pred_tokens) if pred_tokens else 0.0
                set_r = len(common) / len(gt_tokens) if gt_tokens else 0.0
                set_f1 = (2 * set_p * set_r) / (set_p + set_r) if (set_p + set_r) > 0 else 0.0
                
                metrics["set_precision"] = set_p
                metrics["set_recall"] = set_r
                metrics["set_f1"] = set_f1
        else:
            metrics["answer_f1"] = 0.0
            metrics["answer_em"] = 0.0
            metrics["answer_correct"] = False
            if question.query_type == "document_aggregation":
                metrics["set_precision"] = 0.0
                metrics["set_recall"] = 0.0
                metrics["set_f1"] = 0.0

        # Retrieve evidence check
        retrieved_page_ids = {ev.page_id for ev in result.evidences}
        relevant = question.relevant_pages
        if relevant:
            hits = sum(1 for p in relevant if p in retrieved_page_ids)
            metrics["page_recall"] = hits / len(relevant)
        else:
            metrics["page_recall"] = 0.0

        # Evidence object overlap
        retrieved_obj_ids = {ev.object_id for ev in result.evidences}
        relevant_objs = question.relevant_objects
        if relevant_objs:
            obj_hits = sum(1 for o in relevant_objs if o in retrieved_obj_ids)
            metrics["evidence_precision"] = obj_hits / len(retrieved_obj_ids) if retrieved_obj_ids else 0.0
            metrics["evidence_recall"] = obj_hits / len(relevant_objs)
            metrics["evidence_correct"] = obj_hits > 0
        else:
            metrics["evidence_precision"] = 0.0
            metrics["evidence_recall"] = 0.0
            metrics["evidence_correct"] = False

        metrics["wrong_value"] = bool(metrics.get("evidence_correct")) and not bool(metrics.get("answer_correct"))
        metrics["wrong_scope"] = bool(result.evidences) and not bool(metrics.get("evidence_correct"))
        metrics["contradiction_failure"] = (
            float(metrics["trace_summary"].get("max_contradiction_penalty", 0.0) or 0.0) >= 0.18
            and not bool(metrics.get("answer_correct"))
        )

        return metrics
