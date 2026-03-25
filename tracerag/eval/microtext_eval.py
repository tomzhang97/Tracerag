"""
Micro-text evaluation script.

Evaluates TraceRAG's ability to extract small text labels, part numbers,
and configuration codes from technical documents.

Metrics:
- Answer accuracy@1: Exact match of extracted value
- Evidence IoU: Bounding box overlap with ground truth
- Object ID match: Whether evidence points to correct PDF object
- Performance by font size bins
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from loguru import logger
from dataclasses import dataclass

from tracerag.common.types import RegionEvidence, QueryResult
from tracerag.common.geometry import bbox_iou
from tracerag.eval.metrics import precision_at_k, recall_at_k, bbox_iou_score


@dataclass
class MicroTextQuestion:
    """Single micro-text question with ground truth."""
    query_id: str
    query: str
    doc_id: str
    version_id: str
    answer_text: str  # Expected answer
    gt_page_id: str  # Ground truth page
    gt_object_id: str  # Ground truth object ID
    gt_bbox: Tuple[float, float, float, float]  # Ground truth bbox
    font_height_px: float  # Font size for binning
    metadata: Dict[str, Any]


class MicroTextEvaluator:
    """
    Evaluator for micro-text extraction benchmark.
    """

    def __init__(self, system, config: Dict[str, Any] = None):
        """
        Initialize evaluator.

        Args:
            system: TraceRAGSystem instance
            config: Optional configuration
        """
        self.system = system
        self.config = config or {}
        self.iou_threshold = self.config.get("iou_threshold", 0.5)

    def evaluate_question(
        self,
        question: MicroTextQuestion
    ) -> Dict[str, Any]:
        """
        Evaluate single micro-text question.

        Args:
            question: MicroTextQuestion

        Returns:
            Evaluation results dictionary
        """
        # Run query
        result = self.system.answer(question.query)

        # Extract predicted answer
        # Look for extracted values in claims
        predicted_answer = None
        top_evidence = None

        if result.certified_claims:
            # Try to find claim with value
            for claim in result.certified_claims:
                if claim.claim.value:
                    predicted_answer = claim.claim.value
                    if claim.evidences:
                        top_evidence = claim.evidences[0]
                    break

            # Fallback: use first claim text
            if predicted_answer is None:
                predicted_answer = result.certified_claims[0].claim.text
                if result.certified_claims[0].evidences:
                    top_evidence = result.certified_claims[0].evidences[0]

        # Evaluate answer accuracy
        answer_correct = self._check_answer_match(
            predicted_answer,
            question.answer_text
        )

        # Evaluate evidence grounding
        evidence_results = {}
        if top_evidence:
            # Check IoU
            iou = bbox_iou(top_evidence.bbox, question.gt_bbox)
            evidence_results["iou"] = iou
            evidence_results["iou_pass"] = iou >= self.iou_threshold

            # Check object ID match
            evidence_results["object_id_match"] = (
                top_evidence.object_id == question.gt_object_id
            )

            # Check page ID match
            evidence_results["page_id_match"] = (
                top_evidence.page_id == question.gt_page_id
            )
        else:
            evidence_results["iou"] = 0.0
            evidence_results["iou_pass"] = False
            evidence_results["object_id_match"] = False
            evidence_results["page_id_match"] = False

        return {
            "query_id": question.query_id,
            "answer_correct": answer_correct,
            "predicted_answer": predicted_answer,
            "gt_answer": question.answer_text,
            "evidence": evidence_results,
            "font_height_px": question.font_height_px,
        }

    def evaluate_dataset(
        self,
        questions: List[MicroTextQuestion]
    ) -> Dict[str, Any]:
        """
        Evaluate entire micro-text dataset.

        Args:
            questions: List of MicroTextQuestions

        Returns:
            Aggregated evaluation results
        """
        logger.info(f"Evaluating {len(questions)} micro-text questions")

        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Evaluating question {i}/{len(questions)}: {question.query_id}")
            result = self.evaluate_question(question)
            results.append(result)

        # Aggregate metrics
        aggregated = self._aggregate_results(results)

        return aggregated

    def _check_answer_match(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if predicted answer matches ground truth.

        Uses fuzzy matching to handle minor formatting differences.

        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            True if match
        """
        if predicted is None or ground_truth is None:
            return False

        # Normalize
        pred_norm = predicted.strip().lower().replace(" ", "")
        gt_norm = ground_truth.strip().lower().replace(" ", "")

        # Exact match
        if pred_norm == gt_norm:
            return True

        # Substring match (predicted contains ground truth or vice versa)
        if gt_norm in pred_norm or pred_norm in gt_norm:
            return True

        return False

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate evaluation results.

        Args:
            results: List of per-question results

        Returns:
            Aggregated metrics
        """
        total = len(results)

        # Overall metrics
        answer_accuracy = sum(r["answer_correct"] for r in results) / total
        iou_scores = [r["evidence"]["iou"] for r in results]
        mean_iou = np.mean(iou_scores)
        iou_pass_rate = sum(r["evidence"]["iou_pass"] for r in results) / total
        object_id_accuracy = sum(r["evidence"]["object_id_match"] for r in results) / total
        page_id_accuracy = sum(r["evidence"]["page_id_match"] for r in results) / total

        # Performance by font size bins
        font_bins = self._compute_font_size_bins(results)

        aggregated = {
            "total_questions": total,
            "overall": {
                "answer_accuracy@1": answer_accuracy,
                "mean_iou": mean_iou,
                "iou_pass_rate": iou_pass_rate,
                "object_id_accuracy": object_id_accuracy,
                "page_id_accuracy": page_id_accuracy,
            },
            "by_font_size": font_bins,
            "per_question": results,
        }

        return aggregated

    def _compute_font_size_bins(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics binned by font size.

        Args:
            results: Per-question results

        Returns:
            Metrics for each font size bin
        """
        # Define bins: tiny (<8px), small (8-12px), medium (12-16px), large (>16px)
        bins = {
            "tiny_lt8px": [],
            "small_8to12px": [],
            "medium_12to16px": [],
            "large_gt16px": [],
        }

        for r in results:
            font_size = r["font_height_px"]

            if font_size < 8:
                bin_name = "tiny_lt8px"
            elif font_size < 12:
                bin_name = "small_8to12px"
            elif font_size < 16:
                bin_name = "medium_12to16px"
            else:
                bin_name = "large_gt16px"

            bins[bin_name].append(r)

        # Compute metrics for each bin
        bin_metrics = {}
        for bin_name, bin_results in bins.items():
            if len(bin_results) == 0:
                continue

            bin_metrics[bin_name] = {
                "count": len(bin_results),
                "answer_accuracy": sum(r["answer_correct"] for r in bin_results) / len(bin_results),
                "mean_iou": np.mean([r["evidence"]["iou"] for r in bin_results]),
            }

        return bin_metrics

    def print_summary(self, aggregated: Dict[str, Any]):
        """
        Print evaluation summary.

        Args:
            aggregated: Aggregated results
        """
        print("\n" + "=" * 60)
        print("MICRO-TEXT EVALUATION RESULTS")
        print("=" * 60)

        overall = aggregated["overall"]
        print(f"\nOverall Metrics (N={aggregated['total_questions']}):")
        print(f"  Answer Accuracy@1:     {overall['answer_accuracy@1']:.3f}")
        print(f"  Mean IoU:              {overall['mean_iou']:.3f}")
        print(f"  IoU Pass Rate (≥0.5):  {overall['iou_pass_rate']:.3f}")
        print(f"  Object ID Accuracy:    {overall['object_id_accuracy']:.3f}")
        print(f"  Page ID Accuracy:      {overall['page_id_accuracy']:.3f}")

        print("\nPerformance by Font Size:")
        for bin_name, metrics in aggregated["by_font_size"].items():
            print(f"  {bin_name} (N={metrics['count']}):")
            print(f"    Answer Accuracy: {metrics['answer_accuracy']:.3f}")
            print(f"    Mean IoU:        {metrics['mean_iou']:.3f}")

        print("=" * 60 + "\n")
