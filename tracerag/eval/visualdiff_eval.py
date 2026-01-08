"""
Visual diff evaluation script.

Evaluates TraceRAG's ability to detect and describe changes in technical
documents across revisions.

Metrics:
- Change detection recall: Did the system find evidence in both versions?
- Keyword recall: Does the answer mention key change tokens?
- Spatial grounding: Do evidences overlap with ground truth bboxes?
"""

from typing import List, Dict, Any, Tuple, Set
import numpy as np
from loguru import logger
from dataclasses import dataclass

from tracerag.common.types import RegionEvidence, QueryResult
from tracerag.common.utils import bbox_iou


@dataclass
class VisualDiffQuestion:
    """Single visual diff question with ground truth."""
    query_id: str
    query: str
    doc_id: str
    old_version_id: str
    new_version_id: str
    change_type: str  # "added", "removed", "moved", "modified"
    key_tokens: List[str]  # Keywords that should appear in answer
    bbox_old: Tuple[float, float, float, float]  # Old version bbox (None if added)
    bbox_new: Tuple[float, float, float, float]  # New version bbox (None if removed)
    page_old: str  # Old version page ID
    page_new: str  # New version page ID
    metadata: Dict[str, Any]


class VisualDiffEvaluator:
    """
    Evaluator for visual diff detection benchmark.
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
        self.iou_threshold = self.config.get("iou_threshold", 0.3)

    def evaluate_question(
        self,
        question: VisualDiffQuestion
    ) -> Dict[str, Any]:
        """
        Evaluate single visual diff question.

        Args:
            question: VisualDiffQuestion

        Returns:
            Evaluation results dictionary
        """
        # Run query
        result = self.system.answer(question.query)

        # Collect evidences from both versions
        evidences_old = []
        evidences_new = []

        for claim in result.certified_claims:
            for ev in claim.evidences:
                if ev.version_id == question.old_version_id:
                    evidences_old.append(ev)
                elif ev.version_id == question.new_version_id:
                    evidences_new.append(ev)

        # Check if system found evidence in both versions
        has_evidence_old = len(evidences_old) > 0
        has_evidence_new = len(evidences_new) > 0

        # For "added" changes, we don't expect evidence in old version
        if question.change_type == "added":
            change_detected = has_evidence_new
        # For "removed" changes, we don't expect evidence in new version
        elif question.change_type == "removed":
            change_detected = has_evidence_old
        # For other changes, we expect evidence in both
        else:
            change_detected = has_evidence_old and has_evidence_new

        # Check keyword recall
        answer_text = result.answer.lower()
        keywords_found = []
        keywords_missing = []

        for keyword in question.key_tokens:
            if keyword.lower() in answer_text:
                keywords_found.append(keyword)
            else:
                keywords_missing.append(keyword)

        keyword_recall = len(keywords_found) / len(question.key_tokens) if question.key_tokens else 0.0

        # Check spatial grounding (IoU with ground truth bboxes)
        spatial_grounding_old = self._check_spatial_grounding(
            evidences_old,
            question.bbox_old
        ) if question.bbox_old else None

        spatial_grounding_new = self._check_spatial_grounding(
            evidences_new,
            question.bbox_new
        ) if question.bbox_new else None

        return {
            "query_id": question.query_id,
            "change_type": question.change_type,
            "change_detected": change_detected,
            "has_evidence_old": has_evidence_old,
            "has_evidence_new": has_evidence_new,
            "keyword_recall": keyword_recall,
            "keywords_found": keywords_found,
            "keywords_missing": keywords_missing,
            "spatial_grounding_old": spatial_grounding_old,
            "spatial_grounding_new": spatial_grounding_new,
            "num_claims": len(result.certified_claims),
        }

    def evaluate_dataset(
        self,
        questions: List[VisualDiffQuestion]
    ) -> Dict[str, Any]:
        """
        Evaluate entire visual diff dataset.

        Args:
            questions: List of VisualDiffQuestions

        Returns:
            Aggregated evaluation results
        """
        logger.info(f"Evaluating {len(questions)} visual diff questions")

        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Evaluating question {i}/{len(questions)}: {question.query_id}")
            result = self.evaluate_question(question)
            results.append(result)

        # Aggregate metrics
        aggregated = self._aggregate_results(results)

        return aggregated

    def _check_spatial_grounding(
        self,
        evidences: List[RegionEvidence],
        gt_bbox: Tuple[float, float, float, float]
    ) -> Dict[str, Any]:
        """
        Check if evidences spatially overlap with ground truth bbox.

        Args:
            evidences: List of evidences
            gt_bbox: Ground truth bounding box

        Returns:
            Spatial grounding metrics
        """
        if not evidences or gt_bbox is None:
            return {
                "best_iou": 0.0,
                "iou_pass": False,
            }

        # Compute IoU for each evidence
        ious = [bbox_iou(ev.bbox, gt_bbox) for ev in evidences]
        best_iou = max(ious)

        return {
            "best_iou": best_iou,
            "iou_pass": best_iou >= self.iou_threshold,
        }

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
        change_detection_recall = sum(r["change_detected"] for r in results) / total
        mean_keyword_recall = np.mean([r["keyword_recall"] for r in results])

        # Spatial grounding metrics
        spatial_old_ious = [
            r["spatial_grounding_old"]["best_iou"]
            for r in results
            if r["spatial_grounding_old"] is not None
        ]
        spatial_new_ious = [
            r["spatial_grounding_new"]["best_iou"]
            for r in results
            if r["spatial_grounding_new"] is not None
        ]

        mean_spatial_iou_old = np.mean(spatial_old_ious) if spatial_old_ious else 0.0
        mean_spatial_iou_new = np.mean(spatial_new_ious) if spatial_new_ious else 0.0

        # Performance by change type
        change_type_metrics = self._compute_change_type_metrics(results)

        aggregated = {
            "total_questions": total,
            "overall": {
                "change_detection_recall": change_detection_recall,
                "mean_keyword_recall": mean_keyword_recall,
                "mean_spatial_iou_old": mean_spatial_iou_old,
                "mean_spatial_iou_new": mean_spatial_iou_new,
            },
            "by_change_type": change_type_metrics,
            "per_question": results,
        }

        return aggregated

    def _compute_change_type_metrics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each change type.

        Args:
            results: Per-question results

        Returns:
            Metrics for each change type
        """
        # Group by change type
        by_type = {}
        for r in results:
            change_type = r["change_type"]
            if change_type not in by_type:
                by_type[change_type] = []
            by_type[change_type].append(r)

        # Compute metrics for each type
        type_metrics = {}
        for change_type, type_results in by_type.items():
            type_metrics[change_type] = {
                "count": len(type_results),
                "change_detection_recall": sum(r["change_detected"] for r in type_results) / len(type_results),
                "mean_keyword_recall": np.mean([r["keyword_recall"] for r in type_results]),
            }

        return type_metrics

    def print_summary(self, aggregated: Dict[str, Any]):
        """
        Print evaluation summary.

        Args:
            aggregated: Aggregated results
        """
        print("\n" + "=" * 60)
        print("VISUAL DIFF EVALUATION RESULTS")
        print("=" * 60)

        overall = aggregated["overall"]
        print(f"\nOverall Metrics (N={aggregated['total_questions']}):")
        print(f"  Change Detection Recall: {overall['change_detection_recall']:.3f}")
        print(f"  Mean Keyword Recall:     {overall['mean_keyword_recall']:.3f}")
        print(f"  Mean Spatial IoU (Old):  {overall['mean_spatial_iou_old']:.3f}")
        print(f"  Mean Spatial IoU (New):  {overall['mean_spatial_iou_new']:.3f}")

        print("\nPerformance by Change Type:")
        for change_type, metrics in aggregated["by_change_type"].items():
            print(f"  {change_type} (N={metrics['count']}):")
            print(f"    Change Detection: {metrics['change_detection_recall']:.3f}")
            print(f"    Keyword Recall:   {metrics['mean_keyword_recall']:.3f}")

        print("=" * 60 + "\n")
