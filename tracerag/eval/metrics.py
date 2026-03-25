"""
Evaluation metrics for TraceRAG.

Key metrics:
- Precision@K: Fraction of retrieved items that are relevant
- Recall@K: Fraction of relevant items that are retrieved
- BBox IoU: For spatial grounding accuracy
- Vector-Native Accuracy: Fraction of evidences that snap to correct objects
"""

from typing import List, Set, Tuple
import numpy as np

from tracerag.common.types import BBox, RegionEvidence
from tracerag.common.geometry import bbox_iou


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Precision@K.

    Args:
        retrieved: List of retrieved item IDs (in rank order)
        relevant: Set of relevant item IDs
        k: Cutoff

    Returns:
        Precision@K
    """
    if k == 0:
        return 0.0

    retrieved_at_k = retrieved[:k]
    num_relevant = sum(1 for item in retrieved_at_k if item in relevant)

    return num_relevant / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Recall@K.

    Args:
        retrieved: List of retrieved item IDs (in rank order)
        relevant: Set of relevant item IDs
        k: Cutoff

    Returns:
        Recall@K
    """
    if len(relevant) == 0:
        return 0.0

    retrieved_at_k = set(retrieved[:k])
    num_relevant = len(retrieved_at_k & relevant)

    return num_relevant / len(relevant)


def mean_average_precision(
    retrieved_list: List[List[str]],
    relevant_list: List[Set[str]]
) -> float:
    """
    Calculate Mean Average Precision (MAP).

    Args:
        retrieved_list: List of retrieved lists (one per query)
        relevant_list: List of relevant sets (one per query)

    Returns:
        MAP score
    """
    aps = []

    for retrieved, relevant in zip(retrieved_list, relevant_list):
        if len(relevant) == 0:
            continue

        num_relevant = 0
        sum_precisions = 0.0

        for k, item in enumerate(retrieved, 1):
            if item in relevant:
                num_relevant += 1
                precision_at_this_k = num_relevant / k
                sum_precisions += precision_at_this_k

        if num_relevant > 0:
            ap = sum_precisions / len(relevant)
        else:
            ap = 0.0

        aps.append(ap)

    if len(aps) == 0:
        return 0.0

    return np.mean(aps)


def bbox_iou_score(pred_bbox: BBox, gt_bbox: BBox) -> float:
    """
    Calculate bbox IoU score.

    Args:
        pred_bbox: Predicted bounding box
        gt_bbox: Ground truth bounding box

    Returns:
        IoU score [0, 1]
    """
    return bbox_iou(pred_bbox, gt_bbox)


def vector_native_accuracy(
    predicted_evidences: List[RegionEvidence],
    ground_truth_object_ids: Set[str]
) -> float:
    """
    Calculate vector-native accuracy: fraction of predicted evidences
    that correspond to correct ground truth objects.

    Args:
        predicted_evidences: Predicted evidences
        ground_truth_object_ids: Set of correct object IDs

    Returns:
        Accuracy [0, 1]
    """
    if len(predicted_evidences) == 0:
        return 0.0

    correct = sum(1 for ev in predicted_evidences if ev.object_id in ground_truth_object_ids)
    return correct / len(predicted_evidences)


def spatial_grounding_accuracy(
    predicted_evidences: List[RegionEvidence],
    ground_truth_bboxes: List[BBox],
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate spatial grounding accuracy: fraction of predicted evidences
    whose bboxes overlap with ground truth bboxes above threshold.

    Args:
        predicted_evidences: Predicted evidences
        ground_truth_bboxes: Ground truth bounding boxes
        iou_threshold: IoU threshold for match

    Returns:
        Accuracy [0, 1]
    """
    if len(predicted_evidences) == 0:
        return 0.0

    correct = 0

    for ev in predicted_evidences:
        # Check if any ground truth bbox overlaps above threshold
        for gt_bbox in ground_truth_bboxes:
            if bbox_iou(ev.bbox, gt_bbox) >= iou_threshold:
                correct += 1
                break

    return correct / len(predicted_evidences)


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).

    Args:
        retrieved: List of retrieved item IDs (in rank order)
        relevant: Set of relevant item IDs (binary relevance)
        k: Cutoff

    Returns:
        NDCG@K score
    """
    if k == 0 or len(relevant) == 0:
        return 0.0

    retrieved_at_k = retrieved[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(retrieved_at_k, 1):
        rel = 1.0 if item in relevant else 0.0
        dcg += rel / np.log2(i + 1)

    # IDCG (ideal DCG)
    ideal_retrieved = list(relevant) + [None] * (k - len(relevant))
    idcg = 0.0
    for i in range(min(k, len(relevant))):
        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg
