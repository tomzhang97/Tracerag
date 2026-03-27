"""
DocVQA benchmark loader and evaluator.

Supports two modes:
1. Prepared TraceRAG-ready JSONL/JSON created by `tracerag-docvqa-prepare`
2. Flexible direct parsing of common DocVQA question exports
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from loguru import logger

from tracerag.eval.metrics import summarize_route_metrics


@dataclass
class DocVQAQuestion:
    question_id: str
    query: str
    answers: List[str]
    doc_id: str
    version_id: str
    page_id: str
    metadata: Dict[str, Any]


def _normalize_doc_id(value: str) -> str:
    stem = Path(str(value or "")).stem
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or "document"


def _normalize_answer(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if char_a == char_b else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def anls_score(predicted: str, ground_truth: str) -> float:
    pred = _normalize_answer(predicted)
    gold = _normalize_answer(ground_truth)
    if not pred or not gold:
        return 1.0 if pred == gold else 0.0
    if pred == gold:
        return 1.0

    distance = _levenshtein_distance(pred, gold)
    normalized_distance = distance / max(len(pred), len(gold))
    if normalized_distance >= 0.5:
        return 0.0
    return 1.0 - normalized_distance


def load_docvqa_questions(data: Any) -> List[DocVQAQuestion]:
    items = data if isinstance(data, list) else data.get("questions") or data.get("data") or data.get("dataset") or []
    questions: List[DocVQAQuestion] = []

    for index, item in enumerate(items):
        query = item.get("query") or item.get("question") or item.get("query_text")
        if not query:
            logger.warning(f"Skipping DocVQA item {index}: missing question text")
            continue

        raw_answers = item.get("answers")
        if raw_answers is None:
            raw_answers = item.get("answer_text") or item.get("answer") or []

        if isinstance(raw_answers, str):
            answers = [raw_answers]
        else:
            answers = [str(answer) for answer in raw_answers if str(answer).strip()]

        question_id = str(item.get("question_id") or item.get("questionId") or item.get("id") or index)
        version_id = str(item.get("version_id") or item.get("version") or "v1")

        doc_source = (
            item.get("doc_id")
            or item.get("document_id")
            or item.get("documentId")
            or item.get("image")
            or item.get("image_id")
            or item.get("imageId")
            or item.get("file_name")
            or item.get("filename")
            or item.get("image_path")
        )
        doc_id = _normalize_doc_id(str(doc_source or question_id))
        page_id = str(item.get("page_id") or f"{doc_id}_{version_id}_p0")

        questions.append(
            DocVQAQuestion(
                question_id=question_id,
                query=query,
                answers=answers,
                doc_id=doc_id,
                version_id=version_id,
                page_id=page_id,
                metadata=item.get("metadata", item),
            )
        )

    return questions


class DocVQAEvaluator:
    def __init__(self, system):
        self.system = system

    def evaluate_question(self, question: DocVQAQuestion) -> Dict[str, Any]:
        result = self.system.answer(question.query)
        predicted_answer = self._extract_predicted_answer(result)

        best_anls = max((anls_score(predicted_answer, answer) for answer in question.answers), default=0.0)
        exact_match = any(_normalize_answer(predicted_answer) == _normalize_answer(answer) for answer in question.answers)
        evidence_correct = any(
            evidence.doc_id == question.doc_id and evidence.page_id == question.page_id
            for evidence in result.evidences
        )

        trace_summary = result.metadata.get("trace_summary", {})
        contradiction_failure = (
            float(trace_summary.get("max_contradiction_penalty", 0.0) or 0.0) >= 0.18
            and not exact_match
        )

        return {
            "query_id": question.question_id,
            "route_name": result.metadata.get("route_name", "standard_qa"),
            "query_type": "docvqa_qa",
            "predicted_answer": predicted_answer,
            "answers": question.answers,
            "answer_correct": exact_match,
            "exact_match": exact_match,
            "anls": best_anls,
            "evidence_correct": evidence_correct,
            "wrong_value": bool(predicted_answer) and not exact_match and evidence_correct,
            "wrong_scope": bool(result.evidences) and not evidence_correct,
            "contradiction_failure": contradiction_failure,
            "top_candidate_traces": result.metadata.get("top_candidate_traces", []),
            "trace_summary": trace_summary,
            "doc_id": question.doc_id,
            "page_id": question.page_id,
        }

    def evaluate_dataset(self, questions: Iterable[DocVQAQuestion]) -> Dict[str, Any]:
        results = [self.evaluate_question(question) for question in questions]
        total = len(results)
        return {
            "total_questions": total,
            "overall": {
                "exact_match": sum(1.0 for item in results if item["exact_match"]) / total if total else 0.0,
                "anls": sum(item["anls"] for item in results) / total if total else 0.0,
                "evidence_correctness": sum(1.0 for item in results if item["evidence_correct"]) / total if total else 0.0,
            },
            "by_route": summarize_route_metrics(results),
            "per_question": results,
        }

    def print_summary(self, aggregated: Dict[str, Any]):
        overall = aggregated["overall"]
        print("\n" + "=" * 60)
        print("DOCVQA EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nOverall Metrics (N={aggregated['total_questions']}):")
        print(f"  Exact Match:           {overall['exact_match']:.3f}")
        print(f"  ANLS:                  {overall['anls']:.3f}")
        print(f"  Evidence Correctness:  {overall['evidence_correctness']:.3f}")
        print("\nPerformance by Route:")
        for route_name, metrics in aggregated.get("by_route", {}).items():
            print(f"  {route_name} (N={metrics['count']}):")
            print(f"    Answer Accuracy:         {metrics['answer_accuracy']:.3f}")
            print(f"    Evidence Correctness:    {metrics['evidence_correctness']:.3f}")
            print(f"    Wrong Value Rate:        {metrics['wrong_value_rate']:.3f}")
            print(f"    Wrong Scope Rate:        {metrics['wrong_scope_rate']:.3f}")
            print(f"    Contradiction Fail Rate: {metrics['contradiction_failure_rate']:.3f}")
        print("=" * 60 + "\n")

    def _extract_predicted_answer(self, result) -> str:
        if result.answer:
            return str(result.answer).strip()
        for claim in result.certified_claims:
            if claim.claim.value:
                return str(claim.claim.value).strip()
            if claim.claim.text:
                return str(claim.claim.text).strip()
        return ""
