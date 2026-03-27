"""
Unified Evaluation Runner.
Provides a single entry point for running different evaluations across the benchmark.
"""

from typing import List, Dict, Any
from loguru import logger
import time

from tracerag.eval.loader import EngBenchLoader
from tracerag.eval.metrics import summarize_route_metrics
from tracerag.eval.qa import QAEvaluator
from tracerag.retrieval.pipeline import TraceRAGSystem
from tracerag.common.types import EvalResult
from tracerag.common.debug import DebugExporter
import datetime
import subprocess
import json
from pathlib import Path

class EvalRunner:
    """Executes the evaluation pipeline against a benchmark dataset."""

    def __init__(self, system: TraceRAGSystem, config: Dict[str, Any] = None):
        self.system = system
        self.config = config or {}
        
        # Instantiate available evaluators
        self.qa_evaluator = QAEvaluator(self.config)

    def _get_git_hash(self) -> str:
        try:
            return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        except Exception:
            return "unknown"

    def run_eval(self, dataset_path: str) -> EvalResult:
        """
        Run full evaluation on a dataset.

        Args:
            dataset_path: Path to the JSON benchmark
            
        Returns:
            Aggregated results dict
        """
        logger.info(f"Starting evaluation on {dataset_path}")
        loader = EngBenchLoader(dataset_path)
        queries = loader.load()
        
        if not queries:
            logger.warning("No queries found for evaluation.")
            return EvalResult(overall={}, per_query=[])

        all_metrics = []
        start_time = time.time()
        
        debug_dir = self.config.get("debug_export_dir")
        exporter = DebugExporter(debug_dir) if debug_dir else None
        
        for idx, q in enumerate(queries):
            logger.info(f"Evaluating {idx+1}/{len(queries)}: {q.query_id}")
            
            # 1. Run inference
            result_pack = self.system.answer(q.query_text)
            
            # 2. Compute metrics
            metrics = self.qa_evaluator.evaluate_question(q, result_pack)
            all_metrics.append(metrics)
            
            if exporter:
                if metrics.get("page_recall", 0.0) == 0.0 or metrics.get("answer_f1", 1.0) < 0.5:
                    payload = {
                        "query": q.model_dump(),
                        "result_pack": result_pack.model_dump(),
                        "metrics": metrics
                    }
                    exporter.save_failure_trace(q.query_id, payload)

        total_time = time.time() - start_time
        
        overall = {
            "system_name": self.config.get("name", "unknown"),
            "task_name": "unknown",
            "split": "unknown",
            "answer_em": sum(m.get("answer_em", 0.0) for m in all_metrics) / len(queries) if queries else 0,
            "answer_f1": sum(m.get("answer_f1", 0.0) for m in all_metrics) / len(queries) if queries else 0,
            "set_precision": sum(m.get("set_precision", 0.0) for m in all_metrics if "set_precision" in m) / len([m for m in all_metrics if "set_precision" in m]) if any("set_precision" in m for m in all_metrics) else 0.0,
            "set_recall": sum(m.get("set_recall", 0.0) for m in all_metrics if "set_recall" in m) / len([m for m in all_metrics if "set_recall" in m]) if any("set_recall" in m for m in all_metrics) else 0.0,
            "set_f1": sum(m.get("set_f1", 0.0) for m in all_metrics if "set_f1" in m) / len([m for m in all_metrics if "set_f1" in m]) if any("set_f1" in m for m in all_metrics) else 0.0,
            "cer": sum(m.get("cer", 0.0) for m in all_metrics) / len(queries) if queries else 0,
            "page_recall": sum(m.get("page_recall", 0.0) for m in all_metrics) / len(queries) if queries else 0,
            "evidence_precision": sum(m.get("evidence_precision", 0.0) for m in all_metrics) / len(queries) if queries else 0,
            "evidence_recall": sum(m.get("evidence_recall", 0.0) for m in all_metrics) / len(queries) if queries else 0,
            "evidence_iou": sum(m.get("evidence_iou", 0.0) for m in all_metrics) / len(queries) if queries else 0,
            "correct_evidence_rate": sum(1.0 for m in all_metrics if m.get("evidence_correct", False)) / len(queries) if queries else 0,
            "answer_with_correct_evidence_rate": sum(1.0 for m in all_metrics if m.get("evidence_correct", False) and m.get("answer_em", 0) == 1.0) / len(queries) if queries else 0,
            "version_accuracy": sum(m.get("version_accuracy", 0.0) for m in all_metrics) / len(queries) if queries else 0,
            "change_class_accuracy": sum(m.get("change_class_accuracy", 0.0) for m in all_metrics) / len(queries) if queries else 0,
            "latency_ms_mean": (total_time / len(queries)) * 1000.0 if queries else 0,
            "by_route": summarize_route_metrics(all_metrics),
            "storage_mb": 0.0,
            "manifest_path": ""
        }
            
        final_result = EvalResult(overall=overall, per_query=all_metrics)

        # Save manifest if possible
        if exporter:
            manifest = {
                "timestamp": datetime.datetime.now().isoformat(),
                "git_hash": self._get_git_hash(),
                "dataset_path": dataset_path,
                "config": self.config,
                "results": final_result.model_dump()
            }
            manifest_path = Path(debug_dir) / f"eval_manifest_{int(time.time())}.json"
            try:
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)
                logger.info(f"Saved evaluation manifest to {manifest_path}")
                overall["manifest_path"] = str(manifest_path)
            except Exception as e:
                logger.warning(f"Failed to save manifest: {e}")

        logger.info("Evaluation complete.")
        return final_result
