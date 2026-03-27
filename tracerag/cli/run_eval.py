"""
CLI tool for running TraceRAG evaluation benchmarks.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import typer
from loguru import logger

from tracerag.common.config import load_config, setup_logging
from tracerag.eval.docvqa_eval import DocVQAEvaluator, load_docvqa_questions
from tracerag.eval.loader import EngBenchLoader
from tracerag.eval.microtext_eval import MicroTextEvaluator, MicroTextQuestion
from tracerag.eval.visualdiff_eval import VisualDiffEvaluator, VisualDiffQuestion
from tracerag.retrieval.candidate_filter import CandidateFilter
from tracerag.retrieval.llm_answerer import LLMAnswerer
from tracerag.retrieval.pipeline import PatchGridStore, TraceRAGSystem
from tracerag.structural.index import PdfSpatialIndex
from tracerag.visual.encoder import VisualPageEncoder
from tracerag.visual.scorer import VisualScorer


app = typer.Typer()


@app.command()
def main(
    benchmark: str = typer.Option("microtext", help="Benchmark name (eng_bench, microtext, visualdiff, or docvqa)"),
    index_root: str = typer.Option(..., help="Index root directory"),
    data_path: str = typer.Option(..., help="Path to benchmark data file (JSON or JSONL)"),
    config_path: str = typer.Option(None, help="Path to config file (optional)"),
    output_file: str = typer.Option(None, help="Output file for results (JSON)"),
):
    config = load_config(config_path)
    setup_logging(config)

    logger.info(f"Running benchmark: {benchmark}")
    logger.info(f"Data path: {data_path}")

    system = load_system(index_root, config)
    benchmark_data = load_benchmark_data(data_path)

    if benchmark == "eng_bench":
        results = run_eng_bench_eval(system, data_path)
    elif benchmark == "microtext":
        results = run_microtext_eval(system, benchmark_data)
    elif benchmark == "visualdiff":
        results = run_visualdiff_eval(system, benchmark_data)
    elif benchmark == "docvqa":
        results = run_docvqa_eval(system, benchmark_data)
    else:
        logger.error(f"Unknown benchmark: {benchmark}")
        raise typer.Exit(code=1)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")


def load_benchmark_data(data_path: str) -> Any:
    path = Path(data_path)
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() == ".jsonl":
            return {"questions": [json.loads(line) for line in f if line.strip()]}
        return json.load(f)


def load_system(index_root: str, config: dict) -> TraceRAGSystem:
    index_root_path = Path(index_root)
    logger.info(f"Loading TraceRAG system from {index_root_path}")

    text_index_file = index_root_path / "text_index.pkl"
    if not text_index_file.exists():
        raise FileNotFoundError(f"Text index not found: {text_index_file}")
    with open(text_index_file, "rb") as f:
        text_index = pickle.load(f)

    struct_root = index_root_path / "structural"
    if not struct_root.exists():
        raise FileNotFoundError(f"Structural directory not found: {struct_root}")

    spatial_index = PdfSpatialIndex()
    merged_indexes = 0
    for spatial_index_file in struct_root.rglob("spatial_index.pkl"):
        with open(spatial_index_file, "rb") as f:
            loaded_index = pickle.load(f)
        for obj in loaded_index.objects.values():
            spatial_index.add_object(obj)
        merged_indexes += 1
    logger.info(f"Merged {merged_indexes} spatial indexes ({len(spatial_index.objects)} objects)")

    patch_grid_store = PatchGridStore()
    visual_root = index_root_path / "visual"
    if visual_root.exists():
        for patch_file in visual_root.rglob("*.npz"):
            patch_grid_store.add(VisualPageEncoder.load_patch_grid(str(patch_file)))
    logger.info(f"Loaded {len(patch_grid_store.grids)} patch grids")

    candidate_filter = CandidateFilter(text_index, config.get("retrieval", {}))
    visual_encoder = VisualPageEncoder(config.get("visual", {}))
    visual_scorer = VisualScorer(config.get("visual", {}), encoder=visual_encoder)
    llm_answerer = LLMAnswerer(config.get("llm", {}))

    manifest = {}
    manifest_path = index_root_path / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    system = TraceRAGSystem(
        config=config,
        candidate_filter=candidate_filter,
        patch_grid_store=patch_grid_store,
        spatial_index=spatial_index,
        visual_scorer=visual_scorer,
        llm_answerer=llm_answerer,
        manifest=manifest,
    )
    logger.info("System loaded successfully")
    return system


def run_microtext_eval(system: TraceRAGSystem, data: dict) -> dict:
    logger.info("Running micro-text evaluation")
    items = data if isinstance(data, list) else data.get("questions", [])

    questions = []
    for item in items:
        questions.append(
            MicroTextQuestion(
                query_id=item.get("question_id") or item.get("query_id"),
                query=item.get("query_text") or item.get("query"),
                doc_id=item["doc_id"],
                version_id=item["version_id"],
                answer_text=item.get("answer_text") or item.get("answer"),
                gt_page_id=item.get("page_id"),
                gt_object_id=item.get("object_id"),
                gt_bbox=tuple(item.get("bbox") or (0, 0, 0, 0)),
                font_height_px=item.get("font_height_px", 12.0),
                metadata=item,
            )
        )

    evaluator = MicroTextEvaluator(system)
    results = evaluator.evaluate_dataset(questions)
    evaluator.print_summary(results)
    return results


def run_docvqa_eval(system: TraceRAGSystem, data: dict) -> dict:
    logger.info("Running DocVQA evaluation")
    questions = load_docvqa_questions(data)
    evaluator = DocVQAEvaluator(system)
    results = evaluator.evaluate_dataset(questions)
    evaluator.print_summary(results)
    return results


def run_eng_bench_eval(system: TraceRAGSystem, data_path: str) -> dict:
    logger.info("Running engineering benchmark evaluation")
    loader = EngBenchLoader(data_path)
    queries = loader.load()
    if not queries:
        logger.error("No queries loaded from benchmark")
        return {}

    results = []
    for query in queries:
        try:
            result = system.answer(query.query_text)
            predicted_answer = result.answer or ""
            correct = query.ground_truth_answer.lower() in predicted_answer.lower() if query.ground_truth_answer else False
            results.append(
                {
                    "query_id": query.query_id,
                    "query": query.query_text,
                    "predicted_answer": predicted_answer,
                    "ground_truth": query.ground_truth_answer,
                    "correct": correct,
                    "route_name": result.metadata.get("route_name", query.query_type),
                    "top_candidate_traces": result.metadata.get("top_candidate_traces", []),
                }
            )
        except Exception as exc:
            logger.error(f"Error processing query {query.query_id}: {exc}")
            results.append({"query_id": query.query_id, "query": query.query_text, "error": str(exc)})

    valid_results = [item for item in results if "error" not in item]
    accuracy = sum(item["correct"] for item in valid_results) / len(valid_results) if valid_results else 0.0
    aggregated = {
        "total_queries": len(queries),
        "successful": len(valid_results),
        "failed": len(results) - len(valid_results),
        "accuracy": accuracy,
        "per_query": results,
    }

    print("\n" + "=" * 60)
    print("ENGINEERING BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\nTotal Queries: {aggregated['total_queries']}")
    print(f"Successful: {aggregated['successful']}")
    print(f"Failed: {aggregated['failed']}")
    print(f"Accuracy: {aggregated['accuracy']:.3f}")
    print("=" * 60 + "\n")
    return aggregated


def run_visualdiff_eval(system: TraceRAGSystem, data: dict) -> dict:
    logger.info("Running visual diff evaluation")
    items = data if isinstance(data, list) else data.get("questions", [])

    questions = []
    for item in items:
        questions.append(
            VisualDiffQuestion(
                query_id=item.get("question_id") or item.get("query_id"),
                query=item.get("query_text") or item.get("query"),
                doc_id=item["doc_id"],
                old_version_id=item.get("old_version_id") or item.get("revision_a"),
                new_version_id=item.get("new_version_id") or item.get("revision_b"),
                change_type=item.get("change_type"),
                key_tokens=item.get("key_tokens", []),
                bbox_old=tuple(item["bbox_old"]) if item.get("bbox_old") else None,
                bbox_new=tuple(item["bbox_new"]) if item.get("bbox_new") else None,
                page_old=item.get("page_old"),
                page_new=item.get("page_new"),
                metadata=item,
            )
        )

    evaluator = VisualDiffEvaluator(system)
    results = evaluator.evaluate_dataset(questions)
    evaluator.print_summary(results)
    return results


if __name__ == "__main__":
    app()
