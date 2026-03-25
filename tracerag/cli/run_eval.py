"""
CLI tool for running TraceRAG evaluation benchmarks.

Usage:
    tracerag-eval --benchmark microtext --index_root /data/tracerag/index --data_path /path/to/benchmark.json
"""

import typer
import json
import pickle
from pathlib import Path
from typing import Any
from loguru import logger

from tracerag.common.config import load_config, setup_logging
from tracerag.retrieval.pipeline import TraceRAGSystem, PatchGridStore
from tracerag.visual.encoder import VisualPageEncoder
from tracerag.eval.microtext_eval import MicroTextEvaluator, MicroTextQuestion
from tracerag.eval.visualdiff_eval import VisualDiffEvaluator, VisualDiffQuestion
from tracerag.eval.eng_bench_loader import EngBenchLoader


app = typer.Typer()


@app.command()
def main(
    benchmark: str = typer.Option("microtext", help="Benchmark name (eng_bench, microtext, or visualdiff)"),
    index_root: str = typer.Option(..., help="Index root directory"),
    data_path: str = typer.Option(..., help="Path to benchmark data file (JSON)"),
    config_path: str = typer.Option(None, help="Path to config file (optional)"),
    output_file: str = typer.Option(None, help="Output file for results (JSON)"),
):
    """
    Run evaluation benchmarks.

    Benchmarks:
    - eng_bench: General engineering QA benchmark
    - microtext: Micro-text extraction accuracy
    - visualdiff: Visual diff detection across versions
    """
    # Load config
    config = load_config(config_path)
    setup_logging(config)

    logger.info(f"Running benchmark: {benchmark}")
    logger.info(f"Data path: {data_path}")

    # Load TraceRAG system
    system = load_system(index_root, config)

    # Run appropriate benchmark
    if benchmark == "eng_bench":
        results = run_eng_bench_eval(system, data_path)
    elif benchmark == "microtext":
        # Load benchmark data
        benchmark_data = load_benchmark_data(data_path)
        results = run_microtext_eval(system, benchmark_data)
    elif benchmark == "visualdiff":
        # Load benchmark data
        benchmark_data = load_benchmark_data(data_path)
        results = run_visualdiff_eval(system, benchmark_data)
    else:
        logger.error(f"Unknown benchmark: {benchmark}")
        return

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")


def load_benchmark_data(data_path: str) -> Any:
    """
    Load benchmark data from JSON or JSONL file.
    Always returns a list of items or a dict with 'questions' key.
    """
    path = Path(data_path)
    
    if path.suffix == '.jsonl':
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return {"questions": data}  # Wrap in dict to match expected structure
        
    else:
        # Standard JSON
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def load_system(index_root: str, config: dict) -> TraceRAGSystem:
    """Load TraceRAG system from index."""
    index_root = Path(index_root)

    logger.info("Loading indexes...")

    # Load text index
    text_index_file = index_root / "text_index.pkl"
    with open(text_index_file, 'rb') as f:
        text_index = pickle.load(f)

    # Load STLG
    stlg_file = index_root / "stlg.pkl"
    stlg = None
    if stlg_file.exists():
        with open(stlg_file, 'rb') as f:
            stlg = pickle.load(f)

    # Load spatial index
    struct_dir = index_root / "structural"
    spatial_index = None
    if struct_dir.exists():
        for doc_dir in struct_dir.iterdir():
            if not doc_dir.is_dir():
                continue
            for version_dir in doc_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                spatial_index_file = version_dir / "spatial_index.pkl"
                if spatial_index_file.exists():
                    with open(spatial_index_file, 'rb') as f:
                        spatial_index = pickle.load(f)
                    # For now just use the last loaded one - in full system would need multi-doc support in spatial index
                    # But TraceRAGSystem currently takes a single spatial_index. 
                    # We might need to handle this if we are doing multi-doc.
                    # However, to preserve existing logic, we'll keep this but note the limitation.
                    
    # Load patch grids
    visual_dir = index_root / "visual"
    patch_grid_store = PatchGridStore()
    if visual_dir.exists():
        for doc_dir in visual_dir.iterdir():
            if not doc_dir.is_dir():
                continue
            for version_dir in doc_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                for patch_file in version_dir.glob("*.npz"):
                    patch_grid = VisualPageEncoder.load_patch_grid(str(patch_file))
                    patch_grid_store.add(patch_grid)

    # Initialize visual encoder
    visual_encoder = VisualPageEncoder(config.get("visual", {}))

    # Create system
    system = TraceRAGSystem(
        config=config,
        text_index=text_index,
        patch_grid_store=patch_grid_store,
        spatial_index=spatial_index,
        visual_encoder=visual_encoder,
        stlg=stlg,
        ldg=None,
    )

    logger.info("System loaded successfully")
    return system


def run_microtext_eval(system: TraceRAGSystem, data: dict) -> dict:
    """Run micro-text evaluation."""
    logger.info("Running micro-text evaluation")

    # Handle both list and dict input
    items = data if isinstance(data, list) else data.get("questions", [])
    
    # Parse questions
    questions = []
    for item in items:
        # Map fields handling potential naming differences
        question = MicroTextQuestion(
            query_id=item.get("question_id") or item.get("query_id"),
            query=item.get("query_text") or item.get("query"),
            doc_id=item["doc_id"],
            version_id=item["version_id"],
            answer_text=item.get("answer_text") or item.get("answer"),
            gt_page_id=None, # annotations might not have this, harmless if None for now
            gt_object_id=None, # annotations might not have this
            gt_bbox=(0,0,0,0), # Placeholder if missing
            font_height_px=item.get("font_height_px", 12.0),
            metadata=item
        )
        questions.append(question)

    # Run evaluation
    evaluator = MicroTextEvaluator(system)
    results = evaluator.evaluate_dataset(questions)

    # Print summary
    evaluator.print_summary(results)

    return results


def run_eng_bench_eval(system: TraceRAGSystem, data_path: str) -> dict:
    """Run engineering benchmark evaluation."""
    logger.info("Running engineering benchmark evaluation")

    # Load benchmark using robust loader
    loader = EngBenchLoader(data_path)
    queries = loader.load()

    if not queries:
        logger.error("No queries loaded from benchmark")
        return {}

    # Run queries and collect results
    results = []
    for i, query in enumerate(queries, 1):
        logger.info(f"Processing query {i}/{len(queries)}: {query.query_id}")

        try:
            # Run query through TraceRAG
            result = system.answer(query.query_text)

            # Extract predicted answer
            predicted_answer = result.answer

            # Simple correctness check (would need more sophisticated matching in production)
            correct = query.ground_truth_answer.lower() in predicted_answer.lower()

            results.append({
                "query_id": query.query_id,
                "query": query.query_text,
                "predicted_answer": predicted_answer,
                "ground_truth": query.ground_truth_answer,
                "correct": correct,
                "num_claims": len(result.certified_claims),
                "num_evidences": sum(len(c.evidences) for c in result.certified_claims),
            })

        except Exception as e:
            logger.error(f"Error processing query {query.query_id}: {e}")
            results.append({
                "query_id": query.query_id,
                "query": query.query_text,
                "error": str(e),
            })

    # Compute aggregated metrics
    valid_results = [r for r in results if "error" not in r]
    accuracy = sum(r["correct"] for r in valid_results) / len(valid_results) if valid_results else 0.0

    aggregated = {
        "total_queries": len(queries),
        "successful": len(valid_results),
        "failed": len(results) - len(valid_results),
        "accuracy": accuracy,
        "per_query": results,
    }

    # Print summary
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
    """Run visual diff evaluation."""
    logger.info("Running visual diff evaluation")

    # Handle both list and dict input
    items = data if isinstance(data, list) else data.get("questions", [])

    # Parse questions
    questions = []
    for item in items:
        question = VisualDiffQuestion(
            query_id=item.get("question_id") or item.get("query_id"),
            query=item.get("query_text") or item.get("query"),
            doc_id=item["doc_id"],
            old_version_id=item.get("old_version_id") or item.get("revision_a"), # Handle potential schema vars
            new_version_id=item.get("new_version_id") or item.get("revision_b"),
            change_type=item.get("change_type"),
            key_tokens=item.get("key_tokens", []),
            bbox_old=tuple(item["bbox_old"]) if item.get("bbox_old") else None,
            bbox_new=tuple(item["bbox_new"]) if item.get("bbox_new") else None,
            page_old=item.get("page_old"),
            page_new=item.get("page_new"),
            metadata=item
        )
        questions.append(question)

    # Run evaluation
    evaluator = VisualDiffEvaluator(system)
    results = evaluator.evaluate_dataset(questions)

    # Print summary
    evaluator.print_summary(results)

    return results


if __name__ == "__main__":
    app()
