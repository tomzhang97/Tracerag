"""
CLI tool for running TraceRAG evaluation benchmarks.

Usage:
    tracerag-eval --benchmark microtext --index_root /data/tracerag/index --data_path /path/to/benchmark.json
"""

import typer
import json
import pickle
from pathlib import Path
from loguru import logger

from tracerag.common.utils import load_config, setup_logging
from tracerag.retrieval.pipeline import TraceRAGSystem, PatchGridStore
from tracerag.visual.encoder import VisualPageEncoder
from tracerag.eval.microtext_eval import MicroTextEvaluator, MicroTextQuestion
from tracerag.eval.visualdiff_eval import VisualDiffEvaluator, VisualDiffQuestion


app = typer.Typer()


@app.command()
def main(
    benchmark: str = typer.Option("microtext", help="Benchmark name (microtext or visualdiff)"),
    index_root: str = typer.Option(..., help="Index root directory"),
    data_path: str = typer.Option(..., help="Path to benchmark data file (JSON)"),
    config_path: str = typer.Option(None, help="Path to config file (optional)"),
    output_file: str = typer.Option(None, help="Output file for results (JSON)"),
):
    """
    Run evaluation benchmarks.

    Benchmarks:
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

    # Load benchmark data
    with open(data_path, 'r') as f:
        benchmark_data = json.load(f)

    # Run appropriate benchmark
    if benchmark == "microtext":
        results = run_microtext_eval(system, benchmark_data)
    elif benchmark == "visualdiff":
        results = run_visualdiff_eval(system, benchmark_data)
    else:
        logger.error(f"Unknown benchmark: {benchmark}")
        return

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")


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
                break
        if spatial_index:
            break

    # Load patch grids
    visual_dir = index_root / "visual"
    patch_grid_store = PatchGridStore()
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

    # Parse questions
    questions = []
    for item in data.get("questions", []):
        question = MicroTextQuestion(
            query_id=item["query_id"],
            query=item["query"],
            doc_id=item["doc_id"],
            version_id=item["version_id"],
            answer_text=item["answer_text"],
            gt_page_id=item["gt_page_id"],
            gt_object_id=item["gt_object_id"],
            gt_bbox=tuple(item["gt_bbox"]),
            font_height_px=item.get("font_height_px", 12.0),
            metadata=item.get("metadata", {})
        )
        questions.append(question)

    # Run evaluation
    evaluator = MicroTextEvaluator(system)
    results = evaluator.evaluate_dataset(questions)

    # Print summary
    evaluator.print_summary(results)

    return results


def run_visualdiff_eval(system: TraceRAGSystem, data: dict) -> dict:
    """Run visual diff evaluation."""
    logger.info("Running visual diff evaluation")

    # Parse questions
    questions = []
    for item in data.get("questions", []):
        question = VisualDiffQuestion(
            query_id=item["query_id"],
            query=item["query"],
            doc_id=item["doc_id"],
            old_version_id=item["old_version_id"],
            new_version_id=item["new_version_id"],
            change_type=item["change_type"],
            key_tokens=item["key_tokens"],
            bbox_old=tuple(item["bbox_old"]) if item.get("bbox_old") else None,
            bbox_new=tuple(item["bbox_new"]) if item.get("bbox_new") else None,
            page_old=item["page_old"],
            page_new=item["page_new"],
            metadata=item.get("metadata", {})
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
