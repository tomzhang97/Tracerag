"""
CLI tool for interactive search with TraceRAG.

Usage:
    tracerag-search --index_root /data/tracerag/index
"""

import typer
from pathlib import Path
import pickle
import json
from loguru import logger

from tracerag.common.utils import load_config, setup_logging
from tracerag.retrieval.pipeline import TraceRAGSystem, PatchGridStore
from tracerag.visual.encoder import VisualPageEncoder


app = typer.Typer()


@app.command()
def main(
    index_root: str = typer.Option(..., help="Index root directory"),
    config_path: str = typer.Option(None, help="Path to config file (optional)"),
    query: str = typer.Option(None, help="Query string (if None, interactive mode)"),
):
    """
    Search TraceRAG index.

    Interactive mode if no query provided.
    """
    # Load config
    config = load_config(config_path)
    setup_logging(config)

    index_root = Path(index_root)

    logger.info(f"Loading indexes from {index_root}")

    # Load text index
    text_index_file = index_root / "text_index.pkl"
    if not text_index_file.exists():
        logger.error(f"Text index not found: {text_index_file}")
        return

    with open(text_index_file, 'rb') as f:
        text_index = pickle.load(f)
    logger.info("✓ Text index loaded")

    # Load STLG
    stlg_file = index_root / "stlg.pkl"
    stlg = None
    if stlg_file.exists():
        with open(stlg_file, 'rb') as f:
            stlg = pickle.load(f)
        logger.info("✓ STLG loaded")

    # Load spatial index (from first document for demo)
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
                logger.info(f"✓ Spatial index loaded from {version_dir}")
                break
        if spatial_index:
            break

    if spatial_index is None:
        logger.error("No spatial index found")
        return

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

    logger.info(f"✓ Loaded {len(patch_grid_store.grids)} patch grids")

    # Initialize visual encoder
    visual_encoder = VisualPageEncoder(config.get("visual", {}))

    # Create TraceRAG system
    system = TraceRAGSystem(
        config=config,
        text_index=text_index,
        patch_grid_store=patch_grid_store,
        spatial_index=spatial_index,
        visual_encoder=visual_encoder,
        stlg=stlg,
        ldg=None,
    )

    logger.info("✓ TraceRAG system initialized")

    # Query mode
    if query:
        # Single query
        result = system.answer(query)
        print_result(result)
    else:
        # Interactive mode
        print("\nTraceRAG Interactive Search")
        print("=" * 50)
        print("Enter queries (or 'quit' to exit)\n")

        while True:
            try:
                user_query = input("Query> ").strip()

                if not user_query or user_query.lower() in ('quit', 'exit', 'q'):
                    break

                result = system.answer(user_query)
                print_result(result)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")


def print_result(result):
    """Print query result in a readable format."""
    print("\n" + "=" * 50)
    print(f"Query: {result.query}")
    print(f"Type: {result.query_type}")
    print(f"Time: {result.metadata.get('elapsed_time', 0):.2f}s")
    print("-" * 50)
    print(f"\nAnswer: {result.answer}")

    if result.certified_claims:
        print(f"\nCertified Claims ({len(result.certified_claims)}):")
        for i, cc in enumerate(result.certified_claims, 1):
            print(f"\n  {i}. {cc.claim.text}")
            print(f"     Confidence: {cc.confidence:.2f}")
            print(f"     Evidences: {len(cc.evidences)} regions")
            for j, ev in enumerate(cc.evidences[:3], 1):  # Show first 3
                print(f"       - {ev.page_id} [{ev.obj_type}] (score: {ev.score:.2f})")

    print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    app()
