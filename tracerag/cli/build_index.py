"""
CLI tool for building retrieval indexes from ingested documents.

Usage:
    tracerag-build-index --index_root /data/tracerag/index
"""

import typer
from pathlib import Path
import pickle
from loguru import logger

from tracerag.common.utils import load_config, setup_logging
from tracerag.retrieval.text_index import TextIndex
from tracerag.graph.stlg import STLayoutGraph


app = typer.Typer()


@app.command()
def main(
    index_root: str = typer.Option(..., help="Index root directory"),
    config_path: str = typer.Option(None, help="Path to config file (optional)"),
    build_text_index: bool = typer.Option(True, help="Build text index"),
    build_stlg: bool = typer.Option(True, help="Build STLG graph"),
):
    """
    Build retrieval indexes from ingested documents.

    Reads structural artifacts and builds:
    1. Text index (BM25/FAISS)
    2. Spatio-Temporal Layout Graph (STLG)
    """
    # Load config
    config = load_config(config_path)
    setup_logging(config)

    index_root = Path(index_root)

    logger.info(f"Building indexes from {index_root}")

    # Find all structural artifacts
    struct_dir = index_root / "structural"
    if not struct_dir.exists():
        logger.error(f"Structural directory not found: {struct_dir}")
        return

    # Collect all objects from all documents/versions
    all_objects = []
    doc_versions = []

    for doc_dir in struct_dir.iterdir():
        if not doc_dir.is_dir():
            continue

        doc_id = doc_dir.name

        for version_dir in doc_dir.iterdir():
            if not version_dir.is_dir():
                continue

            version_id = version_dir.name
            objects_file = version_dir / "objects.pkl"

            if not objects_file.exists():
                logger.warning(f"Objects file not found: {objects_file}")
                continue

            with open(objects_file, 'rb') as f:
                objects = pickle.load(f)

            all_objects.extend(objects)
            doc_versions.append((doc_id, version_id))

            logger.info(f"Loaded {len(objects)} objects from {doc_id}/{version_id}")

    logger.info(f"Total objects: {len(all_objects)}")

    # Build text index
    if build_text_index:
        logger.info("Building text index...")
        index_type = config.get("retrieval", {}).get("text_index_type", "bm25")
        text_index = TextIndex(index_type=index_type)

        text_index.add_objects_batch(all_objects)
        text_index.build()

        # Save
        text_index_file = index_root / "text_index.pkl"
        with open(text_index_file, 'wb') as f:
            pickle.dump(text_index, f)
        logger.info(f"✓ Text index saved to {text_index_file}")

    # Build STLG
    if build_stlg:
        logger.info("Building STLG...")
        stlg = STLayoutGraph(config.get("graph", {}).get("stlg", {}))

        # Add documents and versions
        doc_map = {}
        for doc_id, version_id in doc_versions:
            if doc_id not in doc_map:
                stlg.add_document(doc_id)
                doc_map[doc_id] = set()

            if version_id not in doc_map[doc_id]:
                stlg.add_version(doc_id, version_id)
                doc_map[doc_id].add(version_id)

        # Add regions
        for obj in all_objects:
            # Add page if needed
            page_id = obj.page_id
            if not stlg.G.has_node(f"page:{page_id}"):
                # Extract page number from page_id
                page_num = int(obj.page_id.split('_p')[-1])
                stlg.add_page(obj.version_id, page_id, page_num)

            # Add region
            stlg.add_region(obj)

        # Save
        stlg_file = index_root / "stlg.pkl"
        with open(stlg_file, 'wb') as f:
            pickle.dump(stlg, f)
        logger.info(f"✓ STLG saved to {stlg_file}")

    logger.info("✓ Index building complete")


if __name__ == "__main__":
    app()
