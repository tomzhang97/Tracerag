"""
CLI tool for ingesting PDF documents into TraceRAG.

Usage:
    tracerag-ingest --doc_id DOC123 --version_id v1 --pdf_path /path/to.pdf
"""

import typer
from pathlib import Path
import pickle
from loguru import logger

from tracerag.common.utils import load_config, setup_logging, ensure_dir
from tracerag.structural.parser import PdfStructuralParser
from tracerag.visual.encoder import VisualPageEncoder


app = typer.Typer()


@app.command()
def main(
    pdf_path: str = typer.Option(..., help="Path to PDF file"),
    doc_id: str = typer.Option(..., help="Document ID"),
    version_id: str = typer.Option(..., help="Version/revision ID"),
    config_path: str = typer.Option(None, help="Path to config file (optional)"),
    output_dir: str = typer.Option(None, help="Output directory (overrides config)"),
):
    """
    Ingest a PDF document into TraceRAG.

    This performs:
    1. Structural parsing (text, vector graphics, tables)
    2. Visual encoding (patch grids)
    3. Saves artifacts to disk
    """
    # Load config
    config = load_config(config_path)
    setup_logging(config)

    logger.info(f"Ingesting PDF: {pdf_path}")
    logger.info(f"Document ID: {doc_id}, Version ID: {version_id}")

    # Determine output directory
    if output_dir is None:
        output_dir = config.get("index_root", "/data/tracerag/index")

    output_dir = Path(output_dir)
    ensure_dir(str(output_dir))

    # Step 1: Structural parsing
    logger.info("Step 1: Structural parsing")
    parser = PdfStructuralParser(config.get("structural", {}))
    objects, spatial_index = parser.parse_document(pdf_path, doc_id, version_id)

    # Save structural artifacts
    struct_dir = output_dir / "structural" / doc_id / version_id
    ensure_dir(str(struct_dir))

    objects_file = struct_dir / "objects.pkl"
    with open(objects_file, 'wb') as f:
        pickle.dump(objects, f)
    logger.info(f"Saved {len(objects)} objects to {objects_file}")

    spatial_index_file = struct_dir / "spatial_index.pkl"
    with open(spatial_index_file, 'wb') as f:
        pickle.dump(spatial_index, f)
    logger.info(f"Saved spatial index to {spatial_index_file}")

    # Step 2: Visual encoding
    logger.info("Step 2: Visual encoding")
    encoder = VisualPageEncoder(config.get("visual", {}))

    visual_dir = output_dir / "visual" / doc_id / version_id
    ensure_dir(str(visual_dir))

    patch_grids = encoder.encode_document(pdf_path, doc_id, version_id, str(visual_dir))
    logger.info(f"Encoded {len(patch_grids)} pages to {visual_dir}")

    logger.info(f"✓ Ingestion complete for {doc_id}/{version_id}")


if __name__ == "__main__":
    app()
