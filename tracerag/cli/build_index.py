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
from tracerag.graph.entity_resolver import EntityResolver


app = typer.Typer()


@app.command()
def main(
    index_root: str = typer.Option(..., help="Index root directory"),
    config_path: str = typer.Option(None, help="Path to config file (optional)"),
    build_text_index: bool = typer.Option(True, help="Build text index"),
    build_stlg: bool = typer.Option(True, help="Build STLG graph"),
    extract_entities: bool = typer.Option(True, help="Extract entities using entity resolver"),
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
    objects_by_doc_version = {}  # (doc_id, version_id) -> objects

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
            objects_by_doc_version[(doc_id, version_id)] = objects

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

        # Extract entities (if enabled)
        if extract_entities:
            logger.info("Extracting entities...")
            entity_config = config.get("graph", {}).get("stlg", {}).get("entity_resolution", {})

            if entity_config.get("enabled", True):
                # Initialize entity resolver (with or without LLM)
                llm_client = None  # Could initialize LLM client here if use_llm=True
                entity_resolver = EntityResolver(llm_client, entity_config)

                # Extract entities per document/version
                all_entities = {}
                for (doc_id, version_id), objects in objects_by_doc_version.items():
                    logger.info(f"Extracting entities from {doc_id}/{version_id}...")
                    entities = entity_resolver.extract_entities_from_objects(
                        objects, doc_id, version_id
                    )
                    all_entities[(doc_id, version_id)] = entities

                    # Add entities to STLG
                    for entity_id, entity_data in entities.items():
                        stlg.add_entity(entity_id, entity_data["label"], metadata=entity_data)

                        # Link regions to entities
                        for obj in entity_data.get("linked_objects", []):
                            stlg.link_region_entity(obj.object_id, entity_id)

                    logger.info(f"  → Added {len(entities)} entities to STLG")

                logger.info(f"✓ Entity extraction complete")

        # Save
        stlg_file = index_root / "stlg.pkl"
        with open(stlg_file, 'wb') as f:
            pickle.dump(stlg, f)
        logger.info(f"✓ STLG saved to {stlg_file}")

    logger.info("✓ Index building complete")


if __name__ == "__main__":
    app()
