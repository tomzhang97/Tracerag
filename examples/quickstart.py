"""
QuickStart example for TraceRAG.

Demonstrates basic usage:
1. Load configuration
2. Ingest a PDF
3. Build indexes
4. Run a query
5. Examine results
"""

from pathlib import Path
from tracerag.common.utils import load_config, setup_logging
from tracerag.structural.parser import PdfStructuralParser
from tracerag.visual.encoder import VisualPageEncoder
from tracerag.retrieval.text_index import TextIndex
from tracerag.graph.stlg import STLayoutGraph
from tracerag.retrieval.pipeline import TraceRAGSystem, PatchGridStore


def quickstart_demo(pdf_path: str, doc_id: str = "DEMO_DOC", version_id: str = "v1"):
    """
    Run a quick demo of TraceRAG.

    Args:
        pdf_path: Path to PDF file
        doc_id: Document ID
        version_id: Version ID
    """
    print("=" * 60)
    print("TraceRAG QuickStart Demo")
    print("=" * 60)

    # Step 1: Load configuration
    print("\n[1/6] Loading configuration...")
    config = load_config()
    setup_logging(config)

    # Step 2: Structural parsing
    print("\n[2/6] Parsing PDF structure (text, tables, vector graphics)...")
    parser = PdfStructuralParser(config.get("structural", {}))
    objects, spatial_index = parser.parse_document(pdf_path, doc_id, version_id)
    print(f"  → Extracted {len(objects)} objects")

    # Step 3: Visual encoding
    print("\n[3/6] Encoding visual appearance (this may take a few minutes)...")
    encoder = VisualPageEncoder(config.get("visual", {}))
    patch_grids = encoder.encode_document(pdf_path, doc_id, version_id)
    print(f"  → Encoded {len(patch_grids)} pages")

    # Step 4: Build indexes
    print("\n[4/6] Building retrieval indexes...")

    # Text index
    text_index = TextIndex(index_type="bm25")
    text_index.add_objects_batch(objects)
    text_index.build()
    print(f"  → Built text index with {len(objects)} objects")

    # Patch grid store
    patch_grid_store = PatchGridStore()
    for pg in patch_grids:
        patch_grid_store.add(pg)

    # STLG
    stlg = STLayoutGraph(config.get("graph", {}).get("stlg", {}))
    stlg.add_document(doc_id)
    stlg.add_version(doc_id, version_id)

    for obj in objects:
        # Add page if needed
        page_id = obj.page_id
        if not stlg.G.has_node(f"page:{page_id}"):
            page_num = int(obj.page_id.split('_p')[-1])
            stlg.add_page(version_id, page_id, page_num)
        stlg.add_region(obj)

    print(f"  → Built STLG with {stlg.G.number_of_nodes()} nodes")

    # Step 5: Create TraceRAG system
    print("\n[5/6] Initializing TraceRAG system...")
    system = TraceRAGSystem(
        config=config,
        text_index=text_index,
        patch_grid_store=patch_grid_store,
        spatial_index=spatial_index,
        visual_encoder=encoder,
        stlg=stlg,
        ldg=None,
    )
    print("  → System ready!")

    # Step 6: Run example queries
    print("\n[6/6] Running example queries...\n")

    example_queries = [
        "What components are shown in this document?",
        "Where is the main diagram?",
        "What specifications are mentioned?",
    ]

    for i, query in enumerate(example_queries, 1):
        print(f"\n{'-' * 60}")
        print(f"Query {i}: {query}")
        print(f"{'-' * 60}")

        result = system.answer(query)

        print(f"\nAnswer:")
        print(f"  {result.answer}")

        if result.certified_claims:
            print(f"\nCertified Claims ({len(result.certified_claims)}):")
            for j, claim in enumerate(result.certified_claims[:3], 1):
                print(f"  {j}. {claim.claim.text}")
                print(f"     Confidence: {claim.confidence:.2f}")
                print(f"     Evidence: {len(claim.evidences)} regions")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

    return system


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python quickstart.py <path_to_pdf>")
        print("\nExample:")
        print("  python quickstart.py /path/to/manual.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not Path(pdf_path).exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    quickstart_demo(pdf_path)
