"""
TraceRAG Main CLI.
Provides a unified entrypoint for core operations (ingest, index, run_eval).

Usage:
  python -m tracerag.cli.main ingest --input <path>
  python -m tracerag.cli.main index --input <path>
  python -m tracerag.cli.main eval --dataset <path>
"""
import argparse
from loguru import logger
import sys
import os
import pickle
from pathlib import Path
from typing import Dict, Any

from tracerag.common.config import load_config, setup_logging
from tracerag.common.io import ensure_dir
from tracerag.structural.extract import PdfExtractor
from tracerag.structural.index import PdfSpatialIndex
from tracerag.structural.handlers.image_handler import ImageHandler
from tracerag.structural.handlers.text_handler import TextHandler
from tracerag.structural.handlers.doc_handler import OfficeHandler
from tracerag.visual.encoder import VisualPageEncoder
from tracerag.visual.scorer import VisualScorer
from tracerag.retrieval.pipeline import TraceRAGSystem, PatchGridStore
from tracerag.retrieval.candidate_filter import CandidateFilter
from tracerag.eval.runner import EvalRunner

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md", ".json", ".docx", ".pptx", ".xlsx"}

def _ingest_single_file(file_path: Path, doc_id: str, version_id: str, output_dir: Path, config: dict, force: bool = False):
    """Ingest a single file (any supported format) into the index."""
    ext = file_path.suffix.lower()

    # Resume support: skip if already ingested
    struct_dir = output_dir / "structural" / doc_id / version_id
    if not force and (struct_dir / "objects.pkl").exists():
        logger.debug(f"  Skipping (already ingested): {file_path.name}")
        return -1  # Signal: skipped

    logger.info(f"  Ingesting [{ext}] {file_path.name} as doc_id={doc_id}")

    # 1. Structural extraction
    if ext == ".pdf":
        extractor = PdfExtractor()
        objects = extractor.extract_document(str(file_path), doc_id, version_id)
    elif ext in {".png", ".jpg", ".jpeg"}:
        handler = ImageHandler()
        objects = handler.process(str(file_path), doc_id, version_id)
    elif ext in {".txt", ".md", ".json"}:
        handler = TextHandler()
        objects = handler.process(str(file_path), doc_id, version_id)
    elif ext in {".docx", ".pptx", ".xlsx"}:
        handler = OfficeHandler()
        objects = handler.process(str(file_path), doc_id, version_id)
    else:
        logger.warning(f"  Skipping unsupported file: {file_path.name}")
        return 0

    ensure_dir(str(struct_dir))
    with open(struct_dir / "objects.pkl", 'wb') as f:
        pickle.dump(objects, f)

    # 2. Visual encoding (only for files that have a visual representation)
    if ext in {".pdf", ".png", ".jpg", ".jpeg"}:
        try:
            encoder = VisualPageEncoder(config.get("visual", {}))
            visual_dir = output_dir / "visual" / doc_id / version_id
            ensure_dir(str(visual_dir))
            encoder.encode_document(str(file_path), doc_id, version_id, str(visual_dir))
        except Exception as e:
            logger.warning(f"  Visual encoding failed for {file_path.name}: {e}")

    return len(objects)


def _sanitize_id(name: str) -> str:
    """Convert a filename into a safe doc_id (alphanumeric + underscores)."""
    import re
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name)


def run_ingest(args):
    config = load_config(args.config)
    setup_logging(config)

    input_path = Path(args.input)
    output_dir = Path(args.output or config.get("index_root", "data/index"))
    ensure_dir(str(output_dir))
    version_id = args.version_id or "v1"
    force = getattr(args, 'force', False)
    start_from = getattr(args, 'start_from', 1)

    if input_path.is_file():
        # --- Single file mode ---
        doc_id = args.doc_id or _sanitize_id(input_path.stem)
        total = _ingest_single_file(input_path, doc_id, version_id, output_dir, config, force)
        if total == -1:
            logger.info("File already ingested. Use --force to re-process.")
        else:
            logger.info(f"Ingestion complete. {total} objects from 1 file.")

    elif input_path.is_dir():
        # --- Recursive directory mode ---
        logger.info(f"Scanning directory: {input_path}")
        files = sorted(
            f for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        logger.info(f"Found {len(files)} supported files.")

        # If resuming from a specific point, clean up artifacts from that point onward
        if start_from > 1:
            import shutil
            cleaned = 0
            for idx, file in enumerate(files, 1):
                if idx < start_from:
                    continue
                doc_id = _sanitize_id(file.stem)
                for sub in ["structural", "visual"]:
                    artifact_dir = output_dir / sub / doc_id / version_id
                    if artifact_dir.exists():
                        shutil.rmtree(artifact_dir)
                        cleaned += 1
            logger.info(f"Cleaned {cleaned} artifact directories from file #{start_from} onward.")

        total_objects = 0
        skipped = 0
        processed = 0
        
        # Load or create manifest for document metadata
        import json
        manifest_path = output_dir / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                pass

        for idx, file in enumerate(files, 1):
            if idx < start_from:
                skipped += 1
                continue
            doc_id = _sanitize_id(file.stem)
            count = _ingest_single_file(file, doc_id, version_id, output_dir, config, force=True if start_from > 1 else force)
            if count == -1:
                skipped += 1
            else:
                processed += 1
                total_objects += count
                logger.info(f"[{idx}/{len(files)}] {file.relative_to(input_path)} → {count} objects")
                manifest[doc_id] = str(file.resolve())

        # Save updated manifest
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Error saving manifest: {e}")

        logger.info(f"Ingestion complete. {total_objects} objects from {processed} new files. {skipped} files skipped.")
        if skipped > 0 and start_from <= 1:
            logger.info("Use --force to re-process all files.")
    else:
        logger.error(f"Input path does not exist: {input_path}")
        return

def run_index(args):
    config = load_config(args.config)
    setup_logging(config)
    logger.info(f"Building index from {args.input}")
    
    input_dir = Path(args.input)
    struct_dir = input_dir / "structural"
    
    all_objects = []
    for doc_dir in struct_dir.iterdir():
        if not doc_dir.is_dir(): continue
        for ver_dir in doc_dir.iterdir():
            if not ver_dir.is_dir(): continue
            obj_file = ver_dir / "objects.pkl"
            if obj_file.exists():
                with open(obj_file, "rb") as f:
                    all_objects.extend(pickle.load(f))
    
    spatial_index = PdfSpatialIndex()
    # In a real system we'd build a global index. For now we assume doc-level.
    # Just a smoke placeholder for the command
    logger.info(f"Loaded {len(all_objects)} objects for indexing.")

def run_evaluate(args):
    config = load_config(args.config)
    setup_logging(config)
    # This would instantiate the system and runner
    logger.info(f"Running evaluation logic for {args.dataset}")

def run_query(args):
    config = load_config(args.config)
    setup_logging(config)
    logger.info(f"Querying: {args.query}")
    
    index_path = Path(args.index)
    
    # 1. Load Structural
    # For demo, we just look for any objects.pkl in the structural subtree
    all_objects = []
    struct_root = index_path / "structural"
    for p in struct_root.rglob("objects.pkl"):
        with open(p, "rb") as f:
            all_objects.extend(pickle.load(f))
    logger.info(f"Loaded {len(all_objects)} structural objects from index.")
    
    spatial_index = PdfSpatialIndex()
    spatial_index.build_index("global", all_objects)
    
    # 2. Load Visual (PatchGridStore)
    from tracerag.visual.encoder import VisualPageEncoder
    grid_store = PatchGridStore()
    visual_root = index_path / "visual"
    grid_count = 0
    for p in visual_root.rglob("*.npz"):
        try:
            grid = VisualPageEncoder.load_patch_grid(str(p))
            grid_store.add(grid)
            grid_count += 1
        except Exception as e:
            logger.warning(f"Failed to load grid {p}: {e}")
            continue
    logger.info(f"Loaded {grid_count} visual patch grids into memory.")
    
    # 3. Instantiate components
    from tracerag.visual.encoder import VisualPageEncoder
    from tracerag.visual.scorer import VisualScorer
    from tracerag.retrieval.llm_answerer import LLMAnswerer
    from tracerag.retrieval.candidate_filter import CandidateFilter
    import re
    
    # Instantiate real components
    encoder = VisualPageEncoder(config.get("visual", {}))
    scorer = VisualScorer(config.get("visual", {}), encoder=encoder)
    llm = LLMAnswerer(config.get("llm", {}))
    
    # Improved candidate filter (scored keyword-based for demo)
    class SimpleFilter(CandidateFilter):
        def __init__(self):
            pass
        def get_candidate_pages(self, query):
            # 1. Extract alphanumeric codes (even short ones like "CSB", min 2 chars)
            codes = re.findall(r'[A-Za-z0-9][A-Za-z0-9\-_.]{1,}', query)
            
            # 2. Extract Chinese segments (split on non-Chinese characters)
            chinese_segments = re.findall(r'[\u4e00-\u9fff]+', query)
            # Break long Chinese segments into overlapping 2-char windows for matching
            cn_terms = []
            for seg in chinese_segments:
                if len(seg) <= 4:
                    cn_terms.append(seg)
                else:
                    # Use the full segment plus sub-segments
                    cn_terms.append(seg)
                    for k in range(len(seg) - 1):
                        cn_terms.append(seg[k:k+2])
            
            # 3. Also keep the raw query terms from whitespace split
            query_terms = [t.lower() for t in query.split() if len(t) > 1]
            
            logger.debug(f"Keyword extraction - codes: {codes}, cn_terms: {cn_terms}, query_terms: {query_terms}")
            
            # Score each page
            page_scores = {}  # page_id -> score
            
            for obj in all_objects:
                text = (obj.text or "")
                text_lower = text.lower()
                score = 0
                
                # Match alphanumeric codes (highest weight - these are identifiers)
                for code in codes:
                    if code.lower() in text_lower:
                        score += 5
                
                # Match Chinese segments
                for cn in cn_terms:
                    if cn in text:
                        score += 3
                
                # Match whole query (for exact substring matches)
                if query.lower().rstrip("？?。") in text_lower:
                    score += 3
                
                # Match individual whitespace-split terms
                for term in query_terms:
                    if term in text_lower:
                        score += 1
                
                if score > 0:
                    page_scores[obj.page_id] = page_scores.get(obj.page_id, 0) + score
            
            if not page_scores:
                logger.warning("No keyword matches found. Falling back to first 50 pages.")
                return list(set(obj.page_id for obj in all_objects))[:50]

            # Sort by score reversed to get top matches
            ranked_pages = sorted(page_scores.items(), key=lambda kv: kv[1], reverse=True)
            logger.info(f"Filter found {len(page_scores)} candidates. Top matches: {ranked_pages[:5]}")
            
            # Document-level expansion: include ALL pages from top-scoring documents
            # This ensures nearby pages (like page 8 when page 4 was matched) are considered
            top_page_ids = [p[0] for p in ranked_pages[:50]]
            
            # Extract base document name from page IDs (normalize across ingestion variants)
            def _get_doc_id(page_id):
                # Strip _vN_pN suffix first
                m = re.match(r'^(.+?)_v\d+_p\d+$', page_id)
                stem = m.group(1) if m else page_id
                # Strip task IDs (e.g., _task-CVYx7I8Ru2TQBZcgv94a224HQyhjF1pA)
                stem = re.sub(r'_task-[A-Za-z0-9]+$', '', stem)
                # Strip trailing underscores/hyphens used as padding
                stem = stem.rstrip('_- ')
                return stem
            
            # Find top documents by highest page score
            # Expand from TOP 3 documents (not just 1) to ensure cross-document diversity
            doc_scores = {}
            for page_id, score in ranked_pages:
                doc_id = _get_doc_id(page_id)
                if doc_id not in doc_scores or score > doc_scores[doc_id]:
                    doc_scores[doc_id] = score
            top_docs = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)
            
            # Take top 3 documents for expansion
            num_expand_docs = min(3, len(top_docs))
            expand_doc_ids = [d[0] for d in top_docs[:num_expand_docs]]
            
            logger.info(f"Top {num_expand_docs} documents for expansion: {[(d[0], d[1]) for d in top_docs[:num_expand_docs]]}")

            # Start with keyword-matched pages (ordered by score)
            expanded = list(top_page_ids)
            seen = set(top_page_ids)
            
            # Proportional budget: doc1 gets 50%, doc2 gets 30%, doc3 gets 20%
            budgets = [50, 30, 20]
            
            for doc_idx, expand_doc_id in enumerate(expand_doc_ids):
                budget = budgets[doc_idx] if doc_idx < len(budgets) else 10
                added = 0
                for obj in all_objects:
                    if added >= budget:
                        break
                    if obj.page_id not in seen and _get_doc_id(obj.page_id) == expand_doc_id:
                        expanded.append(obj.page_id)
                        seen.add(obj.page_id)
                        added += 1
            
            logger.info(f"After multi-document expansion: {len(expanded)} candidates (docs: {expand_doc_ids})")
            
            # Cap at 100 to keep ColPali manageable
            return expanded[:100]



    # Build or load manifest to resolve original document paths
    import json
    manifest_path = index_path / "manifest.json"
    manifest = {}
    
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading manifest: {e}")
            
    if not manifest:
        logger.warning(f"No manifest.json found at {manifest_path}. Source filenames will be marked as 'unknown'. Please reingest to fix.")

    system = TraceRAGSystem(
        config=config,
        candidate_filter=SimpleFilter(),
        patch_grid_store=grid_store,
        spatial_index=spatial_index,
        visual_scorer=scorer,
        llm_answerer=llm,
        manifest=manifest
    )
    
    result = system.answer(args.query)
    
    print("\n" + "="*40)
    print("TraceRAG Answer:")
    print("="*40)
    print(result.answer or "(No answer synthesized)")
    print("\n" + "="*40)
    print(f"Top Evidence ({len(result.evidences)} found):")
    print("="*40)
    for i, ev in enumerate(result.evidences[:10], 1): # Show more for debugging
        obj = spatial_index.get_object(ev.object_id)
        obj_type = obj.obj_type if obj else "unknown"
        
        # Resolve original metadata from manifest
        base_doc_id = ev.doc_id.split("_v")[0].split("_task-")[0] if ev.doc_id else ""
        orig_path_str = manifest.get(base_doc_id)
        file_type = "unknown"
        folder_name = "unknown"
        chinese_title = base_doc_id
        
        if orig_path_str:
            orig_path = Path(orig_path_str)
            file_type = orig_path.suffix.lower().lstrip('.')
            folder_name = orig_path.parent.name
            chinese_title = orig_path.stem

        print(f"{i}. [Score {ev.score:.4f}] {ev.object_id} (Type: {obj_type}, Page: {ev.page_id})")
        print(f"   Source: {chinese_title} | Folder: {folder_name} | Type: {file_type}")
        
        if obj and obj.text and len(obj.text.strip()) > 0:
            snippet = obj.text[:120].replace("\n", " ") + "..."
            print(f"   Text: {snippet}")
        else:
            print(f"   [No text content]")
    print("="*40)

def run_test_llm(args):
    config = load_config(args.config)
    setup_logging(config)
    from tracerag.retrieval.llm_answerer import LLMAnswerer
    
    logger.info(f"Testing LLM API connection with query: {args.text}")
    llm = LLMAnswerer(config.get("llm", {}))
    
    # Simple direct call (bypassing the evidence pipeline)
    try:
        # We need to simulate a dummy call since the LLMAnswerer methods usually expect objects
        # Let's add a direct text-to-text test helper or just use the OpenAI client directly
        if hasattr(llm, 'client') and llm.client:
           response = llm.client.chat.completions.create(
               model=llm.model_name,
               messages=[{"role": "user", "content": args.text}],
               max_tokens=100
           )
           print("\n" + "="*40)
           print(f"LLM Response ({llm.model_name}):")
           print("="*40)
           print(response.choices[0].message.content)
           print("="*40)
        else:
           print("LLM Client not initialized correctly.")
    except Exception as e:
        logger.error(f"LLM Test failed: {e}")

def run_benchmark(args):
    logger.info(f"Running full benchmark using config {args.config}")
    # Run over all splits configured
    pass

def analyze_failures(args):
    import glob
    import json
    from pathlib import Path
    
    logger.info(f"Analyzing failures from traces dir {args.traces_dir}")
    traces = glob.glob(str(Path(args.traces_dir) / "*.json"))
    logger.info(f"Found {len(traces)} failure traces.")
    
    for trace_file in traces:
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                query_id = data.get("query", {}).get("query_id", "unknown")
                metrics = data.get("metrics", {})
                f1 = metrics.get("answer_f1", 0.0)
                recall = metrics.get("page_recall", 0.0)
                logger.info(f"Trace {Path(trace_file).name} | Query {query_id} | F1: {f1:.2f} | Recall: {recall:.2f}")
        except Exception as e:
            logger.warning(f"Could not read trace {trace_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="TraceRAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest Command
    parser_ingest = subparsers.add_parser("ingest", help="Ingest documents (file or folder)")
    parser_ingest.add_argument("--input", required=True, help="Input file or directory path")
    parser_ingest.add_argument("--doc_id", default=None, help="Document ID (auto-generated from filename if omitted)")
    parser_ingest.add_argument("--version_id", default="v1", help="Version ID (default: v1)")
    parser_ingest.add_argument("--output", help="Output directory")
    parser_ingest.add_argument("--config", help="Config path")
    parser_ingest.add_argument("--force", action="store_true", help="Re-process all files even if already ingested")
    parser_ingest.add_argument("--start-from", type=int, default=1, help="Start from file N (e.g. --start-from 88)")

    # Index Command
    parser_index = subparsers.add_parser("index", help="Build search indexes")
    parser_index.add_argument("--input", required=True, help="Ingested artifacts directory")
    parser_index.add_argument("--config", help="Config path")

    # Eval Command
    parser_eval = subparsers.add_parser("eval", help="Run benchmark evaluation")
    parser_eval.add_argument("--dataset", required=True, help="Path to benchmark JSON")
    parser_eval.add_argument("--config", help="Model config to use")

    # Query Command
    parser_query = subparsers.add_parser("query", help="Ask a question")
    parser_query.add_argument("--query", required=True)
    parser_query.add_argument("--index", required=True, help="Index directory")
    parser_query.add_argument("--config", help="Config path")

    # Test LLM Command
    parser_test = subparsers.add_parser("test-llm", help="Test LLM API connection")
    parser_test.add_argument("--text", default="Who are you?", help="Test query")
    parser_test.add_argument("--config", help="Config path")

    # Run Benchmark Command
    parser_bench = subparsers.add_parser("run_benchmark", help="Run full benchmark suite")
    parser_bench.add_argument("--config", required=True, help="System config YAML")

    # Analyze Failures Command
    parser_analyze = subparsers.add_parser("analyze_failures", help="Export and analyze failed cases")
    parser_analyze.add_argument("--traces_dir", required=True, help="Directory containing failure traces")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "index":
        run_index(args)
    elif args.command == "eval":
        run_evaluate(args)
    elif args.command == "query":
        run_query(args)
    elif args.command == "test-llm":
        run_test_llm(args)
    elif args.command == "run_benchmark":
        run_benchmark(args)
    elif args.command == "analyze_failures":
        analyze_failures(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
