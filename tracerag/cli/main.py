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

QUERY_STOPWORDS = (
    "的",
    "是",
    "多少",
    "哪些",
    "什么",
    "怎么",
    "请问",
    "一下",
    "吗",
    "呢",
    "啊",
    "呀",
    "有",
    "和",
    "与",
)


def _load_manifest(index_path: Path) -> Dict[str, str]:
    import json

    manifest_path = index_path / "manifest.json"
    if not manifest_path.exists():
        return {}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _page_to_doc_id(page_id: str) -> str:
    import re

    return re.sub(r"_v\d+_p\d+$", "", page_id)


def _query_codes(query: str):
    import re

    return [code.lower() for code in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_.]{1,}", query)]


def _dedupe_preserve_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _clean_chinese_segment(segment: str) -> str:
    cleaned = segment.strip()
    for stopword in QUERY_STOPWORDS:
        cleaned = cleaned.replace(stopword, "")
    return cleaned


def _extract_query_terms(query: str):
    import re

    codes = _query_codes(query)
    chinese_segments = re.findall(r'[\u4e00-\u9fff]+', query)

    cn_terms = []
    for seg in chinese_segments:
        seg = _clean_chinese_segment(seg)
        if not seg:
            continue
        if len(seg) <= 4:
            cn_terms.append(seg)
            continue

        cn_terms.append(seg)
        for window in (2, 3):
            if len(seg) < window:
                continue
            for k in range(len(seg) - window + 1):
                cn_terms.append(seg[k:k+window])

    query_terms = [t.lower() for t in query.split() if len(t) > 1]

    return codes, _dedupe_preserve_order(cn_terms), _dedupe_preserve_order(query_terms)


def _term_in_text(term: str, text: str, text_lower: str) -> bool:
    if term.isascii():
        return term.lower() in text_lower
    return term in text


def _compute_query_feature_weights(query_features, all_objects):
    import math

    if not query_features:
        return {}

    doc_ids = {obj.doc_id for obj in all_objects if getattr(obj, "doc_id", None)}
    total_docs = max(len(doc_ids), 1)
    doc_hits = {feature: set() for feature in query_features}

    for obj in all_objects:
        text = obj.text or ""
        if not text:
            continue
        text_lower = text.lower()
        doc_id = getattr(obj, "doc_id", None) or _page_to_doc_id(obj.page_id)
        for feature in query_features:
            if _term_in_text(feature, text, text_lower):
                doc_hits[feature].add(doc_id)

    weights = {}
    for feature in query_features:
        df = len(doc_hits[feature])
        idf = math.log((total_docs + 1) / (df + 1)) + 1.0
        if feature.isascii():
            if any(char.isdigit() for char in feature):
                idf += 1.5
        elif len(feature) >= 4:
            idf += 0.4
        weights[feature] = idf

    return weights


def _metadata_doc_bonus(query: str, doc_id: str, manifest: Dict[str, str]) -> int:
    path_str = (manifest.get(doc_id) or "").lower()
    if not path_str:
        return 0

    bonus = 0
    codes, cn_terms, query_terms = _extract_query_terms(query)

    if any(code in path_str for code in codes):
        bonus += 20

    for term in cn_terms + query_terms:
        if len(term) < 2:
            continue
        if term.isascii():
            if term.lower() in path_str:
                bonus += 4
        elif term in path_str:
            bonus += 6 if len(term) >= 4 else 3

    return bonus

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
    """Convert a filename into a safe doc_id while preserving Unicode text."""
    import re
    import unicodedata

    normalized = unicodedata.normalize("NFKC", name)
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", normalized)
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized).strip(" ._")
    return sanitized or "document"


def _render_query_result(result, spatial_index, manifest: Dict[str, str], show_evidence: bool = True, max_evidence: int = 10) -> str:
    lines = [
        "",
        "=" * 40,
        "TraceRAG Answer:",
        "=" * 40,
        result.answer or "(No answer synthesized)",
    ]

    if not show_evidence:
        return "\n".join(lines)

    lines.extend(
        [
            "",
            "=" * 40,
            f"Top Evidence ({len(result.evidences)} found):",
            "=" * 40,
        ]
    )

    for index, ev in enumerate(result.evidences[:max_evidence], 1):
        obj = spatial_index.get_object(ev.object_id)
        obj_type = obj.obj_type if obj else "unknown"

        base_doc_id = ev.doc_id.split("_v")[0].split("_task-")[0] if ev.doc_id else ""
        orig_path_str = manifest.get(base_doc_id)
        file_type = "unknown"
        folder_name = "unknown"
        chinese_title = base_doc_id

        if orig_path_str:
            orig_path = Path(orig_path_str)
            file_type = orig_path.suffix.lower().lstrip(".")
            folder_name = orig_path.parent.name
            chinese_title = orig_path.stem

        lines.append(f"{index}. [Score {ev.score:.4f}] {ev.object_id} (Type: {obj_type}, Page: {ev.page_id})")
        lines.append(f"   Source: {chinese_title} | Folder: {folder_name} | Type: {file_type}")

        if obj and obj.text and obj.text.strip():
            snippet = obj.text[:120].replace("\n", " ") + "..."
            lines.append(f"   Text: {snippet}")
        else:
            lines.append("   [No text content]")

    lines.append("=" * 40)
    return "\n".join(lines)


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
    manifest = _load_manifest(index_path)
    
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
            codes, cn_terms, query_terms = _extract_query_terms(query)
            query_features = _dedupe_preserve_order(codes + cn_terms + query_terms)
            feature_weights = _compute_query_feature_weights(query_features, all_objects)
            
            logger.debug(f"Keyword extraction - codes: {codes}, cn_terms: {cn_terms}, query_terms: {query_terms}")
            
            page_feature_hits = {}  # page_id -> matched query features
            
            for obj in all_objects:
                text = (obj.text or "")
                if not text:
                    continue
                text_lower = text.lower()
                matched_features = []
                for feature in query_features:
                    if _term_in_text(feature, text, text_lower):
                        matched_features.append(feature)

                if matched_features:
                    page_hits = page_feature_hits.setdefault(obj.page_id, set())
                    page_hits.update(matched_features)

            page_scores = {
                page_id: sum(feature_weights.get(feature, 0.0) for feature in matched_features)
                for page_id, matched_features in page_feature_hits.items()
            }
            
            if not page_scores:
                logger.warning("No keyword matches found. Falling back to first 50 pages.")
                return list(set(obj.page_id for obj in all_objects))[:50]

            doc_scores = {}
            for page_id, score in page_scores.items():
                doc_id = _page_to_doc_id(page_id)
                meta_bonus = _metadata_doc_bonus(query, doc_id, manifest)
                doc_scores[doc_id] = max(doc_scores.get(doc_id, 0), score + meta_bonus)

            # Sort by score reversed to get top matches
            ranked_pages = sorted(
                page_scores.items(),
                key=lambda kv: (kv[1] + doc_scores.get(_page_to_doc_id(kv[0]), 0), kv[1]),
                reverse=True,
            )
            logger.info(f"Filter found {len(page_scores)} candidates. Top matches: {ranked_pages[:5]}")
            
            # Find top documents by highest page score plus metadata prior.
            top_docs = sorted(doc_scores.items(), key=lambda kv: kv[1], reverse=True)
            
            # Take top 3 documents for expansion
            num_expand_docs = min(3, len(top_docs))
            expand_doc_ids = [d[0] for d in top_docs[:num_expand_docs]]
            
            logger.info(f"Top {num_expand_docs} documents for expansion: {[(d[0], d[1]) for d in top_docs[:num_expand_docs]]}")

            # Start from the best pages within the strongest documents first.
            expanded_doc_ids = set(expand_doc_ids)
            prioritized_pages = [
                page_id for page_id, _ in ranked_pages
                if _page_to_doc_id(page_id) in expanded_doc_ids
            ]
            fallback_pages = [
                page_id for page_id, _ in ranked_pages
                if _page_to_doc_id(page_id) not in expanded_doc_ids
            ]
            top_page_ids = _dedupe_preserve_order(prioritized_pages + fallback_pages)[:50]

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
                    if obj.page_id not in seen and _page_to_doc_id(obj.page_id) == expand_doc_id:
                        expanded.append(obj.page_id)
                        seen.add(obj.page_id)
                        added += 1
            
            logger.info(f"After multi-document expansion: {len(expanded)} candidates (docs: {expand_doc_ids})")
            
            # Cap at 100 to keep ColPali manageable
            return expanded[:100]

    manifest_path = index_path / "manifest.json"
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
    print(_render_query_result(result, spatial_index, manifest, show_evidence=True, max_evidence=10))

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
