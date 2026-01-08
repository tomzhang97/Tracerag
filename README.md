# TraceRAG: Vector-Native, Revision-Aware, and Layout-Linked Multimodal Retrieval

**High-precision, auditable retrieval for large-scale technical documents (CAD drawings, P&IDs, schematics, manuals)**

## Overview

TraceRAG is a next-generation Retrieval-Augmented Generation (RAG) framework designed specifically for **engineering-grade** technical documentation where:

- **Precision is critical**: Every answer must be exact, auditable, and spatially grounded
- **Documents are multimodal**: Dense mixtures of text, tables, diagrams, photos, and vector schematics
- **Revisions matter**: Track components across document versions with visual diffing
- **Cross-page dependencies exist**: Figures, tables, legends, and procedures are interconnected

Unlike general-purpose RAG systems optimized for semantic "fuzzy" search, TraceRAG provides:

1. **Vector-Native Evidence Alignment**: Grounds VLM attention to exact PDF vector objects (not pixel approximations)
2. **Spatio-Temporal Entity Graphs**: Tracks visual entities (components, symbols) across document revisions
3. **Claim-Level Certificates**: Every factual claim is backed by cryptographically hashed, multi-region evidence

## Key Innovation

### The "Snapper" - Vector-Native Alignment Layer

Traditional multimodal RAG systems (ColPali, BBox-DocVQA, RegionRAG) predict bounding boxes in **pixel space** - they're "guesses" of where pixels are. TraceRAG's Snapper maps VLM patch relevance to **actual PDF vector objects**:

```
VLM Patch Attention (fuzzy, pixel-based)
         ↓
    SNAPPER (φ: patches → vector objects)
         ↓
RegionEvidence (exact, object_id + bbox + hash)
```

This ensures every cited region corresponds to real PDF primitives (text blocks, vector paths, table cells), not hallucinated boxes.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     TraceRAG System                          │
├─────────────────────────────────────────────────────────────┤
│  Dual-Stream Ingestion                                      │
│  ┌──────────────────┐       ┌──────────────────┐           │
│  │  Visual Stream   │       │ Structural Stream│           │
│  │  (ColPali-style) │       │  (PDF Parser)    │           │
│  │                  │       │                  │           │
│  │  • Page renders  │       │  • Vector paths  │           │
│  │  • Patch grids   │       │  • Text blocks   │           │
│  │  • VLM embeddings│       │  • Tables        │           │
│  └────────┬─────────┘       └────────┬─────────┘           │
│           │                          │                      │
│           └──────────┬───────────────┘                      │
│                      ▼                                      │
│         ┌────────────────────────┐                         │
│         │  Snapper (φ)           │                         │
│         │  Vector-Native Align   │                         │
│         └────────────┬───────────┘                         │
│                      ▼                                      │
│         ┌────────────────────────┐                         │
│         │ Spatio-Temporal Layout │                         │
│         │ Graph (STLG) + LDG     │                         │
│         └────────────┬───────────┘                         │
│                      ▼                                      │
│         ┌────────────────────────┐                         │
│         │  Certified Claims      │                         │
│         │  (with evidence hashes)│                         │
│         └────────────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for visual encoding)
- ~10GB disk space for models and indexes

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/tracerag.git
cd tracerag

# Install dependencies
pip install -e .

# Optional: Install development dependencies
pip install -e ".[dev]"
```

### Configuration

Copy and edit the default configuration:

```bash
cp tracerag/config/defaults.yaml my_config.yaml
# Edit my_config.yaml to set paths, model settings, etc.
```

Key configuration sections:
- `data_root`: Where indexes and artifacts are stored
- `visual.model_name`: VLM model (default: ColPali)
- `structural.parser`: PDF parser backend (PyMuPDF or pdfplumber)
- `retrieval.text_index_type`: BM25 (fast) or FAISS (semantic)

## Quick Start

### 1. Ingest a PDF Document

```bash
tracerag-ingest \
  --pdf_path /path/to/manual.pdf \
  --doc_id MANUAL_001 \
  --version_id rev_C \
  --config_path my_config.yaml
```

This performs:
- **Structural parsing**: Extract text, tables, and vector graphics
- **Visual encoding**: Render pages and generate patch embeddings
- **Artifact storage**: Save to `{index_root}/structural/` and `{index_root}/visual/`

### 2. Build Indexes

```bash
tracerag-build-index \
  --index_root /data/tracerag/index \
  --config_path my_config.yaml
```

This builds:
- **Text index** (BM25 or FAISS) for fast keyword/semantic retrieval
- **STLG** (Spatio-Temporal Layout Graph) for version-aware queries

### 3. Search

```bash
# Interactive mode
tracerag-search --index_root /data/tracerag/index

# Single query
tracerag-search \
  --index_root /data/tracerag/index \
  --query "What is the torque setting for coupling bolt CFG-AX14-K2?"
```

Example output:

```
==================================================
Query: What is the torque setting for coupling bolt CFG-AX14-K2?
Type: attribute
Time: 1.23s
--------------------------------------------------

Answer: Based on the documents: Coupling bolt CFG-AX14-K2 has a torque setting of 50 Nm ±2 Nm. Refer to Table 3 for complete specifications...

Certified Claims (2):

  1. Torque setting for CFG-AX14-K2 is 50 Nm ±2 Nm
     Confidence: 0.94
     Evidences: 3 regions
       - MANUAL_001_rev_C_p12 [text_block] (score: 0.94)
       - MANUAL_001_rev_C_p12 [table_cell] (score: 0.87)
       - MANUAL_001_rev_C_p13 [symbol] (score: 0.76)

  2. Reference to Table 3 for complete specifications
     Confidence: 0.82
     Evidences: 2 regions
       - MANUAL_001_rev_C_p12 [text_block] (score: 0.82)
       - MANUAL_001_rev_C_p15 [table] (score: 0.68)

==================================================
```

## Usage Scenarios

### Scenario 1: Locator Query

**Query:** "Where is valve V-101 in the P&ID?"

**TraceRAG Process:**
1. Text index finds pages with "V-101"
2. Visual scorer ranks page regions by relevance
3. Snapper aligns to exact vector symbol object
4. Returns: `(doc_id, version_id, page_id, object_id, bbox, hash)`

### Scenario 2: Revision-Aware Query

**Query:** "What changed for component S1 between Rev A and Rev C?"

**TraceRAG Process:**
1. Classifier detects `revision` query type
2. STLG retrieves entity states across versions
3. Visual diff compares bboxes, styles, positions
4. Returns: Aligned region pairs with change annotations

### Scenario 3: Cross-Page Procedural Query

**Query:** "Show me the assembly procedure for module X, including all referenced figures and warnings"

**TraceRAG Process:**
1. Classifier detects `procedural` query type
2. Retrieves procedure steps
3. LDG expands via `refers_to` edges to figures/tables
4. Returns: Multi-page evidence spanning steps, figures, warnings

## System Components

### 1. Structural Parser (`tracerag/structural/parser.py`)

Extracts deterministic PDF primitives:
- **Text blocks** with exact bboxes (via PyMuPDF)
- **Vector paths** (lines, arcs, curves) from CAD drawings
- **Tables** (cells with coordinates)
- **Spatial index** (R-tree) for fast geometric queries

### 2. Visual Encoder (`tracerag/visual/encoder.py`)

Renders pages and generates patch embeddings:
- Uses ColPali-style VLM (Vision Transformer + LLM)
- Outputs: `PatchGrid` with embeddings `[H, W, d]` and patch bboxes
- Stores patch grids for fast visual retrieval

### 3. Snapper (`tracerag/alignment/snapper.py`)

**The core innovation:** Maps VLM attention to vector objects.

Algorithm:
```python
for each high-attention patch:
    candidates = spatial_index.query(patch_bbox)
    for obj in candidates:
        score += patch_score * IoU(patch_bbox, obj.bbox)
return top_scored_objects_as_RegionEvidence
```

### 4. Spatio-Temporal Layout Graph (`tracerag/graph/stlg.py`)

Graph schema:
- **Nodes:** Document, Version, Page, Region, Entity
- **Edges:** contains, mentions, aligned_with (cross-version), evolved_from

Enables:
- Version-aware retrieval
- Entity state tracking across revisions
- Visual diff detection

### 5. Layout Dependency Graph (`tracerag/graph/ldg.py`)

Models within-document structure:
- **Nodes:** Figure, Table, ProcedureStep, Warning
- **Edges:** refers_to, explained_by, parameterized_by

Auto-detects references like "see Fig. 3" and creates explicit links.

### 6. Retrieval Pipeline (`tracerag/retrieval/pipeline.py`)

Orchestrates end-to-end query flow:
1. Query classification (locator, attribute, procedural, revision, diff)
2. Hybrid retrieval (text + visual)
3. Vector-native alignment
4. Graph expansion
5. Claim extraction and certification

### 7. Claim Certification

Output format:
```json
{
  "query": "...",
  "answer": "...",
  "claims": [
    {
      "claim": {
        "text": "Torque is 50 Nm",
        "value": "50 Nm",
        "entity_id": "CFG-AX14-K2"
      },
      "evidences": [
        {
          "doc_id": "MANUAL_001",
          "version_id": "rev_C",
          "page_id": "MANUAL_001_rev_C_p12",
          "object_id": "obj_1234",
          "bbox": [120.5, 340.2, 280.1, 360.8],
          "obj_type": "text_block",
          "extraction_method": "vector_text",
          "score": 0.94,
          "hash": "a7f3c2...89e1"
        }
      ],
      "confidence": 0.94,
      "reasoning": "Extracted from specification table on page 12"
    }
  ]
}
```

Each evidence has:
- **Exact object ID**: Points to real PDF object
- **Tamper-evident hash**: Cryptographic fingerprint
- **Extraction method**: `vector_text`, `vector_symbol`, or `table`

## API Usage

### Python API

```python
from tracerag.retrieval.pipeline import TraceRAGSystem
from tracerag.common.utils import load_config

# Load config
config = load_config("my_config.yaml")

# Initialize system (assumes indexes are already built)
system = TraceRAGSystem.from_index_root(
    index_root="/data/tracerag/index",
    config=config
)

# Query
result = system.answer("Where is valve V-101?")

# Access results
print(result.answer)
for claim in result.certified_claims:
    print(f"Claim: {claim.claim.text}")
    print(f"Evidence pages: {[e.page_id for e in claim.evidences]}")
```

## Evaluation

### Benchmarks

TraceRAG includes evaluation framework for:

1. **Engineering Document Retrieval**: Precision/Recall on component locator queries
2. **Micro-Text Extraction**: Accuracy on small labels, codes, part numbers
3. **Visual Diff Detection**: Recall on changes across revisions
4. **Spatial Grounding**: Bbox IoU vs ground truth

Run evaluation:

```bash
tracerag-eval \
  --benchmark eng_bench \
  --index_root /data/tracerag/index \
  --output_file results.json
```

### Metrics

- **Precision@K, Recall@K**: Standard retrieval metrics
- **MAP** (Mean Average Precision): Overall ranking quality
- **Vector-Native Accuracy**: Fraction of evidences snapped to correct objects
- **Spatial Grounding Accuracy**: Bbox IoU ≥ 0.5 with ground truth

## Performance

Typical performance (on engineering manuals, 500-1000 pages):

| Operation | Time | Notes |
|-----------|------|-------|
| Ingest (1000-page PDF) | ~15 min | Structural + visual parsing |
| Build index | ~5 min | Text index + STLG |
| Query (simple locator) | ~1-2s | Text → visual → snap |
| Query (revision diff) | ~3-5s | Includes graph traversal |

**Scaling:**
- **Memory**: ~500 MB per 1000 pages (patch grids)
- **Index size**: ~200 MB per 1000 pages (BM25 + STLG)
- **GPU**: Recommended for visual encoding (10-20x faster)

## Advanced Features

### Version Alignment

Automatically align regions across document versions:

```python
stlg.auto_align_versions("rev_A", "rev_B")
```

This uses:
- Text similarity for text blocks
- Bbox IoU + style matching for vector objects
- Layer matching for CAD-derived PDFs

### Custom Reference Patterns

Configure LDG to detect custom cross-references:

```yaml
graph:
  ldg:
    reference_detection:
      patterns:
        - "refer to diagram (\\d+)"
        - "see appendix ([A-Z])"
```

### Hybrid Text + Visual Retrieval

Balance text-based (fast, identifier-focused) and visual (semantic, layout-aware) retrieval:

```yaml
retrieval:
  hybrid_alpha: 0.6  # 0 = pure text, 1 = pure visual
```

## Limitations & Future Work

### Current Limitations

1. **Table extraction**: Basic implementation; advanced tables (merged cells, nested) not fully supported
2. **Symbol recognition**: Generic grouping heuristics; no trained symbol classifier
3. **LLM integration**: Claim extraction is rule-based; future versions will use LLMs
4. **Scalability**: In-memory indexes; large corpora (>10k documents) need distributed backend

### Roadmap

- [ ] Integration with LLMs (GPT-4, Claude) for claim extraction
- [ ] Trained symbol/diagram classification models
- [ ] Multi-document cross-referencing
- [ ] Distributed index backend (Elasticsearch, Weaviate)
- [ ] Web UI for interactive exploration
- [ ] Cloud deployment (AWS, Azure)

## Research & Citations

If you use TraceRAG in your research, please cite:

```bibtex
@article{tracerag2024,
  title={TraceRAG: Vector-Native, Revision-Aware, and Layout-Linked Multimodal Retrieval for Technical Documents},
  author={[Your Name]},
  journal={arXiv preprint},
  year={2024}
}
```

**Related Work:**
- **ColPali**: Visual document retrieval with late interaction
- **BBox-DocVQA**: Bounding-box grounded document QA
- **VersionRAG**: Text-centric version-aware retrieval
- **RegionRAG**: Region-level multimodal RAG

TraceRAG extends these by combining vector-native grounding, spatio-temporal graphs, and claim-level certification.

## Contributing

Contributions welcome! Areas of interest:

- Advanced table extraction algorithms
- Symbol/diagram recognition models
- Benchmark datasets for engineering documents
- Performance optimizations
- Documentation and tutorials

See `CONTRIBUTING.md` for guidelines.

## License

MIT License. See `LICENSE` for details.

## Support

- **Issues**: https://github.com/your-org/tracerag/issues
- **Discussions**: https://github.com/your-org/tracerag/discussions
- **Email**: support@tracerag.ai

---

**TraceRAG**: Engineering-grade retrieval for systems of record.
