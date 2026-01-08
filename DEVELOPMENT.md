# TraceRAG Development Guide

## Iterative Development Order

Based on the technical specification, here's the recommended development order:

### Phase 1: MVP (Minimum Viable Product)

**Goal**: Core functionality for basic retrieval and micro-text extraction

1. **Structural Parser** (`tracerag/structural/parser.py`) ✅
   - PDF text extraction
   - Vector path extraction
   - Spatial indexing (R-tree)

2. **Visual Encoder** (`tracerag/visual/encoder.py`) ✅
   - Page rendering
   - Patch grid generation
   - Basic ColPali integration (or mock)

3. **Snapper** (`tracerag/alignment/snapper.py`) ✅
   - Patch-to-object alignment
   - IoU-based scoring
   - Evidence generation

4. **Basic Retrieval Pipeline** (`tracerag/retrieval/pipeline.py`) ✅
   - Text index (BM25)
   - Simple query flow
   - Evidence collection

5. **Micro-Text Benchmark** ✅
   - Load test data
   - Run evaluation
   - Report metrics

**Deliverables**:
- Can ingest PDFs
- Can answer locator queries ("Where is X?")
- Micro-text accuracy > 70%

### Phase 2: Version Awareness

**Goal**: Add revision tracking and visual diff capabilities

1. **STLG (Spatio-Temporal Layout Graph)** ✅
   - Document/Version/Page hierarchy
   - Region nodes
   - Entity tracking

2. **Version Alignment** ✅
   - Auto-align regions across versions
   - Visual diff detection
   - Change type classification

3. **Revision-Aware Queries** ✅
   - Query classifier updates
   - STLG-based retrieval
   - Graph expansion

4. **Visual Diff Benchmark** ✅
   - Load test data
   - Run evaluation
   - Report metrics

**Deliverables**:
- Can handle multi-version documents
- Can answer "What changed?" queries
- Visual diff recall > 60%

### Phase 3: Advanced Features

**Goal**: Cross-page dependencies and LLM-based answering

1. **LDG (Layout Dependency Graph)** ✅
   - Reference detection
   - Cross-page linking
   - Dependency expansion

2. **LLM Answerer** ✅
   - Prompt construction
   - Claim extraction
   - Certificate mapping

3. **Procedural Queries**
   - Multi-hop retrieval
   - Warning/figure linkage
   - Step-by-step assembly

4. **Full Claim Certification**
   - Evidence hashing
   - Multi-region joins
   - Confidence scoring

**Deliverables**:
- Natural language answers
- Certified claims with evidence
- Procedural query support

## Implementation Checklist

### Core Infrastructure
- [x] Project structure and configuration
- [x] Data models (types.py)
- [x] Utilities (bbox ops, logging, config)
- [x] CLI tools

### Dual-Stream Ingestion
- [x] Structural parser (PyMuPDF)
- [x] Table extraction (pdfplumber)
- [x] Symbol grouping
- [x] Visual encoder (ColPali-style)
- [x] Patch grid storage

### Vector-Native Alignment
- [x] Snapper algorithm
- [x] Spatial index queries
- [x] IoU scoring
- [x] Evidence hashing

### Graph Infrastructure
- [x] STLG implementation
- [x] LDG implementation
- [x] Version alignment
- [x] Graph expansion

### Retrieval
- [x] Text index (BM25/FAISS)
- [x] Query classifier
- [x] Visual scorer
- [x] LLM answerer
- [x] Pipeline orchestration

### Evaluation
- [x] Metrics (Precision, Recall, IoU, etc.)
- [x] Micro-text evaluator
- [x] Visual diff evaluator
- [x] Benchmark loader

### Documentation
- [x] README with usage examples
- [x] Architecture diagrams (ASCII art)
- [x] API documentation
- [x] Development guide (this file)

## Performance Optimization Tips

### Memory
- Store patch grids on disk, memory-map when needed
- Use sparse representations for R-tree indices
- Batch processing for large document sets

### Speed
- Precompute page embeddings offline
- Cache query embeddings
- Use GPU for visual encoding
- Parallelize page processing

### Scaling
- Shard spatial indices by document
- Distributed vector store (Weaviate, Qdrant)
- Async processing pipelines
- Cloud deployment (AWS Lambda, Azure Functions)

## Testing Strategy

### Unit Tests
- Bbox operations (IoU, intersection, etc.)
- Spatial index queries
- Query classification
- Evidence hashing

### Integration Tests
- End-to-end ingestion
- Query pipeline
- Graph construction
- Claim extraction

### Benchmark Tests
- Micro-text accuracy
- Visual diff recall
- Spatial grounding
- Latency measurements

## Debugging Tips

### Common Issues

**Problem**: Low retrieval accuracy
- Check text index quality (try both BM25 and FAISS)
- Verify patch grid embeddings are non-zero
- Inspect Snapper alignment scores
- Validate spatial index coverage

**Problem**: Slow queries
- Profile each pipeline stage
- Check patch grid loading time
- Monitor LLM API latency
- Consider caching strategies

**Problem**: Poor evidence grounding
- Visualize patch attention heatmaps
- Check bbox alignment with PDF objects
- Validate R-tree spatial queries
- Inspect IoU scores

### Logging

Enable debug logging for detailed traces:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Key log points:
- Number of objects extracted
- Patch grid dimensions
- Snapper alignment scores
- Query classification results
- Evidence counts per query

## Contribution Guidelines

### Code Style
- Follow PEP 8
- Use type hints
- Document all public functions
- Keep functions < 50 lines

### Git Workflow
- Feature branches: `feature/your-feature-name`
- Bug fixes: `fix/issue-description`
- Commit messages: Imperative mood ("Add feature" not "Added feature")

### Pull Request Process
1. Create feature branch
2. Implement changes
3. Add tests
4. Update documentation
5. Submit PR with description

### Review Criteria
- Code quality and readability
- Test coverage
- Documentation completeness
- Performance impact

## Future Enhancements

### Short Term (1-3 months)
- [ ] Better table extraction (complex tables, merged cells)
- [ ] Trained symbol recognition models
- [ ] Improved LLM prompt engineering
- [ ] Web UI for interactive exploration

### Medium Term (3-6 months)
- [ ] Multi-document cross-referencing
- [ ] Distributed index backend
- [ ] Cloud deployment templates
- [ ] API server (FastAPI/Flask)

### Long Term (6-12 months)
- [ ] Fine-tuned ColPali for engineering documents
- [ ] End-to-end training on claim extraction
- [ ] Active learning for benchmark expansion
- [ ] Production-ready scalability (10K+ documents)

## Resources

### Papers
- ColPali: Visual Document Retrieval with Late Interaction
- VersionRAG: Version-Aware Retrieval
- BBox-DocVQA: Grounded Document QA
- RegionRAG: Region-Level Multimodal RAG

### Libraries
- PyMuPDF: PDF parsing
- pdfplumber: Table extraction
- rtree: Spatial indexing
- NetworkX: Graph operations
- Transformers: VLM integration

### Datasets
- DocVQA: Document visual question answering
- TabFact: Table fact verification
- Engineering document benchmarks (proprietary)

## Support

For questions and issues:
- GitHub Issues: Technical bugs and feature requests
- Discussions: Architecture questions and design decisions
- Email: Development team contact

---

**Last Updated**: 2024-01-08
**Version**: 0.1.0
