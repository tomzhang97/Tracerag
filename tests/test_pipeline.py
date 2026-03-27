from tracerag.retrieval.pipeline import PatchGridStore, TraceRAGSystem
from tracerag.structural.index import PdfSpatialIndex


class DummyCandidateFilter:
    def get_candidate_pages(self, query: str):
        return []


class DummyVisualScorer:
    def score_pages_batch(self, query: str, patch_grids):
        return []


def test_pipeline_instantiation():
    system = TraceRAGSystem(
        config={},
        candidate_filter=DummyCandidateFilter(),
        patch_grid_store=PatchGridStore(),
        spatial_index=PdfSpatialIndex(),
        visual_scorer=DummyVisualScorer(),
    )
    assert system is not None


def test_pipeline_handles_empty_retrieval():
    system = TraceRAGSystem(
        config={},
        candidate_filter=DummyCandidateFilter(),
        patch_grid_store=PatchGridStore(),
        spatial_index=PdfSpatialIndex(),
        visual_scorer=DummyVisualScorer(),
    )

    pack = system.answer("where is the certificate")

    assert pack.query == "where is the certificate"
    assert pack.evidences == []
    assert "top_candidate_traces" in pack.metadata
    assert pack.metadata["top_candidate_traces"] == []
    assert "trace_summary" in pack.metadata
