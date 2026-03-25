import pytest
from tracerag.retrieval.pipeline import TraceRAGSystem

def test_pipeline_instantiation():
    try:
        config = {
            "retrieval": {"text_index_type": "bm25"},
            "visual": {"model_name": "mock"},
            "debug": False
        }
        system = TraceRAGSystem(config=config)
        assert system is not None
    except Exception as e:
        pytest.fail(f"Pipeline instantiation failed: {e}")
