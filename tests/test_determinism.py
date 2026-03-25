import pytest
from tracerag.common.types import VectorObject
from tracerag.structural.group import VectorGrouper

def test_deterministic_symbol_id():
    grouper = VectorGrouper()
    
    # Create mock VectorObjects
    base1 = VectorObject(
        object_id="f1d1b2_a",
        doc_id="doc1",
        version_id="v1",
        page_id="doc1_v1_0",
        bbox=(10, 10, 20, 20),
        obj_type="path_group"
    )
    
    base2 = VectorObject(
        object_id="c3d3e4_b",
        doc_id="doc1",
        version_id="v1",
        page_id="doc1_v1_0",
        bbox=(20, 20, 30, 30),
        obj_type="path_group"
    )
    
    # The order shouldn't matter for the deterministic ID
    symbol1 = grouper.create_symbol_object([base1, base2])
    symbol2 = grouper.create_symbol_object([base2, base1])
    
    assert symbol1.object_id == symbol2.object_id
    assert len(symbol1.object_id) == 16
    assert symbol1.bbox == (10, 10, 30, 30)

def test_group_paths_deterministic_sort():
    grouper = VectorGrouper()
    
    base1 = VectorObject(object_id="id1", doc_id="d1", version_id="v1", page_id="p1", bbox=(0, 0, 10, 10), obj_type="path_group")
    base2 = VectorObject(object_id="id2", doc_id="d1", version_id="v1", page_id="p1", bbox=(5, 5, 15, 15), obj_type="path_group")
    
    # Shuffle order
    groups1 = grouper.group_paths([base1, base2])
    groups2 = grouper.group_paths([base2, base1])
    
    assert len(groups1) == len(groups2) == 1
    # Check that sorting applied inside means the resulting groups list internal order is the same
    assert groups1[0][0].object_id == groups2[0][0].object_id
    assert groups1[0][1].object_id == groups2[0][1].object_id
