import pytest
import numpy as np
from tracerag.alignment.snapper import snap_page_relevance_to_objects, PatchRelevanceMap, ParsedPageObjects, SnapperConfig
from tracerag.common.types import VectorObject
from tracerag.structural.index import PdfSpatialIndex

def test_snapper_basic_overlap():
    config = SnapperConfig(top_k_patches=10, min_obj_score=0.1, overlap_method="iou")
    
    # Create relevance map
    scores = np.zeros((4, 4))
    scores[1, 1] = 1.0  # High relevance on patch (1,1)
    
    patch_boxes = np.zeros((4, 4, 4))
    patch_boxes[1, 1] = [10, 10, 20, 20]
    
    relevance_map = PatchRelevanceMap(
        page_id="page_1",
        grid_h=4,
        grid_w=4,
        scores=scores,
        patch_boxes=patch_boxes
    )
    
    # Create objects
    obj1 = VectorObject(object_id="obj1", doc_id="d1", version_id="v1", page_id="page_1", bbox=(10, 10, 20, 20), obj_type="path_group")
    obj2 = VectorObject(object_id="obj2", doc_id="d1", version_id="v1", page_id="page_1", bbox=(30, 30, 40, 40), obj_type="text_block")
    
    # Create spatial index
    spatial_index = PdfSpatialIndex()
    spatial_index.build_index("page_1", [obj1, obj2])
    
    page_objects = ParsedPageObjects(page_id="page_1", objects=[obj1, obj2], spatial_index=spatial_index)
    
    evidences = snap_page_relevance_to_objects(relevance_map, page_objects, config)
    
    assert len(evidences) == 1
    assert evidences[0].object_id == "obj1"
    assert evidences[0].score > 0.0
