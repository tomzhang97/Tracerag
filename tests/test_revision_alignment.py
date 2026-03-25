import pytest
import numpy as np
from tracerag.revision.alignment import SpatialRevisionAligner
from tracerag.common.types import VectorObject

def test_revision_alignment():
    aligner = SpatialRevisionAligner(iou_threshold=0.5, translation_threshold=10.0)
    
    # Create mock objects
    obj_old = VectorObject(object_id="old1", doc_id="d1", version_id="v1", page_id="p1", bbox=(10, 10, 20, 20), obj_type="text_block", text="foo")
    
    # Same object in new version
    obj_new = VectorObject(object_id="new1", doc_id="d1", version_id="v2", page_id="p2", bbox=(10, 10, 20, 20), obj_type="text_block", text="foo")
    
    # New object added
    obj_new2 = VectorObject(object_id="new2", doc_id="d1", version_id="v2", page_id="p2", bbox=(30, 30, 40, 40), obj_type="text_block", text="bar")
    
    pairs = aligner.align_pages(old_objects=[obj_old], new_objects=[obj_new, obj_new2])
    
    # We expect obj_old aligned with obj_new, and obj_new2 unaligned
    aligned = [p for p in pairs if p.old_obj and p.new_obj]
    added = [p for p in pairs if not p.old_obj and p.new_obj]
    
    assert len(aligned) == 1
    assert aligned[0].old_obj.object_id == "old1"
    assert aligned[0].new_obj.object_id == "new1"
    
    assert len(added) == 1
    assert added[0].new_obj.object_id == "new2"
