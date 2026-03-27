# Messy Eval Slice

This directory is the curated regression slice for messy real-document QA.

Target size:
- ~100 examples total

Required fields per example:
- `query`
- `expected_answer`
- `expected_answer_type`
- `expected_evidence`
- `route_type`
- `failure_mode`

Recommended `failure_mode` tags:
- `broken_table_layout`
- `key_value_form`
- `ocr_noisy_scan`
- `conflicting_nearby_values`
- `heading_scoped_answer`
- `weak_alias`
- `document_list_aggregation`
- `object_list_aggregation`

Recommended `route_type` values:
- `standard_qa`
- `object_aggregation`
- `document_aggregation`
- `revision_aware_qa`

`expected_evidence` should point to the smallest reliable grounding unit you can annotate:
- `page_id`
- `object_id`
- `bbox`
- `cluster_id`

Workflow:
1. Sample from real corpus files only.
2. Tag each example with exactly one primary failure mode.
3. Keep answer text normalized to the same format expected from `answer_validator.py`.
4. Prefer examples where the evidence region is stable enough to use in regression tests.

See `template.jsonl` for the record shape.
