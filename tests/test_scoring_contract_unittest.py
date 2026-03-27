import unittest

from tracerag.cli.main import _extract_query_terms, _metadata_doc_bonus, _page_to_doc_id, _render_query_result, _sanitize_id
from tracerag.common.types import RegionEvidence, VectorObject
from tracerag.retrieval.evidence_pack import EvidencePack
from tracerag.retrieval.answer_validator import extract_validated_value, validate_answer
from tracerag.retrieval.candidate_extractor import collect_candidate_sets
from tracerag.retrieval.llm_answerer import LLMAnswerer
from tracerag.retrieval.document_aggregation import DocumentAggregationProcessor
from tracerag.retrieval.pipeline import PatchGridStore, TraceRAGSystem
from tracerag.retrieval.triple_ranker import rank_candidate_triples
from tracerag.retrieval.scoring import (
    RouteWeights,
    apply_route_scores,
    build_top_candidate_traces,
    compute_document_priors,
    compute_symbolic_bonus,
    expand_keyword_fallback_candidates,
    extract_query_signals,
)
from tracerag.structural.index import PdfSpatialIndex


class ScoringContractTests(unittest.TestCase):
    def _make_system(self, spatial_index):
        class DummyCandidateFilter:
            def get_candidate_pages(self, query: str):
                return []

        class DummyVisualScorer:
            def score_pages_batch(self, query: str, patch_grids):
                return []

        return TraceRAGSystem(
            config={},
            candidate_filter=DummyCandidateFilter(),
            patch_grid_store=PatchGridStore(),
            spatial_index=spatial_index,
            visual_scorer=DummyVisualScorer(),
        )

    def test_symbolic_bonus_requires_local_support(self):
        spatial_index = PdfSpatialIndex()
        entity_obj = VectorObject(
            object_id="entity",
            doc_id="doc_a",
            version_id="v1",
            page_id="doc_a_v1_p0",
            bbox=(0.0, 0.0, 50.0, 20.0),
            obj_type="text_block",
            text="LEHY-L-S",
        )
        far_attr_obj = VectorObject(
            object_id="attr_far",
            doc_id="doc_a",
            version_id="v1",
            page_id="doc_a_v1_p0",
            bbox=(1000.0, 1000.0, 1050.0, 1020.0),
            obj_type="text_block",
            text="红章",
        )
        near_attr_obj = VectorObject(
            object_id="attr_near",
            doc_id="doc_a",
            version_id="v1",
            page_id="doc_a_v1_p0",
            bbox=(60.0, 0.0, 110.0, 20.0),
            obj_type="text_block",
            text="红章",
        )
        spatial_index.add_object(entity_obj)
        spatial_index.add_object(far_attr_obj)
        signals = extract_query_signals("LEHY-L-S 红章")
        evidence = RegionEvidence(
            doc_id="doc_a",
            version_id="v1",
            page_id="doc_a_v1_p0",
            object_id="entity",
            bbox=entity_obj.bbox,
            obj_type="text_block",
            extraction_method="vector_text",
            score=0.0,
            visual_score=1.0,
        )

        far_bonus, _ = compute_symbolic_bonus(
            evidence,
            spatial_index,
            signals,
            page_objects_cache={"doc_a_v1_p0": [entity_obj, far_attr_obj]},
        )

        spatial_index.add_object(near_attr_obj)
        near_bonus, _ = compute_symbolic_bonus(
            evidence,
            spatial_index,
            signals,
            page_objects_cache={"doc_a_v1_p0": [entity_obj, near_attr_obj]},
        )

        self.assertLess(far_bonus, near_bonus)
        self.assertLess(far_bonus, 0.5)
        self.assertGreater(near_bonus, 0.45)

    def test_query_signals_decompose_entity_attribute_and_answer_type(self):
        signals = extract_query_signals("钢丝绳的直径是多少？")

        self.assertIn("钢丝绳", signals.entity_terms)
        self.assertEqual(signals.canonical_attribute, "diameter")
        self.assertIn("直径", signals.attribute_aliases)
        self.assertIn("Φ", signals.attribute_aliases)
        self.assertEqual(signals.answer_type, "numeric_with_unit")
        self.assertIn("mm", signals.units)

    def test_symbolic_bonus_prefers_same_row_entity_attribute_value_binding(self):
        spatial_index = PdfSpatialIndex()
        entity_obj = VectorObject(
            object_id="entity",
            doc_id="doc_b",
            version_id="v1",
            page_id="doc_b_v1_p0",
            bbox=(0.0, 0.0, 60.0, 20.0),
            obj_type="text_block",
            text="钢丝绳",
        )
        attr_obj = VectorObject(
            object_id="attr",
            doc_id="doc_b",
            version_id="v1",
            page_id="doc_b_v1_p0",
            bbox=(70.0, 0.0, 110.0, 20.0),
            obj_type="text_block",
            text="直径",
        )
        value_obj = VectorObject(
            object_id="value",
            doc_id="doc_b",
            version_id="v1",
            page_id="doc_b_v1_p0",
            bbox=(120.0, 0.0, 170.0, 20.0),
            obj_type="text_block",
            text="10mm",
        )
        far_value_obj = VectorObject(
            object_id="value_far",
            doc_id="doc_b",
            version_id="v1",
            page_id="doc_b_v1_p0",
            bbox=(0.0, 300.0, 50.0, 320.0),
            obj_type="text_block",
            text="10mm",
        )

        for obj in (entity_obj, attr_obj, value_obj):
            spatial_index.add_object(obj)

        signals = extract_query_signals("钢丝绳的直径是多少？")
        evidence = RegionEvidence(
            doc_id="doc_b",
            version_id="v1",
            page_id="doc_b_v1_p0",
            object_id="entity",
            bbox=entity_obj.bbox,
            obj_type="text_block",
            extraction_method="vector_text",
            score=0.0,
            visual_score=1.0,
        )

        same_row_bonus, _ = compute_symbolic_bonus(
            evidence,
            spatial_index,
            signals,
            page_objects_cache={"doc_b_v1_p0": [entity_obj, attr_obj, value_obj]},
        )

        far_bonus, _ = compute_symbolic_bonus(
            evidence,
            spatial_index,
            signals,
            page_objects_cache={"doc_b_v1_p0": [entity_obj, attr_obj, far_value_obj]},
        )

        self.assertGreater(same_row_bonus, far_bonus)
        self.assertGreater(same_row_bonus, 0.65)

    def test_answer_validator_rejects_non_diameter_numbers(self):
        self.assertIsNone(extract_validated_value("10页", "numeric_with_unit", ("mm",)))
        self.assertIsNone(extract_validated_value("10台", "numeric_with_unit", ("mm",)))
        self.assertEqual(extract_validated_value("Φ10", "numeric_with_unit", ("mm",)), "10 mm")

    def test_answer_validator_returns_structured_numeric_plain_result(self):
        result = validate_answer(
            "Qty: 1,250",
            "numeric_plain",
            regex_hints=(r"(?i)\b\d{1,3}(?:,\d{3})+\b",),
            normalization_rules=("strip_commas",),
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(result.normalized_value, "1250")
        self.assertGreater(result.confidence, 0.9)
        self.assertIn("numeric_plain", result.match_reason)

    def test_answer_validator_supports_range_identifier_date_and_certificate_id(self):
        range_result = validate_answer(
            "Working temperature: -20 ~ 60 °C",
            "range_with_unit",
            units=("°c",),
            normalization_rules=("normalize_range_dash", "normalize_unit_spacing", "lowercase"),
        )
        identifier_result = validate_answer(
            "Model No: gsb-282n",
            "identifier",
            regex_hints=(r"(?i)\b[A-Z0-9]+(?:-[A-Z0-9]+)+\b",),
            normalization_rules=("uppercase",),
        )
        date_result = validate_answer(
            "Revision Date: 2024年07月16日",
            "date",
            normalization_rules=("iso_date",),
        )
        cert_result = validate_answer(
            "Certificate No: TSXF32002220160022",
            "certificate_id",
            regex_hints=(r"(?i)\bTSXF[A-Z0-9-]{6,}\b",),
            normalization_rules=("uppercase",),
        )

        self.assertTrue(all(item.is_valid for item in (range_result, identifier_result, date_result, cert_result)))
        self.assertEqual(range_result.normalized_value, "-20-60 °c")
        self.assertEqual(identifier_result.normalized_value, "GSB-282N")
        self.assertEqual(date_result.normalized_value, "2024-07-16")
        self.assertEqual(cert_result.normalized_value, "TSXF32002220160022")

    def test_symbolic_bonus_supports_attribute_alias_binding(self):
        spatial_index = PdfSpatialIndex()
        entity_obj = VectorObject(
            object_id="entity_alias",
            doc_id="doc_c",
            version_id="v1",
            page_id="doc_c_v1_p0",
            bbox=(0.0, 0.0, 60.0, 20.0),
            obj_type="text_block",
            text="钢丝绳",
        )
        attr_obj = VectorObject(
            object_id="attr_alias",
            doc_id="doc_c",
            version_id="v1",
            page_id="doc_c_v1_p0",
            bbox=(70.0, 0.0, 90.0, 20.0),
            obj_type="text_block",
            text="Φ",
        )
        value_obj = VectorObject(
            object_id="value_alias",
            doc_id="doc_c",
            version_id="v1",
            page_id="doc_c_v1_p0",
            bbox=(100.0, 0.0, 150.0, 20.0),
            obj_type="text_block",
            text="10mm",
        )
        for obj in (entity_obj, attr_obj, value_obj):
            spatial_index.add_object(obj)

        signals = extract_query_signals("钢丝绳的直径是多少？")
        evidence = RegionEvidence(
            doc_id="doc_c",
            version_id="v1",
            page_id="doc_c_v1_p0",
            object_id="entity_alias",
            bbox=entity_obj.bbox,
            obj_type="text_block",
            extraction_method="vector_text",
            score=0.0,
            visual_score=1.0,
        )

        bonus, _ = compute_symbolic_bonus(
            evidence,
            spatial_index,
            signals,
            page_objects_cache={"doc_c_v1_p0": [entity_obj, attr_obj, value_obj]},
        )

        self.assertGreater(bonus, 0.60)
        self.assertGreaterEqual(evidence.attribute_match, 1.0)
        self.assertGreater(evidence.local_structure_score, 0.0)

    def test_symbolic_bonus_does_not_cross_clusters_for_value_binding(self):
        spatial_index = PdfSpatialIndex()
        entity_obj = VectorObject(
            object_id="entity_cluster",
            doc_id="doc_d",
            version_id="v1",
            page_id="doc_d_v1_p0",
            bbox=(0.0, 0.0, 60.0, 20.0),
            obj_type="text_block",
            text="钢丝绳",
        )
        attr_obj = VectorObject(
            object_id="attr_cluster",
            doc_id="doc_d",
            version_id="v1",
            page_id="doc_d_v1_p0",
            bbox=(70.0, 0.0, 110.0, 20.0),
            obj_type="text_block",
            text="直径",
        )
        far_value_obj = VectorObject(
            object_id="value_cluster_far",
            doc_id="doc_d",
            version_id="v1",
            page_id="doc_d_v1_p0",
            bbox=(500.0, 400.0, 560.0, 420.0),
            obj_type="text_block",
            text="10mm",
        )
        for obj in (entity_obj, attr_obj, far_value_obj):
            spatial_index.add_object(obj)

        signals = extract_query_signals("钢丝绳的直径是多少？")
        evidence = RegionEvidence(
            doc_id="doc_d",
            version_id="v1",
            page_id="doc_d_v1_p0",
            object_id="entity_cluster",
            bbox=entity_obj.bbox,
            obj_type="text_block",
            extraction_method="vector_text",
            score=0.0,
            visual_score=1.0,
        )

        bonus, _ = compute_symbolic_bonus(
            evidence,
            spatial_index,
            signals,
            page_objects_cache={"doc_d_v1_p0": [entity_obj, attr_obj, far_value_obj]},
        )

        self.assertLess(bonus, 0.70)
        self.assertEqual(evidence.value_match, 0.0)
        self.assertGreater(evidence.local_structure_score, 0.0)

    def test_triple_ranker_penalizes_conflicting_attribute_value_pair(self):
        page_objects = [
            VectorObject(
                object_id="entity_conflict",
                doc_id="doc_e",
                version_id="v1",
                page_id="doc_e_v1_p0",
                bbox=(0.0, 0.0, 60.0, 20.0),
                obj_type="text_block",
                text="\u94a2\u4e1d\u7ef3",
            ),
            VectorObject(
                object_id="attr_diameter",
                doc_id="doc_e",
                version_id="v1",
                page_id="doc_e_v1_p0",
                bbox=(70.0, 0.0, 110.0, 20.0),
                obj_type="text_block",
                text="\u76f4\u5f84",
            ),
            VectorObject(
                object_id="value_diameter",
                doc_id="doc_e",
                version_id="v1",
                page_id="doc_e_v1_p0",
                bbox=(120.0, 0.0, 170.0, 20.0),
                obj_type="text_block",
                text="10mm",
            ),
            VectorObject(
                object_id="attr_length",
                doc_id="doc_e",
                version_id="v1",
                page_id="doc_e_v1_p0",
                bbox=(70.0, 28.0, 110.0, 48.0),
                obj_type="text_block",
                text="\u957f\u5ea6",
            ),
            VectorObject(
                object_id="value_length",
                doc_id="doc_e",
                version_id="v1",
                page_id="doc_e_v1_p0",
                bbox=(120.0, 28.0, 170.0, 48.0),
                obj_type="text_block",
                text="12mm",
            ),
        ]

        signals = extract_query_signals("\u94a2\u4e1d\u7ef3\u7684\u76f4\u5f84\u662f\u591a\u5c11\uff1f")
        candidate_sets = collect_candidate_sets(page_objects, signals)
        triples = rank_candidate_triples(candidate_sets, signals)

        self.assertGreater(len(triples), 0)
        self.assertEqual(triples[0].attribute_atom_id, "attr_diameter")
        self.assertEqual(triples[0].value_atom_id, "value_diameter")
        competing_value_triple = next(triple for triple in triples if triple.value_atom_id == "value_length")
        self.assertLess(
            triples[0].features["contradiction_penalty"],
            competing_value_triple.features["contradiction_penalty"],
        )
        self.assertGreater(triples[0].final_score, competing_value_triple.final_score)

    def test_symbolic_bonus_uses_nearest_previous_heading_band_as_weak_scope(self):
        spatial_index = PdfSpatialIndex()
        heading_obj = VectorObject(
            object_id="heading_scope",
            doc_id="doc_scope",
            version_id="v1",
            page_id="doc_scope_v1_p0",
            bbox=(0.0, 0.0, 120.0, 18.0),
            obj_type="text_block",
            text="\u94a2\u4e1d\u7ef3\u53c2\u6570",
        )
        entity_obj = VectorObject(
            object_id="entity_scope",
            doc_id="doc_scope",
            version_id="v1",
            page_id="doc_scope_v1_p0",
            bbox=(0.0, 50.0, 60.0, 70.0),
            obj_type="text_block",
            text="\u94a2\u4e1d\u7ef3",
        )
        attr_obj = VectorObject(
            object_id="attr_scope",
            doc_id="doc_scope",
            version_id="v1",
            page_id="doc_scope_v1_p0",
            bbox=(70.0, 50.0, 110.0, 70.0),
            obj_type="text_block",
            text="\u76f4\u5f84",
        )
        value_obj = VectorObject(
            object_id="value_scope",
            doc_id="doc_scope",
            version_id="v1",
            page_id="doc_scope_v1_p0",
            bbox=(120.0, 50.0, 170.0, 70.0),
            obj_type="text_block",
            text="10mm",
        )
        for obj in (heading_obj, entity_obj, attr_obj, value_obj):
            spatial_index.add_object(obj)

        signals = extract_query_signals("\u94a2\u4e1d\u7ef3\u7684\u76f4\u5f84\u662f\u591a\u5c11\uff1f")
        evidence = RegionEvidence(
            doc_id="doc_scope",
            version_id="v1",
            page_id="doc_scope_v1_p0",
            object_id="entity_scope",
            bbox=entity_obj.bbox,
            obj_type="text_block",
            extraction_method="vector_text",
            score=0.0,
            visual_score=1.0,
        )

        compute_symbolic_bonus(
            evidence,
            spatial_index,
            signals,
            page_objects_cache={"doc_scope_v1_p0": [heading_obj, entity_obj, attr_obj, value_obj]},
        )

        self.assertGreater(evidence.scope_score, 0.0)
        self.assertLessEqual(evidence.scope_score, 0.35)

    def test_cli_page_to_doc_id_preserves_distinct_sanitized_docs(self):
        self.assertEqual(_page_to_doc_id("25N4L10-182-3_____v1_p9"), "25N4L10-182-3____")
        self.assertEqual(_page_to_doc_id("25N4L10-182-3_______v1_p9"), "25N4L10-182-3______")
        self.assertNotEqual(
            _page_to_doc_id("25N4L10-182-3_____v1_p9"),
            _page_to_doc_id("25N4L10-182-3_______v1_p9"),
        )

    def test_cli_sanitize_id_preserves_chinese(self):
        self.assertEqual(_sanitize_id("装箱清单"), "装箱清单")
        self.assertEqual(_sanitize_id("LEHY-L-S 资料 - 副本"), "LEHY-L-S_资料_-_副本")
        self.assertEqual(
            _sanitize_id('合同号25N4L10-182-3:装箱清单/总毛重?'),
            "合同号25N4L10-182-3_装箱清单_总毛重",
        )

    def test_cli_page_to_doc_id_preserves_chinese_doc_ids(self):
        self.assertEqual(_page_to_doc_id("装箱清单_v1_p9"), "装箱清单")
        self.assertEqual(_page_to_doc_id("维护说明书_v1_p4"), "维护说明书")

    def test_cli_metadata_bonus_prefers_relevant_path_overlap(self):
        manifest = {
            "cert_doc": "/repo/1.型式试验证书/LEHY-L-S门锁证书.pdf",
            "manual_doc": "/repo/7.维护说明书/LEHY-L-S维护说明书.pdf",
        }
        query = "LEHY-L-S的型式试验证书有哪些？"

        cert_bonus = _metadata_doc_bonus(query, "cert_doc", manifest)
        manual_bonus = _metadata_doc_bonus(query, "manual_doc", manifest)

        self.assertGreater(cert_bonus, manual_bonus)
        self.assertGreater(cert_bonus, 0)

    def test_cli_extract_query_terms_removes_query_noise(self):
        codes, cn_terms, query_terms = _extract_query_terms("合同号25N4L10-182-3电梯的总毛重是多少？")

        self.assertIn("25n4l10-182-3", codes)
        self.assertIn("总毛重", cn_terms)
        self.assertIn("毛重", cn_terms)
        self.assertNotIn("多少", cn_terms)
        self.assertNotIn("装箱清单", cn_terms)

    def test_cli_extract_query_terms_keeps_specific_bigrams(self):
        _, cn_terms, _ = _extract_query_terms("合同号25N4L10-182-3电梯的总毛重是多少？")
        self.assertIn("电梯", cn_terms)
        self.assertIn("总毛", cn_terms)

    def test_keyword_fallback_expands_candidates_without_overriding_visual_rank(self):
        spatial_index = PdfSpatialIndex()
        visual_obj = VectorObject(
            object_id="visual_hit",
            doc_id="doc_visual",
            version_id="v1",
            page_id="doc_visual_v1_p0",
            bbox=(0.0, 0.0, 40.0, 20.0),
            obj_type="text_block",
            text="summary cell",
        )
        fallback_obj = VectorObject(
            object_id="fallback_hit",
            doc_id="doc_fallback",
            version_id="v1",
            page_id="doc_fallback_v1_p0",
            bbox=(0.0, 0.0, 40.0, 20.0),
            obj_type="text_block",
            text="LEHY-L-S certificate",
        )
        spatial_index.add_object(visual_obj)
        spatial_index.add_object(fallback_obj)

        evidence = RegionEvidence(
            doc_id="doc_visual",
            version_id="v1",
            page_id="doc_visual_v1_p0",
            object_id="visual_hit",
            bbox=visual_obj.bbox,
            obj_type="text_block",
            extraction_method="vector_text",
            score=5.0,
            visual_score=5.0,
            doc_prior=0.0,
        )
        signals = extract_query_signals("LEHY-L-S certificate")
        evidences = expand_keyword_fallback_candidates(
            candidate_page_ids=["doc_visual_v1_p0", "doc_fallback_v1_p0"],
            evidences=[evidence],
            spatial_index=spatial_index,
            query_signals=signals,
            doc_priors={"doc_visual": 0.0, "doc_fallback": 0.0},
        )
        for item in evidences:
            item.symbolic_bonus, _ = compute_symbolic_bonus(item, spatial_index, signals)

        apply_route_scores(
            evidences,
            RouteWeights(visual=1.0, symbolic=0.35, scope=0.10, document=0.15),
        )
        ranked = sorted(evidences, key=lambda item: item.score, reverse=True)

        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].object_id, "visual_hit")
        self.assertEqual(ranked[1].object_id, "fallback_hit")

    def test_document_prior_uses_metadata_and_stays_bounded(self):
        spatial_index = PdfSpatialIndex()
        title_obj = VectorObject(
            object_id="title",
            doc_id="doc_meta",
            version_id="v1",
            page_id="doc_meta_v1_p0",
            bbox=(0.0, 0.0, 200.0, 40.0),
            obj_type="text_block",
            text="LEHY-L-S 型式证书",
        )
        spatial_index.add_object(title_obj)
        manifest = {
            "doc_meta": r"C:\repo\certificates\LEHY-L-S_type_certificate.pdf",
        }
        signals = extract_query_signals("LEHY-L-S 型式证书")
        profiles, priors = compute_document_priors(
            doc_ids=["doc_meta"],
            manifest=manifest,
            spatial_index=spatial_index,
            query_signals=signals,
        )

        self.assertIn("doc_meta", profiles)
        self.assertGreater(priors["doc_meta"], 0.0)
        self.assertLessEqual(priors["doc_meta"], 1.0)

    def test_document_prior_uses_ancestor_product_folder(self):
        spatial_index = PdfSpatialIndex()
        manifest = {
            "cert_doc": r"C:\repo\LEHY-L-S资料 - 副本\1.型式试验证书\certificate_a.pdf",
            "manual_doc": r"C:\repo\LEHY-L-S资料 - 副本\7.维护说明书\manual_a.pdf",
        }
        signals = extract_query_signals("LEHY-L-S的型式试验证书有哪些？")

        _, priors = compute_document_priors(
            doc_ids=["cert_doc", "manual_doc"],
            manifest=manifest,
            spatial_index=spatial_index,
            query_signals=signals,
        )

        self.assertGreater(priors["cert_doc"], priors["manual_doc"])
        self.assertGreater(priors["cert_doc"], 0.20)

    def test_document_aggregation_uses_shared_support_for_target_doc_type(self):
        spatial_index = PdfSpatialIndex()
        manifest = {
            "doc_a": r"C:\repo\certificates\LEHY-L-S_certificate.pdf",
            "doc_b": r"C:\repo\misc\other.pdf",
        }
        processor = DocumentAggregationProcessor(manifest, spatial_index)
        signals = extract_query_signals("list all LEHY-L-S certificates")
        evidence = RegionEvidence(
            doc_id="doc_a",
            version_id="v1",
            page_id="doc_a_v1_p0",
            object_id="support_a",
            bbox=(0.0, 0.0, 10.0, 10.0),
            obj_type="text_block",
            extraction_method="vector_text",
            score=0.8,
            visual_score=0.8,
            symbolic_bonus=0.4,
            doc_prior=0.0,
        )

        candidates = processor.retrieve_candidate_documents(
            all_evidences=[evidence],
            query_signals=signals,
            route_weights=RouteWeights(visual=0.35, symbolic=0.45, scope=0.35, document=0.9),
        )

        doc_a = next(item for item in candidates if item["doc_id"] == "doc_a")
        self.assertGreater(doc_a["visual_score"], 0.0)
        self.assertGreater(doc_a["score"], 0.0)

    def test_document_aggregation_prefers_certificate_collection_docs(self):
        spatial_index = PdfSpatialIndex()
        manifest = {
            "cert_doc": r"C:\repo\LEHY-L-S资料 - 副本\1.型式试验证书\cert_a.pdf",
            "manual_doc": r"C:\repo\LEHY-L-S资料 - 副本\7.维护说明书\manual_a.pdf",
        }
        processor = DocumentAggregationProcessor(manifest, spatial_index)
        signals = extract_query_signals("LEHY-L-S的型式试验证书有哪些？")

        manual_support = RegionEvidence(
            doc_id="manual_doc",
            version_id="v1",
            page_id="manual_doc_v1_p0",
            object_id="manual_support",
            bbox=(0.0, 0.0, 10.0, 10.0),
            obj_type="text_block",
            extraction_method="vector_text",
            score=0.9,
            visual_score=0.9,
            symbolic_bonus=0.0,
            doc_prior=0.0,
        )

        candidates = processor.retrieve_candidate_documents(
            all_evidences=[manual_support],
            query_signals=signals,
            route_weights=RouteWeights(visual=0.35, symbolic=0.45, scope=0.35, document=0.9),
        )

        self.assertEqual(candidates[0]["doc_id"], "cert_doc")
        self.assertTrue(all(item["doc_id"] == "cert_doc" for item in candidates))

    def test_document_aggregation_expands_all_docs_in_dominant_certificate_facet(self):
        spatial_index = PdfSpatialIndex()
        manifest = {
            "cert_a": r"C:\repo\LEHY-L-S资料 - 副本\1.型式试验证书\安全钳(轿厢侧)  GSB-282N.pdf",
            "cert_b": r"C:\repo\LEHY-L-S资料 - 副本\1.型式试验证书\缓冲器(轿厢侧)  ZOBR-68.pdf",
            "cert_c": r"C:\repo\LEHY-L-S资料 - 副本\1.型式试验证书\分项\门锁(层门)  ZIL-33A.pdf",
            "manual_doc": r"C:\repo\LEHY-L-S资料 - 副本\7.维护说明书\manual_a.pdf",
            "other_product_cert": r"C:\repo\OTHER资料\1.型式试验证书\other_cert.pdf",
        }
        processor = DocumentAggregationProcessor(manifest, spatial_index)
        signals = extract_query_signals("LEHY-L-S的型式试验证书有哪些？")

        seed_support = RegionEvidence(
            doc_id="cert_a",
            version_id="v1",
            page_id="cert_a_v1_p0",
            object_id="seed_support",
            bbox=(0.0, 0.0, 10.0, 10.0),
            obj_type="text_block",
            extraction_method="vector_text",
            score=0.9,
            visual_score=0.9,
            symbolic_bonus=0.2,
            doc_prior=0.0,
        )

        candidates = processor.retrieve_candidate_documents(
            all_evidences=[seed_support],
            query_signals=signals,
            route_weights=RouteWeights(visual=0.35, symbolic=0.45, scope=0.35, document=0.9),
        )

        candidate_ids = {item["doc_id"] for item in candidates}
        self.assertIn("cert_a", candidate_ids)
        self.assertIn("cert_b", candidate_ids)
        self.assertIn("cert_c", candidate_ids)
        self.assertNotIn("manual_doc", candidate_ids)
        self.assertNotIn("other_product_cert", candidate_ids)

    def test_document_identity_uses_concise_filename_over_long_title_blob(self):
        spatial_index = PdfSpatialIndex()
        manifest = {
            "cert_doc": r"C:\repo\LEHY-L-S资料 - 副本\1.型式试验证书\安全钳(轿厢侧)  GSB-282N.pdf",
        }
        processor = DocumentAggregationProcessor(manifest, spatial_index)
        signals = extract_query_signals("LEHY-L-S的型式试验证书有哪些？")
        candidates = processor.retrieve_candidate_documents(
            all_evidences=[],
            query_signals=signals,
            route_weights=RouteWeights(visual=0.35, symbolic=0.45, scope=0.35, document=0.9),
        )

        identity = processor.extract_document_identity(candidates[0])
        self.assertEqual(identity["extracted_title"], "安全钳(轿厢侧) gsb-282n")

    def test_document_aggregation_deduplicates_task_variants(self):
        spatial_index = PdfSpatialIndex()
        processor = DocumentAggregationProcessor({}, spatial_index)
        identities = [
            {
                "doc_id": "doc_a",
                "filename": "安全钳(轿厢侧)  GSB-282N.pdf",
                "folder": "1.型式试验证书",
                "extracted_title": "安全钳(轿厢侧) GSB-282N",
                "certificate_number": "",
                "score": 1.0,
                "doc_prior": 0.4,
                "visual_score": 0.8,
                "symbolic_bonus": 0.3,
            },
            {
                "doc_id": "doc_a_task-abc123",
                "filename": "安全钳(轿厢侧)  GSB-282N_task-abc123.pdf",
                "folder": "1.型式试验证书",
                "extracted_title": "安全钳(轿厢侧) GSB-282N",
                "certificate_number": "",
                "score": 0.9,
                "doc_prior": 0.4,
                "visual_score": 0.7,
                "symbolic_bonus": 0.3,
            },
        ]

        deduped = processor.aggregate_documents(identities)
        self.assertEqual(len(deduped), 1)

    def test_summary_context_adds_full_page_table_rows(self):
        spatial_index = PdfSpatialIndex()
        seed_obj = VectorObject(
            object_id="seed",
            doc_id="doc_table",
            version_id="v1",
            page_id="doc_table_v1_p9",
            bbox=(0.0, 0.0, 100.0, 20.0),
            obj_type="text_block",
            text="箱01 520 kg",
        )
        total_obj = VectorObject(
            object_id="total",
            doc_id="doc_table",
            version_id="v1",
            page_id="doc_table_v1_p9",
            bbox=(0.0, 200.0, 120.0, 220.0),
            obj_type="text_block",
            text="总毛重 7025 kg",
        )
        missing_row = VectorObject(
            object_id="row19",
            doc_id="doc_table",
            version_id="v1",
            page_id="doc_table_v1_p9",
            bbox=(0.0, 180.0, 120.0, 200.0),
            obj_type="text_block",
            text="箱19 310 kg",
        )
        for obj in (seed_obj, total_obj, missing_row):
            spatial_index.add_object(obj)

        system = self._make_system(spatial_index)
        query_signals = extract_query_signals("合同号25N4L10-182-3电梯的总毛重是多少？")
        ranked_evidence = RegionEvidence(
            doc_id="doc_table",
            version_id="v1",
            page_id="doc_table_v1_p9",
            object_id="seed",
            bbox=seed_obj.bbox,
            obj_type="text_block",
            extraction_method="vector_text",
            score=0.9,
            visual_score=0.9,
            symbolic_bonus=0.2,
            doc_prior=0.1,
        )

        expanded = system._expand_summary_context(
            query="合同号25N4L10-182-3电梯的总毛重是多少？",
            ranked_evidences=[ranked_evidence],
            doc_priors={"doc_table": 0.1},
            query_signals=query_signals,
        )

        expanded_ids = {item.object_id for item in expanded}
        self.assertIn("seed", expanded_ids)
        self.assertIn("total", expanded_ids)
        self.assertIn("row19", expanded_ids)

    def test_llm_answerer_parses_fenced_json(self):
        answerer = LLMAnswerer({"provider": "mock"})
        raw = """```json
        {"answer": "7025 kg", "claims": []}
        ```"""

        parsed = answerer._extract_json_payload(raw)
        self.assertEqual(parsed.strip(), '{"answer": "7025 kg", "claims": []}')

    def test_llm_prompt_truncates_evidence_and_requests_concise_answer(self):
        answerer = LLMAnswerer(
            {
                "provider": "mock",
                "max_evidences": 3,
                "max_chars_per_evidence": 80,
                "max_total_evidence_chars": 160,
                "max_prompt_chars": 1200,
            }
        )
        evidences = [
            RegionEvidence(
                doc_id=f"doc_{idx}",
                version_id="v1",
                page_id=f"doc_{idx}_v1_p0",
                object_id=f"obj_{idx}",
                bbox=(0.0, 0.0, 10.0, 10.0),
                obj_type="text_block",
                extraction_method="vector_text",
                score=1.0 - (idx * 0.1),
            )
            for idx in range(5)
        ]
        evidence_texts = {
            f"obj_{idx}": ("very long evidence text " * 40) + str(idx)
            for idx in range(5)
        }

        prompt = answerer._build_prompt("What is the answer?", evidences, evidence_texts, "general")

        self.assertLessEqual(len(prompt), 1200 + len("\n\nJSON Response:"))
        self.assertIn("do not mention evidence", prompt.lower())
        self.assertLessEqual(prompt.count("[E"), 3)

    def test_render_query_result_hides_evidence_by_default(self):
        pack = EvidencePack(
            query="test",
            answer="7025 kg",
            evidences=[
                RegionEvidence(
                    doc_id="doc_a",
                    version_id="v1",
                    page_id="doc_a_v1_p0",
                    object_id="obj_a",
                    bbox=(0.0, 0.0, 10.0, 10.0),
                    obj_type="text_block",
                    extraction_method="vector_text",
                    score=0.9,
                )
            ],
        )
        spatial_index = PdfSpatialIndex()
        spatial_index.add_object(
            VectorObject(
                object_id="obj_a",
                doc_id="doc_a",
                version_id="v1",
                page_id="doc_a_v1_p0",
                bbox=(0.0, 0.0, 10.0, 10.0),
                obj_type="text_block",
                text="sample evidence text",
            )
        )

        rendered = _render_query_result(pack, spatial_index, {}, show_evidence=False)
        rendered_with_evidence = _render_query_result(pack, spatial_index, {}, show_evidence=True, max_evidence=1)

        self.assertIn("TraceRAG Answer:", rendered)
        self.assertNotIn("Top Evidence", rendered)
        self.assertIn("Top Evidence", rendered_with_evidence)

    def test_pipeline_top_candidate_traces_include_normalized_value_and_contradictions(self):
        spatial_index = PdfSpatialIndex()
        entity_obj = VectorObject(
            object_id="trace_entity",
            doc_id="doc_trace",
            version_id="v1",
            page_id="doc_trace_v1_p0",
            bbox=(0.0, 0.0, 60.0, 20.0),
            obj_type="text_block",
            text="钢丝绳",
        )
        attr_obj = VectorObject(
            object_id="trace_attr",
            doc_id="doc_trace",
            version_id="v1",
            page_id="doc_trace_v1_p0",
            bbox=(70.0, 0.0, 110.0, 20.0),
            obj_type="text_block",
            text="直径",
        )
        value_obj = VectorObject(
            object_id="trace_value",
            doc_id="doc_trace",
            version_id="v1",
            page_id="doc_trace_v1_p0",
            bbox=(120.0, 0.0, 170.0, 20.0),
            obj_type="text_block",
            text="10mm",
        )
        for obj in (entity_obj, attr_obj, value_obj):
            spatial_index.add_object(obj)

        signals = extract_query_signals("钢丝绳的直径是多少？")
        evidence = RegionEvidence(
            doc_id="doc_trace",
            version_id="v1",
            page_id="doc_trace_v1_p0",
            object_id="trace_entity",
            bbox=entity_obj.bbox,
            obj_type="text_block",
            extraction_method="vector_text",
            score=0.0,
            visual_score=1.0,
            doc_prior=0.1,
        )

        evidence.symbolic_bonus, _ = compute_symbolic_bonus(
            evidence,
            spatial_index,
            signals,
            page_objects_cache={"doc_trace_v1_p0": [entity_obj, attr_obj, value_obj]},
        )
        apply_route_scores([evidence], RouteWeights(visual=1.0, symbolic=0.45, scope=0.10, document=0.15))

        traces = build_top_candidate_traces([evidence], limit=5)

        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["normalized_value"], "10 mm")
        self.assertEqual(traces[0]["entity_span"]["text"], "钢丝绳")
        self.assertEqual(traces[0]["attribute_span"]["text"], "直径")
        self.assertEqual(traces[0]["value_span"]["text"], "10mm")
        self.assertIn("total", traces[0]["contradiction_penalties"])

    def test_document_aggregation_answer_enumerates_all_items(self):
        spatial_index = PdfSpatialIndex()
        system = self._make_system(spatial_index)

        evidence_a = RegionEvidence(
            doc_id="doc_a",
            version_id="v1",
            page_id="doc_a_v1_p0",
            object_id="doc_agg_a",
            bbox=(0.0, 0.0, 0.0, 0.0),
            obj_type="document_identity",
            extraction_method="document_aggregation",
            score=0.9,
        )
        evidence_b = RegionEvidence(
            doc_id="doc_b",
            version_id="v1",
            page_id="doc_b_v1_p0",
            object_id="doc_agg_b",
            bbox=(0.0, 0.0, 0.0, 0.0),
            obj_type="document_identity",
            extraction_method="document_aggregation",
            score=0.8,
        )

        spatial_index.add_object(
            VectorObject(
                object_id="doc_agg_a",
                doc_id="doc_a",
                version_id="v1",
                page_id="doc_a_v1_p0",
                bbox=(0.0, 0.0, 0.0, 0.0),
                obj_type="document_identity",
                text="[1.型式试验证书] 安全钳(轿厢侧) GSB-282N (No: TSXF32002220160022)",
                meta={"title": "安全钳(轿厢侧) GSB-282N", "folder": "1.型式试验证书", "certificate_number": "TSXF32002220160022"},
            )
        )
        spatial_index.add_object(
            VectorObject(
                object_id="doc_agg_b",
                doc_id="doc_b",
                version_id="v1",
                page_id="doc_b_v1_p0",
                bbox=(0.0, 0.0, 0.0, 0.0),
                obj_type="document_identity",
                text="[1.型式试验证书] 门锁(层门) ZIL-33A (No: TSXF34002220170118)",
                meta={"title": "门锁(层门) ZIL-33A", "folder": "1.型式试验证书", "certificate_number": "TSXF34002220170118"},
            )
        )

        pack = EvidencePack(query="LEHY-L-S的型式试验证书有哪些？", evidences=[evidence_a, evidence_b])
        pack.metadata["query_type"] = "document_aggregation"
        system._synthesize_answer(pack)

        self.assertIn("共找到2份相关文档", pack.answer)
        self.assertIn("GSB-282N", pack.answer)
        self.assertIn("ZIL-33A", pack.answer)
        self.assertNotIn("证书编号", pack.answer)
        self.assertNotIn("目录：", pack.answer)
        self.assertEqual(len(pack.certified_claims), 2)


if __name__ == "__main__":
    unittest.main()
