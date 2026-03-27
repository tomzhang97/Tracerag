import json
import tempfile
import unittest
from pathlib import Path

from tracerag.cli.prepare_docvqa import _load_ocr_objects, _parse_ocr_records
from tracerag.eval.docvqa_eval import anls_score, load_docvqa_questions


class DocVQASetupTests(unittest.TestCase):
    def test_anls_score_handles_exact_and_near_matches(self):
        self.assertEqual(anls_score("invoice 42", "invoice 42"), 1.0)
        self.assertGreater(anls_score("invoice 42", "invoice 43"), 0.5)
        self.assertEqual(anls_score("abc", "completely different"), 0.0)

    def test_load_docvqa_questions_accepts_flexible_schema(self):
        data = {
            "data": [
                {
                    "questionId": 7,
                    "question": "What is the invoice number?",
                    "answers": ["INV-42", "inv-42"],
                    "image": "documents/invoice_42.png",
                }
            ]
        }

        questions = load_docvqa_questions(data)

        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].question_id, "7")
        self.assertEqual(questions[0].doc_id, "invoice_42")
        self.assertEqual(questions[0].page_id, "invoice_42_v1_p0")

    def test_parse_ocr_records_supports_azure_style_layout(self):
        payload = {
            "recognitionResults": [
                {
                    "lines": [
                        {
                            "text": "Invoice Number INV-42",
                            "boundingBox": [0, 0, 100, 0, 100, 20, 0, 20],
                            "words": [
                                {
                                    "text": "Invoice",
                                    "boundingBox": [0, 0, 40, 0, 40, 20, 0, 20],
                                    "confidence": 0.99,
                                },
                                {
                                    "text": "INV-42",
                                    "boundingBox": [50, 0, 100, 0, 100, 20, 50, 20],
                                    "confidence": 0.98,
                                },
                            ],
                        }
                    ]
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            ocr_path = Path(tmp_dir) / "ocr.json"
            ocr_path.write_text(json.dumps(payload), encoding="utf-8")

            lines, words = _parse_ocr_records(ocr_path)

        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["text"], "Invoice Number INV-42")
        self.assertEqual(len(words), 2)
        self.assertEqual(words[1]["text"], "INV-42")

    def test_load_ocr_objects_builds_line_and_word_objects(self):
        payload = {
            "words": [
                {"text": "Invoice", "bbox": [0, 0, 50, 20], "confidence": 0.9},
                {"text": "42", "bbox": [60, 0, 90, 20], "confidence": 0.95},
            ],
            "lines": [
                {"text": "Invoice 42", "bbox": [0, 0, 90, 20]},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            ocr_path = Path(tmp_dir) / "ocr.json"
            ocr_path.write_text(json.dumps(payload), encoding="utf-8")
            objects = _load_ocr_objects("doc_a", "v1", "doc_a_v1_p0", ocr_path, (100, 50))

        texts = [obj.text for obj in objects]
        self.assertIn("Invoice 42", texts)
        self.assertIn("Invoice", texts)
        self.assertIn("42", texts)


if __name__ == "__main__":
    unittest.main()
