"""
Engineering Benchmark Loader.

Robust loader for engineering-specific benchmark datasets with flexible JSON parsing.
"""

import json
from typing import List, Dict, Any
from loguru import logger
from tracerag.common.types import BenchmarkExample


class EngBenchLoader:
    """
    Loader for engineering document retrieval benchmark.

    Supports multiple JSON formats:
    - List format: [{"question": "...", "answer": "..."}, ...]
    - Dict format: {"questions": [{"question": "...", "answer": "..."}, ...]}
    - Flexible field names: question/query, answer/ground_truth
    """

    def __init__(self, data_path: str):
        """
        Initialize loader.

        Args:
            data_path: Path to benchmark JSON file
        """
        self.data_path = data_path
        self.queries: List[BenchmarkExample] = []

    def load(self) -> List[BenchmarkExample]:
        """
        Load benchmark data from JSON.

        Returns:
            List of BenchmarkQuery objects
        """
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                if self.data_path.endswith('.jsonl'):
                    items = [json.loads(line) for line in f if line.strip()]
                else:
                    data = json.load(f)
                    # Handle both list [...] and dict {"questions": [...]} formats
                    items = data.get('questions', []) if isinstance(data, dict) else data

            if not isinstance(items, list):
                logger.error(f"Invalid benchmark format in {self.data_path}")
                return []

            for idx, item in enumerate(items):
                # Flexible field mapping to handle variations in JSON keys
                q_text = item.get('question') or item.get('query') or item.get('query_text')
                q_id = str(item.get('id') or item.get('query_id') or idx)
                ground_truth = (
                    item.get('answer') or
                    item.get('ground_truth') or
                    item.get('ground_truth_answer') or
                    ""
                )
                q_type = item.get('type') or item.get('query_type') or "engineering_qa"
                pages = item.get('pages') or item.get('relevant_pages') or []
                objects = item.get('objects') or item.get('relevant_objects') or []
                metadata = item.get('metadata', {})

                if not q_text:
                    logger.warning(f"Skipping item {idx}: missing query text")
                    continue

                self.queries.append(BenchmarkExample(
                    query_id=q_id,
                    query_text=q_text,
                    ground_truth_answer=ground_truth,
                    query_type=q_type,
                    relevant_pages=pages if isinstance(pages, list) else [],
                    relevant_objects=objects if isinstance(objects, list) else [],
                    metadata=metadata
                ))

            logger.info(f"✓ Loaded {len(self.queries)} questions from Eng_Bench")
            return self.queries

        except FileNotFoundError:
            logger.error(f"Benchmark file not found: {self.data_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {self.data_path}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading Eng_Bench: {e}")
            return []

    def get_queries(self, query_type: str = None) -> List[BenchmarkExample]:
        """
        Get benchmark queries, optionally filtered by type.

        Args:
            query_type: Optional query type filter

        Returns:
            List of queries
        """
        if not self.queries:
            self.load()

        if query_type is None:
            return self.queries

        return [q for q in self.queries if q.query_type == query_type]

    def get_query_by_id(self, query_id: str) -> BenchmarkExample:
        """
        Get specific query by ID.

        Args:
            query_id: Query ID

        Returns:
            BenchmarkExample or None if not found
        """
        if not self.queries:
            self.load()

        for q in self.queries:
            if q.query_id == query_id:
                return q

        return None
