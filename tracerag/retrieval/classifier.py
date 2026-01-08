"""
Query type classifier.

Classifies queries into types:
- locator: "Where is component X?"
- attribute: "What is the torque for bolt Y?"
- procedural: "How to assemble Z?"
- revision: "Show me revision C"
- diff: "What changed between Rev A and Rev B?"
"""

import re
from typing import Optional


class QueryClassifier:
    """
    Classify query into types for specialized retrieval paths.
    """

    def __init__(self, method: str = "heuristic"):
        """
        Initialize classifier.

        Args:
            method: Classification method ('heuristic' or 'trained')
        """
        self.method = method

        # Heuristic patterns
        self.patterns = {
            "revision": [
                r"\brev(?:ision)?\s+[A-Z0-9]",
                r"\bversion\s+[A-Z0-9]",
                r"\b[Vv]\d+",
            ],
            "diff": [
                r"\bdiff(?:erence)?",
                r"\bchange[ds]?",
                r"\bbetween\s+.+\s+and\s+",
                r"\bcompare",
                r"\bwhat.*changed",
            ],
            "procedural": [
                r"\bhow\s+to\b",
                r"\bsteps?\b",
                r"\bprocedure",
                r"\bassembl(?:e|y)",
                r"\binstall",
                r"\boperat(?:e|ion)",
            ],
            "attribute": [
                r"\bwhat\s+is\s+the\s+\w+",
                r"\btorque",
                r"\bvalue",
                r"\bspec(?:ification)?",
                r"\bcolor",
                r"\bsize",
                r"\brating",
                r"\bpressure",
                r"\btemperature",
            ],
            "locator": [
                r"\bwhere\s+is",
                r"\bfind",
                r"\blocate",
                r"\bshow\s+me",
                r"\bidentify",
            ],
        }

        # Compile patterns
        self.compiled_patterns = {
            qtype: [re.compile(p, re.IGNORECASE) for p in patterns]
            for qtype, patterns in self.patterns.items()
        }

    def classify(self, query: str) -> str:
        """
        Classify query.

        Args:
            query: Query string

        Returns:
            Query type: 'locator', 'attribute', 'procedural', 'revision', or 'diff'
        """
        if self.method == "heuristic":
            return self._classify_heuristic(query)
        else:
            # Placeholder for trained classifier
            return self._classify_heuristic(query)

    def _classify_heuristic(self, query: str) -> str:
        """
        Classify using heuristic pattern matching.

        Args:
            query: Query string

        Returns:
            Query type
        """
        # Check patterns in priority order
        priority = ["diff", "revision", "procedural", "attribute", "locator"]

        for qtype in priority:
            patterns = self.compiled_patterns[qtype]
            for pattern in patterns:
                if pattern.search(query):
                    return qtype

        # Default fallback
        return "locator"

    def extract_identifiers(self, query: str) -> list[str]:
        """
        Extract alphanumeric identifiers from query (part numbers, config codes).

        Args:
            query: Query string

        Returns:
            List of extracted identifiers
        """
        # Pattern: alphanumeric codes with hyphens/underscores
        # E.g., CFG-AX14-K2, V-101, S1-B3
        pattern = re.compile(r'\b[A-Z][A-Z0-9\-_]{2,20}\b')
        matches = pattern.findall(query)
        return matches

    def extract_revision_ids(self, query: str) -> list[str]:
        """
        Extract revision identifiers from query.

        Args:
            query: Query string

        Returns:
            List of revision IDs
        """
        identifiers = []

        # Pattern: "Rev A", "Revision C", "V2", etc.
        patterns = [
            r'\bRev\.?\s+([A-Z0-9]+)',
            r'\bRevision\s+([A-Z0-9]+)',
            r'\b[Vv](\d+)',
        ]

        for pattern_str in patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            matches = pattern.findall(query)
            identifiers.extend(matches)

        return identifiers
