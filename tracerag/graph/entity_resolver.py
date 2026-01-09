"""
Entity resolver using LLM for semantic ID extraction.

Implements the "Entity Resolution" step from the design specification:
"An iterative process using an LLM examines text labels and visual context
to assign Semantic IDs (e.g., 'P-101', 'V-101')."

This bridges raw VectorObjects to abstract Entity nodes in the STLG.
"""

import re
import json
from typing import List, Dict, Any, Optional, Set
from loguru import logger

from tracerag.common.types import VectorObject


class EntityResolver:
    """
    LLM-based entity extraction from document regions.

    Identifies semantic entities (components, parts, references) from
    text blocks and assigns unique IDs for graph tracking.
    """

    def __init__(self, llm_client=None, config: Dict[str, Any] = None):
        """
        Initialize entity resolver.

        Args:
            llm_client: Optional LLM client (OpenAI, Anthropic, etc.)
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.config = config or {}
        self.temperature = self.config.get("temperature", 0.1)
        self.use_llm = self.config.get("use_llm", True) and llm_client is not None

        # Fallback regex patterns for common engineering identifiers
        self.identifier_patterns = [
            r'\b[A-Z]{1,3}-\d{1,5}(?:-[A-Z0-9]{1,5})?\b',  # CFG-AX14-K2, V-101
            r'\b[A-Z]\d{1,4}\b',                            # V101, S1
            r'\b[A-Z]{2,4}\d{2,5}\b',                       # ABC123, VALVE101
        ]
        self.compiled_patterns = [re.compile(p) for p in self.identifier_patterns]

    def extract_entities_from_objects(
        self,
        objects: List[VectorObject],
        doc_id: str,
        version_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract entities from a list of VectorObjects.

        Args:
            objects: List of VectorObjects (text blocks, symbols, etc.)
            doc_id: Document ID
            version_id: Version ID

        Returns:
            Dictionary mapping entity_id -> entity metadata
        """
        logger.info(f"Extracting entities from {len(objects)} objects (doc={doc_id}, version={version_id})")

        if self.use_llm:
            entities = self._extract_with_llm(objects, doc_id, version_id)
        else:
            entities = self._extract_with_patterns(objects, doc_id, version_id)

        logger.info(f"Extracted {len(entities)} unique entities")
        return entities

    def _extract_with_llm(
        self,
        objects: List[VectorObject],
        doc_id: str,
        version_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract entities using LLM.

        Args:
            objects: VectorObjects
            doc_id: Document ID
            version_id: Version ID

        Returns:
            Entity dictionary
        """
        # Collect text snippets from objects
        text_snippets = []
        object_map = {}

        for obj in objects[:100]:  # Limit to 100 objects per LLM call
            if obj.text and len(obj.text.strip()) > 0:
                text_snippets.append({
                    "object_id": obj.object_id,
                    "text": obj.text,
                    "type": obj.obj_type
                })
                object_map[obj.object_id] = obj

        if not text_snippets:
            return {}

        # Build LLM prompt
        prompt = self._build_entity_extraction_prompt(text_snippets, doc_id)

        # Call LLM
        try:
            response = self._call_llm(prompt)
            entities_data = json.loads(response)
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}, falling back to patterns")
            return self._extract_with_patterns(objects, doc_id, version_id)

        # Parse LLM response
        entities = {}
        for entity_item in entities_data.get("entities", []):
            entity_id = entity_item.get("entity_id")
            if not entity_id:
                continue

            # Find linked objects
            linked_object_ids = entity_item.get("object_ids", [])
            linked_objects = [object_map[oid] for oid in linked_object_ids if oid in object_map]

            entities[entity_id] = {
                "entity_id": entity_id,
                "label": entity_item.get("label", entity_id),
                "entity_type": entity_item.get("type", "unknown"),
                "linked_objects": linked_objects,
                "doc_id": doc_id,
                "version_id": version_id,
            }

        return entities

    def _extract_with_patterns(
        self,
        objects: List[VectorObject],
        doc_id: str,
        version_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract entities using regex patterns (fallback).

        Args:
            objects: VectorObjects
            doc_id: Document ID
            version_id: Version ID

        Returns:
            Entity dictionary
        """
        entities = {}

        for obj in objects:
            if not obj.text:
                continue

            # Find all matching identifiers in text
            for pattern in self.compiled_patterns:
                matches = pattern.findall(obj.text)
                for match in matches:
                    entity_id = match.strip()

                    if entity_id not in entities:
                        entities[entity_id] = {
                            "entity_id": entity_id,
                            "label": entity_id,
                            "entity_type": "component",
                            "linked_objects": [],
                            "doc_id": doc_id,
                            "version_id": version_id,
                        }

                    # Link this object to the entity
                    entities[entity_id]["linked_objects"].append(obj)

        return entities

    def _build_entity_extraction_prompt(
        self,
        text_snippets: List[Dict[str, str]],
        doc_id: str
    ) -> str:
        """
        Build LLM prompt for entity extraction.

        Args:
            text_snippets: List of text snippets with object IDs
            doc_id: Document ID

        Returns:
            Prompt string
        """
        snippets_str = "\n".join([
            f"[{s['object_id']}] ({s['type']}): {s['text'][:200]}"
            for s in text_snippets
        ])

        prompt = f"""You are analyzing a technical engineering document. Extract semantic entities (components, valves, relays, parts, etc.) from the following text snippets.

Document ID: {doc_id}

Text Snippets:
{snippets_str}

Instructions:
1. Identify all component/part identifiers (e.g., "V-101", "CFG-AX14-K2", "Relay S1")
2. For each entity, provide:
   - entity_id: Unique identifier (e.g., "V-101")
   - label: Human-readable label
   - type: Component type (valve, relay, bolt, sensor, etc.)
   - object_ids: List of snippet IDs that mention this entity

Return JSON with this structure:
{{
  "entities": [
    {{
      "entity_id": "V-101",
      "label": "Valve V-101",
      "type": "valve",
      "object_ids": ["obj_123", "obj_456"]
    }}
  ]
}}

Only include entities that are clearly engineering components. Ignore generic words.

JSON Response:"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API.

        Args:
            prompt: Prompt string

        Returns:
            LLM response
        """
        if self.llm_client is None:
            raise ValueError("LLM client not available")

        # Detect client type and call appropriately
        if hasattr(self.llm_client, 'chat'):
            # OpenAI-style
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",  # Fast model for entity extraction
                messages=[
                    {"role": "system", "content": "You are a technical documentation entity extractor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        elif hasattr(self.llm_client, 'messages'):
            # Anthropic-style
            response = self.llm_client.messages.create(
                model="claude-3-haiku-20240307",  # Fast model
                max_tokens=2000,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        else:
            raise ValueError("Unknown LLM client type")

    def match_entities_across_versions(
        self,
        entities_v1: Dict[str, Dict[str, Any]],
        entities_v2: Dict[str, Dict[str, Any]]
    ) -> List[tuple[str, str]]:
        """
        Match entities across two versions (for STLG alignment).

        Args:
            entities_v1: Entities from version 1
            entities_v2: Entities from version 2

        Returns:
            List of (entity_id_v1, entity_id_v2) pairs
        """
        matches = []

        for e1_id, e1_data in entities_v1.items():
            # Exact match
            if e1_id in entities_v2:
                matches.append((e1_id, e1_id))
                continue

            # Fuzzy match on label
            best_match = None
            best_score = 0.0

            for e2_id, e2_data in entities_v2.items():
                # Simple similarity: longest common substring / total length
                label1 = e1_data["label"].lower()
                label2 = e2_data["label"].lower()

                # Calculate similarity
                common = self._longest_common_substring(label1, label2)
                similarity = 2 * len(common) / (len(label1) + len(label2))

                if similarity > best_score and similarity > 0.7:
                    best_score = similarity
                    best_match = e2_id

            if best_match:
                matches.append((e1_id, best_match))

        return matches

    def _longest_common_substring(self, s1: str, s2: str) -> str:
        """Find longest common substring."""
        m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
        longest, x_longest = 0, 0

        for x in range(1, 1 + len(s1)):
            for y in range(1, 1 + len(s2)):
                if s1[x - 1] == s2[y - 1]:
                    m[x][y] = m[x - 1][y - 1] + 1
                    if m[x][y] > longest:
                        longest = m[x][y]
                        x_longest = x
                else:
                    m[x][y] = 0

        return s1[x_longest - longest: x_longest]
