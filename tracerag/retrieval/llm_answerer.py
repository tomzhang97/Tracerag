"""
LLM wrapper for answer generation and claim extraction.

Provides unified interface for calling LLMs (OpenAI, Anthropic, or local models)
to generate answers and extract certified claims.
"""

import json
from typing import List, Dict, Any, Optional
from loguru import logger

from tracerag.common.types import RegionEvidence, Claim, CertifiedClaim


class LLMAnswerer:
    """
    LLM wrapper for generating answers with claim extraction.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM answerer.

        Args:
            config: Configuration dictionary (from llm section)
        """
        self.config = config
        self.model_name = config.get("model_name", "gpt-4-turbo")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2000)
        self.claim_extraction_enabled = config.get("claim_extraction", {}).get("enabled", True)
        self.max_claims = config.get("claim_extraction", {}).get("max_claims_per_answer", 5)

        self.client = self._init_client()

    def _init_client(self):
        """Initialize LLM client based on model name."""
        if "gpt" in self.model_name or "openai" in self.model_name:
            try:
                from openai import OpenAI
                return OpenAI()
            except ImportError:
                logger.warning("OpenAI client not available, using mock")
                return None
        elif "claude" in self.model_name:
            try:
                from anthropic import Anthropic
                return Anthropic()
            except ImportError:
                logger.warning("Anthropic client not available, using mock")
                return None
        else:
            # Local model or mock
            logger.warning("Using mock LLM client")
            return None

    def generate_answer_with_claims(
        self,
        query: str,
        evidences: List[RegionEvidence],
        evidence_texts: Dict[str, str],
        query_type: str
    ) -> Dict[str, Any]:
        """
        Generate answer and extract claims using LLM.

        Args:
            query: User query
            evidences: List of evidence regions
            evidence_texts: Mapping from evidence ID to text content
            query_type: Query type (locator, attribute, etc.)

        Returns:
            Dictionary with 'answer' and 'claims' (list of claim dicts)
        """
        # Build prompt
        prompt = self._build_prompt(query, evidences, evidence_texts, query_type)

        # Call LLM
        response = self._call_llm(prompt)

        # Parse response
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON response, returning raw answer")
            return {
                "answer": response,
                "claims": []
            }

    def _build_prompt(
        self,
        query: str,
        evidences: List[RegionEvidence],
        evidence_texts: Dict[str, str],
        query_type: str
    ) -> str:
        """
        Build LLM prompt with query and evidence snippets.

        Args:
            query: User query
            evidences: List of evidences
            evidence_texts: Evidence ID -> text mapping
            query_type: Query type

        Returns:
            Prompt string
        """
        # Create evidence snippets with IDs
        evidence_snippets = []
        for i, ev in enumerate(evidences[:20], 1):  # Limit to 20 evidences
            ev_id = f"E{i}"
            text = evidence_texts.get(ev.object_id, "")

            snippet = f"[{ev_id}] (Page: {ev.page_id}, Type: {ev.obj_type})\n{text}\n"
            evidence_snippets.append(snippet)

        evidences_str = "\n".join(evidence_snippets)

        prompt = f"""You are a technical documentation assistant. Answer the user's query based on the provided evidence from technical documents.

Query: {query}
Query Type: {query_type}

Evidence:
{evidences_str}

Instructions:
1. Provide a clear, concise answer to the query
2. Extract factual claims from your answer
3. For each claim, list the evidence IDs (E1, E2, etc.) that support it
4. Return your response as JSON with this exact structure:

{{
  "answer": "Your natural language answer here",
  "claims": [
    {{
      "text": "First factual claim",
      "value": "Structured value if applicable (e.g., '50 Nm'), otherwise null",
      "entity_id": "Entity this claim is about (e.g., 'V-101'), otherwise null",
      "claim_type": "attribute|location|procedure_step|general",
      "evidence_ids": ["E1", "E3"]
    }}
  ]
}}

Important:
- Only include claims that are directly supported by the evidence
- Each claim should reference at least one evidence ID
- Keep answer concise and technical
- Maximum {self.max_claims} claims

JSON Response:"""

        return prompt

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API.

        Args:
            prompt: Prompt string

        Returns:
            LLM response text
        """
        if self.client is None:
            # Mock response for development
            return self._mock_response()

        try:
            if "gpt" in self.model_name:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a technical documentation assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content

            elif "claude" in self.model_name:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text

            else:
                return self._mock_response()

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._mock_response()

    def _mock_response(self) -> str:
        """Generate mock response for testing."""
        return json.dumps({
            "answer": "Based on the provided evidence, the requested information is available in the technical documentation.",
            "claims": [
                {
                    "text": "Information found in technical documentation",
                    "value": None,
                    "entity_id": None,
                    "claim_type": "general",
                    "evidence_ids": ["E1"]
                }
            ]
        })

    def map_claims_to_evidences(
        self,
        claims_data: List[Dict[str, Any]],
        evidences: List[RegionEvidence]
    ) -> List[CertifiedClaim]:
        """
        Map LLM claim output to CertifiedClaim objects.

        Args:
            claims_data: List of claim dicts from LLM
            evidences: List of all evidences (with E1, E2, ... indexing)

        Returns:
            List of CertifiedClaim objects
        """
        certified_claims = []

        for claim_dict in claims_data:
            # Create Claim
            claim = Claim(
                text=claim_dict.get("text", ""),
                value=claim_dict.get("value"),
                entity_id=claim_dict.get("entity_id"),
                claim_type=claim_dict.get("claim_type", "general")
            )

            # Map evidence IDs to RegionEvidence objects
            evidence_ids = claim_dict.get("evidence_ids", [])
            claim_evidences = []

            for ev_id in evidence_ids:
                # Parse "E1" -> index 0, "E2" -> index 1, etc.
                if ev_id.startswith("E"):
                    try:
                        idx = int(ev_id[1:]) - 1
                        if 0 <= idx < len(evidences):
                            claim_evidences.append(evidences[idx])
                    except ValueError:
                        logger.warning(f"Invalid evidence ID: {ev_id}")

            # Calculate confidence as average of evidence scores
            if claim_evidences:
                confidence = sum(e.score for e in claim_evidences) / len(claim_evidences)
            else:
                confidence = 0.5  # Default

            # Create CertifiedClaim
            certified_claim = CertifiedClaim(
                claim=claim,
                evidences=claim_evidences,
                confidence=confidence,
                reasoning=f"Supported by {len(claim_evidences)} evidence regions"
            )

            certified_claims.append(certified_claim)

        return certified_claims
