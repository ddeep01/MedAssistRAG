import json
import re
import logging
from typing import Dict, List, Any, Optional

from src.memory.models import EntityStore

logger = logging.getLogger("MedAssistRAG.EntityMemory")


def build_entity_extraction_prompt(user_text: str) -> str:
    """Prompt for strict medical entity extraction without inference."""
    return f"""You are a medical entity extraction component.

Extract ONLY medical entities that are EXPLICITLY mentioned in the user text.

Rules:
1. Do NOT infer unstated diagnoses or conditions.
2. Do NOT invent medications, symptoms, or facts.
3. If no entities are present in a category, return an empty array [].
4. Return ONLY valid JSON matching this exact structure:
{{
    "conditions": [],
    "symptoms": [],
    "medications": [],
    "tests": [],
    "procedures": [],
    "body_parts": []
}}

User text:
"{user_text}"

JSON output:"""


class EntityMemory:
    """
    Manages lightweight medical entity tracking per conversation_id.
    Merges and deduplicates extracted entities across turns.
    """

    def __init__(self, llm: Optional[Any] = None, enabled: bool = True):
        self.enabled = enabled
        self._llm = llm
        self._stores: Dict[str, EntityStore] = {}

    @property
    def llm(self) -> Any:
        """Lazily instantiates local LLM if not injected."""
        if self._llm is None:
            from src.generator.llm import LLM
            self._llm = LLM()
        return self._llm

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extracts medical entities from text using local LLM."""
        if not text or not text.strip() or not self.enabled:
            return {
                "conditions": [], "symptoms": [], "medications": [],
                "tests": [], "procedures": [], "body_parts": []
            }

        prompt = build_entity_extraction_prompt(text)

        try:
            raw_out = self.llm.generate_raw(prompt)
            # Find JSON block in raw output
            json_match = re.search(r"\{.*\}", raw_out, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict):
                    return {
                        "conditions": data.get("conditions", []) if isinstance(data.get("conditions"), list) else [],
                        "symptoms": data.get("symptoms", []) if isinstance(data.get("symptoms"), list) else [],
                        "medications": data.get("medications", []) if isinstance(data.get("medications"), list) else [],
                        "tests": data.get("tests", []) if isinstance(data.get("tests"), list) else [],
                        "procedures": data.get("procedures", []) if isinstance(data.get("procedures"), list) else [],
                        "body_parts": data.get("body_parts", []) if isinstance(data.get("body_parts"), list) else [],
                    }
        except Exception as e:
            logger.warning(f"[EntityMemory] Failed entity extraction: {e}")

        # Basic fallback regex extraction for common medical patterns if LLM fails
        return self._regex_fallback_extract(text)

    def _regex_fallback_extract(self, text: str) -> Dict[str, List[str]]:
        """Simple fallback regex entity extraction."""
        text_lower = text.lower()
        extracted = {
            "conditions": [], "symptoms": [], "medications": [],
            "tests": [], "procedures": [], "body_parts": []
        }

        condition_patterns = ["hypertension", "diabetes", "asthma", "covid-19", "copd"]
        med_patterns = ["metformin", "amlodipine", "lisinopril", "insulin", "aspirin", "ibuprofen"]
        symptom_patterns = ["headache", "fever", "cough", "nausea", "diarrhea", "chest pain"]

        for cond in condition_patterns:
            if cond in text_lower:
                extracted["conditions"].append(cond)
        for med in med_patterns:
            if med in text_lower:
                extracted["medications"].append(med)
        for sym in symptom_patterns:
            if sym in text_lower:
                extracted["symptoms"].append(sym)

        return extracted

    def update_entities(self, conversation_id: str, text: str) -> Dict[str, List[str]]:
        """Extracts entities from new text and merges into conversation's EntityStore."""
        if conversation_id not in self._stores:
            self._stores[conversation_id] = EntityStore()

        extracted = self.extract_entities(text)
        self._stores[conversation_id].merge(extracted)
        return self._stores[conversation_id].to_dict()

    def get_entities(self, conversation_id: str) -> Dict[str, List[str]]:
        """Returns recorded entities for given conversation_id."""
        if conversation_id in self._stores:
            return self._stores[conversation_id].to_dict()
        return {
            "conditions": [], "symptoms": [], "medications": [],
            "tests": [], "procedures": [], "body_parts": []
        }

    def clear(self, conversation_id: str) -> None:
        """Clears entity store for given conversation_id."""
        if conversation_id in self._stores:
            del self._stores[conversation_id]
