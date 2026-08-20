import re
import logging
from typing import List, Dict, Any, Optional

from src.utils.config import load_retry_config

logger = logging.getLogger("MedAssistRAG.QueryRewriter")


def build_rewrite_prompt(query: str, conversation_context: Optional[str] = None) -> str:
    """Constructs prompt for medical query rewriting."""
    ctx_str = conversation_context if conversation_context and conversation_context.strip() else "None"
    return f"""You are a medical search query rewriting component.

Rewrite the user's query into a clear, standalone query that can be used to search a medical knowledge base.

Rules:
1. Preserve the user's original intent.
2. Resolve ambiguous references (e.g. "it", "this", "that", "they", "high BP") using conversation context when available.
3. Include important medical entities when known.
4. Do not answer the question.
5. Do not add new medical information or invent facts.
6. Return ONLY the rewritten query text.

Conversation context:
{ctx_str}

Original query:
{query}

Rewritten query:"""


class QueryRewriter:
    """
    Medical query rewriter using local LLM to transform weak/ambiguous queries into
    clear, standalone knowledge base search queries.
    """

    def __init__(self, llm: Optional[Any] = None, config_path: Optional[str] = None):
        self.config = load_retry_config(config_path)
        self.enabled = self.config.get("query_rewriting", {}).get("enabled", True)
        self._llm = llm

    @property
    def llm(self) -> Any:
        """Lazily instantiates the LLM instance if not injected."""
        if self._llm is None:
            from src.generator.llm import LLM
            self._llm = LLM()
        return self._llm

    def validate_and_clean_rewrite(self, raw_output: str, original_query: str) -> str:
        """
        Validates LLM rewrite output to prevent empty output, answer leaks,
        excessive length, or prompt artifacts. Returns cleaned query or original on failure.
        """
        if not raw_output or not isinstance(raw_output, str):
            return original_query

        cleaned = raw_output.strip()

        # Remove common prefix artifacts
        prefixes_to_strip = [
            r"^rewritten query:\s*",
            r"^rewritten:\s*",
            r"^query:\s*",
            r"^standalone query:\s*",
            r"^output:\s*"
        ]
        for p in prefixes_to_strip:
            cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()

        # Remove surrounding quotes
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()

        # Validation rules:
        # 1. Non-empty & length bounds
        if len(cleaned) < 3 or len(cleaned) > 300:
            logger.warning(f"Invalid rewrite length ({len(cleaned)} chars). Fallback to original.")
            return original_query

        # 2. Must not look like an answer or conversational reply
        answer_indicators = [
            "the answer is", "this condition is caused by", "treatment includes",
            "according to medical literature", "here is the rewritten query"
        ]
        if any(ind in cleaned.lower() for ind in answer_indicators):
            logger.warning("Rewrite contained answer indicator. Fallback to original.")
            return original_query

        return cleaned

    def rewrite(
        self,
        query: str,
        conversation_context: Optional[str] = None,
        retrieval_results: Optional[List[Dict[str, Any]]] = None,
        confidence: Optional[float] = None
    ) -> str:
        """
        Transforms original query into a refined standalone medical query.
        Returns original query if query rewriting is disabled or validation fails.
        """
        if not query or not query.strip():
            return query

        if not self.enabled:
            return query

        prompt = build_rewrite_prompt(query, conversation_context)

        try:
            raw_response = self.llm.generate_raw(prompt)
            rewritten_query = self.validate_and_clean_rewrite(raw_response, query)
            logger.debug(f"[QueryRewriter] original='{query}' -> rewritten='{rewritten_query}'")
            return rewritten_query
        except Exception as e:
            logger.error(f"[QueryRewriter] Exception during rewrite: {e}. Fallback to original.")
            return query
