import re
import logging
from typing import List, Dict, Any, Optional, Union

from src.utils.config import load_retry_config

logger = logging.getLogger("MedAssistRAG.QueryRewriter")


def format_memory_context(conversation_context: Any) -> str:
    """Formats string, dict, or MemoryContext into clean context string for LLM prompt."""
    if not conversation_context:
        return "None"

    if isinstance(conversation_context, str):
        return conversation_context.strip() if conversation_context.strip() else "None"

    # If MemoryContext object or dict
    if hasattr(conversation_context, "to_dict"):
        ctx_dict = conversation_context.to_dict()
    elif isinstance(conversation_context, dict):
        ctx_dict = conversation_context
    else:
        return str(conversation_context)

    lines = []

    entities = ctx_dict.get("entities", {})

    if entities:
        ent_strs = []
        for category, items in entities.items():
            if items:
                ent_strs.append(f"{category}: {', '.join(items)}")
        if ent_strs:
            lines.append("Tracked Medical Entities: " + "; ".join(ent_strs))

    recent = ctx_dict.get("recent_messages", [])
    if recent:
        lines.append("Recent Conversation:")
        for m in recent:
            role = m.get("role", "user").capitalize()
            content = m.get("content", "")
            lines.append(f"  {role}: {content}")

    return "\n".join(lines) if lines else "None"


def build_rewrite_prompt(query: str, conversation_context: Any = None) -> str:
    """Constructs prompt for medical query rewriting with memory context."""
    ctx_str = format_memory_context(conversation_context)
    return f"""You are a medical search query rewriting component.

Rewrite the user's query into a clear, standalone query that can be used to search a medical knowledge base.

Rules:
1. Preserve the user's original intent.
2. Resolve ambiguous references (e.g. "it", "this", "that", "they", "high BP", "the medication") using conversation context when available.
3. Include important medical entities when known.
4. If multiple entities are mentioned and the question is ambiguous, do not force an incorrect single entity.
5. Do not answer the question.
6. Do not add new medical information or invent facts.
7. Return ONLY the rewritten query text.

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
        conversation_context: Any = None,
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
