import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("MedAssistRAG.SummaryMemory")


def build_summary_prompt(history_text: str) -> str:
    """Prompt for conversation summarization without fact invention."""
    return f"""You are a conversation summarization component.

Summarize the following medical conversation history concisely.

Rules:
1. Summarize ONLY facts explicitly stated in the conversation history.
2. Do NOT invent new medical symptoms, drugs, or diagnoses.
3. Keep the summary under 100 words.

Conversation history:
{history_text}

Summary:"""


class SummaryMemory:
    """
    Manages conversation summary memory per conversation_id.
    Triggers summary generation when message history reaches trigger_messages limit.
    """

    def __init__(self, llm: Optional[Any] = None, trigger_messages: int = 10, enabled: bool = True):
        self.enabled = enabled
        self.trigger_messages = trigger_messages
        self._llm = llm
        self._summaries: Dict[str, str] = {}

    @property
    def llm(self) -> Any:
        """Lazily instantiates local LLM if not injected."""
        if self._llm is None:
            from src.generator.llm import LLM
            self._llm = LLM()
        return self._llm

    def update_summary(self, conversation_id: str, recent_messages: List[Dict[str, str]]) -> str:
        """
        Generates/updates conversation summary if message count >= trigger_messages.
        """
        if not self.enabled or not recent_messages or len(recent_messages) < self.trigger_messages:
            return self._summaries.get(conversation_id, "")

        history_lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent_messages]
        history_text = "\n".join(history_lines)

        prompt = build_summary_prompt(history_text)

        try:
            summary_raw = self.llm.generate_raw(prompt)
            summary_clean = summary_raw.strip()
            if summary_clean:
                self._summaries[conversation_id] = summary_clean
                logger.debug(f"[SummaryMemory] Updated summary for {conversation_id}: '{summary_clean}'")
        except Exception as e:
            logger.warning(f"[SummaryMemory] Failed summary generation: {e}")

        return self._summaries.get(conversation_id, "")

    def get_summary(self, conversation_id: str) -> str:
        """Returns recorded summary for given conversation_id."""
        return self._summaries.get(conversation_id, "")

    def set_summary(self, conversation_id: str, summary_text: str) -> None:
        """Directly sets summary string (used in testing/mocking)."""
        self._summaries[conversation_id] = summary_text

    def clear(self, conversation_id: str) -> None:
        """Clears summary memory for given conversation_id."""
        if conversation_id in self._summaries:
            del self._summaries[conversation_id]
