import uuid
import logging
from typing import Dict, List, Any, Optional

from src.memory.models import MemoryContext
from src.memory.short_term_memory import ShortTermMemory
from src.memory.entity_memory import EntityMemory
from src.utils.config import load_memory_config

logger = logging.getLogger("MedAssistRAG.MemoryManager")


class MemoryManager:
    """
    Central Memory Manager for MedAssistRAG.
    Coordinates short-term message history and medical entity tracking
    per conversation_id while keeping conversation states strictly isolated.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        config_path: Optional[str] = None,
        short_term_memory: Optional[ShortTermMemory] = None,
        entity_memory: Optional[EntityMemory] = None,
    ):
        self.config = load_memory_config(config_path)
        m_cfg = self.config.get("memory", {})

        self.enabled = m_cfg.get("enabled", True)

        st_max = int(m_cfg.get("short_term", {}).get("max_messages", 10))
        ent_enabled = m_cfg.get("entities", {}).get("enabled", True) and self.enabled

        self.short_term = short_term_memory or ShortTermMemory(max_messages=st_max)
        self.entity_memory = entity_memory or EntityMemory(llm=llm, enabled=ent_enabled)

        self._active_conversations: set = set()

    def create_conversation(self, conversation_id: Optional[str] = None) -> str:
        """Creates or registers a conversation_id."""
        cid = conversation_id or str(uuid.uuid4())
        self._active_conversations.add(cid)
        return cid

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """
        Appends a message to the conversation history and updates entity memory for user turns.
        """
        if not self.enabled or not conversation_id or not content:
            return

        self._active_conversations.add(conversation_id)

        # 1. Add message to short-term memory
        self.short_term.add_message(conversation_id, role, content)

        # 2. Extract and merge entities if user message
        if role == "user":
            self.entity_memory.update_entities(conversation_id, content)

    def get_context(self, conversation_id: Optional[str]) -> MemoryContext:
        """
        Returns structured MemoryContext (recent messages, entities) for given conversation_id.
        Returns empty MemoryContext if conversation_id is None or memory is disabled.
        """
        if not self.enabled or not conversation_id:
            return MemoryContext(recent_messages=[], entities={})

        recent_msgs = self.short_term.get_recent_messages(conversation_id)
        entities = self.entity_memory.get_entities(conversation_id)

        return MemoryContext(
            recent_messages=recent_msgs,
            entities=entities
        )

    def update_entities(self, conversation_id: str, text: str) -> Dict[str, List[str]]:
        """Directly extracts and updates entity memory for given text."""
        if not self.enabled or not conversation_id:
            return {}
        return self.entity_memory.update_entities(conversation_id, text)

    def clear_conversation(self, conversation_id: str) -> None:
        """
        Clears all short-term history and entity memory for given conversation_id.
        """
        if not conversation_id:
            return

        self.short_term.clear(conversation_id)
        self.entity_memory.clear(conversation_id)
        if conversation_id in self._active_conversations:
            self._active_conversations.remove(conversation_id)

        logger.debug(f"[MemoryManager] Cleared conversation {conversation_id}")

