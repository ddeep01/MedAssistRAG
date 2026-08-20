from typing import List, Dict, Any, Optional
from src.memory.models import Message


class ShortTermMemory:
    """
    Manages bounded recent message history per conversation_id.
    Drops oldest messages when capacity exceeds max_messages.
    """

    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self._storage: Dict[str, List[Message]] = {}

    def add_message(self, conversation_id: str, role: str, content: str) -> Message:
        """Appends a new message to the conversation history and enforces max_messages capacity."""
        if conversation_id not in self._storage:
            self._storage[conversation_id] = []

        msg = Message(role=role, content=content)
        self._storage[conversation_id].append(msg)

        # Enforce max capacity by truncating oldest messages
        if len(self._storage[conversation_id]) > self.max_messages:
            self._storage[conversation_id] = self._storage[conversation_id][-self.max_messages:]

        return msg

    def get_recent_messages(self, conversation_id: str) -> List[Dict[str, str]]:
        """Returns the list of recent messages formatted as dictionaries."""
        messages = self._storage.get(conversation_id, [])
        return [m.to_dict() for m in messages]

    def get_message_count(self, conversation_id: str) -> int:
        """Returns total message count for the conversation."""
        return len(self._storage.get(conversation_id, []))

    def clear(self, conversation_id: str) -> None:
        """Clears short-term message history for the given conversation_id."""
        if conversation_id in self._storage:
            del self._storage[conversation_id]
