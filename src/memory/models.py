import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_id": self.message_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            message_id=data.get("message_id", str(uuid.uuid4()))
        )


@dataclass
class EntityStore:
    conditions: List[str] = field(default_factory=list)
    symptoms: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)
    procedures: List[str] = field(default_factory=list)
    body_parts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "conditions": sorted(list(set(self.conditions))),
            "symptoms": sorted(list(set(self.symptoms))),
            "medications": sorted(list(set(self.medications))),
            "tests": sorted(list(set(self.tests))),
            "procedures": sorted(list(set(self.procedures))),
            "body_parts": sorted(list(set(self.body_parts))),
        }

    def merge(self, new_entities: Dict[str, List[str]]) -> None:
        """Merges and deduplicates new extracted entities into existing store."""
        for key in ["conditions", "symptoms", "medications", "tests", "procedures", "body_parts"]:
            items = new_entities.get(key, [])
            current_list = getattr(self, key)
            for item in items:
                if item and item.strip() and item.strip().lower() not in [c.lower() for c in current_list]:
                    current_list.append(item.strip())


@dataclass
class MemoryContext:
    recent_messages: List[Dict[str, str]] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recent_messages": self.recent_messages,
            "entities": self.entities,
            "summary": self.summary
        }
