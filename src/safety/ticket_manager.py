import os
import json
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger("MedAssistRAG.TicketManager")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TICKETS_FILE = os.path.join(BASE_DIR, "data", "tickets", "tickets.json")


class TicketManager:
    """
    Manages creation and persistence of support tickets for queries requiring
    personalized medical support (MEDIUM risk).
    """

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path or DEFAULT_TICKETS_FILE
        self._ensure_storage_directory()

    def _ensure_storage_directory(self) -> None:
        """Creates parent directory for tickets JSON file if it does not exist."""
        try:
            dir_path = os.path.dirname(self.file_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            if not os.path.exists(self.file_path):
                with open(self.file_path, "w", encoding="utf-8") as f:
                    json.dump([], f)
        except Exception as e:
            logger.warning(f"[TicketManager] Could not initialize ticket storage file {self.file_path}: {e}")

    def create_ticket(
        self,
        conversation_id: str,
        query: str,
        risk_level: str = "MEDIUM"
    ) -> Dict[str, Any]:
        """
        Creates and stores a support ticket for a query.
        Returns structured ticket dictionary.
        """
        ticket_id = f"TICKET-{str(uuid.uuid4())[:8].upper()}"
        ticket = {
            "ticket_id": ticket_id,
            "conversation_id": conversation_id,
            "query": query,
            "risk_level": risk_level,
            "status": "OPEN",
            "created_at": datetime.utcnow().isoformat()
        }

        try:
            tickets = self.list_tickets()
            tickets.append(ticket)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(tickets, f, indent=2)
            logger.info(f"[TicketManager] Created support ticket: {ticket_id}")
        except Exception as e:
            logger.error(f"[TicketManager] Failed to persist ticket {ticket_id}: {e}")

        return ticket

    def list_tickets(self) -> List[Dict[str, Any]]:
        """Reads and returns all persisted support tickets."""
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[TicketManager] Could not read tickets from {self.file_path}: {e}")
            return []
