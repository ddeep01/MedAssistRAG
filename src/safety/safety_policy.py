from enum import Enum
from typing import Dict, Any, Optional
from src.utils.config import load_safety_config


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SafetyAction(str, Enum):
    RAG = "RAG"
    CREATE_TICKET = "CREATE_TICKET"
    SAFETY_WARNING = "SAFETY_WARNING"


class SafetyPolicy:
    """
    Evaluates risk level classification and deterministically maps to system actions.
    Provides standard controlled responses for MEDIUM and HIGH risk cases.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_safety_config(config_path)
        s_cfg = self.config.get("safety", {})
        self.enabled = s_cfg.get("enabled", True)
        self.fallback_risk_level = s_cfg.get("fallback_risk_level", "HIGH")

    def get_action(self, risk_level: str) -> SafetyAction:
        """Deterministically maps risk level to system action."""
        rl = risk_level.upper() if isinstance(risk_level, str) else "HIGH"

        if rl == RiskLevel.LOW:
            return SafetyAction.RAG
        elif rl == RiskLevel.MEDIUM:
            return SafetyAction.CREATE_TICKET
        else:
            # Default to SAFETY_WARNING for HIGH or invalid levels
            return SafetyAction.SAFETY_WARNING

    @staticmethod
    def get_medium_response(ticket: Optional[Dict[str, Any]] = None) -> str:
        """Returns standard controlled response for MEDIUM risk queries."""
        if ticket and "ticket_id" in ticket:
            ticket_id = ticket["ticket_id"]
            return (
                f"This query requires personalized medical assessment. MedAssistRAG cannot "
                f"provide a reliable answer for this situation. A support request (ID: {ticket_id}) "
                f"has been created so that you can be connected with a qualified healthcare professional."
            )
        return (
            "This query requires personalized medical assessment. MedAssistRAG cannot provide a "
            "reliable answer for this situation. Please consult a qualified healthcare professional."
        )

    @staticmethod
    def get_high_response() -> str:
        """Returns standard controlled safety response for HIGH risk queries."""
        return (
            "[HIGH-RISK MEDICAL WARNING]\n\n"
            "This query may involve a situation that requires urgent professional medical attention "
            "or critical medication management. MedAssistRAG cannot safely assess or manage this "
            "situation through automated chat.\n\n"
            "Please seek immediate medical attention or contact your local emergency medical service "
            "if you may be experiencing an emergency, or consult your prescribing physician before "
            "making any changes to prescribed medications or dosages."
        )
