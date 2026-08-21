import json
import re
import logging
from typing import Dict, Any, Optional

from src.safety.safety_policy import SafetyPolicy, RiskLevel, SafetyAction
from src.config import load_safety_config

logger = logging.getLogger("MedAssistRAG.RiskClassifier")


def build_risk_classifier_prompt(query: str, conversation_context: Any = None) -> str:
    """Builds prompt for medical risk classification."""
    ctx_str = "None"
    if conversation_context:
        if isinstance(conversation_context, str):
            ctx_str = conversation_context.strip()
        elif hasattr(conversation_context, "to_dict"):
            ctx_dict = conversation_context.to_dict()
            recent = ctx_dict.get("recent_messages", [])
            if recent:
                ctx_str = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent])
        elif isinstance(conversation_context, dict):
            recent = conversation_context.get("recent_messages", [])
            if recent:
                ctx_str = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent])

    return f"""You are the Medical Risk Classification component of MedAssistRAG.

Your ONLY task is to classify the user's query into exactly one of:
LOW
MEDIUM
HIGH

Do NOT answer the medical question.

LOW:
Use LOW for general educational or informational medical questions that can normally be handled by a knowledge-base RAG system.
Examples:
- What is hypertension?
- What is diabetes?
- What is metformin?
- What are common symptoms of diabetes?
- How does insulin work?

MEDIUM:
Use MEDIUM when the user is asking about their personal medical situation or seeking personalized medical guidance that should be handled by a qualified healthcare professional rather than answered directly by the automated system.
Examples:
- I have been having headaches for three weeks. What should I do?
- I have high blood pressure. What should I do?
- What treatment should I personally take for my condition?

HIGH:
Use HIGH when the query involves potentially dangerous medical decisions, urgent/emergency symptoms, medication changes, dosage changes, or other situations where incorrect automated guidance could cause significant harm.
Examples:
- Should I stop taking my medication?
- Can I double my medication dose?
- Can I replace my prescribed medicine?
- Should I change my insulin dose?
- I have severe chest pain and difficulty breathing.
- I think I am having a heart attack. What should I do?

IMPORTANT RULES:
1. Classify the risk of the REQUEST, not the seriousness of the disease mentioned.
2. A general educational question about a serious disease can still be LOW.
3. Do not diagnose the user.
4. Do not answer the user's medical question.
5. Medication changes and dosage changes should generally be HIGH.
6. Potential emergency situations should be HIGH.
7. Personal medical guidance should generally be MEDIUM.
8. General educational questions should generally be LOW.
9. Return ONLY JSON.
10. The risk_level must be exactly LOW, MEDIUM, or HIGH.

Context:
{ctx_str}

USER QUERY:
{query}

OUTPUT:
{{"risk_level": "LOW | MEDIUM | HIGH"}}"""


class RiskClassifier:
    """
    LLM-based Risk Classifier that evaluates medical risk level of user queries.
    Deterministically maps to system actions with conservative fallback on failure.
    """

    def __init__(self, llm: Optional[Any] = None, config_path: Optional[str] = None):
        self.config = load_safety_config(config_path)
        s_cfg = self.config.get("safety", {})
        self.enabled = s_cfg.get("enabled", True)
        self.fallback_risk_level = s_cfg.get("fallback_risk_level", "HIGH")
        self.policy = SafetyPolicy(config_path=config_path)
        self._llm = llm

    @property
    def llm(self) -> Any:
        """Lazily instantiates local LLM if not injected."""
        if self._llm is None:
            from src.generator.llm import LLM
            self._llm = LLM()
        return self._llm

    def classify(self, query: str, conversation_context: Any = None) -> Dict[str, Any]:
        """
        Classifies query into LOW, MEDIUM, or HIGH.
        Returns structured dictionary containing risk_level, action, and query details.
        """
        if not query or not query.strip():
            return {
                "risk_level": RiskLevel.LOW,
                "action": SafetyAction.RAG,
                "query": query,
                "is_fallback": False
            }

        if not self.enabled:
            return {
                "risk_level": RiskLevel.LOW,
                "action": SafetyAction.RAG,
                "query": query,
                "is_fallback": False
            }

        prompt = build_risk_classifier_prompt(query, conversation_context)

        try:
            raw_output = self.llm.generate_raw(prompt)
            risk_level = self._validate_and_parse_output(raw_output)

            if risk_level is not None:
                action = self.policy.get_action(risk_level)
                logger.info(f"[RiskClassifier] Classified query '{query[:40]}...' -> {risk_level} (Action: {action})")
                return {
                    "risk_level": risk_level,
                    "action": action,
                    "query": query,
                    "is_fallback": False
                }
        except Exception as e:
            logger.error(f"[RiskClassifier] Exception during risk classification: {e}. Applying fallback.")

        # Fallback conservatively if classification fails or LLM output is invalid
        fallback_action = self.policy.get_action(self.fallback_risk_level)
        logger.warning(f"[RiskClassifier] Fallback triggered -> {self.fallback_risk_level} (Action: {fallback_action})")
        return {
            "risk_level": self.fallback_risk_level,
            "action": fallback_action,
            "query": query,
            "is_fallback": True
        }

    def _validate_and_parse_output(self, raw_output: str) -> Optional[str]:
        """Validates LLM output and extracts valid risk level ('LOW', 'MEDIUM', 'HIGH')."""
        if not raw_output or not isinstance(raw_output, str):
            return None

        # Search for JSON block
        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict) and "risk_level" in data:
                    val = str(data["risk_level"]).strip().upper()
                    if val in ["LOW", "MEDIUM", "HIGH"]:
                        return val
            except Exception:
                pass

        # Strict regex fallback if raw string explicitly specifies level
        raw_upper = raw_output.strip().upper()
        if raw_upper in ["LOW", "MEDIUM", "HIGH"]:
            return raw_upper

        return None
