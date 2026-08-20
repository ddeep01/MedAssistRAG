import pytest
from unittest.mock import MagicMock

from src.safety.risk_classifier import RiskClassifier
from src.safety.safety_policy import SafetyPolicy, RiskLevel, SafetyAction
from src.safety.ticket_manager import TicketManager
from src.pipeline.rag_pipeline import RAGPipeline


# ----------------------------------------------------
# TEST 1: LOW RISK EDUCATIONAL QUESTION
# ----------------------------------------------------
def test_low_risk_educational_query():
    mock_llm = MagicMock()
    mock_llm.generate_raw.return_value = '{"risk_level": "LOW"}'

    classifier = RiskClassifier(llm=mock_llm)
    res = classifier.classify("What is hypertension?")

    assert res["risk_level"] == RiskLevel.LOW
    assert res["action"] == SafetyAction.RAG
    assert not res["is_fallback"]


# ----------------------------------------------------
# TEST 2: MEDIUM RISK PERSONAL GUIDANCE
# ----------------------------------------------------
def test_medium_risk_personal_guidance():
    mock_llm = MagicMock()
    mock_llm.generate_raw.return_value = '{"risk_level": "MEDIUM"}'

    classifier = RiskClassifier(llm=mock_llm)
    res = classifier.classify("I have been having headaches for three weeks. What should I do?")

    assert res["risk_level"] == RiskLevel.MEDIUM
    assert res["action"] == SafetyAction.CREATE_TICKET


# ----------------------------------------------------
# TEST 3: HIGH RISK MEDICATION CHANGE
# ----------------------------------------------------
def test_high_risk_medication_change():
    mock_llm = MagicMock()
    mock_llm.generate_raw.return_value = '{"risk_level": "HIGH"}'

    classifier = RiskClassifier(llm=mock_llm)
    res = classifier.classify("Should I double my medication dose?")

    assert res["risk_level"] == RiskLevel.HIGH
    assert res["action"] == SafetyAction.SAFETY_WARNING


# ----------------------------------------------------
# TEST 4: HIGH RISK EMERGENCY SYMPTOMS
# ----------------------------------------------------
def test_high_risk_emergency_symptoms():
    mock_llm = MagicMock()
    mock_llm.generate_raw.return_value = '{"risk_level": "HIGH"}'

    classifier = RiskClassifier(llm=mock_llm)
    res = classifier.classify("I have severe chest pain and difficulty breathing.")

    assert res["risk_level"] == RiskLevel.HIGH
    assert res["action"] == SafetyAction.SAFETY_WARNING


# ----------------------------------------------------
# TEST 5: INVALID LLM OUTPUT (SAFE FALLBACK TO HIGH)
# ----------------------------------------------------
def test_invalid_llm_output_fallback():
    mock_llm = MagicMock()
    mock_llm.generate_raw.return_value = "maybe HIGH or MEDIUM"

    classifier = RiskClassifier(llm=mock_llm)
    res = classifier.classify("What is hypertension?")

    assert res["risk_level"] == "HIGH"
    assert res["action"] == SafetyAction.SAFETY_WARNING
    assert res["is_fallback"] is True


# ----------------------------------------------------
# TEST 6: INVALID JSON OUTPUT (SAFE FALLBACK TO HIGH)
# ----------------------------------------------------
def test_invalid_json_fallback():
    mock_llm = MagicMock()
    mock_llm.generate_raw.return_value = "Not a json output"

    classifier = RiskClassifier(llm=mock_llm)
    res = classifier.classify("What is metformin?")

    assert res["risk_level"] == "HIGH"
    assert res["action"] == SafetyAction.SAFETY_WARNING
    assert res["is_fallback"] is True


# ----------------------------------------------------
# TEST 7: LLM EXCEPTION/TIMEOUT (SAFE FALLBACK TO HIGH)
# ----------------------------------------------------
def test_llm_exception_fallback():
    mock_llm = MagicMock()
    mock_llm.generate_raw.side_effect = Exception("LLM connection timeout")

    classifier = RiskClassifier(llm=mock_llm)
    res = classifier.classify("What is diabetes?")

    assert res["risk_level"] == "HIGH"
    assert res["action"] == SafetyAction.SAFETY_WARNING
    assert res["is_fallback"] is True


# ----------------------------------------------------
# TEST 8: TICKET CREATION FAILURE FALLBACK
# ----------------------------------------------------
def test_ticket_creation_failure_handling():
    mock_ticket_mgr = MagicMock()
    mock_ticket_mgr.create_ticket.side_effect = Exception("Storage read-only")

    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = {
        "risk_level": "MEDIUM",
        "action": SafetyAction.CREATE_TICKET,
        "query": "Headaches for weeks",
        "is_fallback": False
    }

    mock_llm = MagicMock()
    pipeline = RAGPipeline(llm=mock_llm, risk_classifier=mock_classifier, ticket_manager=mock_ticket_mgr)

    # When ticket creation throws, SafetyPolicy.get_medium_response handles None ticket gracefully
    res = pipeline.query("I have had headaches for three weeks.")
    assert "personalized medical assessment" in res["answer"].lower()
    assert res["risk_level"] == "MEDIUM"


# ----------------------------------------------------
# TEST 9: CONVERSATION CONTEXT PRONOUN RESOLUTION
# ----------------------------------------------------
def test_context_pronoun_resolution():
    mock_llm = MagicMock()
    mock_llm.generate_raw.return_value = '{"risk_level": "HIGH"}'

    classifier = RiskClassifier(llm=mock_llm)
    context = {"recent_messages": [{"role": "user", "content": "I take metformin."}]}

    res = classifier.classify("Should I stop it?", context)

    assert res["risk_level"] == RiskLevel.HIGH
    assert mock_llm.generate_raw.called
    assert "metformin" in str(mock_llm.generate_raw.call_args) or "Context" in str(mock_llm.generate_raw.call_args)


# ----------------------------------------------------
# TEST 10: SERIOUS DISEASE EDUCATIONAL QUESTION (LOW RISK)
# ----------------------------------------------------
def test_serious_disease_educational_question():
    mock_llm = MagicMock()
    mock_llm.generate_raw.return_value = '{"risk_level": "LOW"}'

    classifier = RiskClassifier(llm=mock_llm)
    res = classifier.classify("What is cancer?")

    # Disease severity alone does not force HIGH risk
    assert res["risk_level"] == RiskLevel.LOW
    assert res["action"] == SafetyAction.RAG
