import os
import sys

# Ensure workspace root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.safety.risk_classifier import RiskClassifier
from src.safety.safety_policy import SafetyPolicy, SafetyAction
from src.safety.ticket_manager import TicketManager


def main():
    print("=" * 75)
    print("      MEDICAL SAFETY & RISK CLASSIFICATION MANUAL INTEGRATION TEST     ")
    print("=" * 75)

    classifier = RiskClassifier()
    ticket_manager = TicketManager()

    test_queries = [
        ("What is hypertension?", "LOW"),
        ("I have been having frequent headaches for three weeks. What should I do?", "MEDIUM"),
        ("Should I change my prescribed medication dose?", "HIGH"),
        ("I have severe chest pain and difficulty breathing.", "HIGH")
    ]

    for i, (query, expected_level) in enumerate(test_queries, start=1):
        print(f"\n----------------------------------------------------------------------")
        print(f"TEST CASE {i}: \"{query}\"")
        print(f"Expected Level: {expected_level}")
        print(f"----------------------------------------------------------------------")

        res = classifier.classify(query)
        risk_level = res["risk_level"]
        action = res["action"]

        print(f"  [Assigned Risk Level]: {risk_level}")
        print(f"  [System Action]:        {action}")

        if action == SafetyAction.RAG:
            print("  [Pipeline Branch]:     Execute Normal RAG Pipeline (SearchKB -> Reranker -> Confidence)")
            print("  [Response Output]:     (Normal Knowledge Base Answer Generated)")
        elif action == SafetyAction.CREATE_TICKET:
            ticket = ticket_manager.create_ticket("manual-test-conv", query, risk_level="MEDIUM")
            response = SafetyPolicy.get_medium_response(ticket)
            print("  [Pipeline Branch]:     Bypass RAG Answer -> Create Support Ticket & Professional Referral")
            print(f"  [Ticket Created]:      ID={ticket['ticket_id']} | Status={ticket['status']}")
            print(f"  [Response Output]:     {response}")
        elif action == SafetyAction.SAFETY_WARNING:
            response = SafetyPolicy.get_high_response()
            print("  [Pipeline Branch]:     Bypass RAG Answer -> Return Controlled High-Risk Safety Warning")
            print(f"  [Response Output]:\n{response}")

        print("-" * 75)


if __name__ == "__main__":
    main()
