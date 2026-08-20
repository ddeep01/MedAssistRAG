# src/tools/executor.py

from src.retriever.search import Retriever
from src.confidence.scorer import ConfidenceScorer

retriever = Retriever()
confidence_scorer = ConfidenceScorer()


def execute_tool(tool_name, args):
    if tool_name == "SearchKB":
        query = args.get("query", "")
        k = args.get("k", 5)
        mode = args.get("mode", None)
        return_structured = args.get("return_structured", False)
        include_confidence = args.get("include_confidence", True)

        raw_candidates = retriever.search_structured(query, k=k, mode=mode)
        confidence_info = confidence_scorer.evaluate_retrieval(query, raw_candidates)

        if return_structured:
            return {
                "results": raw_candidates,
                "confidence": confidence_info
            }

        # Legacy backward compatible list of text strings with attached confidence metadata
        texts = [c.get("text", "") for c in raw_candidates if "text" in c]

        if include_confidence:
            return {
                "sources": texts,
                "confidence": confidence_info
            }

        return texts

    elif tool_name == "CreateTicket":
        return {
            "status": "ticket_created",
            "issue": args.get("issue", "")
        }
    elif tool_name == "MedicalDisclaimerTool":
        return {
            "disclaimer":
            "⚠️ This information is AI-generated and should not be considered professional medical advice. Please consult a qualified doctor for accurate diagnosis and treatment."
        }

    else:
        return {"error": "Unknown tool"}