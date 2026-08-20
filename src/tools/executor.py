# src/tools/executor.py

from src.retriever.search import Retriever

retriever = Retriever()


def execute_tool(tool_name, args):
    if tool_name == "SearchKB":
        query = args.get("query", "")
        k = args.get("k", 3)
        mode = args.get("mode", None)
        return_structured = args.get("return_structured", False)
        return retriever.search(query, k=k, mode=mode, return_structured=return_structured)

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