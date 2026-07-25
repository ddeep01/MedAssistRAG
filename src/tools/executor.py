# src/tools/executor.py

from src.retriever.search import Retriever

retriever = Retriever()


def execute_tool(tool_name, args):
    if tool_name == "SearchKB":
        return retriever.search(args["query"], k=3)

    elif tool_name == "CreateTicket":
        return {
            "status": "ticket_created",
            "issue": args["issue"]
        }
    elif tool_name == "MedicalDisclaimerTool":
        return {
            "disclaimer":
            "⚠️ This information is AI-generated and should not be considered professional medical advice. Please consult a qualified doctor for accurate diagnosis and treatment."
        }

    else:
        return {"error": "Unknown tool"}