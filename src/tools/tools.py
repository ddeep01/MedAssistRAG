TOOLS = [
    {
        "name": "SearchKB",
        "description": "Search knowledge base for relevant documents",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "CreateTicket",
        "description": "Create a support ticket",
        "parameters": {
            "type": "object",
            "properties": {
                "issue": {"type": "string"}
            },
            "required": ["issue"]
        }
    },
    # =========================
    # NEW TOOL
    # =========================
    {
        "name": "MedicalDisclaimerTool",
        "description": "Adds medical safety disclaimer for symptom-related queries",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
]

