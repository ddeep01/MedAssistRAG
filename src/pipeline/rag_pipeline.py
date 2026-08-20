import json
from src.generator.llm import LLM
from src.tools.executor import execute_tool


def tool_prompt(query):
    return f"""
You are an AI assistant with access to tools.

Available tools:
1. SearchKB(query) - search knowledge base
2. CreateTicket(issue) - create support ticket
3. MedicalDisclaimerTool(query) - add disclaimer for medical symptom related questions

Decide the best tool.

Return ONLY JSON:
{{"tool": "...", "args": {{...}}}}

Query: {query}
"""


class RAGPipeline:
    def __init__(self):
        self.llm = LLM()

    def query(self, query):
        # =========================
        # STEP 1: TOOL DECISION
        # =========================
        decision_raw = self.llm.generate_raw(tool_prompt(query))

        try:
            decision = json.loads(decision_raw)
        except Exception:
            decision = {"tool": "SearchKB", "args": {"query": query}}

        # =========================
        # STEP 2: EXECUTE TOOL
        # =========================
        try:
            tool_output = execute_tool(decision["tool"], decision["args"])
        except Exception:
            tool_output = []

        # Process tool output structure
        if isinstance(tool_output, dict):
            sources = tool_output.get("sources", [])
            confidence_info = tool_output.get("confidence", {
                "confidence": 0.0,
                "level": "LOW",
                "needs_retry": True,
                "needs_query_rewrite": True
            })
        elif isinstance(tool_output, list):
            sources = tool_output
            confidence_info = {
                "confidence": 0.0,
                "level": "LOW",
                "needs_retry": True,
                "needs_query_rewrite": True
            }
        else:
            sources = [str(tool_output)]
            confidence_info = {
                "confidence": 0.0,
                "level": "LOW",
                "needs_retry": True,
                "needs_query_rewrite": True
            }

        # =========================
        # STEP 3: CONFIDENCE DECISION & GENERATION
        # =========================
        if decision["tool"] == "SearchKB":
            if confidence_info.get("level") == "LOW" or confidence_info.get("needs_retry"):
                answer = "Insufficient evidence found to answer the query with confidence."
            elif not sources:
                answer = "No relevant information found."
            else:
                context = "\n".join(sources)
                answer = self.llm.generate(query, context)

        elif decision["tool"] == "CreateTicket":
            answer = f"Your issue has been registered: {tool_output.get('issue', 'Unknown')}"
            sources = []
            confidence_info = {
                "confidence": 1.0,
                "level": "HIGH",
                "needs_retry": False,
                "needs_query_rewrite": False
            }

        elif decision["tool"] == "MedicalDisclaimerTool":
            # Search KB with confidence checking
            kb_output = execute_tool("SearchKB", {"query": query})
            if isinstance(kb_output, dict):
                sources = kb_output.get("sources", [])
                confidence_info = kb_output.get("confidence", confidence_info)
            else:
                sources = kb_output if isinstance(kb_output, list) else [str(kb_output)]

            if confidence_info.get("level") == "LOW" or confidence_info.get("needs_retry"):
                answer = "Insufficient medical evidence found to answer the query with confidence."
            elif not sources:
                answer = "No relevant medical information found."
            else:
                context = "\n".join(sources)
                base_answer = self.llm.generate(query, context)
                disclaimer = tool_output.get("disclaimer", "")
                answer = (
                    base_answer
                    + "\n\n"
                    + disclaimer
                )
        else:
            answer = "Something went wrong."
            sources = []
            confidence_info = {
                "confidence": 0.0,
                "level": "LOW",
                "needs_retry": True,
                "needs_query_rewrite": True
            }

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence_info.get("confidence", 0.0),
            "level": confidence_info.get("level", "LOW"),
            "needs_retry": confidence_info.get("needs_retry", True),
            "needs_query_rewrite": confidence_info.get("needs_query_rewrite", True),
            "confidence_details": confidence_info
        }


def main():
    rag = RAGPipeline()
    print("🧠 Confidence-Aware RAG ready (type 'exit')\n")

    while True:
        query = input("Ask question: ")
        if query.lower() in ["exit", "quit"]:
            break

        result = rag.query(query)
        print(f"\n📊 Confidence Level: {result['level']} ({result['confidence']})")
        print("📄 Sources:", result["sources"])
        print("\n💡 Answer:\n", result["answer"])
        print("-" * 50)


if __name__ == "__main__":
    main()