import json
from typing import Optional, Dict, Any
from src.generator.llm import LLM
from src.tools.executor import execute_tool
from src.retry.retry_controller import RetryController


def tool_prompt(query: str) -> str:
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
        self.retry_controller = RetryController()

    def query(self, query: str, conversation_context: Optional[str] = None) -> Dict[str, Any]:
        # =========================
        # STEP 1: TOOL DECISION
        # =========================
        decision_raw = self.llm.generate_raw(tool_prompt(query))

        try:
            decision = json.loads(decision_raw)
        except Exception:
            decision = {"tool": "SearchKB", "args": {"query": query}}

        tool_name = decision.get("tool", "SearchKB")
        tool_args = decision.get("args", {"query": query})

        # =========================
        # STEP 2: EXECUTE TOOL & RETRY CONTROLLER
        # =========================
        retry_state = None

        if tool_name in ["SearchKB", "MedicalDisclaimerTool"]:
            target_query = tool_args.get("query", query)
            retry_state = self.retry_controller.execute_with_retry(
                query=target_query,
                conversation_context=conversation_context
            )
            sources = [c["text"] for c in retry_state["best_results"] if "text" in c]
            confidence_info = retry_state["confidence_info"]
        elif tool_name == "CreateTicket":
            tool_output = execute_tool("CreateTicket", tool_args)
            sources = []
            confidence_info = {
                "confidence": 1.0,
                "level": "HIGH",
                "needs_retry": False,
                "needs_query_rewrite": False
            }
        else:
            sources = []
            confidence_info = {
                "confidence": 0.0,
                "level": "LOW",
                "needs_retry": True,
                "needs_query_rewrite": True
            }

        # =========================
        # STEP 3: GENERATE ANSWER / ABSTAIN
        # =========================
        if tool_name == "SearchKB":
            if confidence_info.get("level") == "LOW" or (retry_state and retry_state.get("needs_abstention")):
                answer = "Insufficient evidence found to answer the query with confidence after self-correction retry attempts."
            elif not sources:
                answer = "No relevant information found."
            else:
                context = "\n".join(sources)
                answer = self.llm.generate(query, context)

        elif tool_name == "CreateTicket":
            answer = f"Your issue has been registered: {tool_args.get('issue', 'Unknown')}"

        elif tool_name == "MedicalDisclaimerTool":
            if confidence_info.get("level") == "LOW" or (retry_state and retry_state.get("needs_abstention")):
                answer = "Insufficient medical evidence found to answer the query with confidence after self-correction retry attempts."
            elif not sources:
                answer = "No relevant medical information found."
            else:
                context = "\n".join(sources)
                base_answer = self.llm.generate(query, context)
                disclaimer_info = execute_tool("MedicalDisclaimerTool", tool_args)
                disclaimer = disclaimer_info.get("disclaimer", "") if isinstance(disclaimer_info, dict) else ""
                answer = (
                    base_answer
                    + "\n\n"
                    + disclaimer
                )
        else:
            answer = "Something went wrong."

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence_info.get("confidence", 0.0),
            "level": confidence_info.get("level", "LOW"),
            "needs_retry": confidence_info.get("needs_retry", True),
            "needs_query_rewrite": confidence_info.get("needs_query_rewrite", True),
            "confidence_details": confidence_info,
            "retry_state": retry_state
        }


def main():
    rag = RAGPipeline()
    print("🧠 Self-Correction Bounded RAG ready (type 'exit')\n")

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