from src.generator.llm import LLM
from src.tools.executor import execute_tool
import json


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
        except:
            decision = {"tool": "SearchKB", "args": {"query": query}}

        # =========================
        # STEP 2: EXECUTE TOOL
        # =========================
        try:
            result = execute_tool(decision["tool"], decision["args"])
        except:
            result = []

        # =========================
        # STEP 3: GENERATE ANSWER
        # =========================
        if decision["tool"] == "SearchKB":
            if not result:
                answer = "No relevant information found."
            else:
                context = "\n".join(result)
                answer = self.llm.generate(query, context)

        elif decision["tool"] == "CreateTicket":
            answer = f"Your issue has been registered: {result.get('issue', 'Unknown')}"
            context = ""

        elif decision["tool"] == "MedicalDisclaimerTool":
            # First search KB normally
            kb_result = execute_tool("SearchKB", {"query": query})

            if not kb_result:
                answer = "No relevant medical information found."
            else:
                context = "\n".join(kb_result)
                base_answer = self.llm.generate(query, context)
                disclaimer = result["disclaimer"]
                answer = (
                    base_answer
                    + "\n\n"
                    + disclaimer
                )
        else:
            answer = "Something went wrong."
            context = ""

        return {
            "answer": answer,
            "sources": result if isinstance(result, list) else [str(result)]
        }


# Optional CLI (keep this if you want manual testing)
def main():
    rag = RAGPipeline()

    print("🧠 Tool-enabled RAG ready (type 'exit')\n")

    while True:
        query = input("Ask question: ")

        if query.lower() in ["exit", "quit"]:
            break

        result = rag.query(query)

        print("\n📄 Sources:", result["sources"])
        print("\n💡 Answer:\n", result["answer"])
        print("-" * 50)


if __name__ == "__main__":
    main()