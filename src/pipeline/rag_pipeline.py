import json
from typing import Optional, Dict, Any

from src.generator.llm import LLM
from src.tools.executor import execute_tool
from src.retry.retry_controller import RetryController
from src.query.query_rewriter import QueryRewriter
from src.memory.memory_manager import MemoryManager


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
    def __init__(self, llm: Optional[LLM] = None, memory_manager: Optional[MemoryManager] = None):
        self._llm = llm
        self.retry_controller = RetryController()
        self.query_rewriter = QueryRewriter(llm=self._llm)
        self.memory_manager = memory_manager or MemoryManager(llm=self._llm)

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = LLM()
        return self._llm

    def query(self, query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes conversational memory-aware RAG pipeline.
        Medical answers strictly come from retrieved RAG evidence. Memory is used ONLY
        for conversation continuity and query rewriting.
        """
        # 1. Initialize or resolve conversation_id
        cid = self.memory_manager.create_conversation(conversation_id)

        # 2. Get existing memory context prior to current turn
        prior_memory_context = self.memory_manager.get_context(cid)

        # 3. Record user message in memory
        self.memory_manager.add_message(cid, "user", query)

        # 4. Generate standalone query if conversation context exists
        if prior_memory_context.recent_messages or prior_memory_context.entities:
            standalone_query = self.query_rewriter.rewrite(query, prior_memory_context)
        else:
            standalone_query = query

        # =========================
        # STEP 5: TOOL DECISION
        # =========================
        decision_raw = self.llm.generate_raw(tool_prompt(standalone_query))

        try:
            decision = json.loads(decision_raw)
        except Exception:
            decision = {"tool": "SearchKB", "args": {"query": standalone_query}}

        tool_name = decision.get("tool", "SearchKB")
        tool_args = decision.get("args", {"query": standalone_query})

        # =========================
        # STEP 6: EXECUTE TOOL & SELF-CORRECTION RETRY CONTROLLER
        # =========================
        retry_state = None

        if tool_name in ["SearchKB", "MedicalDisclaimerTool"]:
            target_query = tool_args.get("query", standalone_query)
            retry_state = self.retry_controller.execute_with_retry(
                query=target_query,
                conversation_context=prior_memory_context
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
        # STEP 7: GENERATE ANSWER / ABSTAIN
        # =========================
        if tool_name == "SearchKB":
            if confidence_info.get("level") == "LOW" or (retry_state and retry_state.get("needs_abstention")):
                answer = "Insufficient evidence found to answer the query with confidence after self-correction retry attempts."
            elif not sources:
                answer = "No relevant information found."
            else:
                context = "\n".join(sources)
                answer = self.llm.generate(standalone_query, context)

        elif tool_name == "CreateTicket":
            answer = f"Your issue has been registered: {tool_args.get('issue', 'Unknown')}"

        elif tool_name == "MedicalDisclaimerTool":
            if confidence_info.get("level") == "LOW" or (retry_state and retry_state.get("needs_abstention")):
                answer = "Insufficient medical evidence found to answer the query with confidence after self-correction retry attempts."
            elif not sources:
                answer = "No relevant medical information found."
            else:
                context = "\n".join(sources)
                base_answer = self.llm.generate(standalone_query, context)
                disclaimer_info = execute_tool("MedicalDisclaimerTool", tool_args)
                disclaimer = disclaimer_info.get("disclaimer", "") if isinstance(disclaimer_info, dict) else ""
                answer = (
                    base_answer
                    + "\n\n"
                    + disclaimer
                )
        else:
            answer = "Something went wrong."

        # 8. Record assistant response in memory
        self.memory_manager.add_message(cid, "assistant", answer)

        return {
            "conversation_id": cid,
            "query": query,
            "standalone_query": standalone_query,
            "answer": answer,
            "sources": sources,
            "confidence": confidence_info.get("confidence", 0.0),
            "level": confidence_info.get("level", "LOW"),
            "needs_retry": confidence_info.get("needs_retry", True),
            "needs_query_rewrite": confidence_info.get("needs_query_rewrite", True),
            "confidence_details": confidence_info,
            "retry_state": retry_state,
            "memory_context": self.memory_manager.get_context(cid).to_dict()
        }


def main():
    rag = RAGPipeline()
    print("🧠 Conversational Memory-Aware RAG ready (type 'exit' or 'clear')\n")

    current_cid = rag.memory_manager.create_conversation()

    while True:
        query = input(f"[{current_cid[:8]}] Ask question: ")
        if query.lower() in ["exit", "quit"]:
            break
        if query.lower() == "clear":
            rag.memory_manager.clear_conversation(current_cid)
            current_cid = rag.memory_manager.create_conversation()
            print("Cleared conversation memory.\n")
            continue

        result = rag.query(query, conversation_id=current_cid)
        print(f"\n🔄 Standalone Query: '{result['standalone_query']}'")
        print(f"📊 Confidence Level: {result['level']} ({result['confidence']})")
        print("📄 Sources:", result["sources"])
        print("\n💡 Answer:\n", result["answer"])
        print("-" * 50)


if __name__ == "__main__":
    main()