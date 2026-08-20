import json
import logging
from typing import Optional, Dict, Any, List

from src.generator.llm import LLM
from src.tools.executor import execute_tool
from src.retry.retry_controller import RetryController
from src.query.query_rewriter import QueryRewriter
from src.memory.memory_manager import MemoryManager
from src.safety.risk_classifier import RiskClassifier
from src.safety.safety_policy import SafetyPolicy, SafetyAction, RiskLevel
from src.safety.ticket_manager import TicketManager
from src.citations.citation_manager import CitationManager

logger = logging.getLogger("MedAssistRAG.Pipeline")


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
    def __init__(
        self,
        llm: Optional[LLM] = None,
        memory_manager: Optional[MemoryManager] = None,
        risk_classifier: Optional[RiskClassifier] = None,
        ticket_manager: Optional[TicketManager] = None,
        retry_controller: Optional[RetryController] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        citation_manager: Optional[CitationManager] = None,
    ):
        self._llm = llm
        self._retry_controller = retry_controller
        self._query_rewriter = query_rewriter
        self.memory_manager = memory_manager or MemoryManager(llm=self._llm)
        self.risk_classifier = risk_classifier or RiskClassifier(llm=self._llm)
        self.ticket_manager = ticket_manager or TicketManager()
        self.citation_manager = citation_manager or CitationManager()

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = LLM()
        return self._llm

    @property
    def retry_controller(self) -> RetryController:
        if self._retry_controller is None:
            self._retry_controller = RetryController()
        return self._retry_controller

    @property
    def query_rewriter(self) -> QueryRewriter:
        if self._query_rewriter is None:
            self._query_rewriter = QueryRewriter(llm=self._llm)
        return self._query_rewriter

    def query(self, query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes Medical Risk Classification -> Memory-Aware RAG Pipeline with Citation Attribution.
        Classifies risk into LOW (RAG + Citations), MEDIUM (Ticket), or HIGH (Safety Warning).
        """
        # 1. Initialize or resolve conversation_id
        cid = self.memory_manager.create_conversation(conversation_id)

        # 2. Get existing memory context prior to current turn
        prior_memory_context = self.memory_manager.get_context(cid)

        # 3. Perform LLM Risk Classification BEFORE running SearchKB/QueryRewriter
        risk_res = self.risk_classifier.classify(query, prior_memory_context)
        risk_level = risk_res["risk_level"]
        action = risk_res["action"]

        logger.info(f"----------------------------------------")
        logger.info(f"Risk Classification | Level: {risk_level} | Action: {action}")
        logger.info(f"----------------------------------------")

        # ====================================================
        # HIGH RISK: CONTROLLED EMERGENCY SAFETY WARNING
        # ====================================================
        if action == SafetyAction.SAFETY_WARNING:
            self.memory_manager.add_message(cid, "user", query)
            answer = SafetyPolicy.get_high_response()
            self.memory_manager.add_message(cid, "assistant", answer)

            return {
                "conversation_id": cid,
                "query": query,
                "standalone_query": query,
                "risk_level": risk_level,
                "action": action.value if hasattr(action, "value") else str(action),
                "answer": answer,
                "sources": [],
                "citations": [],
                "confidence": 0.0,
                "level": "N/A",
                "ticket": None,
                "retry_state": None,
                "memory_context": self.memory_manager.get_context(cid).to_dict()
            }

        # ====================================================
        # MEDIUM RISK: CREATE SUPPORT TICKET & REFERRAL
        # ====================================================
        elif action == SafetyAction.CREATE_TICKET:
            self.memory_manager.add_message(cid, "user", query)
            try:
                ticket = self.ticket_manager.create_ticket(cid, query, risk_level="MEDIUM")
            except Exception as e:
                logger.error(f"[RAGPipeline] Ticket creation failed: {e}")
                ticket = None

            answer = SafetyPolicy.get_medium_response(ticket)
            self.memory_manager.add_message(cid, "assistant", answer)

            return {
                "conversation_id": cid,
                "query": query,
                "standalone_query": query,
                "risk_level": risk_level,
                "action": action.value if hasattr(action, "value") else str(action),
                "answer": answer,
                "sources": [],
                "citations": [],
                "confidence": 1.0,
                "level": "N/A",
                "ticket": ticket,
                "retry_state": None,
                "memory_context": self.memory_manager.get_context(cid).to_dict()
            }

        # ====================================================
        # LOW RISK: NORMAL MEMORY-AWARE RAG PIPELINE
        # ====================================================
        # 4. Record user message in memory
        self.memory_manager.add_message(cid, "user", query)

        # 5. Generate standalone query if conversation context exists
        if prior_memory_context.recent_messages or prior_memory_context.entities:
            standalone_query = self.query_rewriter.rewrite(query, prior_memory_context)
        else:
            standalone_query = query

        # 6. Tool decision
        decision_raw = self.llm.generate_raw(tool_prompt(standalone_query))

        try:
            decision = json.loads(decision_raw)
        except Exception:
            decision = {"tool": "SearchKB", "args": {"query": standalone_query}}

        tool_name = decision.get("tool", "SearchKB")
        tool_args = decision.get("args", {"query": standalone_query})

        # 7. Execute Tool & Self-Correction Retry Controller
        retry_state = None
        raw_candidates: List[Dict[str, Any]] = []

        if tool_name in ["SearchKB", "MedicalDisclaimerTool"]:
            target_query = tool_args.get("query", standalone_query)
            retry_state = self.retry_controller.execute_with_retry(
                query=target_query,
                conversation_context=prior_memory_context
            )
            raw_candidates = retry_state.get("best_results", [])
            sources = [c["text"] for c in raw_candidates if "text" in c]
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

        # 8. Generate Answer with Citation Attribution
        validation_res = None
        citations = []

        if tool_name in ["SearchKB", "MedicalDisclaimerTool"]:
            if confidence_info.get("level") == "LOW" or (retry_state and retry_state.get("needs_abstention")):
                answer = "Insufficient evidence found to answer the query with confidence after self-correction retry attempts."
            elif not raw_candidates:
                answer = "No relevant information found."
            else:
                # Create evidence objects with temporary IDs (E1, E2, ...)
                evidence_objects = self.citation_manager.create_evidence_objects(raw_candidates)
                evidence_ctx_str = self.citation_manager.build_evidence_context(evidence_objects)

                # Generate draft response from LLM with [E#] citation tags
                raw_draft_answer = self.llm.generate_with_evidence(standalone_query, evidence_ctx_str)

                # Validate citations, replace [E1] -> [1], consolidate sources, append Source List
                validation_res = self.citation_manager.validate_and_format_citations(raw_draft_answer, evidence_objects)
                answer = validation_res.formatted_text
                citations = [c.to_dict() for c in validation_res.valid_citations]

                if tool_name == "MedicalDisclaimerTool":
                    disclaimer_info = execute_tool("MedicalDisclaimerTool", tool_args)
                    disclaimer = disclaimer_info.get("disclaimer", "") if isinstance(disclaimer_info, dict) else ""
                    if disclaimer:
                        answer += f"\n\n{disclaimer}"

        elif tool_name == "CreateTicket":
            answer = f"Your issue has been registered: {tool_args.get('issue', 'Unknown')}"
        else:
            answer = "Something went wrong."

        # 9. Record assistant response in memory
        self.memory_manager.add_message(cid, "assistant", answer)

        return {
            "conversation_id": cid,
            "query": query,
            "standalone_query": standalone_query,
            "risk_level": risk_level,
            "action": action.value if hasattr(action, "value") else str(action),
            "answer": answer,
            "sources": sources,
            "citations": citations,
            "validation_result": validation_res.to_dict() if validation_res else None,
            "confidence": confidence_info.get("confidence", 0.0),
            "level": confidence_info.get("level", "LOW"),
            "ticket": None,
            "needs_retry": confidence_info.get("needs_retry", True),
            "needs_query_rewrite": confidence_info.get("needs_query_rewrite", True),
            "confidence_details": confidence_info,
            "retry_state": retry_state,
            "memory_context": self.memory_manager.get_context(cid).to_dict()
        }


def main():
    rag = RAGPipeline()
    print("[CITATIONS] Memory-Aware & Citation-Attributed RAG ready (type 'exit' or 'clear')\n")

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
        print(f"\n[SAFETY] Risk Level: {result['risk_level']} -> Action: {result['action']}")
        if result['action'] == "RAG":
            print(f"[REWRITE] Standalone Query: '{result['standalone_query']}'")
            print(f"[CONFIDENCE] Level: {result['level']} ({result['confidence']})")
        elif result['ticket']:
            print(f"[TICKET] Support Ticket Created: {result['ticket']['ticket_id']}")
        print("\n[RESPONSE]:\n", result["answer"])
        print("-" * 50)


if __name__ == "__main__":
    main()