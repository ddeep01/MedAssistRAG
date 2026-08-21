import logging
from typing import List, Dict, Any, Optional, Callable

from src.query.query_rewriter import QueryRewriter
from src.confidence.scorer import ConfidenceScorer
from src.retriever.search import Retriever
from src.config import load_retry_config

logger = logging.getLogger("MedAssistRAG.RetryController")


class RetryController:
    """
    Self-correction controller that manages bounded query rewriting retry loops
    when initial retrieval confidence is LOW (< 0.60).
    Tracks best attempt, loop prevention, and evidence-aware abstention.
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        scorer: Optional[ConfidenceScorer] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        config_path: Optional[str] = None,
    ):
        self.config = load_retry_config(config_path)
        r_cfg = self.config.get("retry", {})

        self.enabled = r_cfg.get("enabled", True)
        self.max_retries = int(r_cfg.get("max_retries", 2))
        self.min_confidence_improvement = float(r_cfg.get("min_confidence_improvement", 0.05))

        self.retriever = retriever or Retriever(config_path)
        self.scorer = scorer or ConfidenceScorer(config_path=config_path)
        self.query_rewriter = query_rewriter or QueryRewriter(config_path=config_path)

    def execute_with_retry(
        self,
        query: str,
        conversation_context: Optional[str] = None,
        k: int = 5,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes retrieval with self-correction retry loop if initial confidence is LOW.
        Tracks best attempt and returns structured state.
        """
        if not query or not query.strip():
            return {
                "status": "INSUFFICIENT_EVIDENCE",
                "original_query": query,
                "best_query": query,
                "best_confidence": 0.0,
                "best_level": "LOW",
                "best_results": [],
                "confidence_info": self.scorer.evaluate_retrieval(query, []),
                "retry_count": 0,
                "confidence_improved": False,
                "needs_abstention": True,
                "attempts": [],
            }

        attempted_queries = set()
        history_attempts = []

        # ====================================================
        # ATTEMPT 0: INITIAL RETRIEVAL (ORIGINAL QUERY)
        # ====================================================
        curr_query = query.strip()
        attempted_queries.add(curr_query.lower())

        initial_raw_candidates = self.retriever.search_structured(curr_query, k=k, mode=mode)
        initial_conf_info = self.scorer.evaluate_retrieval(curr_query, initial_raw_candidates)

        initial_conf = initial_conf_info["confidence"]
        initial_level = initial_conf_info["level"]

        history_attempts.append({
            "attempt": 0,
            "query": curr_query,
            "confidence": initial_conf,
            "level": initial_level,
            "results": initial_raw_candidates
        })

        best_attempt = {
            "query": curr_query,
            "results": initial_raw_candidates,
            "confidence": initial_conf,
            "level": initial_level,
            "confidence_info": initial_conf_info,
            "retry_count": 0,
        }

        # Log Attempt 0
        logger.info(
            f"\n========================================\n"
            f"SELF-CORRECTION\n"
            f"========================================\n"
            f"Attempt: 0\n"
            f"Query: '{curr_query}'\n"
            f"Confidence: {initial_conf:.4f}\n"
            f"Level: {initial_level}\n"
        )

        # Check if initial retrieval is already HIGH or MEDIUM
        if initial_level in ["HIGH", "MEDIUM"] or not self.enabled or self.max_retries <= 0:
            logger.info("Action: ACCEPT (Initial retrieval sufficient)\n========================================\n")
            return {
                "status": "SUCCESS",
                "original_query": query,
                "best_query": best_attempt["query"],
                "best_confidence": best_attempt["confidence"],
                "best_level": best_attempt["level"],
                "best_results": best_attempt["results"],
                "confidence_info": best_attempt["confidence_info"],
                "retry_count": 0,
                "confidence_improved": False,
                "needs_abstention": False,
                "attempts": history_attempts,
            }

        # ====================================================
        # SELF-CORRECTION RETRY LOOP (LOW CONFIDENCE)
        # ====================================================
        logger.info("Action: QUERY_REWRITE (Triggered on LOW confidence)\n")

        retry_count = 0
        while retry_count < self.max_retries:
            retry_count += 1

            # 1. Rewrite query using LLM
            rewritten_query = self.query_rewriter.rewrite(
                query=curr_query,
                conversation_context=conversation_context,
                retrieval_results=best_attempt["results"],
                confidence=best_attempt["confidence"]
            )

            # 2. Check Loop Prevention / Repeated Rewrites
            if rewritten_query.lower() in attempted_queries:
                logger.info(f"Loop prevention triggered: '{rewritten_query}' already attempted. Stopping retry loop.")
                break

            attempted_queries.add(rewritten_query.lower())

            # 3. Perform search with rewritten query
            new_raw_candidates = self.retriever.search_structured(rewritten_query, k=k, mode=mode)
            new_conf_info = self.scorer.evaluate_retrieval(rewritten_query, new_raw_candidates)

            new_conf = new_conf_info["confidence"]
            new_level = new_conf_info["level"]
            improvement = new_conf - best_attempt["confidence"]

            history_attempts.append({
                "attempt": retry_count,
                "query": rewritten_query,
                "confidence": new_conf,
                "level": new_level,
                "results": new_raw_candidates
            })

            logger.info(
                f"----------------------------------------\n"
                f"Attempt: {retry_count}\n"
                f"Query: '{rewritten_query}'\n"
                f"Confidence: {new_conf:.4f}\n"
                f"Level: {new_level}\n"
                f"Improvement: {improvement:+.4f}\n"
            )

            # 4. Check if score improved and update best attempt
            if new_conf > best_attempt["confidence"]:
                best_attempt = {
                    "query": rewritten_query,
                    "results": new_raw_candidates,
                    "confidence": new_conf,
                    "level": new_level,
                    "confidence_info": new_conf_info,
                    "retry_count": retry_count,
                }

            # 5. Success Condition (HIGH or MEDIUM reached)
            if new_level in ["HIGH", "MEDIUM"]:
                logger.info(f"Action: ACCEPT (Reached {new_level} confidence)\n========================================\n")
                break

            # 6. Check if improvement was below minimum threshold
            if improvement < self.min_confidence_improvement:
                logger.info("Action: CONTINUE (Improvement below threshold)")

            curr_query = rewritten_query

        # Determine final status
        is_successful = best_attempt["level"] in ["HIGH", "MEDIUM"]
        overall_improvement = best_attempt["confidence"] - initial_conf

        status_str = "SUCCESS" if is_successful else "INSUFFICIENT_EVIDENCE"
        logger.info(
            f"Final Status: {status_str} | Best Level: {best_attempt['level']} | Best Conf: {best_attempt['confidence']:.4f}\n"
            f"========================================\n"
        )

        return {
            "status": status_str,
            "original_query": query,
            "best_query": best_attempt["query"],
            "best_confidence": best_attempt["confidence"],
            "best_level": best_attempt["level"],
            "best_results": best_attempt["results"],
            "confidence_info": best_attempt["confidence_info"],
            "retry_count": best_attempt["retry_count"],
            "confidence_improved": (overall_improvement >= self.min_confidence_improvement),
            "needs_abstention": not is_successful,
            "attempts": history_attempts,
        }
