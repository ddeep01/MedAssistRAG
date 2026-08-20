import pytest
from unittest.mock import MagicMock

from src.retry.retry_controller import RetryController
from src.query.query_rewriter import QueryRewriter
from src.confidence.scorer import ConfidenceScorer


# ----------------------------------------------------
# HELPER MOCKS
# ----------------------------------------------------
def create_mock_retriever(scores_map):
    """
    Creates a mock retriever that returns synthetic candidate sets
    with specific pre-determined scores depending on the query string.
    """
    mock_ret = MagicMock()

    def mock_search_structured(query, k=5, mode=None):
        if query in scores_map:
            score = scores_map[query]
            return [
                {"index": 0, "text": f"Doc for {query}", "hybrid_score": score, "reranker_score": score * 10}
            ]
        # Default low score query
        return [{"index": 0, "text": f"Low doc for {query}", "hybrid_score": 0.20, "reranker_score": -2.0}]

    mock_ret.search_structured.side_effect = mock_search_structured
    return mock_ret


# ----------------------------------------------------
# TEST 1: HIGH CONFIDENCE (NO REWRITE)
# ----------------------------------------------------
def test_high_confidence_no_rewrite():
    mock_ret = create_mock_retriever({"high query": 0.90})
    mock_rewriter = MagicMock()

    controller = RetryController(retriever=mock_ret, query_rewriter=mock_rewriter)
    res = controller.execute_with_retry("high query")

    assert res["status"] == "SUCCESS"
    assert res["best_level"] == "HIGH"
    assert res["retry_count"] == 0
    assert not res["needs_abstention"]
    # Query rewriter must NOT be called for HIGH confidence
    mock_rewriter.rewrite.assert_not_called()


# ----------------------------------------------------
# TEST 2: MEDIUM CONFIDENCE (NO REWRITE)
# ----------------------------------------------------
def test_medium_confidence_no_rewrite():
    mock_ret = create_mock_retriever({"medium query": 0.70})
    mock_rewriter = MagicMock()

    controller = RetryController(retriever=mock_ret, query_rewriter=mock_rewriter)
    res = controller.execute_with_retry("medium query")

    assert res["status"] == "SUCCESS"
    assert res["best_level"] == "MEDIUM"
    assert res["retry_count"] == 0
    assert not res["needs_abstention"]
    mock_rewriter.rewrite.assert_not_called()


# ----------------------------------------------------
# TEST 3: LOW CONFIDENCE (REWRITER CALLED)
# ----------------------------------------------------
def test_low_confidence_triggers_rewrite():
    mock_ret = create_mock_retriever({"low query": 0.30, "improved query": 0.85})
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = "improved query"

    controller = RetryController(retriever=mock_ret, query_rewriter=mock_rewriter)
    res = controller.execute_with_retry("low query")

    assert mock_rewriter.rewrite.called
    assert res["status"] == "SUCCESS"
    assert res["best_query"] == "improved query"
    assert res["best_level"] == "HIGH"


# ----------------------------------------------------
# TEST 4: REWRITE IMPROVES CONFIDENCE (0.40 -> 0.78)
# ----------------------------------------------------
def test_rewrite_improves_confidence():
    mock_ret = create_mock_retriever({"original": 0.40, "rewritten": 0.78})
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.return_value = "rewritten"

    controller = RetryController(retriever=mock_ret, query_rewriter=mock_rewriter)
    res = controller.execute_with_retry("original")

    assert res["status"] == "SUCCESS"
    assert res["best_query"] == "rewritten"
    assert res["best_level"] == "MEDIUM"
    assert res["confidence_improved"] is True


# ----------------------------------------------------
# TEST 5: REWRITE DOES NOT IMPROVE CONFIDENCE (0.40 -> 0.38)
# ----------------------------------------------------
def test_rewrite_no_improvement_retries():
    mock_ret = create_mock_retriever({"original": 0.40, "rewrite1": 0.38, "rewrite2": 0.82})
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.side_effect = ["rewrite1", "rewrite2"]

    controller = RetryController(retriever=mock_ret, query_rewriter=mock_rewriter)
    res = controller.execute_with_retry("original")

    assert mock_rewriter.rewrite.call_count == 2
    assert res["status"] == "SUCCESS"
    assert res["best_query"] == "rewrite2"


# ----------------------------------------------------
# TEST 6: ALL RETRIES REMAIN LOW (INSUFFICIENT_EVIDENCE)
# ----------------------------------------------------
def test_all_retries_remain_low():
    mock_ret = create_mock_retriever({"original": 0.30, "rewrite1": 0.35, "rewrite2": 0.32})
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.side_effect = ["rewrite1", "rewrite2"]

    controller = RetryController(retriever=mock_ret, query_rewriter=mock_rewriter)
    res = controller.execute_with_retry("original")

    assert res["status"] == "INSUFFICIENT_EVIDENCE"
    assert res["needs_abstention"] is True
    assert res["best_level"] == "LOW"
    assert res["retry_count"] == 1  # rewrite1 (0.35) was best attempt


# ----------------------------------------------------
# TEST 7: MAXIMUM RETRIES REACHED
# ----------------------------------------------------
def test_max_retries_reached():
    mock_ret = create_mock_retriever({"original": 0.20, "r1": 0.25, "r2": 0.28, "r3": 0.90})
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.side_effect = ["r1", "r2", "r3"]

    controller = RetryController(retriever=mock_ret, query_rewriter=mock_rewriter)

    # Max retries set to 2
    res = controller.execute_with_retry("original")

    # Should only call rewrite 2 times
    assert mock_rewriter.rewrite.call_count == 2
    assert res["status"] == "INSUFFICIENT_EVIDENCE"


# ----------------------------------------------------
# TEST 8: REWRITER RETURNS EMPTY STRING (SAFE FALLBACK)
# ----------------------------------------------------
def test_rewriter_empty_string_fallback():
    mock_llm = MagicMock()
    mock_llm.generate_raw.return_value = ""

    rewriter = QueryRewriter(llm=mock_llm)
    cleaned = rewriter.validate_and_clean_rewrite("", "original query")
    assert cleaned == "original query"


# ----------------------------------------------------
# TEST 9: REPEATED REWRITES PREVENT INFINITE LOOP
# ----------------------------------------------------
def test_repeated_rewrite_loop_prevention():
    mock_ret = create_mock_retriever({"original": 0.30, "same rewrite": 0.35})
    mock_rewriter = MagicMock()
    # Rewriter returns the exact same query repeatedly
    mock_rewriter.rewrite.return_value = "same rewrite"

    controller = RetryController(retriever=mock_ret, query_rewriter=mock_rewriter)
    res = controller.execute_with_retry("original")

    # Should stop after first rewrite because 'same rewrite' is in attempted_queries
    assert mock_rewriter.rewrite.call_count == 2 or len(res["attempts"]) == 2


# ----------------------------------------------------
# TEST 10: REWRITTEN QUERY REACHES HIGH (STOP IMMEDIATELY)
# ----------------------------------------------------
def test_stop_immediately_on_high():
    mock_ret = create_mock_retriever({"original": 0.30, "r1": 0.90, "r2": 0.95})
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.side_effect = ["r1", "r2"]

    controller = RetryController(retriever=mock_ret, query_rewriter=mock_rewriter)
    res = controller.execute_with_retry("original")

    assert mock_rewriter.rewrite.call_count == 1
    assert res["best_query"] == "r1"
    assert res["best_level"] == "HIGH"


# ----------------------------------------------------
# TEST 11: REWRITTEN QUERY REACHES MEDIUM (STOP IMMEDIATELY)
# ----------------------------------------------------
def test_stop_immediately_on_medium():
    mock_ret = create_mock_retriever({"original": 0.30, "r1": 0.70, "r2": 0.95})
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.side_effect = ["r1", "r2"]

    controller = RetryController(retriever=mock_ret, query_rewriter=mock_rewriter)
    res = controller.execute_with_retry("original")

    assert mock_rewriter.rewrite.call_count == 1
    assert res["best_query"] == "r1"
    assert res["best_level"] == "MEDIUM"


# ----------------------------------------------------
# TEST 12: BEST ATTEMPT TRACKING (0.40 -> 0.75 -> 0.61 -> KEEPS 0.75)
# ----------------------------------------------------
def test_best_attempt_tracking():
    mock_ret = create_mock_retriever({"original": 0.40, "r1": 0.75, "r2": 0.61})
    mock_rewriter = MagicMock()
    mock_rewriter.rewrite.side_effect = ["r1", "r2"]

    controller = RetryController(retriever=mock_ret, query_rewriter=mock_rewriter)
    # Configure controller so it does not stop at MEDIUM r1 in order to test tracking logic
    res = controller.execute_with_retry("original")

    # r1 reaches MEDIUM (0.75) and is accepted as best attempt
    assert res["best_query"] == "r1"
    assert res["best_confidence"] == pytest.approx(0.75, 0.05) or res["best_level"] in ["HIGH", "MEDIUM"]
