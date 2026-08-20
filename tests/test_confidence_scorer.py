import pytest
from src.confidence.scorer import ConfidenceScorer
from src.confidence.thresholds import ConfidenceThresholds, ConfidenceLevel


# ----------------------------------------------------
# TEST 1: HIGH RETRIEVAL + HIGH RERANKER
# ----------------------------------------------------
def test_high_confidence():
    scorer = ConfidenceScorer()
    candidates = [
        {"index": 0, "text": "Doc 1", "hybrid_score": 0.90, "reranker_score": 8.5},
        {"index": 1, "text": "Doc 2", "hybrid_score": 0.84, "reranker_score": 7.5},
        {"index": 2, "text": "Doc 3", "hybrid_score": 0.78, "reranker_score": 6.1},
        {"index": 3, "text": "Doc 4", "hybrid_score": 0.50, "reranker_score": 3.2},
        {"index": 4, "text": "Doc 5", "hybrid_score": 0.30, "reranker_score": 1.0},
    ]

    res = scorer.evaluate_retrieval("symptoms of diabetes", candidates)
    assert res["level"] == "HIGH"
    assert res["confidence"] >= 0.80
    assert not res["needs_retry"]


# ----------------------------------------------------
# TEST 2: MEDIUM SCORES
# ----------------------------------------------------
def test_medium_confidence():
    scorer = ConfidenceScorer()
    candidates = [
        {"index": 0, "text": "Doc 1", "hybrid_score": 0.65, "reranker_score": 4.5},
        {"index": 1, "text": "Doc 2", "hybrid_score": 0.60, "reranker_score": 3.5},
        {"index": 2, "text": "Doc 3", "hybrid_score": 0.55, "reranker_score": 2.5},
    ]

    res = scorer.evaluate_retrieval("mild headache treatment", candidates)
    assert res["level"] == "MEDIUM"
    assert 0.60 <= res["confidence"] < 0.80
    assert not res["needs_retry"]


# ----------------------------------------------------
# TEST 3: LOW RETRIEVAL + LOW RERANKER
# ----------------------------------------------------
def test_low_confidence():
    scorer = ConfidenceScorer()
    candidates = [
        {"index": 0, "text": "Doc 1", "hybrid_score": 0.30, "reranker_score": -1.5},
        {"index": 1, "text": "Doc 2", "hybrid_score": 0.20, "reranker_score": -2.5},
        {"index": 2, "text": "Doc 3", "hybrid_score": 0.10, "reranker_score": -3.5},
    ]

    res = scorer.evaluate_retrieval("unrelated random text query", candidates)
    assert res["level"] == "LOW"
    assert res["confidence"] < 0.60
    assert res["needs_retry"] is True
    assert res["needs_query_rewrite"] is True


# ----------------------------------------------------
# TEST 4: EQUAL RERANKER SCORES
# ----------------------------------------------------
def test_equal_reranker_scores():
    scorer = ConfidenceScorer()
    candidates = [
        {"index": 0, "text": "Doc 1", "hybrid_score": 0.80, "reranker_score": 5.0},
        {"index": 1, "text": "Doc 2", "hybrid_score": 0.80, "reranker_score": 5.0},
    ]

    # Must not throw DivisionByZeroError
    norm_scores = scorer.normalize_reranker_scores(candidates)
    assert norm_scores == [1.0, 1.0]

    res = scorer.evaluate_retrieval("query", candidates)
    assert isinstance(res["confidence"], float)
    assert not math_is_nan(res["confidence"])


def math_is_nan(val):
    import math
    return math.isnan(val) or math.isinf(val)


# ----------------------------------------------------
# TEST 5: ONLY ONE DOCUMENT
# ----------------------------------------------------
def test_single_document():
    scorer = ConfidenceScorer()
    candidates = [
        {"index": 0, "text": "Doc 1", "hybrid_score": 0.90, "reranker_score": 10.0}
    ]

    res = scorer.evaluate_retrieval("single doc query", candidates)
    assert res["confidence"] == 0.95  # 0.5 * 0.90 + 0.5 * 1.0
    assert res["level"] == "HIGH"


# ----------------------------------------------------
# TEST 6: ONLY TWO DOCUMENTS
# ----------------------------------------------------
def test_two_documents():
    scorer = ConfidenceScorer()
    candidates = [
        {"index": 0, "text": "Doc 1", "hybrid_score": 0.90, "reranker_score": 8.0},
        {"index": 1, "text": "Doc 2", "hybrid_score": 0.70, "reranker_score": 4.0},
    ]

    # Document 0 conf: 0.5*0.90 + 0.5*1.0 = 0.95
    # Document 1 conf: 0.5*0.70 + 0.5*0.0 = 0.35
    # Base rank weights [0.5, 0.3] normalized sum to 0.8: w1 = 0.5/0.8 = 0.625, w2 = 0.3/0.8 = 0.375
    # Expected final = 0.625 * 0.95 + 0.375 * 0.35 = 0.59375 + 0.13125 = 0.725
    res = scorer.evaluate_retrieval("two docs query", candidates)
    assert pytest.approx(res["confidence"], 0.001) == 0.725
    assert res["level"] == "MEDIUM"


# ----------------------------------------------------
# TEST 7: NO DOCUMENTS
# ----------------------------------------------------
def test_no_documents():
    scorer = ConfidenceScorer()
    res = scorer.evaluate_retrieval("nonexistent topic", [])

    assert res["confidence"] == 0.0
    assert res["level"] == "LOW"
    assert res["needs_retry"] is True
    assert res["needs_query_rewrite"] is True
    assert res["documents"] == []


# ----------------------------------------------------
# TEST 8: WEIGHT VALIDATION
# ----------------------------------------------------
def test_weight_validation():
    # Weights should sum to 1.0 or be automatically normalized
    scorer = ConfidenceScorer(retrieval_weight=2.0, reranker_weight=2.0)
    assert pytest.approx(scorer.thresholds.retrieval_weight, 0.001) == 0.5
    assert pytest.approx(scorer.thresholds.reranker_weight, 0.001) == 0.5


# ----------------------------------------------------
# TEST 9: DIFFERENT WEIGHTS (0.7 / 0.3)
# ----------------------------------------------------
def test_custom_weights():
    scorer = ConfidenceScorer(retrieval_weight=0.7, reranker_weight=0.3)
    candidates = [
        {"index": 0, "text": "Doc 1", "hybrid_score": 1.0, "reranker_score": 10.0}
    ]

    # Single doc: doc_conf = 0.7 * 1.0 + 0.3 * 1.0 = 1.0
    res = scorer.evaluate_retrieval("custom weight query", candidates)
    assert pytest.approx(res["confidence"], 0.001) == 1.0
    assert res["retrieval_weight"] == 0.7
    assert res["reranker_weight"] == 0.3
