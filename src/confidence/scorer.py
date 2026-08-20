import math
import logging
from typing import List, Dict, Any, Optional

from src.confidence.thresholds import ConfidenceThresholds, ConfidenceLevel

logger = logging.getLogger("MedAssistRAG.ConfidenceScorer")


def sanitize_float(val: Any, default: float = 0.0) -> float:
    """Safely converts input to float and replaces NaN/inf with default."""
    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return default
        return f_val
    except (ValueError, TypeError):
        return default


class ConfidenceScorer:
    """
    Programmatic Confidence Scorer for MedAssistRAG retrieval results.
    Evaluates confidence strictly from Hybrid Retrieval scores and Cross-Encoder Reranker scores.
    """

    def __init__(
        self,
        retrieval_weight: Optional[float] = None,
        reranker_weight: Optional[float] = None,
        thresholds: Optional[ConfidenceThresholds] = None,
        config_path: Optional[str] = None,
    ):
        self.thresholds = thresholds or ConfidenceThresholds(config_path)

        if retrieval_weight is not None:
            self.thresholds.retrieval_weight = float(retrieval_weight)
        if reranker_weight is not None:
            self.thresholds.reranker_weight = float(reranker_weight)

        # Validate weights sum to 1.0
        total_w = self.thresholds.retrieval_weight + self.thresholds.reranker_weight
        if abs(total_w - 1.0) > 1e-5:
            if total_w > 0:
                self.thresholds.retrieval_weight /= total_w
                self.thresholds.reranker_weight /= total_w
            else:
                self.thresholds.retrieval_weight = 0.5
                self.thresholds.reranker_weight = 0.5

    def normalize_reranker_scores(self, candidates: List[Dict[str, Any]]) -> List[float]:
        """
        Applies Min-Max normalization to the Cross-Encoder reranker scores
        across the current candidate set. Handles equal scores safely.
        """
        if not candidates:
            return []

        raw_scores = [
            sanitize_float(c.get("reranker_score", c.get("hybrid_score", 0.0)))
            for c in candidates
        ]

        min_s = min(raw_scores)
        max_s = max(raw_scores)

        if max_s == min_s:
            # If all reranker scores are equal, return 1.0 if min_s > 0 else 0.5
            norm_val = 1.0 if min_s > 0 else 0.5
            return [norm_val for _ in raw_scores]

        return [(s - min_s) / (max_s - min_s) for s in raw_scores]

    def calculate_document_confidence(
        self, candidate: Dict[str, Any], norm_reranker_score: float
    ) -> float:
        """
        Calculates document confidence from hybrid retrieval score and normalized reranker score.
        Formula: retrieval_weight * hybrid_score + reranker_weight * normalized_reranker_score
        """
        hybrid_s = sanitize_float(
            candidate.get(
                "hybrid_score",
                candidate.get("dense_score", candidate.get("bm25_score", 0.0)),
            )
        )
        norm_r = sanitize_float(norm_reranker_score)

        doc_conf = (
            self.thresholds.retrieval_weight * hybrid_s
            + self.thresholds.reranker_weight * norm_r
        )
        return max(0.0, min(1.0, doc_conf))

    def calculate_final_confidence(self, doc_confidences: List[float]) -> float:
        """
        Calculates the aggregate top-K confidence using rank weights (default top 3).
        Normalizes rank weights if fewer documents than weight array length exist.
        """
        if not doc_confidences:
            return 0.0

        base_weights = self.thresholds.rank_weights
        # Restrict calculation to top-K documents corresponding to rank_weights
        k = min(len(doc_confidences), len(base_weights))
        top_k_confidences = doc_confidences[:k]
        available_weights = base_weights[:k]

        weight_sum = sum(available_weights)
        if weight_sum > 0:
            norm_weights = [w / weight_sum for w in available_weights]
        else:
            norm_weights = [1.0 / k for _ in range(k)]

        final_score = sum(
            w * conf for w, conf in zip(norm_weights, top_k_confidences)
        )
        return max(0.0, min(1.0, final_score))

    def classify_confidence(self, score: float) -> ConfidenceLevel:
        """Classifies a confidence score into HIGH, MEDIUM, or LOW."""
        return self.thresholds.classify(score)

    def evaluate_retrieval(
        self, query: str, candidate_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates retrieval results and returns structured confidence metadata.
        """
        if not query or not query.strip() or not candidate_results:
            return {
                "confidence": 0.0,
                "level": ConfidenceLevel.LOW.value,
                "needs_retry": True,
                "needs_query_rewrite": True,
                "retrieval_score": 0.0,
                "reranker_score": 0.0,
                "normalized_reranker_score": 0.0,
                "retrieval_weight": self.thresholds.retrieval_weight,
                "reranker_weight": self.thresholds.reranker_weight,
                "documents": [],
            }

        norm_reranker_scores = self.normalize_reranker_scores(candidate_results)

        enriched_docs = []
        doc_confidences = []

        for cand, norm_r in zip(candidate_results, norm_reranker_scores):
            doc_conf = self.calculate_document_confidence(cand, norm_r)
            doc_confidences.append(doc_conf)

            c_info = {
                "index": cand.get("index", -1),
                "text": cand.get("text", ""),
                "hybrid_score": round(
                    sanitize_float(
                        cand.get(
                            "hybrid_score",
                            cand.get(
                                "dense_score", cand.get("bm25_score", 0.0)
                            ),
                        )
                    ),
                    4,
                ),
                "reranker_score": round(
                    sanitize_float(cand.get("reranker_score", 0.0)), 4
                ),
                "normalized_reranker_score": round(sanitize_float(norm_r), 4),
                "document_confidence": round(doc_conf, 4),
            }
            enriched_docs.append(c_info)

        final_conf = self.calculate_final_confidence(doc_confidences)
        level = self.classify_confidence(final_conf)
        needs_retry = level == ConfidenceLevel.LOW

        top_hybrid = enriched_docs[0]["hybrid_score"] if enriched_docs else 0.0
        top_raw_r = enriched_docs[0]["reranker_score"] if enriched_docs else 0.0
        top_norm_r = (
            enriched_docs[0]["normalized_reranker_score"]
            if enriched_docs
            else 0.0
        )

        result = {
            "confidence": round(final_conf, 4),
            "level": level.value,
            "needs_retry": needs_retry,
            "needs_query_rewrite": needs_retry,
            "retrieval_score": top_hybrid,
            "reranker_score": top_raw_r,
            "normalized_reranker_score": top_norm_r,
            "retrieval_weight": self.thresholds.retrieval_weight,
            "reranker_weight": self.thresholds.reranker_weight,
            "documents": enriched_docs,
        }

        # Debug logging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"\nConfidence Analysis:\n"
                f"--------------------------------\n"
                f"Documents: {len(enriched_docs)}\n"
                f"Top Hybrid Score: {top_hybrid:.4f}\n"
                f"Top Reranker Score: {top_raw_r:.4f}\n"
                f"Normalized Reranker: {top_norm_r:.4f}\n"
                f"Final Confidence: {final_conf:.4f}\n"
                f"Level: {level.value}\n"
                f"Needs Retry: {needs_retry}\n"
                f"--------------------------------"
            )

        return result
