import os
from typing import List, Dict, Any, Union
from sentence_transformers import CrossEncoder
from src.utils.config import load_retrieval_config


class Reranker:
    """
    Cross-Encoder Reranker for refining hybrid/dense/keyword retrieval results.
    """

    def __init__(self, model_name_or_path: str = None):
        config = load_retrieval_config()
        reranker_cfg = config.get("reranker", {})

        target_path = model_name_or_path or reranker_cfg.get("model_path", "models/reranker_finetuned")
        fallback_path = reranker_cfg.get("fallback_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        if target_path and os.path.exists(target_path):
            model_to_load = target_path
        else:
            model_to_load = fallback_path

        try:
            self.model = CrossEncoder(model_to_load)
        except Exception as e:
            print(f"[Warning] Failed to load CrossEncoder from {model_to_load}: {e}. Falling back to {fallback_path}")
            self.model = CrossEncoder(fallback_path)

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[str]:
        """
        Legacy method: Takes query and list of text documents,
        returns top_n text documents sorted by Cross-Encoder score descending.
        """
        if not documents:
            return []

        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)

        # Handle single document prediction returning float scalar instead of array
        if not hasattr(scores, "__len__"):
            scores = [float(scores)]

        ranked = sorted(zip(documents, scores), key=lambda x: float(x[1]), reverse=True)
        return [doc for doc, _ in ranked[:top_n]]

    def rerank_candidates(
        self, query: str, candidates: List[Dict[str, Any]], top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate dictionaries containing at least a 'text' key.
        Appends 'reranker_score' to each dictionary and returns top_n candidates
        sorted by 'reranker_score' descending.
        """
        if not candidates:
            return []

        pairs = [[query, c.get("text", "")] for c in candidates]
        scores = self.model.predict(pairs)

        if not hasattr(scores, "__len__"):
            scores = [float(scores)]

        # Attach reranker_score to a shallow copy of candidates
        updated_candidates = []
        for cand, score in zip(candidates, scores):
            c_copy = dict(cand)
            c_copy["reranker_score"] = float(score)
            updated_candidates.append(c_copy)

        updated_candidates.sort(key=lambda x: x["reranker_score"], reverse=True)
        return updated_candidates[:top_n]