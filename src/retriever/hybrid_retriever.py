import os
import time
import pickle
import logging
from typing import List, Dict, Any, Optional

from src.retriever.dense_retriever import DenseRetriever
from src.retriever.bm25_retriever import BM25Retriever
from src.reranker.reranker import Reranker
from src.config import load_retrieval_config

logger = logging.getLogger("MedAssistRAG.HybridRetriever")


def min_max_normalize(scores: List[float]) -> List[float]:
    """
    Applies min-max score normalization to a list of floats.
    Handles edge case where max_score == min_score.
    """
    if not scores:
        return []

    min_s = min(scores)
    max_s = max(scores)

    if max_s == min_s:
        return [1.0 if min_s > 0 else 0.5 for _ in scores]

    return [(s - min_s) / (max_s - min_s) for s in scores]


class HybridRetriever:
    """
    Hybrid Retriever combining FAISS Dense Vector Search, BM25 Keyword Search,
    Score Normalization, Weighted Score Fusion, and Cross-Encoder Reranking.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        texts: Optional[List[str]] = None,
        faiss_index: Optional[Any] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        reranker: Optional[Reranker] = None,
    ):
        self.config = config or load_retrieval_config()
        r_cfg = self.config.get("retrieval", {})
        rr_cfg = self.config.get("reranker", {})

        self.mode = r_cfg.get("mode", "hybrid")
        self.dense_top_k = r_cfg.get("dense_top_k", 20)
        self.bm25_top_k = r_cfg.get("bm25_top_k", 20)
        self.candidate_k = r_cfg.get("candidate_k", 20)
        self.final_top_k = r_cfg.get("final_top_k", 5)
        self.alpha = r_cfg.get("alpha", 0.5)

        # Paths & Models
        self.texts_path = r_cfg.get("texts_path", "data/embeddings/texts.pkl")
        self.faiss_index_path = r_cfg.get("faiss_index_path", "data/embeddings/faiss_index.bin")
        self.bm25_index_path = r_cfg.get("bm25_index_path", "data/embeddings/bm25_index.pkl")
        self.model_path = r_cfg.get("embedding_model", "BAAI/bge-small-en")

        # Components initialization
        self.dense_retriever = DenseRetriever(
            model_name=self.model_path,
            index_path=self.faiss_index_path,
            texts_path=self.texts_path,
            index=faiss_index,
            texts=texts
        )

        self.bm25_retriever = bm25_retriever or BM25Retriever(
            texts=self.dense_retriever.texts,
            index_path=self.bm25_index_path
        )

        # Reranker
        self.reranker_enabled = rr_cfg.get("enabled", True)
        self.reranker = reranker if reranker is not None else (Reranker() if self.reranker_enabled else None)

    @property
    def texts(self):
        return self.dense_retriever.texts

    @texts.setter
    def texts(self, value):
        self.dense_retriever.texts = value

    @property
    def faiss_index(self):
        return self.dense_retriever.index

    @faiss_index.setter
    def faiss_index(self, value):
        self.dense_retriever.index = value

    @property
    def model(self):
        return self.dense_retriever.model

    @model.setter
    def model(self, value):
        self.dense_retriever.model = value

    def search_dense(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Performs FAISS dense search using cosine similarity (IndexFlatIP)."""
        return self.dense_retriever.search(query, top_k=top_k)


    def search_bm25(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Performs BM25 keyword search."""
        if not query or not self.bm25_retriever:
            return []
        return self.bm25_retriever.search(query, top_k=top_k)

    def search(
        self,
        query: str,
        mode: Optional[str] = None,
        alpha: Optional[float] = None,
        final_top_k: Optional[int] = None,
        enable_reranker: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes search pipeline according to mode:
        - 'dense': FAISS search (+ optional reranking)
        - 'bm25': BM25 search (+ optional reranking)
        - 'hybrid': FAISS + BM25 + Score Normalization + Weighted Fusion (+ optional reranking)
        """
        start_time = time.time()
        curr_mode = mode or self.mode
        curr_alpha = alpha if alpha is not None else self.alpha
        curr_final_k = final_top_k if final_top_k is not None else self.final_top_k
        use_reranker = enable_reranker if enable_reranker is not None else self.reranker_enabled

        if not query or not query.strip():
            return []

        # ==========================================
        # MODE 1: DENSE FAISS ONLY
        # ==========================================
        if curr_mode == "dense":
            dense_res = self.search_dense(query, top_k=self.candidate_k)
            candidates = []
            for item in dense_res:
                candidates.append({
                    "index": item["index"],
                    "text": item["text"],
                    "dense_score": item["score"],
                    "bm25_score": 0.0,
                    "hybrid_score": item["score"]
                })

        # ==========================================
        # MODE 2: BM25 ONLY
        # ==========================================
        elif curr_mode == "bm25":
            bm25_res = self.search_bm25(query, top_k=self.candidate_k)
            candidates = []
            for item in bm25_res:
                candidates.append({
                    "index": item["index"],
                    "text": item["text"],
                    "dense_score": 0.0,
                    "bm25_score": item["score"],
                    "hybrid_score": item["score"]
                })

        # ==========================================
        # MODE 3: HYBRID (FAISS + BM25 + FUSION)
        # ==========================================
        else:
            dense_res = self.search_dense(query, top_k=self.dense_top_k)
            bm25_res = self.search_bm25(query, top_k=self.bm25_top_k)

            # Collect candidates and raw scores by document index
            candidates_dict = {}

            for d in dense_res:
                idx = d["index"]
                candidates_dict[idx] = {
                    "index": idx,
                    "text": d["text"],
                    "raw_dense": d["score"],
                    "raw_bm25": 0.0
                }

            for b in bm25_res:
                idx = b["index"]
                if idx in candidates_dict:
                    candidates_dict[idx]["raw_bm25"] = b["score"]
                else:
                    candidates_dict[idx] = {
                        "index": idx,
                        "text": b["text"],
                        "raw_dense": 0.0,
                        "raw_bm25": b["score"]
                    }

            if not candidates_dict:
                return []

            candidate_list = list(candidates_dict.values())

            # Extract raw scores for normalization
            raw_dense_scores = [c["raw_dense"] for c in candidate_list]
            raw_bm25_scores = [c["raw_bm25"] for c in candidate_list]

            # Min-Max Normalization
            norm_dense_scores = min_max_normalize(raw_dense_scores)
            norm_bm25_scores = min_max_normalize(raw_bm25_scores)

            # Weighted Score Fusion
            for i, c in enumerate(candidate_list):
                c["dense_score"] = float(norm_dense_scores[i])
                c["bm25_score"] = float(norm_bm25_scores[i])
                c["hybrid_score"] = float(curr_alpha * c["dense_score"] + (1.0 - curr_alpha) * c["bm25_score"])

                # Remove raw internal keys
                del c["raw_dense"]
                del c["raw_bm25"]

            # Sort candidate pool by hybrid score descending
            candidate_list.sort(key=lambda x: x["hybrid_score"], reverse=True)
            candidates = candidate_list[:self.candidate_k]

        # ==========================================
        # STEP 4: CROSS-ENCODER RERANKING
        # ==========================================
        if use_reranker and self.reranker:
            final_results = self.reranker.rerank_candidates(query, candidates, top_n=curr_final_k)
        else:
            final_results = candidates[:curr_final_k]

        elapsed = time.time() - start_time

        # Debug logging
        if logger.isEnabledFor(logging.DEBUG):
            top_dense = max([c.get("dense_score", 0.0) for c in final_results], default=0.0)
            top_bm25 = max([c.get("bm25_score", 0.0) for c in final_results], default=0.0)
            top_hybrid = max([c.get("hybrid_score", 0.0) for c in final_results], default=0.0)
            top_reranker = max([c.get("reranker_score", 0.0) for c in final_results if "reranker_score" in c], default=0.0)
            logger.debug(
                f"[HybridRetriever] query='{query}' mode={curr_mode} alpha={curr_alpha} "
                f"candidates={len(candidates)} final={len(final_results)} latency={elapsed:.4f}s "
                f"top_dense={top_dense:.4f} top_bm25={top_bm25:.4f} top_hybrid={top_hybrid:.4f} top_reranker={top_reranker:.4f}"
            )

        return final_results
