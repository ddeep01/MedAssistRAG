import os
import pickle
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("MedAssistRAG.DenseRetriever")


def load_embedding_model(model_name: str = "BAAI/bge-small-en") -> SentenceTransformer:
    """Loads SentenceTransformer model for dense embedding generation."""
    return SentenceTransformer(model_name)


def create_embeddings(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Generates L2-normalized dense embeddings for a corpus of texts."""
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Builds a FAISS IndexFlatIP index using L2-normalized embeddings.
    Inner product on L2-normalized vectors mathematically equals cosine similarity.
    """
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_faiss_index(
    index: faiss.IndexFlatIP,
    texts: List[str],
    index_path: str = "data/embeddings/faiss_index.bin",
    texts_path: str = "data/embeddings/texts.pkl"
) -> None:
    """Saves FAISS index and texts pickle to disk."""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    with open(texts_path, "wb") as f:
        pickle.dump(texts, f)


class DenseRetriever:
    """
    Dense Vector Retriever utilizing BAAI/bge-small-en embeddings
    and FAISS IndexFlatIP for cosine similarity search.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        index_path: str = "data/embeddings/faiss_index.bin",
        texts_path: str = "data/embeddings/texts.pkl",
        model: Optional[SentenceTransformer] = None,
        index: Optional[faiss.IndexFlatIP] = None,
        texts: Optional[List[str]] = None
    ):
        self.model_name = model_name
        self.index_path = index_path
        self.texts_path = texts_path

        self.model = model or load_embedding_model(self.model_name)
        self.texts = texts
        self.index = index

        self._initialize_resources()

    def _initialize_resources(self):
        """Loads corpus texts and FAISS index from disk if not supplied."""
        if self.texts is None and os.path.exists(self.texts_path):
            try:
                with open(self.texts_path, "rb") as f:
                    self.texts = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load texts from {self.texts_path}: {e}")
                self.texts = []

        if self.index is None and os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception as e:
                logger.warning(f"Failed to load FAISS index from {self.index_path}: {e}")
                self.index = None

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Performs FAISS dense search using cosine similarity (inner product on L2-normalized vectors).
        Returns List of dicts containing index, text, and cosine similarity score.
        """
        if not query or not query.strip() or self.index is None or not self.texts:
            return []

        try:
            query_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        except TypeError:
            query_emb = self.model.encode([query], normalize_embeddings=True)

        if not isinstance(query_emb, np.ndarray):
            query_emb = np.array(query_emb, dtype=np.float32)

        query_emb = np.ascontiguousarray(query_emb.astype(np.float32))
        faiss.normalize_L2(query_emb)

        scores, indices = self.index.search(query_emb, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append({
                "index": int(idx),
                "text": self.texts[idx],
                "score": float(score)  # Cosine similarity in [-1.0, 1.0]
            })

        return results
