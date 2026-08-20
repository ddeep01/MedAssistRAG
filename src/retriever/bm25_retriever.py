import os
import re
import pickle
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


def tokenize_medical_text(text: str) -> List[str]:
    """
    Medical-aware tokenizer that lowercases and extracts alphanumeric words
    including hyphenated medical terms (e.g. SARS-CoV-2, COVID-19, HbA1c, metformin).
    """
    if not text:
        return []
    # Match words and hyphenated compound terms (e.g., SARS-CoV-2, long-term, HbA1c)
    tokens = re.findall(r'[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*', text.lower())
    return tokens


def build_bm25(texts: List[str]) -> BM25Okapi:
    """Builds a BM25Okapi index from a list of corpus texts."""
    tokenized_corpus = [tokenize_medical_text(t) for t in texts]
    return BM25Okapi(tokenized_corpus)


def save_bm25(bm25_index: BM25Okapi, filepath: str) -> None:
    """Saves the BM25Okapi index object to disk via pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(bm25_index, f)


def load_bm25(filepath: str) -> BM25Okapi:
    """Loads a BM25Okapi index object from disk."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


class BM25Retriever:
    """
    BM25 retriever using rank-bm25 for keyword-based search.
    Preserves 1-to-1 document index mapping with the input corpus texts.
    """

    def __init__(self, texts: List[str] = None, index_path: str = None):
        self.texts = texts if texts is not None else []
        self.bm25 = None
        self.index_path = index_path

        if index_path and os.path.exists(index_path):
            self.bm25 = load_bm25(index_path)
        elif self.texts:
            self.bm25 = build_bm25(self.texts)

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Searches the BM25 index with the given query.

        Returns list of structured dicts:
        {
            "index": 123,
            "text": "...",
            "score": float
        }
        """
        if not query or not query.strip():
            return []

        if not self.bm25:
            if self.index_path and os.path.exists(self.index_path):
                self.bm25 = load_bm25(self.index_path)
            elif self.texts:
                self.bm25 = build_bm25(self.texts)
            else:
                return []

        tokenized_query = tokenize_medical_text(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by BM25 score in descending order
        # Ensure top_k does not exceed number of documents
        top_k = min(top_k, len(scores))
        if top_k <= 0:
            return []

        # Sort indices by score descending
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc_idx = int(idx)
            score_val = float(scores[doc_idx])
            text_val = self.texts[doc_idx] if doc_idx < len(self.texts) else ""
            results.append({
                "index": doc_idx,
                "text": text_val,
                "score": score_val
            })

        return results
