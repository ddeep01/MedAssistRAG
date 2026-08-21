from typing import List, Dict, Any, Union, Optional
from src.retriever.hybrid_retriever import HybridRetriever
from src.config import load_retrieval_config


class Retriever:
    """
    Retriever facade that wraps HybridRetriever and preserves
    backward compatibility with legacy callers expecting list of text strings.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_retrieval_config(config_path)
        self.hybrid_retriever = HybridRetriever(config=self.config)

    def search(
        self,
        query: str,
        k: int = 5,
        mode: Optional[str] = None,
        alpha: Optional[float] = None,
        return_structured: bool = False
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """
        Searches the knowledge base using the configured retrieval mode (dense, bm25, hybrid).

        If return_structured is False (default): Returns List[str] (texts).
        If return_structured is True: Returns List[Dict] with scores & metadata.
        """
        results = self.hybrid_retriever.search(
            query=query,
            mode=mode,
            alpha=alpha,
            final_top_k=k
        )

        if return_structured:
            return results

        return [item["text"] for item in results]

    def search_structured(
        self,
        query: str,
        k: int = 5,
        mode: Optional[str] = None,
        alpha: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Direct helper to get structured results with detailed scores."""
        return self.search(query=query, k=k, mode=mode, alpha=alpha, return_structured=True)