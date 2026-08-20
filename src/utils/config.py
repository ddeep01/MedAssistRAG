import os
import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "retrieval.yaml"
)

DEFAULT_CONFIG = {
    "retrieval": {
        "mode": "hybrid",
        "dense_top_k": 20,
        "bm25_top_k": 20,
        "candidate_k": 20,
        "final_top_k": 5,
        "alpha": 0.5,
        "faiss_index_path": "data/embeddings/faiss_index_after_retriever_finetuning.bin",
        "faiss_index_fallback_path": "data/embeddings/faiss_index.bin",
        "texts_path": "data/embeddings/texts.pkl",
        "bm25_index_path": "data/embeddings/bm25_index.pkl",
        "embedding_model": "BAAI/bge-small-en",
        "finetuned_embedding_model": "models/retriever_finetuned"
    },
    "reranker": {
        "enabled": True,
        "model_path": "models/reranker_finetuned",
        "fallback_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_n": 5
    }
}


def load_retrieval_config(config_path=None):
    """Loads retrieval configuration from YAML file or returns defaults."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config:
                    return config
        except Exception as e:
            print(f"[Warning] Failed to load config from {config_path}: {e}. Using defaults.")

    return DEFAULT_CONFIG
