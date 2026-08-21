import os
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_RETRIEVAL_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "retrieval.yaml")
DEFAULT_CONFIDENCE_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "confidence.yaml")
DEFAULT_RETRY_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "retry.yaml")
DEFAULT_MEMORY_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "memory.yaml")
DEFAULT_SAFETY_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "safety.yaml")
DEFAULT_CITATIONS_CONFIG_PATH = os.path.join(BASE_DIR, "configs", "citations.yaml")

DEFAULT_RETRIEVAL_CONFIG = {
    "retrieval": {
        "mode": "hybrid",
        "dense_top_k": 20,
        "bm25_top_k": 20,
        "candidate_k": 20,
        "final_top_k": 5,
        "alpha": 0.5,
        "faiss_index_path": "data/embeddings/faiss_index.bin",
        "texts_path": "data/embeddings/texts.pkl",
        "bm25_index_path": "data/embeddings/bm25_index.pkl",
        "embedding_model": "BAAI/bge-small-en"
    },
    "reranker": {
        "enabled": True,
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "top_n": 5
    }
}

DEFAULT_CONFIDENCE_CONFIG = {
    "confidence": {
        "retrieval_weight": 0.5,
        "reranker_weight": 0.5,
        "high_threshold": 0.80,
        "medium_threshold": 0.60,
        "rank_weights": [0.5, 0.3, 0.2]
    }
}

DEFAULT_RETRY_CONFIG = {
    "retry": {
        "enabled": True,
        "max_retries": 2,
        "min_confidence_improvement": 0.05
    },
    "query_rewriting": {
        "enabled": True
    }
}

DEFAULT_MEMORY_CONFIG = {
    "memory": {
        "enabled": True,
        "short_term": {
            "max_messages": 10
        },
        "entities": {
            "enabled": True
        }
    }
}


DEFAULT_SAFETY_CONFIG = {
    "safety": {
        "enabled": True,
        "fallback_risk_level": "HIGH",
        "risk_levels": {
            "low": "RAG",
            "medium": "CREATE_TICKET",
            "high": "SAFETY_WARNING"
        }
    }
}

DEFAULT_CITATIONS_CONFIG = {
    "citations": {
        "enabled": True,
        "max_evidence_chunks": 5,
        "strip_invalid_citations": True
    }
}


def load_retrieval_config(config_path=None):
    """Loads retrieval configuration from YAML file or returns defaults."""
    if config_path is None:
        config_path = DEFAULT_RETRIEVAL_CONFIG_PATH

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config:
                    return config
        except Exception as e:
            print(f"[Warning] Failed to load config from {config_path}: {e}. Using defaults.")

    return DEFAULT_RETRIEVAL_CONFIG


def load_confidence_config(config_path=None):
    """Loads confidence configuration from YAML file or returns defaults."""
    if config_path is None:
        config_path = DEFAULT_CONFIDENCE_CONFIG_PATH

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config:
                    return config
        except Exception as e:
            print(f"[Warning] Failed to load confidence config from {config_path}: {e}. Using defaults.")

    return DEFAULT_CONFIDENCE_CONFIG


def load_retry_config(config_path=None):
    """Loads retry configuration from YAML file or returns defaults."""
    if config_path is None:
        config_path = DEFAULT_RETRY_CONFIG_PATH

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config:
                    return config
        except Exception as e:
            print(f"[Warning] Failed to load retry config from {config_path}: {e}. Using defaults.")

    return DEFAULT_RETRY_CONFIG


def load_memory_config(config_path=None):
    """Loads memory configuration from YAML file or returns defaults."""
    if config_path is None:
        config_path = DEFAULT_MEMORY_CONFIG_PATH

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config:
                    return config
        except Exception as e:
            print(f"[Warning] Failed to load memory config from {config_path}: {e}. Using defaults.")

    return DEFAULT_MEMORY_CONFIG


def load_safety_config(config_path=None):
    """Loads safety configuration from YAML file or returns defaults."""
    if config_path is None:
        config_path = DEFAULT_SAFETY_CONFIG_PATH

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config:
                    return config
        except Exception as e:
            print(f"[Warning] Failed to load safety config from {config_path}: {e}. Using defaults.")

    return DEFAULT_SAFETY_CONFIG


def load_citations_config(config_path=None):
    """Loads citations configuration from YAML file or returns defaults."""
    if config_path is None:
        config_path = DEFAULT_CITATIONS_CONFIG_PATH

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                if config:
                    return config
        except Exception as e:
            print(f"[Warning] Failed to load citations config from {config_path}: {e}. Using defaults.")

    return DEFAULT_CITATIONS_CONFIG
