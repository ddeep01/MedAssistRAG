import os
import pickle
import sys

# Ensure src is on pythonpath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retriever.bm25_retriever import build_bm25, save_bm25, load_bm25
from src.utils.config import load_retrieval_config


def main():
    config = load_retrieval_config()
    texts_path = config["retrieval"].get("texts_path", "data/embeddings/texts.pkl")
    bm25_index_path = config["retrieval"].get("bm25_index_path", "data/embeddings/bm25_index.pkl")

    if not os.path.exists(texts_path):
        print(f"[Error] Corpus texts file not found at: {texts_path}")
        print("Please build or place texts.pkl prior to building the BM25 index.")
        sys.exit(1)

    print(f"Loading texts from {texts_path}...")
    with open(texts_path, "rb") as f:
        texts = pickle.load(f)

    print(f"Building BM25 index for {len(texts)} documents...")
    bm25_index = build_bm25(texts)

    print(f"Saving BM25 index to {bm25_index_path}...")
    save_bm25(bm25_index, bm25_index_path)

    print(f"[SUCCESS] Successfully built and saved BM25 index with {len(texts)} documents!")


if __name__ == "__main__":
    main()
