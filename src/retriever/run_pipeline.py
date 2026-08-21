from src.data_utils.load_data import load_data
from src.data_utils.chunk_data import chunk_text
from src.retriever.dense_retriever import (
    load_embedding_model,
    create_embeddings,
    build_faiss_index,
    save_faiss_index
)
from src.retriever.bm25_retriever import build_bm25, save_bm25


def main():
    print("Loading data...")
    df = load_data("data/processed/qa_dataset_large.csv")

    print("Chunking text...")
    texts = chunk_text(df)

    print("Loading embedding model (BAAI/bge-small-en)...")
    model = load_embedding_model()

    print("Creating dense embeddings...")
    embeddings = create_embeddings(model, texts)

    print("Building FAISS IndexFlatIP (Cosine Similarity)...")
    faiss_idx = build_faiss_index(embeddings)

    print("Saving FAISS index & texts...")
    save_faiss_index(faiss_idx, texts)

    print("Building & Saving BM25 index...")
    bm25_idx = build_bm25(texts)
    save_bm25(bm25_idx, "data/embeddings/bm25_index.pkl")

    print("✅ Retrieval index build pipeline completed successfully!")


if __name__ == "__main__":
    main()