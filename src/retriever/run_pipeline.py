from src.data_utils.load_data import load_data
from src.data_utils.chunk_data import chunk_text
from src.retriever.embed import load_model, create_embeddings
from src.retriever.build_faiss import build_index, save_index

def main():
    print("Loading data...")
    df = load_data("data/processed/qa_dataset_large.csv")

    print("Chunking...")
    texts = chunk_text(df)

    print("Loading embedding model...")
    model = load_model()

    print("Creating embeddings...")
    embeddings = create_embeddings(model, texts)

    print("Building FAISS index...")
    index = build_index(embeddings)

    print("Saving...")
    save_index(index, texts)

    print("Done!")

if __name__ == "__main__":
    main()