from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader

from src.reranker.train_data import load_reranker_data


def main():
    print("Loading base reranker...")
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # FORCE CPU (important for Mac)
    model.model.to("cpu")

    print("Loading training data...")
    train_data = load_reranker_data(
        "data/processed/qa_dataset_large.csv",
        max_samples=2000
    )

    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=4
    )

    print("Training started...")

    model.fit(
        train_dataloader=train_dataloader,
        epochs=3,
        warmup_steps=50,
        show_progress_bar=True
    )

    print("Saving model...")
    model.save("models/reranker_finetuned")

    print("Done!")


if __name__ == "__main__":
    main()