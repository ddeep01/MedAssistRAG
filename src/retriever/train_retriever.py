from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from torch.utils.data import DataLoader

from src.retriever.train_data import load_training_data


def main():
    print("Loading base model...")
    model = SentenceTransformer("BAAI/bge-small-en")

    # FORCE CPU (CRITICAL for Mac stability)
    model = model.to("cpu")

    print("Loading training data...")
    train_examples = load_training_data(
        "data/processed/qa_dataset_large.csv",
        max_samples=2000   # 🔥 KEY CHANGE
    )

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=4   # safe
    )

    train_loss = MultipleNegativesRankingLoss(model)

    print("Training started...")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=5,                # ✅ as you wanted
        warmup_steps=50,
        show_progress_bar=True
    )

    print("Saving model...")
    model.save("models/retriever_finetuned")

    print("Done!")


if __name__ == "__main__":
    main()