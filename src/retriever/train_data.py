import pandas as pd
from sentence_transformers import InputExample


def load_training_data(path, max_samples=2000):
    df = pd.read_csv(path)

    # 🔥 LIMIT DATASET SIZE
    df = df.sample(n=min(max_samples, len(df)), random_state=42)

    train_examples = []

    for _, row in df.iterrows():
        query = str(row["question"])
        doc = str(row["answer"])

        train_examples.append(
            InputExample(texts=[query, doc])
        )

    return train_examples