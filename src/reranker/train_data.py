import pandas as pd
from sentence_transformers import InputExample


def load_reranker_data(path, max_samples=2000):
    df = pd.read_csv(path)

    # limit dataset (important for Mac)
    df = df.sample(n=min(max_samples, len(df)), random_state=42)

    examples = []

    for _, row in df.iterrows():
        query = str(row["question"])
        pos_doc = str(row["answer"])

        # positive example (label = 1)
        examples.append(InputExample(texts=[query, pos_doc], label=1.0))

        # simple negative (random answer)
        neg_doc = df.sample(1).iloc[0]["answer"]
        examples.append(InputExample(texts=[query, str(neg_doc)], label=0.0))

    return examples