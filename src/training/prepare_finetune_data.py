import json
from tqdm import tqdm

# ✅ Correct import
from src.retriever.search import Retriever


import pandas as pd

import pandas as pd

def load_dataset():
    df = pd.read_csv("data/processed/qa_dataset_large.csv")

    data = []

    for _, row in df.iterrows():
        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()

        if question and answer:
            data.append({
            "question": question,
            "answer": f"{answer} (Source: {row['source']})"
})
    return data


def create_training_data(data, max_samples=200):
    formatted = []

    # ✅ Initialize retriever ONCE (important for speed)
    retriever = Retriever()

    for i, sample in enumerate(tqdm(data)):
        if i >= max_samples:
            break

        # Extract fields safely
        question = sample.get("question", "")
        answer = sample.get("answer", "No answer available")

        if not question:
            continue

        # ✅ Retrieve context from FAISS
        docs = retriever.search(question, k=3)
        context = " ".join(docs)

        # ✅ Format for training
        formatted.append({
            "input": f"Question: {question}\nContext: {context}",
            "output": f"{answer} (Source: doc_1)"
        })

    return formatted


if __name__ == "__main__":
    data = load_dataset()

    formatted = create_training_data(data, max_samples=200)

    with open("data/finetune_data.json", "w") as f:
        json.dump(formatted, f, indent=2)

    print(f"✅ Generated {len(formatted)} REAL samples!")