import json
import pandas as pd
from tqdm import tqdm

from src.retriever.search import Retriever


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "data/processed/qa_dataset_large.csv"
OUTPUT_FILE = "data/finetune_data.json"

MAX_SAMPLES = 200
TOP_K = 3


# ============================================================
# Load QA Dataset
# ============================================================

def load_dataset():
    df = pd.read_csv(INPUT_FILE)

    data = []

    for _, row in df.iterrows():

        question = str(row["question"]).strip()
        answer = str(row["answer"]).strip()
        source = str(row["source"]).strip()

        if not question or not answer:
            continue

        data.append({
            "question": question,
            "answer": answer,
            "source": source
        })

    return data


# ============================================================
# Create TinyLlama Fine-Tuning Dataset
# ============================================================

def create_training_data(data, max_samples=MAX_SAMPLES):

    formatted = []

    # Initialize retriever once
    retriever = Retriever()

    for i, sample in enumerate(tqdm(data, desc="Creating training data")):

        if i >= max_samples:
            break

        question = sample["question"]
        answer = sample["answer"]
        source = sample["source"]

        if not question or not answer:
            continue

        # ----------------------------------------------------
        # Retrieve context using existing FAISS retriever
        # ----------------------------------------------------

        docs = retriever.search(
            question,
            k=TOP_K
        )

        context = " ".join(docs)

        # ----------------------------------------------------
        # TinyLlama prompt format
        # ----------------------------------------------------

        prompt = f"""### Instruction:
You are a medical assistant. Answer the user's question using only the provided context.

If the answer cannot be found in the context, say "I don't know."

### Context:
{context}

### Question:
{question}

### Answer:
"""

        # ----------------------------------------------------
        # Final training text
        # ----------------------------------------------------

        output = f"{answer} (Source: {source})"

        formatted.append({
            "prompt": prompt,
            "response": output
        })

    return formatted


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("Loading QA dataset...")

    data = load_dataset()

    print(f"Loaded {len(data)} QA samples.")

    formatted = create_training_data(
        data,
        max_samples=MAX_SAMPLES
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            formatted,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"✅ Generated {len(formatted)} TinyLlama training samples!"
    )

    print(f"✅ Saved to: {OUTPUT_FILE}")