import pandas as pd

from src.retriever.search import Retriever


# ============================================================
# Configuration
# ============================================================

DEFAULT_MAX_SAMPLES = 5000
TOP_K = 3


# ============================================================
# Load Generator Training Data
# ============================================================

def load_generator_data(
    path,
    max_samples=DEFAULT_MAX_SAMPLES
):
    df = pd.read_csv(path)

    # Limit dataset size
    df = df.sample(
        n=min(max_samples, len(df)),
        random_state=42
    )

    # Initialize retriever once
    retriever = Retriever()

    dataset = []

    for _, row in df.iterrows():

        question = str(
            row["question"]
        ).strip()

        answer = str(
            row["answer"]
        ).strip()

        if not question or not answer:
            continue

        # ----------------------------------------------------
        # Retrieve relevant context
        # ----------------------------------------------------

        docs = retriever.search(
            question,
            k=TOP_K
        )

        context = " ".join(docs)

        # ----------------------------------------------------
        # TinyLlama instruction prompt
        # ----------------------------------------------------

        prompt = f"""### Instruction:
You are a medical assistant.

Use ONLY the provided context to answer the question.

If the answer cannot be found in the context, say "I don't know."

### Context:
{context}

### Question:
{question}

### Answer:
"""

        dataset.append({
            "prompt": prompt,
            "response": answer
        })

    return dataset