import pandas as pd


def load_generator_data(path, max_samples=5000):
    df = pd.read_csv(path)

    # limit dataset
    df = df.sample(n=min(max_samples, len(df)), random_state=42)

    dataset = []

    for _, row in df.iterrows():
        question = str(row["question"])
        answer = str(row["answer"])

        prompt = f"""
            You are a medical assistant.

            Answer the question based on your knowledge.

            Question:
            {question}

            Answer:
        """

        dataset.append({
            "prompt": prompt,
            "response": answer
        })

    return dataset