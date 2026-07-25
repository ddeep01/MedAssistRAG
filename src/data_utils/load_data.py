import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    # basic cleaning
    df = df.dropna()
    df = df.drop_duplicates()

    # combine question + answer (IMPORTANT for retrieval)
    df["text"] = df["question"] + " " + df["answer"]

    return df

if __name__ == "__main__":
    df = load_data("data/processed/qa_dataset_large.csv")
    print(df.head())