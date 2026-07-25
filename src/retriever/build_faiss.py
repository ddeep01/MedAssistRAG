import faiss
import numpy as np
import pickle

def build_index(embeddings):
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index

def save_index(index, texts):
    faiss.write_index(index, "data/embeddings/faiss_index_after_retriever_finetuning.bin")

    with open("data/embeddings/texts.pkl", "wb") as f:
        pickle.dump(texts, f)