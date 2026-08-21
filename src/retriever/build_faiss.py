import faiss
import numpy as np
import pickle

def build_index(embeddings):
    """
    Builds a FAISS IndexFlatIP index using L2-normalized document embeddings.
    Because vectors are L2-normalized, inner product corresponds to cosine similarity.
    """
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index

def save_index(index, texts):
    faiss.write_index(index, "data/embeddings/faiss_index_after_retriever_finetuning.bin")

    with open("data/embeddings/texts.pkl", "wb") as f:
        pickle.dump(texts, f)