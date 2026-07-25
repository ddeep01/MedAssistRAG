

import faiss
import pickle
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self):
        # Load FAISS index
        #self.index = faiss.read_index("data/embeddings/faiss_index.bin")
        self.index = faiss.read_index("data/embeddings/faiss_index_after_retriever_finetuning.bin")

        # Load stored texts
        with open("data/embeddings/texts.pkl", "rb") as f:
            self.texts = pickle.load(f)

        # Load embedding model
        #self.model = SentenceTransformer("BAAI/bge-small-en")
        self.model = SentenceTransformer("models/retriever_finetuned")

    def search(self, query, k=5):
        # Encode query
        q_emb = self.model.encode([query], normalize_embeddings=True)

        # Search FAISS
        D, I = self.index.search(q_emb, k)

        # Return top-k documents
        return [self.texts[i] for i in I[0]]


'''
import faiss
import pickle
from sentence_transformers import SentenceTransformer

from src.reranker.reranker import Reranker  # 🔥 NEW


class Retriever:
    def __init__(self):
        # Load FAISS index
        self.index = faiss.read_index("data/embeddings/faiss_index_after_retriever_finetuning.bin")

        # Load stored texts
        with open("data/embeddings/texts.pkl", "rb") as f:
            self.texts = pickle.load(f)

        # Load embedding model (your fine-tuned one)
        self.model = SentenceTransformer("models/retriever_finetuned")

        # 🔥 Initialize reranker
        self.reranker = Reranker()

    def search(self, query, k=30):
        # =========================
        # STEP 1: RETRIEVAL
        # =========================
        q_emb = self.model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(q_emb, k)

        candidates = [self.texts[i] for i in I[0]]

        # =========================
        # STEP 2: RERANKING
        # =========================
        reranked = self.reranker.rerank(query, candidates, top_n=5)

        return reranked
'''