from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        # Lightweight + good performance
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        #self.model = CrossEncoder("models/reranker_finetuned")

    def rerank(self, query, documents, top_n=5):
        pairs = [[query, doc] for doc in documents]

        scores = self.model.predict(pairs)

        # sort by score (descending)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked[:top_n]]