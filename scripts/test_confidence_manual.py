import os
import sys

# Ensure workspace root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retriever.hybrid_retriever import HybridRetriever
from src.confidence.scorer import ConfidenceScorer
import faiss
from sentence_transformers import SentenceTransformer


def main():
    print("=" * 70)
    print("      PART 2 — CONFIDENCE-AWARE RETRIEVAL MANUAL QUERY TEST     ")
    print("=" * 70)

    # Medical corpus for end-to-end evaluation
    corpus_texts = [
        "Metformin hydrochloride is an oral antidiabetic drug used to treat type 2 diabetes and manage HbA1c levels. Side effects include nausea, abdominal pain, diarrhea, and rare lactic acidosis.",
        "Hypertension or long-term high blood pressure increases the risk of stroke, myocardial infarction, congestive heart failure, and chronic kidney disease.",
        "Diabetes mellitus symptoms include frequent urination (polyuria), excessive thirst (polydipsia), increased hunger (polyphagia), fatigue, and blurred vision.",
        "Asthma is a chronic respiratory condition characterized by airway inflammation, bronchospasm, wheezing, coughing, and shortness of breath.",
        "Lisinopril is an ACE inhibitor used in the treatment of hypertension and heart failure."
    ]

    # Initialize index and components
    emb_model = SentenceTransformer("BAAI/bge-small-en")
    embeddings = emb_model.encode(corpus_texts, normalize_embeddings=True)
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    faiss_idx = faiss.IndexFlatIP(dim)
    faiss_idx.add(embeddings)

    retriever = HybridRetriever(
        texts=corpus_texts,
        faiss_index=faiss_idx
    )
    scorer = ConfidenceScorer()

    queries = [
        "What are the symptoms of diabetes?",
        "What are the side effects of metformin?",
        "What complications can hypertension cause?",
        "What is asthma?",
        "Who won the 1998 FIFA World Cup in France?"
    ]

    for q_idx, q in enumerate(queries, start=1):
        print(f"\nQUERY {q_idx}: \"{q}\"")
        print("-" * 70)

        raw_candidates = retriever.search(q, mode="hybrid", final_top_k=5)
        eval_result = scorer.evaluate_retrieval(q, raw_candidates)

        print(f"Top Documents Retained: {len(eval_result['documents'])}")
        for d_idx, doc in enumerate(eval_result['documents'], start=1):
            doc_snippet = doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text']
            print(f"\n  [{d_idx}] Snippet: {doc_snippet}")
            print(f"      Hybrid Score:           {doc['hybrid_score']}")
            print(f"      Raw Reranker Score:     {doc['reranker_score']}")
            print(f"      Norm Reranker Score:    {doc['normalized_reranker_score']}")
            print(f"      Document Confidence:    {doc['document_confidence']}")

        print(f"\n  ==> FINAL CONFIDENCE:       {eval_result['confidence']}")
        print(f"  ==> CONFIDENCE LEVEL:       {eval_result['level']}")
        print(f"  ==> NEEDS RETRY / REWRITE:  {eval_result['needs_retry']}")
        print("-" * 70)


if __name__ == "__main__":
    main()
