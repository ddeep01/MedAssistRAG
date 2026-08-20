import os
import sys
import time
import math
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Add workspace root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.retriever.hybrid_retriever import HybridRetriever
from src.retriever.bm25_retriever import BM25Retriever
from src.utils.config import load_retrieval_config
import faiss
from sentence_transformers import SentenceTransformer


def compute_recall_at_k(retrieved_docs: List[str], ground_truth: str, k: int) -> float:
    top_k_docs = retrieved_docs[:k]
    gt_words = set(ground_truth.lower().split())
    if not gt_words:
        return 0.0

    for doc in top_k_docs:
        doc_words = set(doc.lower().split())
        overlap = len(gt_words & doc_words)
        if overlap / max(1, len(gt_words)) >= 0.25 or ground_truth.lower() in doc.lower():
            return 1.0
    return 0.0


def compute_precision_at_k(retrieved_docs: List[str], ground_truth: str, k: int) -> float:
    top_k_docs = retrieved_docs[:k]
    if not top_k_docs:
        return 0.0

    gt_words = set(ground_truth.lower().split())
    if not gt_words:
        return 0.0

    hits = 0
    for doc in top_k_docs:
        doc_words = set(doc.lower().split())
        overlap = len(gt_words & doc_words)
        if overlap / max(1, len(gt_words)) >= 0.25 or ground_truth.lower() in doc.lower():
            hits += 1

    return hits / len(top_k_docs)


def compute_mrr(retrieved_docs: List[str], ground_truth: str) -> float:
    gt_words = set(ground_truth.lower().split())
    if not gt_words:
        return 0.0

    for rank_idx, doc in enumerate(retrieved_docs, start=1):
        doc_words = set(doc.lower().split())
        overlap = len(gt_words & doc_words)
        if overlap / max(1, len(gt_words)) >= 0.25 or ground_truth.lower() in doc.lower():
            return 1.0 / rank_idx
    return 0.0


def compute_ndcg_at_k(retrieved_docs: List[str], ground_truth: str, k: int) -> float:
    top_k_docs = retrieved_docs[:k]
    gt_words = set(ground_truth.lower().split())
    if not gt_words:
        return 0.0

    dcg = 0.0
    for rank_idx, doc in enumerate(top_k_docs, start=1):
        doc_words = set(doc.lower().split())
        overlap = len(gt_words & doc_words)
        rel = 1.0 if (overlap / max(1, len(gt_words)) >= 0.25 or ground_truth.lower() in doc.lower()) else 0.0
        dcg += rel / math.log2(rank_idx + 1)

    idcg = 1.0 / math.log2(2)
    return dcg / idcg if idcg > 0 else 0.0


def run_experiments(eval_dataset: List[Dict[str, str]], retriever: HybridRetriever) -> pd.DataFrame:
    experiments = [
        {"name": "EXP 1: Dense FAISS", "mode": "dense", "enable_reranker": False},
        {"name": "EXP 2: FAISS + Cross-Encoder", "mode": "dense", "enable_reranker": True},
        {"name": "EXP 3: BM25", "mode": "bm25", "enable_reranker": False},
        {"name": "EXP 4: FAISS + BM25 (Fusion)", "mode": "hybrid", "enable_reranker": False},
        {"name": "EXP 5: FAISS + BM25 + Cross-Encoder", "mode": "hybrid", "enable_reranker": True},
    ]

    results = []

    for exp in experiments:
        exp_name = exp["name"]
        mode = exp["mode"]
        enable_reranker = exp["enable_reranker"]

        recalls_5 = []
        recalls_10 = []
        recalls_20 = []
        precisions_5 = []
        mrrs = []
        ndcgs_10 = []
        latencies = []

        for sample in eval_dataset:
            query = sample["question"]
            gt = sample["answer"]

            start_t = time.time()
            retrieved = retriever.search(
                query=query,
                mode=mode,
                final_top_k=20,
                enable_reranker=enable_reranker
            )
            lat = time.time() - start_t

            doc_texts = [r["text"] for r in retrieved]

            recalls_5.append(compute_recall_at_k(doc_texts, gt, 5))
            recalls_10.append(compute_recall_at_k(doc_texts, gt, 10))
            recalls_20.append(compute_recall_at_k(doc_texts, gt, 20))
            precisions_5.append(compute_precision_at_k(doc_texts, gt, 5))
            mrrs.append(compute_mrr(doc_texts, gt))
            ndcgs_10.append(compute_ndcg_at_k(doc_texts, gt, 10))
            latencies.append(lat)

        n = max(1, len(eval_dataset))
        results.append({
            "Experiment": exp_name,
            "Recall@5": round(sum(recalls_5) / n, 4),
            "Recall@10": round(sum(recalls_10) / n, 4),
            "Recall@20": round(sum(recalls_20) / n, 4),
            "Precision@5": round(sum(precisions_5) / n, 4),
            "MRR": round(sum(mrrs) / n, 4),
            "NDCG@10": round(sum(ndcgs_10) / n, 4),
            "Avg Latency (ms)": round((sum(latencies) / n) * 1000, 2),
        })

    return pd.DataFrame(results)


def main():
    print("Initializing evaluation...")
    config = load_retrieval_config()

    sample_eval_data = [
        {
            "question": "What are the side effects of metformin?",
            "answer": "Metformin side effects include gastrointestinal upset, nausea, diarrhea, abdominal pain, and rare risk of lactic acidosis."
        },
        {
            "question": "What problems can long-term high blood pressure cause?",
            "answer": "Hypertension causes long-term complications such as heart failure, stroke, myocardial infarction, and kidney failure."
        },
        {
            "question": "How is type 2 diabetes managed?",
            "answer": "Type 2 diabetes is managed with lifestyle modifications, regular exercise, blood glucose monitoring, and oral antidiabetic drugs like metformin."
        },
        {
            "question": "What are symptoms of COVID-19 infection?",
            "answer": "Common symptoms of SARS-CoV-2 coronavirus infection include fever, cough, fatigue, dyspnea, and loss of taste or smell."
        }
    ]

    corpus_texts = [
        "Metformin hydrochloride is an oral antidiabetic drug used to treat type 2 diabetes, manage HbA1c levels, with side effects of nausea and diarrhea.",
        "Hypertension or long-term high blood pressure increases the risk of stroke, myocardial infarction, chronic kidney disease, and heart failure.",
        "SARS-CoV-2 is the coronavirus strain responsible for the COVID-19 pandemic causing fever, cough, and acute respiratory distress.",
        "Lisinopril is an ACE inhibitor used in the treatment of hypertension and congestive heart failure.",
        "Common side effects of metformin include nausea, abdominal pain, diarrhea, and rare lactic acidosis."
    ]

    # Build temporary in-memory FAISS & BM25 index for benchmark evaluation
    emb_model = SentenceTransformer("BAAI/bge-small-en")
    embeddings = emb_model.encode(corpus_texts, normalize_embeddings=True)
    dim = embeddings.shape[1]
    faiss_idx = faiss.IndexFlatL2(dim)
    faiss_idx.add(embeddings)

    bm25_ret = BM25Retriever(texts=corpus_texts)

    retriever = HybridRetriever(
        texts=corpus_texts,
        faiss_index=faiss_idx,
        bm25_retriever=bm25_ret
    )

    print("\nRunning Retrieval Experiments...")
    df_results = run_experiments(sample_eval_data, retriever)

    print("\n========================================================")
    print("          RETRIEVAL EXPERIMENT EVALUATION RESULTS        ")
    print("========================================================")
    print(df_results.to_string(index=False))
    print("========================================================\n")

    output_dir = "evaluation/reports"
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "hybrid_retrieval_experiments.csv")
    df_results.to_csv(report_file, index=False)
    print(f"Results saved to {report_file}")


if __name__ == "__main__":
    main()
