import time

from .advanced_metrics import (
    compute_f1,
    exact_match,
    compute_bleu,
    semantic_similarity,
    grounding_score,
    hallucination_score
)


def evaluate_model(name, dataset, retriever, generator):
    total = {
        "EM": 0,
        "F1": 0,
        "BLEU": 0,
        "Semantic": 0,
        "Grounding": 0,
        "Hallucination": 0,
        "Latency": 0
    }

    for sample in dataset:
        q = sample["question"]
        gt = sample["answer"]

        start = time.time()

        docs = retriever.retrieve(q)
        context = " ".join(docs)

        pred = generator.generate(q, context)

        latency = time.time() - start

        total["EM"] += exact_match(pred, gt)
        total["F1"] += compute_f1(pred, gt)
        total["BLEU"] += compute_bleu(pred, gt)
        total["Semantic"] += semantic_similarity(pred, gt)
        total["Grounding"] += grounding_score(pred, context)
        total["Hallucination"] += hallucination_score(pred, context)
        total["Latency"] += latency

    n = len(dataset)

    return {
        "Model": name,
        "EM": total["EM"] / n,
        "F1": total["F1"] / n,
        "BLEU": total["BLEU"] / n,
        "Semantic": total["Semantic"] / n,
        "Grounding": total["Grounding"] / n,
        "Hallucination": total["Hallucination"] / n,
        "Latency (sec)": total["Latency"] / n
    }