import numpy as np
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load once
model = SentenceTransformer("all-MiniLM-L6-v2")


# 🔹 F1
def compute_f1(pred, truth):
    pred_tokens = pred.lower().split()
    truth_tokens = truth.lower().split()

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)


# 🔹 Exact Match
def exact_match(pred, truth):
    return int(pred.strip().lower() == truth.strip().lower())


# 🔹 BLEU (FIXED)
def compute_bleu(pred, truth):
    smoothie = SmoothingFunction().method1
    return sentence_bleu(
        [truth.split()],
        pred.split(),
        smoothing_function=smoothie
    )


# 🔹 Semantic Similarity
def semantic_similarity(pred, truth):
    emb1 = model.encode([pred])
    emb2 = model.encode([truth])
    return cosine_similarity(emb1, emb2)[0][0]


# 🔹 Grounding
def grounding_score(answer, context):
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())

    if len(answer_tokens) == 0:
        return 0

    return len(answer_tokens & context_tokens) / len(answer_tokens)


# 🔹 Hallucination
def hallucination_score(answer, context):
    return 1 - grounding_score(answer, context)