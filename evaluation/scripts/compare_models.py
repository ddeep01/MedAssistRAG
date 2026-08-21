import argparse
from collections import Counter
from pathlib import Path
import re
import sys

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None):
        return iterable


# =========================
# PATH SETUP
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]

# ✅ USE YOUR SHARED SUBSET FILE HERE
DATA_PATH = BASE_DIR / "data" / "processed" / "qa_dataset_evaluation.csv"

RESULTS_PATH = BASE_DIR / "evaluation" / "reports" / "evaluation_results_finetuned.csv"
PLOT_PATH = BASE_DIR / "evaluation" / "plots" / "evaluation_plot_finetuned.png"

SIMILARITY_MODEL_NAME = "BAAI/bge-base-en-v1.5"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# =========================
# DATA LOADING
# =========================
def load_clean_data():
    df = pd.read_csv(DATA_PATH)
    return df.reset_index(drop=True)


def clean_text(text):
    if pd.isna(text):
        return ""
    return str(text).replace("\n", " ").strip()


# =========================
# METRICS
# =========================
def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def compute_f1(pred, truth):
    pred_tokens = normalize(pred).split()
    truth_tokens = normalize(truth).split()

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)


def exact_match(pred, truth):
    return int(normalize(pred) == normalize(truth))


def is_hallucinated(answer, context):
    answer_words = set(normalize(answer).split())
    context_words = set(normalize(context).split())

    overlap = len(answer_words & context_words)

    return int(overlap < 5)


def compute_bleu(pred, truth):
    from nltk.translate.bleu_score import sentence_bleu
    return sentence_bleu([normalize(truth).split()], normalize(pred).split())


# =========================
# SEMANTIC SIMILARITY
# =========================
def get_similarity_model():
    from sentence_transformers import SentenceTransformer

    if not hasattr(get_similarity_model, "_model"):
        get_similarity_model._model = SentenceTransformer(SIMILARITY_MODEL_NAME)
    return get_similarity_model._model


def semantic_similarity_batch(preds, truths):
    from sentence_transformers import util

    model = get_similarity_model()

    emb1 = model.encode(preds, convert_to_tensor=True)
    emb2 = model.encode(truths, convert_to_tensor=True)

    scores = util.cos_sim(emb1, emb2)

    return [float(scores[i][i]) for i in range(len(preds))]


# =========================
# BUILD YOUR RAG PIPELINE
# =========================
def build_rag_pipeline():
    from src.pipeline.rag_pipeline import RAGPipeline
    return RAGPipeline()


# =========================
# MAIN EVALUATION
# =========================
def evaluate_model():
    print("\nEvaluating your model on shared subset...\n")

    rag = build_rag_pipeline()
    df = load_clean_data()

    preds = []
    truths = []
    contexts = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = row["question"]
        ground_truth = row["answer"]

        result = rag.query(question)

        preds.append(result["answer"])
        truths.append(ground_truth)
        contexts.append(" ".join(result["sources"]))

    f1_scores = [compute_f1(p, t) for p, t in zip(preds, truths)]
    em_scores = [exact_match(p, t) for p, t in zip(preds, truths)]
    hallucinations = [is_hallucinated(p, c) for p, c in zip(preds, contexts)]
    bleu_scores = [compute_bleu(p, t) for p, t in zip(preds, truths)]
    similarities = semantic_similarity_batch(preds, truths)

    result = {
        "Model": "my_model",  # 👈 change name if needed
        "F1 Score": sum(f1_scores) / len(f1_scores),
        "Exact Match": sum(em_scores) / len(em_scores),
        "Hallucination Rate": sum(hallucinations) / len(hallucinations),
        "Semantic Similarity": sum(similarities) / len(similarities),
        "BLEU Score": sum(bleu_scores) / len(bleu_scores),
    }

    return result


# =========================
# SAVE + PLOT
# =========================
def plot_results(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        "F1 Score",
        "Exact Match",
        "Hallucination Rate",
        "Semantic Similarity",
        "BLEU Score",
    ]

    ax = df.set_index("Model")[metrics].plot(kind="bar")
    ax.set_title("Model Evaluation (Shared Subset)")
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()


def save_results(result):
    df = pd.DataFrame([result])
    df.to_csv(RESULTS_PATH, index=False)
    plot_results(df)
    return df


# =========================
# ENTRY POINT
# =========================
def main():
    result = evaluate_model()
    df = save_results(result)

    print("\nFinal Evaluation using Finetuned Models:\n")
    print(df)


if __name__ == "__main__":
    main()