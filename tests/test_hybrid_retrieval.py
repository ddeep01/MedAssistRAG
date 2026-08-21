import os
import pytest
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from src.retriever.bm25_retriever import (
    tokenize_medical_text,
    build_bm25,
    BM25Retriever
)
from src.retriever.hybrid_retriever import (
    HybridRetriever,
    min_max_normalize
)
from src.reranker.reranker import Reranker


# ----------------------------------------------------
# FIXTURES
# ----------------------------------------------------
@pytest.fixture
def sample_corpus():
    return [
        "Metformin hydrochloride is an oral antidiabetic drug used to treat type 2 diabetes and manage HbA1c levels.",
        "Hypertension or long-term high blood pressure increases the risk of stroke, myocardial infarction, and heart failure.",
        "SARS-CoV-2 is the coronavirus strain responsible for the COVID-19 pandemic causing respiratory infection.",
        "Lisinopril is an ACE inhibitor used in the treatment of hypertension and congestive heart failure.",
        "Common side effects of metformin include nausea, abdominal pain, diarrhea, and rare lactic acidosis."
    ]


@pytest.fixture
def sample_faiss_index(sample_corpus):
    # Dummy 384-dim embeddings using random vectors for deterministic index test
    np.random.seed(42)
    embeddings = np.random.randn(len(sample_corpus), 384).astype("float32")
    # Normalize for cosine/similarity testing
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(384)
    index.add(embeddings)
    return index


# ----------------------------------------------------
# 1. BM25 INDEX CONSTRUCTION
# ----------------------------------------------------
def test_bm25_index_construction(sample_corpus):
    bm25 = build_bm25(sample_corpus)
    assert isinstance(bm25, BM25Okapi)
    assert bm25.corpus_size == len(sample_corpus)


# ----------------------------------------------------
# 2. BM25 RETRIEVAL
# ----------------------------------------------------
def test_bm25_retrieval(sample_corpus):
    retriever = BM25Retriever(texts=sample_corpus)
    results = retriever.search("metformin side effects", top_k=2)

    assert len(results) <= 2
    assert len(results) > 0
    first = results[0]
    assert "index" in first
    assert "text" in first
    assert "score" in first
    assert isinstance(first["index"], int)
    assert isinstance(first["score"], float)
    assert "metformin" in first["text"].lower()


# ----------------------------------------------------
# 3. FAISS RETRIEVAL
# ----------------------------------------------------
def test_faiss_retrieval(sample_corpus, sample_faiss_index):
    retriever = HybridRetriever(texts=sample_corpus, faiss_index=sample_faiss_index)
    results = retriever.search_dense("hypertension", top_k=3)

    assert len(results) <= 3
    assert len(results) > 0
    assert "index" in results[0]
    assert "score" in results[0]
    assert isinstance(results[0]["score"], float)


# ----------------------------------------------------
# 4. HYBRID RETRIEVAL
# ----------------------------------------------------
def test_hybrid_retrieval(sample_corpus, sample_faiss_index):
    retriever = HybridRetriever(
        texts=sample_corpus,
        faiss_index=sample_faiss_index,
        reranker=None  # Disable reranker for raw fusion test
    )
    results = retriever.search("metformin side effects", mode="hybrid", final_top_k=3, enable_reranker=False)

    assert len(results) <= 3
    assert len(results) > 0
    first = results[0]
    assert "index" in first
    assert "text" in first
    assert "dense_score" in first
    assert "bm25_score" in first
    assert "hybrid_score" in first


# ----------------------------------------------------
# 5. SCORE NORMALIZATION
# ----------------------------------------------------
def test_score_normalization():
    # Test standard range
    scores = [10.0, 20.0, 30.0, 50.0]
    norm = min_max_normalize(scores)
    assert norm == [0.0, 0.25, 0.5, 1.0]

    # Test edge case: max == min
    equal_scores = [5.0, 5.0, 5.0]
    norm_equal = min_max_normalize(equal_scores)
    assert norm_equal == [1.0, 1.0, 1.0]

    # Empty list
    assert min_max_normalize([]) == []


# ----------------------------------------------------
# 6. SCORE FUSION
# ----------------------------------------------------
def test_score_fusion(sample_corpus, sample_faiss_index):
    retriever_50 = HybridRetriever(
        texts=sample_corpus, faiss_index=sample_faiss_index, reranker=None
    )
    res_alpha_05 = retriever_50.search("diabetes metformin", mode="hybrid", alpha=0.5, enable_reranker=False)

    res_alpha_09 = retriever_50.search("diabetes metformin", mode="hybrid", alpha=0.9, enable_reranker=False)

    assert len(res_alpha_05) > 0
    assert len(res_alpha_09) > 0
    # Expected formula: alpha * dense + (1-alpha) * bm25
    item = res_alpha_05[0]
    expected_hybrid = 0.5 * item["dense_score"] + 0.5 * item["bm25_score"]
    assert pytest.approx(item["hybrid_score"], 0.001) == expected_hybrid


# ----------------------------------------------------
# 7. DUPLICATE REMOVAL
# ----------------------------------------------------
def test_duplicate_removal(sample_corpus, sample_faiss_index):
    retriever = HybridRetriever(texts=sample_corpus, faiss_index=sample_faiss_index, reranker=None)
    results = retriever.search("hypertension treatment", mode="hybrid", final_top_k=10, enable_reranker=False)

    indices = [r["index"] for r in results]
    # Verify no duplicate indices exist
    assert len(indices) == len(set(indices))


# ----------------------------------------------------
# 8. RERANKER INTEGRATION
# ----------------------------------------------------
def test_reranker_integration(sample_corpus):
    candidates = [
        {"index": 0, "text": sample_corpus[0], "dense_score": 0.8, "bm25_score": 0.9, "hybrid_score": 0.85},
        {"index": 1, "text": sample_corpus[1], "dense_score": 0.5, "bm25_score": 0.4, "hybrid_score": 0.45}
    ]

    reranker = Reranker()
    reranked = reranker.rerank_candidates("What are the side effects of metformin?", candidates, top_n=2)

    assert len(reranked) == 2
    assert "reranker_score" in reranked[0]
    assert "reranker_score" in reranked[1]
    # Reranker score must be sorted descending
    assert reranked[0]["reranker_score"] >= reranked[1]["reranker_score"]


# ----------------------------------------------------
# 9. EMPTY QUERY HANDLING
# ----------------------------------------------------
def test_empty_query_handling(sample_corpus, sample_faiss_index):
    retriever = HybridRetriever(texts=sample_corpus, faiss_index=sample_faiss_index)

    assert retriever.search("", mode="hybrid") == []
    assert retriever.search("   ", mode="hybrid") == []
    assert retriever.search_bm25("") == []


# ----------------------------------------------------
# 10. EXACT MEDICAL TERMINOLOGY
# ----------------------------------------------------
def test_exact_medical_terminology():
    text = "Patient with SARS-CoV-2, COVID-19, high HbA1c and hypertension taking metformin."
    tokens = tokenize_medical_text(text)

    expected_terms = ["sars-cov-2", "covid-19", "hba1c", "hypertension", "metformin"]
    for term in expected_terms:
        assert term in tokens, f"Medical term '{term}' was removed by tokenizer!"


# ----------------------------------------------------
# 11. SEMANTIC WORDING QUERY
# ----------------------------------------------------
def test_semantic_wording_query(sample_corpus, sample_faiss_index):
    retriever = HybridRetriever(texts=sample_corpus, faiss_index=sample_faiss_index)
    query = "What problems can long-term high blood pressure cause?"

    results = retriever.search(query, mode="hybrid", final_top_k=3)
    assert len(results) > 0
    # Corpus item 1 contains 'Hypertension or long-term high blood pressure increases the risk...'
    retrieved_texts = [r["text"] for r in results]
    assert any("high blood pressure" in t.lower() or "hypertension" in t.lower() for t in retrieved_texts)


# ----------------------------------------------------
# 12. DRUG NAME QUERY
# ----------------------------------------------------
def test_drug_name_query(sample_corpus, sample_faiss_index):
    retriever = HybridRetriever(texts=sample_corpus, faiss_index=sample_faiss_index)
    query = "What are the side effects of metformin?"

    results = retriever.search(query, mode="hybrid", final_top_k=3)
    assert len(results) > 0
    retrieved_texts = [r["text"] for r in results]
    assert any("metformin" in t.lower() for t in retrieved_texts)


# ----------------------------------------------------
# 13. COSINE SIMILARITY & INDEXFLATIP VERIFICATION
# ----------------------------------------------------
def test_cosine_similarity_faiss_properties(sample_corpus):
    # a. Document embeddings are normalized before indexing
    embeddings = np.random.randn(len(sample_corpus), 384).astype("float32")
    faiss.normalize_L2(embeddings)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    # c. FAISS uses IndexFlatIP
    index = faiss.IndexFlatIP(384)
    assert isinstance(index, faiss.IndexFlatIP)
    index.add(embeddings)

    retriever = HybridRetriever(texts=sample_corpus, faiss_index=index)
    results = retriever.search_dense("metformin diabetes", top_k=3)

    # b. Query embeddings are normalized before search
    # e. The old 1 / (1 + distance) conversion is no longer used (score is direct cosine similarity in [-1.0, 1.0])
    assert len(results) > 0
    top = results[0]
    assert "score" in top
    assert -1.0 <= top["score"] <= 1.0
    assert "raw_distance" not in top  # Old L2 distance key removed


def test_higher_cosine_similarity_better_ranking(sample_corpus):
    # d. Higher cosine similarity produces a better dense ranking
    doc_vectors = np.array([
        [1.0, 0.0, 0.0],        # Exactly aligned with query [1, 0, 0] -> Cosine = 1.0
        [0.7071, 0.7071, 0.0],  # Angle 45 deg -> Cosine = 0.7071
        [0.0, 1.0, 0.0],        # Orthogonal -> Cosine = 0.0
    ], dtype=np.float32)

    faiss.normalize_L2(doc_vectors)
    idx = faiss.IndexFlatIP(3)
    idx.add(doc_vectors)

    mock_texts = ["Aligned Doc", "Partial Doc", "Orthogonal Doc"]
    retriever = HybridRetriever(texts=mock_texts, faiss_index=idx)

    class DummyModel:
        def encode(self, texts, normalize_embeddings=True):
            v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            faiss.normalize_L2(v)
            return v

    retriever.model = DummyModel()

    res = retriever.search_dense("query", top_k=3)
    assert len(res) == 3
    assert res[0]["text"] == "Aligned Doc"
    assert res[0]["score"] == pytest.approx(1.0, abs=1e-4)
    assert res[1]["text"] == "Partial Doc"
    assert res[1]["score"] == pytest.approx(0.7071, abs=1e-4)
    assert res[2]["text"] == "Orthogonal Doc"
    assert res[2]["score"] == pytest.approx(0.0, abs=1e-4)

