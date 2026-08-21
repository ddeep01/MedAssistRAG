import pytest
from unittest.mock import MagicMock

from src.citations.models import Evidence, Citation, ValidationResult
from src.citations.citation_manager import CitationManager
from src.safety.safety_policy import SafetyAction, RiskLevel
from src.pipeline.rag_pipeline import RAGPipeline


# ----------------------------------------------------
# HELPER FIXTURE
# ----------------------------------------------------
def get_sample_evidence():
    return [
        Evidence(
            evidence_id="E1",
            document_id="doc_101",
            chunk_id="chunk_1",
            text="Metformin treats type 2 diabetes.",
            source="MedlinePlus",
            title="Metformin Guide",
            url="https://medlineplus.gov/metformin.html",
            hybrid_score=0.90,
            reranker_score=0.95
        ),
        Evidence(
            evidence_id="E2",
            document_id="doc_102",
            chunk_id="chunk_2",
            text="Side effects include nausea and diarrhea.",
            source="PubMed",
            title="Metformin Adverse Effects",
            url="https://pubmed.ncbi.nlm.nih.gov/12345",
            hybrid_score=0.85,
            reranker_score=0.88
        )
    ]


# ----------------------------------------------------
# TEST 1: VALID SINGLE CITATION ([E1] -> [1])
# ----------------------------------------------------
def test_valid_single_citation():
    cm = CitationManager()
    ev_list = get_sample_evidence()
    raw_answer = "Metformin is used for diabetes. [E1]"

    res = cm.validate_and_format_citations(raw_answer, ev_list)

    assert "[1]" in res.cleaned_answer
    assert "[E1]" not in res.cleaned_answer
    assert len(res.valid_citations) == 1
    assert res.valid_citations[0].citation_number == 1
    assert "MedlinePlus — Metformin Guide" in res.formatted_text


# ----------------------------------------------------
# TEST 2: MULTIPLE VALID CITATIONS ([E1][E2] -> [1][2])
# ----------------------------------------------------
def test_multiple_valid_citations():
    cm = CitationManager()
    ev_list = get_sample_evidence()
    raw_answer = "Metformin causes nausea. [E1][E2]"

    res = cm.validate_and_format_citations(raw_answer, ev_list)

    assert "[1][2]" in res.cleaned_answer
    assert len(res.valid_citations) == 2
    assert "Sources:" in res.formatted_text
    assert "[1] MedlinePlus — Metformin Guide" in res.formatted_text
    assert "[2] PubMed — Metformin Adverse Effects" in res.formatted_text


# ----------------------------------------------------
# TEST 3: INVALID CITATION REJECTION ([E99])
# ----------------------------------------------------
def test_invalid_citation_rejection():
    cm = CitationManager()
    ev_list = get_sample_evidence()
    raw_answer = "Metformin cures diabetes instantly. [E99]"

    res = cm.validate_and_format_citations(raw_answer, ev_list)

    assert "[E99]" not in res.cleaned_answer
    assert "E99" in res.invalid_citation_ids
    assert len(res.valid_citations) == 0


# ----------------------------------------------------
# TEST 4: MIXED CITATIONS ([E1][E99])
# ----------------------------------------------------
def test_mixed_citations():
    cm = CitationManager()
    ev_list = get_sample_evidence()
    raw_answer = "Metformin is used for diabetes. [E1][E99]"

    res = cm.validate_and_format_citations(raw_answer, ev_list)

    assert "[1]" in res.cleaned_answer
    assert "[E99]" not in res.cleaned_answer
    assert "E99" in res.invalid_citation_ids
    assert len(res.valid_citations) == 1


# ----------------------------------------------------
# TEST 5: NO CITATIONS
# ----------------------------------------------------
def test_no_citations():
    cm = CitationManager()
    ev_list = get_sample_evidence()
    raw_answer = "Metformin is a medication for diabetes."

    res = cm.validate_and_format_citations(raw_answer, ev_list)

    assert len(res.valid_citations) == 0
    assert len(res.invalid_citation_ids) == 0
    assert "Sources:" not in res.formatted_text


# ----------------------------------------------------
# TEST 6: DUPLICATE EVIDENCE (CONSOLIDATE SAME DOC)
# ----------------------------------------------------
def test_duplicate_document_consolidation():
    cm = CitationManager()
    # E1 and E2 come from same document doc_101
    ev_list = [
        Evidence("E1", "doc_101", "c1", "Chunk 1 text", "MedlinePlus", "Metformin", "url1"),
        Evidence("E2", "doc_101", "c2", "Chunk 2 text", "MedlinePlus", "Metformin", "url1")
    ]
    raw_answer = "Part 1 [E1] Part 2 [E2]"

    res = cm.validate_and_format_citations(raw_answer, ev_list)

    # Both E1 and E2 map to user-facing source number [1]
    assert "[1]" in res.cleaned_answer
    assert len(res.valid_citations) == 1
    assert res.valid_citations[0].evidence_ids == ["E1", "E2"]
    assert res.formatted_text.count("[1] MedlinePlus — Metformin") == 1


# ----------------------------------------------------
# TEST 7: MISSING URL (SAFE FORMATTING)
# ----------------------------------------------------
def test_missing_url_fallback():
    cm = CitationManager()
    ev_list = [Evidence("E1", "doc1", "c1", "Text", "MedlinePlus", "Metformin", url=None)]
    raw_answer = "Metformin is used. [E1]"

    res = cm.validate_and_format_citations(raw_answer, ev_list)

    assert "[1] MedlinePlus — Metformin" in res.formatted_text
    assert "None" not in res.formatted_text


# ----------------------------------------------------
# TEST 8: MISSING TITLE (SAFE FALLBACK)
# ----------------------------------------------------
def test_missing_title_fallback():
    cm = CitationManager()
    ev_list = [Evidence("E1", "doc1", "c1", "Text", "MedlinePlus", title="", url=None)]
    raw_answer = "Metformin is used. [E1]"

    res = cm.validate_and_format_citations(raw_answer, ev_list)

    assert "[1] MedlinePlus" in res.formatted_text


# ----------------------------------------------------
# TEST 9: EMPTY EVIDENCE LIST
# ----------------------------------------------------
def test_empty_evidence_list():
    cm = CitationManager()
    res = cm.validate_and_format_citations("Some text [E1]", [])

    assert res.cleaned_answer == "Some text"
    assert len(res.valid_citations) == 0


# ----------------------------------------------------
# TEST 10: LLM INVENTS SOURCE TEXT ([E99] REJECTED)
# ----------------------------------------------------
def test_llm_invents_source():
    cm = CitationManager()
    ev_list = get_sample_evidence()
    raw_answer = "According to Smith et al. [E99] metformin cures diabetes."

    res = cm.validate_and_format_citations(raw_answer, ev_list)

    assert "[E99]" not in res.cleaned_answer
    assert "E99" in res.invalid_citation_ids


# ----------------------------------------------------
# TEST 11: LOW RISK PIPELINE INTEGRATION (RAG + CITATIONS)
# ----------------------------------------------------
def test_low_risk_pipeline_citations():
    mock_risk_classifier = MagicMock()
    mock_risk_classifier.classify.return_value = {
        "risk_level": RiskLevel.LOW,
        "action": SafetyAction.RAG,
        "query": "What is hypertension?",
        "is_fallback": False
    }

    mock_llm = MagicMock()
    mock_llm.generate_with_evidence.return_value = 'Hypertension is high blood pressure. [E1]'

    mock_retry = MagicMock()
    mock_retry.execute_with_retry.return_value = {
        "best_results": [
            {"index": 0, "text": "Hypertension snippet", "document_id": "doc1", "source": "MedlinePlus", "title": "Hypertension"}
        ],
        "confidence_info": {"confidence": 0.85, "level": "HIGH", "needs_retry": False}
    }

    pipeline = RAGPipeline(
        llm=mock_llm,
        risk_classifier=mock_risk_classifier,
        retry_controller=mock_retry
    )

    res = pipeline.query("What is hypertension?")

    assert res["action"] == "RAG"
    assert "[1]" in res["answer"]
    assert "Sources:" in res["answer"]
    assert "[1] MedlinePlus — Hypertension" in res["answer"]


# ----------------------------------------------------
# TEST 12: MEDIUM RISK PIPELINE INTEGRATION (TICKET PATH)
# ----------------------------------------------------
def test_medium_risk_pipeline_no_citations():
    mock_risk_classifier = MagicMock()
    mock_risk_classifier.classify.return_value = {
        "risk_level": RiskLevel.MEDIUM,
        "action": SafetyAction.CREATE_TICKET,
        "query": "Headache 3 weeks",
        "is_fallback": False
    }

    mock_llm = MagicMock()
    pipeline = RAGPipeline(llm=mock_llm, risk_classifier=mock_risk_classifier)

    res = pipeline.query("I have had headaches for three weeks.")

    assert res["action"] == "CREATE_TICKET"
    assert res["citations"] == []
    assert "Sources:" not in res["answer"]


# ----------------------------------------------------
# TEST 13: HIGH RISK PIPELINE INTEGRATION (SAFETY WARNING)
# ----------------------------------------------------
def test_high_risk_pipeline_no_citations():
    mock_risk_classifier = MagicMock()
    mock_risk_classifier.classify.return_value = {
        "risk_level": RiskLevel.HIGH,
        "action": SafetyAction.SAFETY_WARNING,
        "query": "Double dose",
        "is_fallback": False
    }

    mock_llm = MagicMock()
    pipeline = RAGPipeline(llm=mock_llm, risk_classifier=mock_risk_classifier)

    res = pipeline.query("Should I double my medication dose?")

    assert res["action"] == "SAFETY_WARNING"
    assert res["citations"] == []
    assert "Sources:" not in res["answer"]
