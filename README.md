  # MedAssistRAG

## Overview

**MedAssistRAG** is a production-designed Retrieval-Augmented Generation (RAG) framework engineered for safe, grounded, and confidence-aware medical question answering. It integrates multi-stage hybrid retrieval (dense FAISS + sparse BM25), Cross-Encoder reranking, programmatic confidence scoring, self-correction retry loops, multi-turn conversation memory, medical safety risk classification, support ticket escalation, and verifiable source citation attribution.

---

## Problem Statement

Medical AI systems powered solely by ungrounded Large Language Models (LLMs) are prone to hallucinations, outdated medical facts, and unsafe medical advice. In healthcare, providing unverified answers for high-risk symptoms or medication changes can pose severe risks to patient safety.

MedAssistRAG addresses these challenges by enforcing strict evidence-grounded generation:
- **Retrieving Grounded Knowledge**: Pulling factual passages from biomedical knowledge bases before generating answers.
- **Assessing Confidence**: Programmatically evaluating retrieval relevance before allowing the LLM to respond.
- **Enforcing Safety Escalation**: Intercepting personal medical queries and emergency scenarios to bypass RAG and route to human support or emergency warnings.
- **Attributing Sources**: Validating and formatting strict inline citations mapped to verified medical source references.

---

## Implemented Features

| Feature | Implementation Component | Status |
|---|---|---|
| **FAISS Vector Search** | `src/retriever/hybrid_retriever.py` | Implemented |
| **BM25 Keyword Search** | `src/retriever/bm25_retriever.py` | Implemented |
| **Hybrid Score Fusion** | `min_max_normalize` + Weighted Fusion ($\alpha = 0.5$) | Implemented |
| **Cross-Encoder Reranking** | `src/reranker/reranker.py` | Implemented |
| **Confidence Scoring** | `src/confidence/scorer.py` | Implemented |
| **Self-Correction Retry Loop** | `src/retry/retry_controller.py` | Implemented |
| **Query Rewriting** | `src/query/query_rewriter.py` | Implemented |
| **Multi-Turn Memory** | `src/memory/memory_manager.py` (Short-Term, Entity) | Implemented |
| **Risk Classification** | `src/safety/risk_classifier.py` (LOW, MEDIUM, HIGH) | Implemented |
| **Support Ticket Manager** | `src/safety/ticket_manager.py` | Implemented |
| **Citation Attribution** | `src/citations/citation_manager.py` | Implemented |
| **LLM Generation** | `src/generator/llm.py` (TinyLlama-1.1B + LoRA adapter) | Implemented |

---

## Current Architecture

```text
                                ┌─────────────────┐
                                │   User Query    │
                                └────────┬────────┘
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │   Conversation Memory  │
                            └────────┬───────────────┘
                                     │
                                     ▼
                            ┌────────────────────────┐
                            │    Risk Classifier     │
                            └────────┬───────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │ (LOW Risk)                │ (MEDIUM Risk)             │ (HIGH Risk)
         ▼                           ▼                           ▼
┌─────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│ Query Rewriter  │        │ Create Ticket    │        │ Emergency Warning│
└────────┬────────┘        │ Referral Flow    │        │ Safety Response  │
         │                 └──────────────────┘        └──────────────────┘
         ▼
┌─────────────────────────────────────────────────────┐
│                  Hybrid Retrieval                   │
│   ┌──────────────────────┐  ┌────────────────────┐   │
│   │ FAISS (bge-small-en) │  │ BM25 (Rank-BM25)   │   │
│   └──────────┬───────────┘  └─────────┬──────────┘   │
│              └────────────┬───────────┘              │
│                           ▼                          │
│               Min-Max Normalization &                │
│                 Weighted Score Fusion                │
└───────────────────────────┬─────────────────────────┘
                            │
                            ▼
               ┌──────────────────────────┐
               │ Cross-Encoder Reranker   │
               └────────────┬─────────────┘
                            │
                            ▼
               ┌──────────────────────────┐
               │    Confidence Scorer     │
               └────────────┬─────────────┘
                            │
            ┌───────────────┴───────────────┐
            │ (LOW Confidence)              │ (HIGH / MEDIUM)
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│ Self-Correction Loop  │       │     LLM Generator     │
│  (Query Retry / Rewr) │       │   (TinyLlama-1.1B)    │
└───────────────────────┘       └───────────┬───────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │   Citation Manager    │
                                │ ([E1] -> [1] Sources) │
                                └───────────┬───────────┘
                                            │
                                            ▼
                                ┌───────────────────────┐
                                │     Final Answer      │
                                └───────────────────────┘
```

---

## End-to-End Workflow

1. **User Query Input**: The user sends a query (and optional `conversation_id`) to `RAGPipeline.query()`.
2. **Conversation Context Lookup**: Memory context (short-term history, tracked entities) is resolved.
3. **Medical Risk Classification**: `RiskClassifier` evaluates query risk:
   - **HIGH Risk** (e.g. emergency symptoms or dosage changes): Bypasses retrieval and returns a controlled emergency warning (`SafetyPolicy.get_high_response()`).
   - **MEDIUM Risk** (e.g. personal clinical guidance): Bypasses retrieval, creates a support ticket in `data/tickets/tickets.json`, and returns a professional referral response (`SafetyPolicy.get_medium_response()`).
   - **LOW Risk** (e.g. general medical education): Proceeds through the RAG pipeline.
4. **Query Rewriting**: If prior context exists, `QueryRewriter` generates a standalone search query.
5. **Hybrid Retrieval**: Candidate passages are retrieved using FAISS (dense) and BM25 (sparse keyword), normalized via Min-Max scaling, and combined with weight $\alpha = 0.5$.
6. **Cross-Encoder Reranking**: `Reranker` scores candidate pairs and returns top $N$ passages.
7. **Confidence Evaluation**: `ConfidenceScorer` calculates final confidence score.
8. **Self-Correction Retry Controller**: If confidence is `LOW` (< 0.60), `RetryController` executes bounded query rewriting retries (up to 2 retries).
9. **LLM Evidence Generation**: `LLM.generate_with_evidence()` generates an answer tagged with temporary evidence markers (`[E1]`, `[E2]`).
10. **Citation Validation & Attribution**: `CitationManager` validates `[E#]` markers, strips invalid markers, converts valid markers to `[1]`, `[2]`, consolidates duplicate document sources, and appends the formatted `Sources:` list.

---

## Retrieval System

The retrieval layer (`src/retriever/hybrid_retriever.py`) supports three operational modes:

### 1. FAISS Dense Retrieval
- Embeds queries and text chunks using `BAAI/bge-small-en` (or fine-tuned `models/retriever_finetuned`).
- Searches index using L2 distance and converts distances to similarity scores:
  $$\text{sim\_score} = \frac{1}{1.0 + \text{L2\_distance}}$$

### 2. BM25 Sparse Retrieval
- Uses `rank-bm25` (BM25Okapi) with a medical tokenizer (`tokenize_medical_text`) that handles hyphenated medical terms (e.g., `SARS-CoV-2`, `HbA1c`).

### 3. Hybrid Score Fusion
- Normalizes raw scores using Min-Max scaling:
  $$\text{score}_{\text{norm}} = \frac{s - s_{\min}}{s_{\max} - s_{\min}}$$
- Combines scores using weighted fusion ($\alpha = 0.5$):
  $$\text{score}_{\text{hybrid}} = \alpha \cdot \text{score}_{\text{dense}} + (1 - \alpha) \cdot \text{score}_{\text{bm25}}$$

### 4. Cross-Encoder Reranking
- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (or `models/reranker_finetuned`) to re-score candidate pairs `[query, candidate_text]`.

---

## Query Processing & Retry Controller

- **Query Rewriter (`src/query/query_rewriter.py`)**: Uses the LLM to rewrite ambiguous queries into standalone search queries using conversational context while preserving intent.
- **Retry Controller (`src/retry/retry_controller.py`)**: Executes self-correction loops when initial retrieval yields `LOW` confidence. Features:
  - Bounded retry loop limit (`max_retries = 2`).
  - Loop prevention (tracks attempted query hashes).
  - Best-attempt tracking across iterations.
  - Insufficient evidence abstention when confidence remains low.

---

## Confidence Calculation

Confidence (`src/confidence/scorer.py`) is computed programmatically without relying on LLM self-reporting.

### Document Confidence Formula
$$\text{Conf}_{\text{doc}} = w_{\text{retrieval}} \cdot \text{score}_{\text{hybrid}} + w_{\text{reranker}} \cdot \text{score}_{\text{norm\_reranker}}$$
Default weights: $w_{\text{retrieval}} = 0.5$, $w_{\text{reranker}} = 0.5$.

### Final Aggregate Confidence Formula
Top-K document confidences are weighted using rank weights ($[0.5, 0.3, 0.2]$ for top 3 documents):
$$\text{Conf}_{\text{final}} = \frac{\sum_{i=1}^{K} w_{\text{rank}, i} \cdot \text{Conf}_{\text{doc}, i}}{\sum_{i=1}^{K} w_{\text{rank}, i}}$$

### Thresholds (`configs/confidence.yaml`)
- **HIGH**: Score $\ge 0.80$
- **MEDIUM**: $0.60 \le \text{Score} < 0.80$
- **LOW**: Score $< 0.60$ (triggers self-correction retry)

---

## Conversation Memory

Conversation state (`src/memory/memory_manager.py`) is tracked per `conversation_id`:
- **Short-Term Memory (`short_term_memory.py`)**: Bounded sliding window of recent messages (max 10 messages).
- **Entity Memory (`entity_memory.py`)**: Tracks medical entities (`conditions`, `symptoms`, `medications`, `tests`, `procedures`, `body_parts`).

---

## Medical Safety & Escalation

- **Risk Classifier (`src/safety/risk_classifier.py`)**: Classifies query intent into `LOW`, `MEDIUM`, or `HIGH` risk levels.
- **Safety Policy (`src/safety/safety_policy.py`)**: Maps risk levels to actions (`RAG`, `CREATE_TICKET`, `SAFETY_WARNING`).
- **Ticket Manager (`src/safety/ticket_manager.py`)**: Creates and persists support tickets (`TICKET-XXXX`) to `data/tickets/tickets.json` for human clinical escalation.

---

## Citations & Source Attribution

- **Evidence Objects (`src/citations/models.py`)**: Assigns temporary identifiers (`[E1]`, `[E2]`) to retrieved passages.
- **Citation Validation (`src/citations/citation_manager.py`)**:
  - Validates `[E#]` markers generated by the LLM.
  - Strips hallucinated markers (e.g. `[E99]`).
  - Replaces `[E1]` markers with human-readable numbers `[1]`.
  - Consolidates multiple chunks from the same document into a single citation number.
  - Appends a clean `Sources:` list with titles, publishers, and URLs.

---

## LLM Generator

Implemented in `src/generator/llm.py`:
- **Base Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Fine-Tuned Adapter**: Automatically loads LoRA adapter from `models/generator_finetuned` if present, with fallback to base model.

---

## Data Pipeline & Index Building

Medical datasets (PubMedQA, MedQuAD, MedMCQA) are unified into `data/processed/qa_dataset_large.csv`.

To build or rebuild the retrieval index:
```bash
python src/retriever/run_pipeline.py
python src/retriever/build_bm25_index.py
```

---

## Project Structure

```text
MedAssistRAG/
│
├── configs/                  # Configuration YAML files
│   ├── citations.yaml
│   ├── confidence.yaml
│   ├── memory.yaml
│   ├── retrieval.yaml
│   ├── retry.yaml
│   └── safety.yaml
│
├── data/                     # Data stores and indexes
│   ├── embeddings/
│   │   ├── bm25_index.pkl
│   │   └── texts.pkl
│   ├── tickets/
│   │   └── tickets.json
│   └── processed/
│
├── docs/                     # Technical documentation
│   └── ARCHITECTURE_AND_INTERVIEW_GUIDE.md
│
├── evaluation/               # Benchmark evaluation scripts & reports
│   ├── plots/
│   ├── reports/
│   └── scripts/
│       └── eval_retrieval.py
│
├── models/                   # Model weight checkpoints
│   ├── generator/
│   ├── reranker/
│   └── retriever/
│
├── scripts/                  # Manual component test scripts
│   ├── test_citations_manual.py
│   ├── test_confidence_manual.py
│   ├── test_memory_manual.py
│   └── test_safety_manual.py
│
├── src/                      # Core application source modules
│   ├── citations/
│   ├── confidence/
│   ├── data_utils/
│   ├── evaluation/
│   ├── generator/
│   ├── memory/
│   ├── pipeline/
│   │   └── rag_pipeline.py
│   ├── query/
│   ├── reranker/
│   ├── retriever/
│   ├── retry/
│   ├── safety/
│   ├── tools/
│   ├── training/
│   └── utils/
│
├── tests/                    # Automated Pytest unit test suite
│   ├── test_citation_manager.py
│   ├── test_confidence_scorer.py
│   ├── test_hybrid_retrieval.py
│   ├── test_llm.py
│   ├── test_memory_manager.py
│   ├── test_retry_controller.py
│   └── test_risk_classifier.py
│
├── .gitignore
├── main.py                   # Primary application entry point
├── README.md
└── requirements.txt
```

---

## Technologies Used

- **Python 3.11+**
- **PyTorch** & **Transformers**
- **PEFT (Parameter-Efficient Fine-Tuning / LoRA)**
- **Sentence-Transformers** (`BAAI/bge-small-en`, `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **FAISS (CPU)**
- **Rank-BM25**
- **PyYAML**, **Pandas**, **NumPy**, **Scikit-Learn**
- **Pytest**

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd MedAssistRAG
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

System parameters are configured via YAML files in `configs/`:
- `retrieval.yaml`: Mode (`hybrid`), top-K limits, alpha weighting, model paths.
- `confidence.yaml`: Retrieval/reranker weights, threshold levels (`HIGH: 0.80`, `MEDIUM: 0.60`).
- `retry.yaml`: Max retries (`2`), improvement threshold (`0.05`).
- `memory.yaml`: Short-term max messages (`10`), entity extraction configuration.
- `safety.yaml`: Fallback risk level (`HIGH`), risk mappings.
- `citations.yaml`: Max evidence chunks (`5`), citation formatting rules.

---

## Running the Application

To run the interactive CLI interface:
```bash
python main.py
```

Or run `src/pipeline/rag_pipeline.py` directly:
```bash
python -m src.pipeline.rag_pipeline
```

---

## Testing & Verification

### Automated Pytest Suite
Run unit tests across all system components:
```bash
python -m pytest
```

### Manual Integration Test Scripts
```bash
python scripts/test_confidence_manual.py
python scripts/test_safety_manual.py
python scripts/test_memory_manual.py
python scripts/test_citations_manual.py
```

---

## Limitations

- **CPU Latency**: Running HuggingFace local models (TinyLlama and Cross-Encoder) on CPU introduces inference latency per query.
- **Static Knowledge Base**: Embeddings reflect the offline processed medical corpus and require explicit re-indexing to add new medical literature.
- **Model Capacity**: TinyLlama-1.1B is lightweight; complex medical reasoning benefits from higher parameter models when hardware allows.

---

## Future Work / Not Currently Implemented

The following features were explored during design discussions but are **NOT** currently implemented in the active codebase:
- **Agentic RAG / Multi-Agent Orchestrators**: Currently uses a deterministic multi-stage pipeline (`RAGPipeline`), not an autonomous multi-agent graph.
- **Multi-Source Real-Time Retrieval**: Currently retrieves from the configured local medical index; live API integration with PubMed/MedlinePlus APIs is planned future work.
- **Claim-Level NLI Verification**: Currently validates citation markers `[E1]`, but full formal NLI claim-level entailment checking is planned future work.
- **Knowledge Graph / GraphRAG**: Currently uses hybrid vector + BM25 search rather than graph database retrieval.
- **Long-Term User Preference Memory**: Memory is currently scoped per active conversation session.

---

## Medical Disclaimer

> **IMPORTANT MEDICAL DISCLAIMER**:
> MedAssistRAG is designed strictly for research, educational, and experimental purposes. It does NOT provide certified medical diagnosis, treatment, or professional clinical advice. Always consult a qualified physician or healthcare provider for personal medical concerns. In case of a medical emergency, contact emergency medical services immediately.
