# 🎓 MedAssistRAG: System Architecture & Interview Guide

This guide provides a comprehensive breakdown of the project architecture, directory roles, file functionality, and key interview explanations.

---

## 📌 1. Project Directory Roles & Structure

```text
MedAssistRAG/
│
├── README.md                 # Primary project overview, setup, and usage instructions
├── requirements.txt          # Python dependencies required to run the application
├── .gitignore                # Specifies intentionally untracked files to keep Git clean
├── main.py                   # Main CLI entry point for running the medical RAG assistant
│
├── configs/                  # Holds YAML/JSON configuration files (hyperparameters, model paths)
├── data/
│   ├── raw/                  # Original raw medical datasets (PubMedQA, MedQuAD, MedMCQA)
│   ├── processed/            # Unified, cleaned CSV/JSONL datasets ready for training/retrieval
│   └── embeddings/           # FAISS vector database indices (.bin) and serialized texts (.pkl)
│
├── docs/
│   ├── images/               # Visual diagrams and graphics for documentation
│   ├── screenshots/          # Application demo screenshots and evaluation charts
│   └── architecture/         # Architectural flowcharts and system design documents
│
├── evaluation/
│   ├── plots/                # Generated visual benchmark comparisons (PNG charts)
│   ├── reports/              # Quantitative evaluation metric outputs (CSV summary files)
│   ├── metrics/              # Metric definitions and scoring configurations
│   └── scripts/              # Independent evaluation benchmark execution scripts
│
├── models/
│   ├── retriever/            # Fine-tuned dense retriever model checkpoints (SentenceTransformer)
│   ├── reranker/             # Fine-tuned cross-encoder reranker model checkpoints
│   └── generator/            # Fine-tuned LLM generator weights & LoRA adapters
│
├── outputs/
│   ├── predictions/          # Model inference generated prediction outputs
│   ├── comparison/           # Side-by-side baseline vs RAG response output files
│   └── tickets/              # Logged support escalation tickets created by tools
│
├── scripts/                  # Helper automation scripts for setup, dockerization, or batch jobs
├── tests/                    # Unit and integration test suite (pytest)
│
└── src/                      # Core production Python package
    ├── data_utils/           # Data loading, cleaning, merging, and chunking modules
    ├── retriever/            # FAISS vector indexing, dense search, and fine-tuning logic
    ├── reranker/             # Cross-Encoder re-ranking logic and training utilities
    ├── generator/            # LLM interface (Ollama/HuggingFace) and instruction tuning
    ├── tools/                # Tool schema definitions and execution router
    ├── preference_alignment/ # Direct Preference Optimization (DPO) alignment scripts
    ├── training/             # Single Canonical LLM LoRA Fine-Tuning Pipeline
    ├── config.py             # Single Canonical System Configuration Module
    ├── pipeline/             # Memory-aware RAG orchestrator pipelineties and tokenization scripts
    └── evaluation/           # Metric calculation functions and model comparison routines
```

---

## ❓ 2. Interview Q&A: "Why are there empty folders in your repository?"

### **Interview Answer:**

> "In professional MLOps and Machine Learning software engineering, repositories follow standardized project templates (such as Cookiecutter Data Science). 
> 
> Heavy artifacts such as **raw datasets (multi-GB CSVs)**, **model weights (PyTorch/HuggingFace checkpoints)**, **FAISS indices**, and **generated output logs** are explicitly excluded from Git using `.gitignore` because version control systems are designed for code, not massive binary files.
> 
> We keep placeholder files (`.gitkeep`) inside directories like `configs/`, `data/raw/`, `models/`, `outputs/`, and `docs/` so that when a team member or recruiter clones the project repository:
> 1. The exact folder hierarchy is immediately initialized on their local machine.
> 2. Automated data pipelines and training scripts can read from `data/raw/` and output model checkpoints to `models/` without failing with `FileNotFoundError` or missing directory errors.
> 3. It demonstrates production-grade project organization and clean environment practices."

---

## 📄 3. File-by-File Functionality Overview

### 🏁 **Root & Pipeline Entry Points**

- **`main.py`**:
  - The CLI user interface. Imports `RAGPipeline` from `src.pipeline.rag_pipeline`, accepts user queries interactively in a loop, calls `rag.query(query)`, and displays retrieved context sources alongside the generated answer.

- **`src/pipeline/rag_pipeline.py`**:
  - The main pipeline orchestrator (`RAGPipeline`). Uses an LLM to select an appropriate tool (`SearchKB`, `CreateTicket`, `MedicalDisclaimerTool`) based on user query intent, executes the selected tool via `execute_tool`, constructs context, and generates evidence-grounded answers.

- **`tests/test_llm.py`**:
  - Unit test verifying LLM initialization and response generation functionality.

---

### 🧹 **Data Processing (`src/data_utils/`)**

- **`load_data.py`**: Reads raw CSV datasets, performs missing-value removal, deduplication, and creates a combined search text field (`question` + `answer`).
- **`chunk_data.py`**: Splits medical documents into chunks using `RecursiveCharacterTextSplitter` with chunk size of 300 and overlap of 50 tokens.
- **`prepare_data.py`**: Extracts medical Q&A pairs from PubMedQA, MedQuAD XML files, and MedMCQA datasets, merging them into a clean unified dataset (`qa_dataset_large.csv`).

---

### 🔍 **Dense Retrieval (`src/retriever/`)**

- **`embed.py`**: Encodes text chunks into dense vector embeddings using `BAAI/bge-small-en`.
- **`build_faiss.py`**: Builds an L2 vector index (`faiss.IndexFlatL2`) from embeddings and serializes the index (`faiss_index.bin`) and text list (`texts.pkl`).
- **`search.py`**: Encapsulates the `Retriever` class that loads FAISS indices and performs similarity searches to return top-k relevant document passages.
- **`run_pipeline.py`**: Runs end-to-end embedding creation and FAISS index build workflow.
- **`train_data.py`**: Formats question-answer pairs into `InputExample` objects for retriever training.
- **`train_retriever.py`**: Fine-tunes the dense retriever model using `MultipleNegativesRankingLoss`.

---

### 🎯 **Reranking (`src/reranker/`)**

- **`reranker.py`**: Implements the `Reranker` class using Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to re-score and sort initial FAISS candidate documents for high retrieval precision.
- **`train_data.py`**: Generates positive (`label=1.0`) and negative (`label=0.0`) query-document pairs for training.
- **`train_reranker.py`**: Fine-tunes the cross-encoder model on medical QA pairs.

---

### 🤖 **Answer Generation (`src/generator/`)**

- **`llm.py`**: Implements the `LLM` class supporting local inference via **Ollama (Llama 3)** or local HuggingFace/PEFT models, providing raw prompt generation and context-grounded medical answer generation.
- **`training/train_data.py`**: Prepares prompt-response datasets for fine-tuning generator models.
- **`training/train_generator.py`**: Fine-tunes TinyLlama model using LoRA (`peft`) for domain-adapted medical answer generation.

---

### 🛠️ **Tool Framework (`src/tools/`)**

- **`tools.py`**: Declares JSON schemas for available system tools (`SearchKB`, `CreateTicket`, `MedicalDisclaimerTool`).
- **`executor.py`**: Contains `execute_tool(tool_name, args)` router function that executes FAISS retrieval, ticket registration, or disclaimer attachment.

---

### ⚖️ **Preference Alignment (`src/preference_alignment/`)**

- **`config.py`**: Configuration parameters (learning rate, batch size, paths) for Direct Preference Optimization (DPO).
- **`format_data.py`**: Loads and formats preference data into `{prompt, chosen, rejected}` structures.
- **`train_dpo.py`**: Executes DPO training using `TRL`'s `DPOTrainer` and LoRA to reduce hallucinations and align LLM answers with human preferences.

---

### 🏋️ **Model Training (`src/training/`)**

- **`prepare_finetune_data.py`**: Retrieves context from FAISS for training samples and constructs prompt-target pairs for Supervised Fine-Tuning (SFT).
- **`tokenize_data.py`**: Tokenizes instruction datasets using HuggingFace tokenizers (`google/flan-t5-large`) and saves tokenized data to disk.
- **`train_lora.py`**: Fine-tunes sequence-to-sequence LLMs using PEFT LoRA adapters.

---

### 📊 **Evaluation & Metrics (`src/evaluation/`)**

- **`advanced_metrics.py`**: Defines core evaluation metrics:
  - **F1 Score**: Token-level precision & recall overlap.
  - **Exact Match**: Binary string match indicator.
  - **BLEU Score**: N-gram similarity against ground truth.
  - **Semantic Similarity**: Cosine similarity using sentence embeddings.
  - **Grounding Score & Hallucination Rate**: Measures answer token coverage within retrieved context.
- **`final_evaluate.py`**: Evaluates end-to-end RAG pipeline across dataset samples, computing average latency and performance metrics.
- **`compare_models.py`**: Benchmarks models, exports evaluation CSV reports to `evaluation/reports/`, and generates bar plots saved in `evaluation/plots/`.

---

## 💡 Key Architectural Highlights to Mention in Interviews

1. **RAG Architecture**: Mitigates LLM hallucinations by retrieving factual evidence before generating answers.
2. **Two-Stage Retrieval (Retrieve & Rerank)**: Combines fast FAISS dense vector search (Recall) with computationally accurate Cross-Encoder reranking (Precision).
3. **Tool-Augmented Reasoning**: Uses LLM JSON tool selection to dynamically choose between Knowledge Base Search, Ticket Escalation, or Medical Safety Warning generation.
4. **DPO Preference Alignment**: Leverages Direct Preference Optimization to train the generator model to prefer factual, grounded medical answers over speculative ones.
