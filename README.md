# 🏥 MedAssistRAG: Retrieval Augmented Generation System for Safe and Grounded Medical Question Answering

MedAssistRAG is a Retrieval-Augmented Generation (RAG) based medical question-answering system designed to provide safe, reliable, and evidence-grounded healthcare information. The system combines dense retrieval, reranking, tool-based reasoning, and Large Language Models (LLMs) to reduce hallucinations and improve answer quality in medical conversations.

Traditional LLMs often generate plausible but factually incorrect medical responses. MedAssistRAG addresses this challenge by retrieving relevant evidence from trusted medical datasets before generating an answer. The system also incorporates safety mechanisms such as medical disclaimers and human escalation workflows.

---

## 📌 Features

* Retrieval-Augmented Generation (RAG)
* Dense semantic retrieval using Sentence Transformers
* FAISS-based vector search
* Cross-Encoder reranking for improved retrieval quality
* Tool-Augmented reasoning and execution
* Medical safety disclaimer generation
* Human support ticket escalation
* Fine-tuned retriever model
* DPO (Direct Preference Optimization) experimentation
* Support for Ollama and local Transformer models
* Evidence-grounded medical answer generation

---

## 🏗️ System Architecture

```text
                    ┌─────────────────┐
                    │   User Query    │
                    └────────┬────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │ Tool Selection Module  │
                └────────┬───────────────┘
                         │
         ┌───────────────┼─────────────────┐
         │               │                 │
         ▼               ▼                 ▼
    SearchKB      CreateTicket   MedicalDisclaimer
         │
         ▼
 ┌───────────────────────┐
 │ Dense Retriever (BGE) │
 └──────────┬────────────┘
            │
            ▼
      FAISS Search
            │
            ▼
    Top-K Passages
            │
            ▼
  Cross Encoder Reranker
            │
            ▼
   Top Relevant Context
            │
            ▼
     Llama 3 / TinyLlama
            │
            ▼
       Final Response
```

---

## 📁 Directory Structure

```text
MedAssistRAG/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── main.py
│
├── configs/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│
├── docs/
│   ├── images/
│   ├── screenshots/
│   └── architecture/
│
├── evaluation/
│   ├── plots/
│   ├── reports/
│   ├── metrics/
│   └── scripts/
│
├── models/
│   ├── retriever/
│   ├── reranker/
│   └── generator/
│
├── outputs/
│   ├── predictions/
│   ├── comparison/
│   └── tickets/
│
├── scripts/
│
├── tests/
│
└── src/
    ├── data_utils/
    ├── retriever/
    ├── reranker/
    ├── generator/
    ├── tools/
    ├── preference_alignment/
    ├── pipeline/
    ├── training/
    ├── evaluation/
    └── __init__.py
```

---

# 🎯 Problem Statement

Medical question-answering systems powered solely by Large Language Models are susceptible to hallucinations, misinformation, and unsafe recommendations. In healthcare settings, such inaccuracies can have serious consequences.

The objective of MedAssistRAG is to:

* Retrieve trustworthy medical evidence.
* Generate context-aware answers.
* Reduce hallucinated responses.
* Provide safety warnings when necessary.
* Escalate unresolved cases to human experts.

---

# ⚙️ How It Works

The system follows a multi-stage pipeline:

### 1. Query Processing

The user submits a medical question.

Example:

```text
What are the symptoms of hypertension?
```

The query is normalized and passed to the Tool Selection Module.

---

### 2. Tool Selection

An LLM decides which tool should handle the request.

Available tools:

| Tool                  | Purpose                         |
| --------------------- | ------------------------------- |
| SearchKB              | Search medical knowledge base   |
| CreateTicket          | Escalate issue for human review |
| MedicalDisclaimerTool | Add safety disclaimer           |

Example tool response:

```json
{
  "tool": "SearchKB",
  "args": {
    "query": "What are the symptoms of hypertension?"
  }
}
```

---

### 3. Dense Retrieval

The query is converted into embeddings using:

```text
BAAI/bge-small-en
```

These embeddings are searched against a FAISS vector database.

---

### 4. Re-Ranking

Retrieved passages are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This improves retrieval precision by selecting the most relevant medical evidence.

---

### 5. Context Construction

Top-ranked passages are combined into a structured prompt.

Example:

```text
You are a medical assistant.

Use ONLY the context below to answer the question.

If the answer is not in the context,
say "I don't know."
```

---

### 6. Response Generation

The final prompt is passed to:

* Llama 3 (via Ollama)
* TinyLlama (Fine-Tuned)
* DPO Aligned Models

The generated answer is returned along with retrieved evidence.

---

# 🧩 Tool-Augmented Framework

## 🔍 SearchKB

Retrieves relevant medical information from the knowledge base.

### Input

```json
{
  "query": "What causes diabetes?"
}
```

### Output

Relevant medical passages retrieved from FAISS.

---

## 🎫 CreateTicket

Used when the system lacks sufficient confidence.

### Input

```json
{
  "issue": "Complex medical case"
}
```

### Output

Structured escalation ticket.

---

## ⚠️ MedicalDisclaimerTool

Appends safety warnings for sensitive medical queries.

### Example Disclaimer

```text
This information is provided for educational purposes only and should not replace professional medical advice.
```

---

# 📚 Datasets Used

The system combines multiple biomedical datasets.

## PubMedQA

* Biomedical research questions
* Evidence-based answers
* Derived from PubMed abstracts

## MedQuAD

* NIH healthcare question-answer pairs
* Consumer-oriented medical information

## MedMCQA

* Medical entrance examination questions
* Clinical reasoning focused

These datasets are merged into a unified corpus for retrieval and training.

---

# 🔄 Data Processing Pipeline

```text
Raw Datasets
      │
      ▼
Cleaning & Normalization
      │
      ▼
Question + Answer Merge
      │
      ▼
Chunking
      │
      ▼
Embedding Generation
      │
      ▼
FAISS Index Creation
      │
      ▼
Retrieval Corpus
```

---

# 🧠 Model Components

## Retriever

### Base Model

```text
BAAI/bge-small-en
```

### Purpose

* Semantic search
* Embedding generation
* Medical document retrieval

---

## Vector Database

### FAISS

Stores:

* Dense embeddings
* Medical text chunks


---

## Re-Ranker

### Model

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Purpose

* Improve ranking quality
* Increase retrieval precision
* Select best passages

---

## Generator

### Supported Models

#### Ollama

```text
llama3
```

#### Local Models

* TinyLlama-1.1B-Chat-v1.0
* Fine-Tuned Generator
* DPO Aligned Model

---

# 🎯 Fine-Tuning

## Retriever Fine-Tuning

### Base Model

```text
BAAI/bge-small-en
```

### Training Objective

```text
MultipleNegativesRankingLoss
```

### Configuration

| Parameter       | Value |
| --------------- | ----- |
| Samples         | 2000  |
| Epochs          | 5     |
| Batch Size      | 4     |
| Training Device | CPU   |

---

## DPO Alignment

Direct Preference Optimization (DPO) was explored to improve:

* Alignment
* Response quality
* Hallucination reduction
* User preference satisfaction


---

# 📊 Evaluation Metrics

The system is evaluated using multiple metrics.

| Metric              | Description                   |
| ------------------- | ----------------------------- |
| F1 Score            | Token overlap                 |
| Exact Match         | Exact answer correctness      |
| BLEU Score          | Text similarity               |
| Semantic Similarity | Meaning preservation          |
| Hallucination Rate  | Unsupported content detection |

---

## 📈 Best Model Results

| Metric              | Score |
| ------------------- | ----- |
| F1 Score            | 0.41  |
| Exact Match         | 0.174 |
| Semantic Similarity | 0.79  |
| BLEU Score          | 0.15  |
| Hallucination Rate  | 0.29  |

---

# 🚀 Installation

## Clone Repository

```bash
git clone <repository-url>
cd MedAssistRAG
```

## Create Virtual Environment

```bash
python -m venv .venv
```

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔧 Build Embeddings and FAISS Index

Run:

```bash
python src/retriever/run_pipeline.py
```

This will:

* Generate embeddings
* Build FAISS index
* Store vector database artifacts

---

# ▶️ Running the Project

## Start Ollama

```bash
ollama run llama3
```

## Launch Application

```bash
python main.py
```

---

# 💡 Example Query

```text
What are the symptoms of hypertension?
```

### Workflow

```text
User Query
     │
     ▼
SearchKB
     │
     ▼
Retrieve Documents
     │
     ▼
Rerank Passages
     │
     ▼
Generate Response
     │
     ▼
Add Disclaimer
     │
     ▼
Return Answer
```

---

# 🔒 Safety Features

* Context-grounded generation
* Hallucination reduction
* Medical disclaimers
* Human escalation workflow
* Tool-based reasoning
* Evidence-backed answers

---

# 👥 Contributors

* Xena Doris Pereira
* Angel Manoj
* Deep Patel
* Pronnati Tapaswi

---

# ⚠️ Disclaimer

This project is intended for research and educational purposes only.

The generated responses should not be considered professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for healthcare decisions.

---
