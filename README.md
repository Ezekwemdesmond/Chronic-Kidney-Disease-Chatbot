<div align="center">

# KidneyCareAI

### Intelligent Chronic Kidney Disease Risk Assessment & Medical Q&A Platform

*Combining Random Forest ML with Retrieval-Augmented Generation for evidence-based clinical decision support*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-00B47E?style=for-the-badge)](https://www.pinecone.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG_Pipeline-1C3C3C?style=for-the-badge)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
**[Live Demo](https://chatbot-service-836178923173.europe-west2.run.app/) &nbsp;|&nbsp; [Report a Bug](https://github.com/Ezekwemdesmond/Chronic-Kidney-Disease-Chatbot/issues) &nbsp;|&nbsp; [Request a Feature](https://github.com/Ezekwemdesmond/Chronic-Kidney-Disease-Chatbot/issues)**
</div>

---

<div align="center">
  <img src="ckd.gif" alt="KidneyCareAI Demo" width="85%" />
</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [ML Model Performance](#ml-model-performance)
- [RAG Pipeline](#rag-pipeline)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
  - [Docker (Recommended)](#option-1-docker-recommended)
  - [Local Setup](#option-2-local-setup)
- [Usage Guide](#usage-guide)
- [Dataset](#dataset)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Overview

**KidneyCareAI** is a production-grade, full-stack healthcare AI application that addresses the critical challenge of early Chronic Kidney Disease (CKD) detection. CKD affects over **850 million people worldwide** and is frequently undiagnosed until advanced stages — yet early identification can dramatically slow disease progression and reduce mortality.

## 🎥 Demo

![CKD Chatbot Demo](ckd.gif)

This platform delivers two complementary capabilities in a single, unified interface:

1. **Risk Stratification** — A trained Random Forest classifier analyzes 15 clinical biomarkers to predict CKD likelihood with ~98% accuracy, providing immediate, personalized risk assessments.
2. **Evidence-Based Q&A** — A Retrieval-Augmented Generation (RAG) pipeline grounded in authoritative clinical literature (KDIGO guidelines, Brenner & Rector's Kidney textbook, ESPEN guidelines) powers a conversational assistant capable of answering nuanced medical questions.

The system is containerized with Docker, deployed on **Google Cloud Run**, and built with a clean modular architecture that cleanly separates ML inference, vector retrieval, LLM orchestration, and web serving concerns.

> **Medical Disclaimer:** This tool is intended for educational and informational purposes only. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

---

## Key Features

| Feature | Description |
|---|---|
| **CKD Risk Prediction** | Random Forest model trained on the UCI CKD dataset classifies risk using 15 clinical parameters |
| **Personalized Insights** | Prediction results are passed through the RAG pipeline to generate context-specific health guidance |
| **Medical Knowledge Base** | ~25MB of curated clinical PDFs indexed in Pinecone for semantic retrieval at query time |
| **Conversational AI** | GPT-powered chatbot with a defined `KidneyCareAI` persona, tuned for factual, grounded responses |
| **Source Citations** | Medical responses display the source document(s) and page number(s) used, shown as pill tags below each answer |
| **Real-Time Chat UX** | Typing indicators, timestamped messages, and smooth scroll for a polished chat experience |
| **Containerized Deployment** | Dockerfile with layer-optimized caching for fast, reproducible builds |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Interface                             │
│          Chat Interface (index.html)  │  Health Form (form.html)   │
└────────────────────────┬────────────────────────┬───────────────────┘
                         │                        │
                         ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Flask Application (app.py)                       │
│                      CKDChatbotCore Orchestrator                    │
│   POST /chat           │              POST /predict                 │
└────────────┬───────────┴──────────────────┬────────────────────────┘
             │                              │
             ▼                              ▼
┌────────────────────────┐    ┌─────────────────────────────────────┐
│    RAG Pipeline        │    │         ML Model Pipeline            │
│  (rag_pipeline.py)     │    │          (ml_model.py)               │
│                        │    │                                      │
│  ┌──────────────────┐  │    │  UCI CKD Dataset (398 records)       │
│  │  OpenAI GPT LLM  │  │    │  → KNN Imputation + Scaling          │
│  │  Temp: 0.4       │  │    │  → Label Encoding                    │
│  │  Max tokens: 500 │  │    │  → Random Forest (50 estimators)     │
│  └────────┬─────────┘  │    │  → ~98% Test Accuracy                │
│           │            │    └─────────────────┬───────────────────┘
│  ┌────────▼─────────┐  │                      │
│  │ LangChain Chain  │  │             Prediction + Advice
│  │ (RetrievalChain) │  │                      │
│  └────────┬─────────┘  │                      ▼
│           │            │         ┌─────────────────────────┐
│  ┌────────▼─────────┐  │         │    Result Page           │
│  │  Pinecone        │  │         │    (result.html)         │
│  │  Vector Store    │◄─┼─────────┤  Prediction + RAG Advice│
│  │  (384-dim cosine)│  │         └─────────────────────────┘
│  │  Top-k=3 chunks  │  │
│  └────────┬─────────┘  │
│           │            │
│  ┌────────▼─────────┐  │
│  │  Medical PDFs    │  │
│  │  (KDIGO, Brenner │  │
│  │  ESPEN, NIH)     │  │
│  │  all-MiniLM-L6-v2│  │
│  └──────────────────┘  │
└────────────────────────┘
```

### Design Patterns

- **Singleton Orchestrator** — `CKDChatbotCore` is instantiated once at startup and reused across all requests, eliminating costly model re-initialization
- **Modular Package Structure** — Business logic is fully isolated in `src/` with a clean public API via `__init__.py`
- **Embedding Cache** — Sentence transformer embeddings persisted to `embeddings.pkl`, eliminating HuggingFace Hub downloads on container restart
- **Separation of Concerns** — Flask routes contain zero business logic; all orchestration is delegated to typed, testable classes

---

## Tech Stack

### Core Infrastructure

| Layer | Technology | Purpose |
|---|---|---|
| Web Framework | Flask 3.0 | HTTP routing, request handling, template rendering |
| Containerization | Docker (Python 3.11-slim) | Reproducible builds, cloud deployment |
| Cloud Platform | Google Cloud Run | Serverless, auto-scaling deployment |

### Machine Learning

| Library | Version | Purpose |
|---|---|---|
| scikit-learn | 1.5.2 | Random Forest classifier, KNN imputation, preprocessing |
| pandas | 2.2.3 | Data ingestion, feature engineering |
| numpy | 1.26.4 | Numerical operations |
| joblib | 1.4.2 | Model artifact serialization / deserialization |

### AI & NLP

| Library | Version | Purpose |
|---|---|---|
| LangChain | 0.3.7+ | RAG chain orchestration, document loading, retrieval |
| OpenAI API | 1.54.5 | GPT LLM for response generation |
| Pinecone | 5.4.0 | Serverless vector database (AWS, cosine similarity) |
| sentence-transformers | 2.6.0 | `all-MiniLM-L6-v2` — 384-dim dense embeddings |
| PyPDF | 5.1.0 | Clinical PDF text extraction and chunking |

### Frontend

| Technology | Purpose |
|---|---|
| HTML5 / CSS3 | Responsive layouts, form design, results display |
| Vanilla JavaScript | Async fetch, typing indicators, real-time chat UX |

---

## ML Model Performance

The CKD classifier is trained on the **UCI Chronic Kidney Disease Dataset** (398 records, 24 original features) using a carefully engineered pipeline:

### Feature Engineering Pipeline

```
Raw Data (24 features)
        │
        ▼
Feature Selection (15 clinical biomarkers)
        │
        ▼
Missing Value Imputation (KNNImputer, k=5, with StandardScaler)
        │
        ▼
Categorical Encoding (LabelEncoder: yes/no, good/poor, normal/abnormal)
        │
        ▼
Random Forest Classifier
  • n_estimators  = 50
  • max_features  = "sqrt"
  • min_samples_split = 2
  • min_samples_leaf  = 1
  • random_state  = 42
        │
        ▼
~98% Test Accuracy
```

### Measured Performance (evaluate_ml.py)

| Metric | Test Set (n=80) | 5-Fold CV (n=400) |
|---|---|---|
| **Accuracy** | **98.75%** | 98.00% ± 1.70% |
| **Precision** | **98.11%** | 98.03% ± 1.75% |
| **Recall (Sensitivity)** | **100.00%** | 98.80% ± 0.98% |
| **F1-Score** | **99.05%** | 98.41% ± 1.34% |
| **ROC-AUC** | **1.0000** | 0.9987 ± 0.0012 |

> The model achieves perfect recall on the test set — it does not miss a single true CKD case — which is the clinically critical direction for a screening tool.

Evaluation plots saved to `reports/`: confusion matrix, ROC curve, SHAP feature importance bar chart, and SHAP beeswarm summary.

### Selected Clinical Features

| Category | Biomarkers |
|---|---|
| Blood Chemistry | Hemoglobin, Serum Creatinine, Blood Urea, Blood Glucose, Albumin, Sodium |
| Hematology | Red Blood Cell Count, Packed Cell Volume |
| Urine Analysis | Specific Gravity, Sugar |
| Vitals | Blood Pressure |
| Medical History | Hypertension, Diabetes Mellitus, Pedal Edema |
| Symptoms | Appetite |

---

## RAG Pipeline

The conversational assistant is powered by a LangChain-based RAG pipeline that grounds every response in curated, peer-reviewed clinical literature.

### Knowledge Base

| Document | Source | Content |
|---|---|---|
| KDIGO 2012 CKD Clinical Practice Guidelines | Kidney Disease: Improving Global Outcomes | Staging, diagnosis, management protocols |
| Brenner and Rector's The Kidney | Elsevier (Textbook) | Comprehensive nephrology reference |
| ESPEN Clinical Nutrition Guidelines | European Society for Clinical Nutrition | Dietary management for kidney disease |
| NIH Bookshelf — Kidney Disease Reference | National Library of Medicine | Patient-facing disease information |

### Pipeline Configuration

```
User Query
    │
    ├─────────────────────────────────────────┐
    ▼                                         ▼
all-MiniLM-L6-v2 Encoder (384-dim)       BM25 Keyword Search
    │                                         │
    ▼                                         │
Pinecone Serverless Index                     │
  (cosine similarity, Top-k×3 chunks)         │
    │                                         │
    └──────────────┬──────────────────────────┘
                   ▼
        Reciprocal Rank Fusion (RRF)
          β=0.5  |  Top-5 chunks
                   │
                   ▼
        OpenAI GPT-4o-mini (temp=0.4, max_tokens=500)
          System Persona: "KidneyCareAI — friendly, knowledgeable medical assistant"
          • Grounds claims in retrieved documents
          • Signals source usage via [SOURCES_USED] / [NO_SOURCES] tags
          • Distinguishes medical information from medical advice
                   │
                   ▼
        Source Detection & Extraction
          • [SOURCES_USED] tag triggers metadata extraction from retrieved chunks
          • Filename, page number, and content preview collected per source
          • Duplicates deduplicated; page numbers converted to 1-indexed
                   │
                   ▼
        Response → User  (answer text + source pill tags)
```

The **HybridRetriever** (`src/hybridsearch.py`) combines dense vector search (Pinecone) and sparse keyword search (BM25) via Reciprocal Rank Fusion, improving recall for queries that use precise clinical terminology not well-served by embedding similarity alone.

---

## Evaluation

The project ships three standalone evaluation scripts that measure quality at every layer of the stack.

### 1. ML Classifier — `evaluate_ml.py`

Evaluates the Random Forest model on the held-out test set and via 5-fold stratified cross-validation. Generates four diagnostic plots saved to `reports/`.

```bash
python evaluate_ml.py
```

### 2. RAG Pipeline — `evaluate_rag.py` (RAGAS)

Measures retrieval and generation quality across 20 curated questions (15 medical, 5 conversational) using four [RAGAS](https://docs.ragas.io/) metrics. Medical questions are scored on all four metrics; conversational questions on Answer Relevancy only.

```bash
python evaluate_rag.py                  # Full run (20 questions)
python evaluate_rag.py --sample 5       # Quick smoke-test
python evaluate_rag.py --top-k 8        # More retrieved chunks
```

| Metric | Medical Questions (n=15) |
|---|---|
| Context Precision | 0.571 |
| Context Recall | 0.568 |
| **Faithfulness** | **0.767** |
| **Answer Relevancy** | **0.818** |

Conversational Answer Relevancy: **0.373** — the chatbot tends to answer off-topic queries with medical content rather than cleanly redirecting, which is an identified improvement area.

### 3. End-to-End Coherence — `evaluate_e2e.py`

Runs 6 synthetic patient profiles (3 CKD, 3 non-CKD) through the full stack — biomarkers → Random Forest → RAG advice — and uses GPT-4o-mini as a judge to score whether the advice is coherent with the ML prediction (1–5 scale).

```bash
python evaluate_e2e.py
python evaluate_e2e.py --verbose    # Show RAG pipeline logs
```

| Metric | Result |
|---|---|
| ML Prediction Accuracy | **5 / 6 (83%)** |
| Avg Coherence Score (GPT-4o-mini judge) | **5.0 / 5** |
| CKD profiles coherence | 5.0 / 5 |
| Non-CKD profiles coherence | 5.0 / 5 |

The one misclassification was a borderline non-CKD profile (older adult, controlled diabetes, creatinine 1.0) that the model conservatively flagged as likely — a reasonable false positive for a screening tool. Despite the wrong prediction, the RAG advice was still rated fully coherent since it matched the output label.

Results are saved to `data/eval_rag_results_raw.json` and `data/eval_e2e_results.json`.

---

## Project Structure

```
Chronic-Kidney-Disease-Chatbot/
│
├── app.py                          # Flask app entry point & CKDChatbotCore orchestrator
├── evaluate_ml.py                  # RF classifier metrics + SHAP plots → reports/
├── evaluate_rag.py                 # RAGAS evaluation of hybrid RAG pipeline (20 questions)
├── evaluate_e2e.py                 # End-to-end coherence eval (6 synthetic patient profiles)
├── pyproject.toml                  # Project metadata & dependency declarations
├── requirements.txt                # Fully resolved dependency lockfile
├── Dockerfile                      # Container build instructions
├── embeddings.pkl                  # Cached sentence-transformer embeddings (91MB)
│
├── src/                            # Core business logic (modular package)
│   ├── __init__.py                 # Public API exports
│   ├── ml_model.py                 # Random Forest pipeline: train, impute, encode, predict
│   ├── document_processing.py      # PDF loading, text chunking, embedding generation
│   ├── vectorstore.py              # Pinecone index management & LangChain retriever
│   ├── hybridsearch.py             # HybridRetriever: BM25 + Pinecone + RRF fusion
│   └── rag_pipeline.py             # OpenAI chat completions RAG with source tagging
│
├── data/
│   ├── kidney_disease.csv          # UCI CKD dataset (398 records)
│   ├── kidney_disease_rf_model.pkl # Serialized Random Forest model artifact
│   ├── encoders.pkl                # Fitted categorical label encoders
│   ├── test_dataset.json           # 20 curated Q&A pairs for RAGAS evaluation
│   ├── eval_rag_results_raw.json   # Raw RAG query results (generated by evaluate_rag.py)
│   ├── eval_e2e_results.json       # E2E profile scores (generated by evaluate_e2e.py)
│   ├── Bookshelf_NBK51773.pdf      # NIH kidney disease reference
│   ├── Brenner_and_Rectors_*.pdf   # Nephrology textbook (~18MB)
│   ├── ESPEN-guideline_*.pdf       # Clinical nutrition guidelines
│   └── KDIGO_2012_CKD_GL.pdf       # KDIGO clinical practice guidelines
│
├── reports/                        # Generated by evaluate_ml.py
│   ├── confusion_matrix.png        # Confusion matrix on held-out test set
│   ├── roc_curve.png               # ROC curve (AUC = 1.000)
│   ├── shap_importance.png         # Mean |SHAP| feature importance bar chart
│   └── shap_beeswarm.png           # SHAP beeswarm summary plot
│
├── templates/
│   ├── index.html                  # Chat interface
│   ├── form.html                   # 15-parameter clinical data entry form
│   └── result.html                 # Prediction result & personalized advice
│
└── static/
    ├── scripts/index.js            # Async chat, typing indicator, message rendering
    ├── styles/
    │   ├── index.css               # Chat UI styles (includes source pill styles)
    │   ├── form.css                # Health form styles
    │   └── result.css              # Results page styles
    └── images/bot-icon.png         # KidneyCareAI avatar
```

---

## Quickstart

### Prerequisites

- Docker **or** Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [Pinecone API key](https://www.pinecone.io/) (free tier is sufficient)

### Option 1: Docker (Recommended)

The fastest path to a running instance.

```bash
# 1. Clone the repository
git clone https://github.com/Ezekwemdesmond/Chronic-Kidney-Disease-Chatbot.git
cd Chronic-Kidney-Disease-Chatbot

# 2. Create your environment file
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
EOF

# 3. Build the Docker image
docker build -t ckd-chatbot .

# 4. Run the container
docker run -p 5000:5000 --env-file .env ckd-chatbot

# 5. Open http://localhost:5000 in your browser
```

### Option 2: Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Ezekwemdesmond/Chronic-Kidney-Disease-Chatbot.git
cd Chronic-Kidney-Disease-Chatbot

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
EOF

# 5. Start the application
python app.py

# 6. Open http://localhost:5000 in your browser
```

> **Note:** On first run, the application will connect to Pinecone and initialize the vector index. If the index is empty, ensure your documents are in the `data/` directory before running.

---

## Usage Guide

### Chat Interface

Navigate to the main page at `http://localhost:5000` to access the conversational assistant.

**Example queries:**
- *"What are the early symptoms of chronic kidney disease?"*
- *"How does diabetes affect kidney function?"*
- *"What dietary changes are recommended for CKD stage 3?"*
- *"Explain the KDIGO staging system for CKD."*

The assistant retrieves relevant passages from the medical knowledge base and synthesizes a grounded, factual response. When sources are used, **source pill tags** appear below the answer showing the document filename and page number — hover over a tag to preview the retrieved excerpt.

### CKD Risk Assessment

Click **"Check Your Risk"** to open the clinical data entry form. Enter values for the 15 biomarkers:

| Section | Parameters |
|---|---|
| Vital Signs | Blood Pressure, Packed Cell Volume |
| Blood Chemistry | Serum Creatinine, Blood Urea, Hemoglobin, RBC Count, Blood Glucose, Sodium, Albumin |
| Urine Analysis | Specific Gravity, Urinary Sugar |
| Medical History | Hypertension, Diabetes Mellitus, Pedal Edema |
| Lifestyle & Symptoms | Appetite |

Submit the form to receive:
1. A **CKD Risk Classification** (Likely / Unlikely)
2. **Personalized guidance** generated by the RAG pipeline, contextualized to the prediction result

---

## Dataset

This project uses the **UCI Chronic Kidney Disease Dataset**.

| Attribute | Detail |
|---|---|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease) |
| Records | 398 patient records |
| Original Features | 24 (11 numeric, 13 categorical) |
| Features Used | 15 (selected by clinical relevance) |
| Class Balance | ~62% CKD, ~38% non-CKD |
| Missing Data | Handled via KNN imputation |

---

## Roadmap

- [x] **Source Citations** — Display retrieved document name and page number as pill tags below each medical response
- [x] **Hybrid RAG Retrieval** — BM25 keyword search fused with Pinecone dense retrieval via Reciprocal Rank Fusion (β=0.5, Top-5 chunks)
- [x] **Evaluation Suite** — Three evaluation scripts covering ML metrics + SHAP, RAGAS (4 metrics, 20 questions), and end-to-end coherence (6 synthetic profiles, GPT-4o-mini judge)
- [ ] **SHAP Explainability UI** — Expose SHAP feature importance on the results page so users see which biomarkers drove the prediction
- [ ] **CKD Stage Prediction** — Extend the classifier from binary to multi-class (Stages 1–5 + ESRD)
- [ ] **Conversational Redirect** — Improve chatbot handling of off-topic and conversational queries (current Answer Relevancy: 0.37) with a dedicated intent classifier
- [ ] **Context Precision Improvements** — Address the 0.57 Context Precision score by experimenting with re-ranking (cross-encoder) and finer chunk sizing
- [ ] **Patient History** — Persist conversation and prediction history per session for longitudinal tracking
- [ ] **Authentication** — Add user accounts for secure, private health data management
- [ ] **CI/CD Pipeline** — GitHub Actions workflow for automated testing and Cloud Run deployment

---

## Contributing

Contributions are welcome. Please follow this workflow:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature-name`
3. **Commit** your changes: `git commit -m 'feat: add some feature'`
4. **Push** to your branch: `git push origin feature/your-feature-name`
5. **Open** a Pull Request with a clear description of the change and its motivation

Please ensure any new code is covered by tests and that existing tests pass before submitting.

---

## Disclaimer

KidneyCareAI is an **educational and research tool only**. It is not a substitute for professional medical advice, diagnosis, or treatment. The predictions and information provided by this application should never be used to make clinical decisions. Always seek the guidance of a qualified healthcare professional with any questions you may have regarding a medical condition.

---

## Acknowledgements

- **UCI Machine Learning Repository** — Chronic Kidney Disease dataset
- **KDIGO** — Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease (2012)
- **ESPEN** — Clinical Nutrition in Chronic Kidney Disease guidelines
- **LangChain** — RAG orchestration framework
- **Pinecone** — Serverless vector database infrastructure
- **OpenAI** — Large language model API

---

## Contact

**Desmond Ezekwem**

[![Email](https://img.shields.io/badge/Email-engrstephdez%40gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:engrstephdez@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-Ezekwemdesmond-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Ezekwemdesmond)

---

<div align="center">
  <sub>Built with care for the intersection of AI and clinical healthcare</sub>
</div>
