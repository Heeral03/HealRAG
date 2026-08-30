# HealRAG 🏥⚖️
### Production-Grade Digital Health Corrective RAG (CRAG) Engine

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-green.svg)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-orange.svg)](https://github.com/facebookresearch/faiss)
[![Groq LLM](https://img.shields.io/badge/LLM-Groq--120B-purple.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**HealRAG** is an advanced, production-grade **Corrective Retrieval-Augmented Generation (CRAG)** framework tailored for digital health regulations, European health data governance, and clinical interoperability standards (GDPR, EU EHDS, HL7 FHIR, IPS).

Standard Vanilla RAG systems suffer from critical failure modes in zero-trust domains:
- **Blind Trust**: Feeding low-confidence retrieved chunks into an LLM, leading to hallucinations or misleading compliance advice.
- **Jargon & Semantic Drift**: Failing when industry jargon (*"break-glass emergency"*) differs from statutory text (*"EHDS Chapter II Article 7 emergency access override"*).
- **Static Corpus Limits**: Complete inability to answer out-of-corpus queries due to lack of external search fallbacks.
- **Document Distraction**: Diluting LLM attention with dense, noisy patient record dumps.

HealRAG resolves these failure modes by combining an active **Retrieval Evaluator**, a **Knowledge Refiner** (sentence-level noise stripper), a **Query Expansion Engine**, and an **External Web Search Fallback**.

---

## 🏛️ System Architecture

```
                            +-------------------+
                            |    User Query     |
                            +---------+---------+
                                      |
                                      v
                            +---------+---------+
                            | FAISS Vector DB   |
                            | (all-MiniLM-L6)   |
                            +---------+---------+
                                      |
                                      v
                            +---------+---------+
                            | CRAG Evaluator    |
                            | (Score Chunks)    |
                            +---------+---------+
                                      |
        +-----------------------------+-----------------------------+
        | (Score >= 0.60)             | (0.45 <= Score < 0.60)      | (Score < 0.45)
        v                             v                             v
  [ CORRECT ]                   [ AMBIGUOUS ]                 [ INCORRECT ]
        |                             |                             |
        v                             v                             v
+-------+-------+             +-------+-------+             +-------+-------+
| Knowledge     |             | Query         |             | External Web  |
| Refinement    |             | Expansion     |             | Search        |
| (Strip Noise) |             | + Search      |             | Fallback      |
+-------+-------+             +-------+-------+             +-------+-------+
        |                             |                             |
        +-----------------------------+-----------------------------+
                                      |
                                      v
                            +---------+---------+
                            | Groq LLM          |
                            | (openai/gpt-120b) |
                            +-------------------+
```

---

## 📊 Benchmark & Quantitative Ablation Study

HealRAG was benchmarked against a **24-question test battery** (18 baseline regulatory queries + 6 specialized failure-mode stress queries).

| Benchmark Metric | Vanilla RAG Baseline | Corrective RAG (CRAG) | Net Impact |
| :--- | :--- | :--- | :--- |
| **Standard Baseline Suite (18 Qs)** | **88.9%** (16/18) | **88.9%** (16/18) | **Preserved Precision**: 0% regression on clean local data. |
| **Failure Mode Stress Suite (6 Qs)** | **0.0%** (0/6) | **50.0%** (3/6) | **+50.0% Resolution Rate** on structural RAG failures. |
| **Overall Battery Accuracy (24 Qs)** | **66.7%** (16/24) | **79.2%** (19/24) | **+12.5% Absolute Improvement** overall. |
| **Average Query Latency** | **4.11s** | **4.63s** | **+12.6% Overhead (1.13x multiplier)**: Controlled latency trade-off. |
| **Prompt Token Footprint (`CORRECT`)** | ~1,200 tokens | **~450 tokens** | **~62.5% Token Cost Reduction** via Knowledge Refinement. |

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Heeral03/HealRAG.git
cd HealRAG

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root with your Groq API key:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

---

## 💻 Usage Options

### A. Run CLI

```bash
# Execute CRAG Pipeline (Default)
python3 src/main.py --query "What is break-glass emergency?"

# Run Vanilla RAG Baseline for comparison
python3 src/main.py --query "What is break-glass emergency?" --vanilla
```

### B. Start FastAPI REST Service

```bash
# Launch FastAPI server on port 8000
python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```
- **Interactive Swagger Documentation**: Open [`http://localhost:8000/docs`](http://localhost:8000/docs) in your browser.
- **Health Check**:
  ```bash
  curl http://localhost:8000/health
  ```
- **Query Endpoint**:
  ```bash
  curl -X POST "http://localhost:8000/query" \
       -H "Content-Type: application/json" \
       -d '{"query": "What is GDPR Article 9?", "top_k": 3}'
  ```

### C. Run Full Ablation Study

```bash
# Run 24-question quantitative benchmark
python3 eval/run_full_ablation_benchmark.py

# Run side-by-side qualitative comparison script
python3 eval/compare_vanilla_vs_crag.py
```

---

## 📁 Repository Structure

```
HealRAG/
├── src/
│   ├── config.py             # Centralized settings & evaluator thresholds
│   ├── seeder.py             # 110-document synthetic digital health regulatory corpus
│   ├── chunker.py            # Sliding-window word chunker
│   ├── embedder.py           # FAISS vector database builder
│   ├── retriever.py          # FAISS vector search engine (IndexFlatIP)
│   ├── generator.py          # Groq API LLM interface with citation formatting
│   ├── evaluator.py          # Retrieval Evaluator (Heuristic & T5 seq2seq options)
│   ├── refiner.py            # Knowledge Refiner (Sentence-level noise stripper)
│   ├── searcher.py           # Web Search Fallback & Query Expansion Engine
│   ├── crag_pipeline.py      # Core CRAG Orchestrator
│   ├── api.py                # Production FastAPI REST Web Service
│   └── main.py               # CLI Entrypoint
├── eval/
│   ├── eval_dataset.json            # 18 curated benchmark questions
│   ├── stress_test_dataset.json     # 6 failure mode stress queries
│   ├── run_full_ablation_benchmark.py # Quantitative ablation study runner
│   ├── compare_vanilla_vs_crag.py   # Side-by-side comparative runner
│   └── ablation_benchmark_results.json # Full benchmark output metrics
├── tests/
│   └── test_rag.py           # Automated unit tests
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
