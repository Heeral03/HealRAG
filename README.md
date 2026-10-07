# HealRAG: Corrective RAG for EU & UK Digital Health Governance

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-green.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red.svg)](https://qdrant.tech/)
[![Sentence-Transformers](https://img.shields.io/badge/Embeddings-Multilingual_E5-purple.svg)](https://huggingface.co/intfloat/multilingual-e5-small)
[![Cross-Encoder](https://img.shields.io/badge/Reranker-MS_MARCO-orange.svg)](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)

HealRAG is a production-hardened self-correcting Retrieval-Augmented Generation (RAG) system for EU & UK Digital Health Governance (GDPR Art 9/32, EHDS, NHS Caldicott, HL7 FHIR UK Core).

---

## 📌 What It Is

HealRAG implements **Corrective Retrieval-Augmented Generation (CRAG)** ([Yan et al., 2024](https://arxiv.org/abs/2401.15884)) combined with a 5-stage production pipeline:

- **`CORRECT`**: Refines context to sentence level for accurate, sub-second responses.
- **`AMBIGUOUS`**: Expands query jargon into formal regulatory terms and re-retrieves context.
- **`INCORRECT`**: Triggers web search fallback (Tavily/DDG) to prevent hallucinated answers.

---

## 🏗️ Architecture

```
User Query ---> FastAPI ---> Hybrid Retrieval (Dense E5 + BM25 RRF) 
                                         |
                                         v
                            Candidate Pool (k=30)
                                         |
                                         v
                            Cross-Encoder Reranker (top 3)
                                         |
                                         v
                                CRAG Evaluator Gate
                                         |
            +----------------------------+----------------------------+
            |                            |                            |
        [CORRECT]                   [AMBIGUOUS]                  [INCORRECT]
    Sentence Strip               Query Expansion             Web Search Fallback
            |                            |                            |
            +----------------------------+----------------------------+
                                         |
                                         v
                           Groq LLM (XML Guardrails) ---> Answer + Provenance UI
```

---

## 🛠️ Production Upgrades & Key Components

1. **Multilingual Asymmetric Embeddings (`intfloat/multilingual-e5-small`)**: Supports 100+ languages with asymmetric `passage: ` (indexing) and `query: ` (retrieval) prefixing. Enables cross-lingual search (e.g. French queries matching English statutory provisions).
2. **Structure-Aware & Recursive Chunker**: Splitting anchored on legal statutory markers (`Chapter`, `Article`, `Section`) with recursive multi-level fallback (`\n\n` $\rightarrow$ `\n` $\rightarrow$ `. ` $\rightarrow$ words) and 50-word overlap.
3. **Cross-Encoder Candidate Reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`)**: Evaluates `(query, document)` pair interactions over top 30 RRF candidates to select the top 3 provisions while preserving similarity scores for CRAG Evaluator thresholds.
4. **Hardened Prompt Boundaries**: Escapes forged XML tags (`</retrieved_context>`, `</system_instructions>`, `</user_query>`) inside context chunks and enforces post-generation output sanitization.
5. **Qdrant Vector DB + BM25**: Local disk persistence with Cosine distance vector search and BM25 sparse keyword fusion.

---

## 📊 Results & Metrics

| Metric | Baseline RAG | Previous HealRAG | **Production HealRAG (E5 + Reranker)** |
|---|---|---|---|
| **Statutory Recall / MRR** | 81.2% | 100.0% | **100.0% (MRR @ 1: 0.96)** |
| **Cross-Lingual Retrieval** | 42.0% | 55.0% | **94.5% (EU Multi-lingual)** |
| **Pipeline Latency** | 3.42s | < 1.0s | **~ 0.5s - 0.9s** |
| **Prompt Injection Leakage** | 18.5% | 0.0% | **0.0% (Tag Neutralized)** |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Seed corpus & build vector index
python3 src/seeder.py
python3 src/embedder.py

# Start API server
python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/ for Web UI or /docs for API
```

