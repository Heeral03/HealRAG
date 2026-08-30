# HealRAG: A Corrective RAG System for EU & UK Digital Health Governance

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-green.svg)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-orange.svg)](https://github.com/facebookresearch/faiss)
[![Groq LLM](https://img.shields.io/badge/LLM-Groq--120B-purple.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**What it is:** HealRAG is a Retrieval-Augmented Generation system that answers complex queries about **EU and UK digital health regulations** (GDPR, EHDS, UK NHS Caldicott Principles, NHS DSPT, National Data Opt-out, and HL7 FHIR / UK Core standards) — but unlike a standard RAG pipeline, it doesn't blindly trust whatever it retrieves. It implements the core idea from Yan et al.'s *Corrective Retrieval Augmented Generation* (CRAG) paper: **evaluate retrieval quality before generating an answer, and self-correct when that retrieval is weak.**

---

### The Problem It Addresses

Standard RAG systems are structurally naive — they retrieve top-k chunks and feed them into an LLM regardless of whether those chunks actually answer the question. In a regulatory/healthcare domain, that's a real liability: a confidently wrong answer about GDPR consent rules or emergency data-access provisions isn't a minor bug, it's a compliance risk. I set out to build a system that knows when its own retrieval has failed, rather than papering over that failure with a hallucinated answer.

---

### System Architecture

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

The pipeline has three stages beyond a standard RAG system:

1. **Retrieval Evaluator** — after FAISS-based vector retrieval (`all-MiniLM-L6-v2` embeddings, cosine similarity), a lightweight evaluator scores retrieval confidence using a weighted combination of similarity score (70%) and query-term coverage (30%), classifying the result into three tiers: **CORRECT** (≥0.60), **AMBIGUOUS** (0.45–0.60), or **INCORRECT** (<0.45).
2. **Knowledge Refiner** — for high-confidence retrievals, decomposes chunks down to the sentence level and strips irrelevant surrounding text before it ever reaches the LLM, reducing prompt noise and token footprint.
3. **Query Expansion + Web Fallback** — for ambiguous or incorrect retrievals, a query-expansion step rewrites informal or jargon-heavy queries into formal terminology (e.g., "break-glass" → "EHDS Chapter II Article 7 emergency access override"), and the system falls back to external web search when the local corpus genuinely doesn't contain the answer — rather than returning a dead-end refusal.

These three stages route dynamically based on the evaluator's decision, forming a self-correcting pipeline rather than a fixed, one-shot retrieve-and-generate flow.

---

### A Key Engineering Decision — And Why It Matters

The CRAG paper uses a fine-tuned model as its retrieval evaluator. I made a deliberate trade-off instead: a **fast heuristic evaluator** (weighted similarity + keyword coverage) rather than an LLM call on every retrieved chunk. Calling an LLM to judge every chunk would add 1.5–3 seconds of latency *per chunk* — untenable in a system meant to feel responsive. The heuristic evaluator runs in sub-millisecond time while still reliably separating clean retrievals from weak ones — a conscious latency-vs-fidelity trade-off, not a simplification made out of convenience. *(Note: Option for T5 seq2seq evaluator backend `T5RetrievalEvaluator` is also implemented in `src/evaluator.py`).*

---

### Proving It Works: Methodology Before Implementation

Before writing any correction logic, I built the evidence that correction was needed. I ran an 18-question baseline suite (Easy/Hard/Adversarial tiers) against vanilla RAG, then designed a **separate 6-question stress suite specifically targeting four failure modes** I'd identified through manual analysis: semantic drift between jargon and formal legal text, static-corpus dead-ends, noisy document distraction, and blind one-shot generation with no self-awareness of retrieval quality. This let me build CRAG against concrete, empirically-observed failures rather than the paper's abstract description of the problem.

---

### Results — A Full Ablation Study, Not A Single Demo

| Benchmark | Vanilla RAG | CRAG | Delta |
|---|---|---|---|
| Standard baseline (18 Qs) | 88.9% | 88.9% | No regression on clean queries |
| Failure-mode stress suite (6 Qs) | 0.0% | 50.0% | +50% resolution on structural failures |
| Combined (24 Qs) | 66.7% | 79.2% | +12.5% absolute improvement |
| Avg. latency | 4.11s | 4.63s | +12.6% aggregate overhead |
| Latency on clean queries (fast path) | 4.11s | 1.21–1.76s | **~2.5x faster** — noise stripping shrinks the prompt |
| Latency on fallback queries | N/A | 6.06–14.23s | Overhead isolated to genuinely low-confidence cases |
| Prompt tokens (fast path) | ~1,200 | ~450 | ~62.5% cost reduction |

The counterintuitive finding worth highlighting: **CRAG is faster and cheaper than vanilla RAG on the majority of queries**, because the knowledge refiner strips noise before generation. The latency overhead only appears on the minority of queries where retrieval genuinely failed — exactly where spending an extra few seconds to get a correct answer instead of a hallucinated one is the right trade-off.

---

### The Engineering Trade-Off, Stated Plainly

CRAG's net cost is close to neutral-to-positive on a real query distribution: most production queries hit the fast path and get *faster, cheaper, cleaner* answers than vanilla RAG; the latency cost is concentrated entirely on the edge cases where correction genuinely matters. You're trading a few extra seconds on a minority of queries for a 50% resolution rate on failure modes that would otherwise silently return wrong or dead-end answers.

---

## 💻 Usage Options

### 1. Installation & Environment

```bash
git clone https://github.com/Heeral03/HealRAG.git
cd HealRAG
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

### 2. CLI Execution

```bash
# Execute CRAG Pipeline (Default)
python3 src/main.py --query "What is break-glass emergency?"

# Run Vanilla RAG Baseline for comparison
python3 src/main.py --query "What is break-glass emergency?" --vanilla
```

### 3. Start FastAPI REST Service

```bash
python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```
- Interactive Swagger Documentation: [`http://localhost:8000/docs`](http://localhost:8000/docs)
- Health Check: `curl http://localhost:8000/health`
- Query Endpoint:
  ```bash
  curl -X POST "http://localhost:8000/query" \
       -H "Content-Type: application/json" \
       -d '{"query": "What is GDPR Article 9?", "top_k": 3}'
  ```

### 4. Run Full Ablation Study

```bash
python3 eval/run_full_ablation_benchmark.py
```

---

## Repository Structure

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

## License

Distributed under the MIT License. See `LICENSE` for more information.
