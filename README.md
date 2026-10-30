# HealRAG: Corrective RAG for EU & UK Digital Health Governance

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red.svg)](https://qdrant.tech/)
[![Embeddings](https://img.shields.io/badge/Embeddings-E5%20%2F%20MiniLM%20(ONNX)-purple.svg)](https://huggingface.co/intfloat/multilingual-e5-small)
[![Reranker](https://img.shields.io/badge/Reranker-MS_MARCO_Cross--Encoder-orange.svg)](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)

HealRAG is a self-correcting Retrieval-Augmented Generation (RAG) system for EU & UK digital health governance: GDPR (Art. 5/6/9/32 and the scope and definition articles), the European Health Data Space (EHDS), NHS Caldicott and data-security guidance, and HL7 FHIR UK Core.

---

## What it is

HealRAG implements **Corrective RAG (CRAG)** ([Yan et al., 2024](https://arxiv.org/abs/2401.15884)). Plain RAG retrieves and then generates, even when the retrieval was poor. HealRAG scores the retrieved evidence first and chooses a different path depending on how good it is:

| Route | Meaning | Action |
|---|---|---|
| **CORRECT** | Local evidence is relevant | Strip each chunk down to its supporting sentences, then generate |
| **AMBIGUOUS** | Partial match | Refine local chunks, add web search results, then generate |
| **INCORRECT** | Local evidence is irrelevant | Discard it, use web search only, then generate |

A second safety net runs **after** generation (see [Answerability check](#4-answerability-check-generation-time-correction)): if the model says the context cannot answer the question, the pipeline retrieves again with an expanded query and regenerates once.

---

## Architecture

### Offline: build the index

```
 seeder.py                chunker.py                  embedder.py                    storage
┌──────────────┐   ┌─────────────────────┐   ┌───────────────────────┐   ┌──────────────────────────┐
│ ~110 docs:   │   │ split on Chapter /  │   │ ONNX embeddings       │   │ Qdrant collection        │
│ GDPR, EHDS,  │──▶│ Article / Section,  │──▶│ ("passage: " prefix   │──▶│  vector + payload/point  │
│ NHS, FHIR    │   │ recursive fallback, │   │  for E5), L2-normed   │   │  (cosine, db/qdrant)     │
│ + .json      │   │ 400 words, 55 ovlp  │   │                       │   │ metadata.json (for BM25) │
│ sidecars     │   │ + metadata per chunk│   │                       │   │                          │
└──────────────┘   └─────────────────────┘   └───────────────────────┘   └──────────────────────────┘
```

### Online: answer a query

```
User query ─▶ FastAPI (auth, rate limit) ─▶ Semantic cache ──hit──▶ cached answer
                                                  │ miss
                                                  ▼
                     ┌──────────────── Hybrid retrieval ────────────────┐
                     │  Dense: Qdrant (cosine)     Sparse: BM25         │
                     │  incl. article-reference tokens e.g. ref_9_2_a   │
                     │          └──── Reciprocal Rank Fusion ────┘      │
                     │  Candidate pool (k=30-50) ─▶ Cross-encoder rerank│
                     └──────────────────────────┬───────────────────────┘
                                                ▼
                                  CRAG evaluator (confidence gate)
                      ┌─────────────────────────┼─────────────────────────┐
                  [CORRECT]                [AMBIGUOUS]               [INCORRECT]
              sentence-level            refine local +             web search only
              knowledge strip           web search (Tavily/DDG)    (local discarded)
                      └─────────────────────────┼─────────────────────────┘
                                                ▼
                           Generator (Groq LLM, XML-bounded prompt)
                                                ▼
                         Answerability check ── refusal? ──▶ expand query,
                                                │            re-retrieve, add web,
                                                │            regenerate once
                                                ▼
                       Answer + provenance + per-stage latencies ─▶ SQLite log
```

---

## How it works

### 1. Indexing (offline)

- **Corpus (`seeder.py`)**: authoritative provisions condensed into the repo, plus 75 synthetic FHIR patient records used to exercise record-style retrieval. Each document is written as a `.txt` file with a `.json` sidecar holding publisher, jurisdiction, document type, version, effective date, source URL, a SHA-256 `content_hash`, and a `hierarchy_rank` (primary law = 1 < regulator guidance < technical standard < commentary = 4).
- **Chunking (`chunker.py`)**: structure-aware first, splitting before `Chapter`, `Article` and `Section` markers so a chunk lines up with a legal provision. Anything over 400 words is split recursively (paragraph, line, sentence, word) with a 55-word overlap. Every chunk carries the full sidecar metadata.
- **Embedding (`embedder.py`)**: `fastembed` (ONNX, no PyTorch) behind a small SentenceTransformer-compatible wrapper, shared as a single instance to save RAM. E5 models get asymmetric prefixes: `passage: ` at index time and `query: ` at search time.

### 2. Retrieval (online)

- **Qdrant (dense)**: one point per chunk, a normalised vector with the chunk metadata as its payload, cosine distance. It runs in **embedded mode** (`QdrantClient(path=...)`), so it is a persistent on-disk database inside the Python process with no separate server and no network hop. Payload comes back with every hit, so provenance (publisher, jurisdiction, version, hierarchy rank) needs no second lookup.
- **BM25 (sparse)**: built in memory from `metadata.json`. The tokenizer also emits citation tokens (`9(2)(a)` becomes `ref_9_2_a`), so exact article references match exactly.
- **Reciprocal Rank Fusion**: `1/(60 + dense_rank) + 1/(60 + bm25_rank)` merges the two ranked lists without having to normalise their different score scales.
- **Cross-encoder rerank**: `ms-marco-MiniLM-L-6-v2` reads each (query, chunk) pair jointly and picks the final top-k. Chunks whose heading carries the exact article reference named in the query are placed first, because a semantic model cannot tell `9(2)(a)` from `9(2)(h)`.

### 3. Evaluate, route, generate

- **Evaluator (`evaluator.py`)**: a deterministic score per chunk, `0.7 x dense_similarity + 0.3 x keyword_coverage`. A numeric term in the query (for example `32`) that is missing from the chunk halves the similarity. The best chunk decides the route: `>= 0.60` CORRECT, `0.45 to 0.60` AMBIGUOUS, `< 0.45` INCORRECT (thresholds in `config.py`). A T5 backend like the paper's is available as an option.
- **Refiner (`refiner.py`)**: keeps only sentences that overlap with the query terms (knowledge stripping).
- **Web fallback (`searcher.py`)**: Tavily, then DuckDuckGo, then an offline mock, with a TTL cache. Queries are expanded for known jargon and acronyms (`gdpr`, `ehds`, `caldicott`, `break-glass`, ...).
- **Generator (`generator.py`)**: Groq LLM via LangChain. The prompt uses `<system_instructions>`, `<retrieved_context>` and `<user_query>` boundaries. Forged boundary tags inside retrieved text are neutralised, and the output is sanitised, including while streaming. If the model call fails, a structured mock answer is returned.

### 4. Answerability check (generation-time correction)

The evaluator measures how close the chunks are to the query, not whether they answer it. A question such as *"What is GDPR?"* can retrieve chunks that mention the GDPR without defining it and still score as CORRECT. So after generation the pipeline checks whether the model replied with the mandated refusal sentence (`NO_ANSWER_SENTENCE` in `generator.py`). If it did:

1. re-retrieve locally with an expanded query,
2. add web results if the web fallback has not run yet,
3. regenerate **once**, and only if there is new evidence.

If this recovers an answer the route becomes AMBIGUOUS; if nothing can answer, the route becomes INCORRECT and the honest refusal is kept. The initial route is preserved as `observability.initial_eval_action`. Refusals are never written to the semantic cache. The corpus also carries the GDPR scope and definition articles (Art. 1-4) and a short GDPR overview so that definitional questions can be answered locally.

### 5. Serving and observability

- **FastAPI** endpoints: `POST /query`, `GET /health`, `GET /analytics`, `GET /rate-limit-status`, `POST /auth/generate`, `POST /auth/revoke`, and the web UI at `/`.
- **Security**: `X-API-Key` authentication with scopes, a token-bucket rate limiter priced by estimated token cost, hardened prompt boundaries.
- **Semantic cache** (cosine >= 0.95) skips the whole pipeline for near-duplicate queries.
- **Telemetry**: per-stage latencies, trust grade, and the winning chunk's provenance are returned in the response and logged to SQLite (`db.py`).

---

## Evaluation

Two generations of evaluation live in the repo.

### `evals/` (DeepEval, LLM-as-judge) with golden datasets in `goldens/`

Components are tested separately so a failure points at one stage.

| Script | Golden set | Measures |
|---|---|---|
| `eval_retriever.py` | `golden_retriever.json` (20) | Contextual Recall and Precision |
| `eval_generator.py` | `golden_generator.json` (20) | Faithfulness and Answer Relevancy, fed the *gold* chunks so generation is isolated |
| `eval_pipeline.py` | `golden_retriever.json` | End-to-end Contextual Relevancy, Faithfulness, Answer Relevancy, plus deterministic Hit@k / MRR (`--retrieval-only` uses no tokens) |
| `geval_correctness.py` | `golden_retriever.json` | G-Eval correctness against the ideal answer |
| `eval_scope.py` | `golden_scope.json` (14) | Answers in-scope questions, declines off-topic ones, does not over-refuse |
| `eval_toxicity.py` | `golden_toxicity.json` (9) | No toxic replies when provoked |
| `eval_leakage.py` | `golden_leakage.json` (13) | No system prompt, credential, or identifier leaks (8-word shingle match, secret patterns, planted canaries, plus a judge rubric and poisoned-chunk cases) |

The judge is a Groq-hosted model wrapped in `src/groq_judge.py` (OpenAI is used if `OPENAI_API_KEY` is set). The safety evals share `safety_runner.py` and exit non-zero on any failure, so they can gate CI. A saved run in `scope_results.json` passed 13 of 14 scope cases.

### `eval/` (benchmark scripts)

Information-retrieval metrics, vanilla-vs-CRAG ablations, stress tests, concurrency benchmarks, and a pipeline profiler. Hybrid retrieval clearly beat dense-only on the 50-query IR set (measured with `all-MiniLM-L6-v2` on the earlier FAISS build, before the move to Qdrant):

| Cutoff | Dense Hit Rate | Hybrid (BM25 + RRF) Hit Rate | Hybrid MRR |
|---|---|---|---|
| k=1 | 66% | **88%** | 0.88 |
| k=3 | 78% | **94%** | 0.91 |
| k=5 | 86% | **94%** | 0.91 |
| k=10 | 88% | **100%** | 0.92 |

> These numbers pre-date the Qdrant migration and the added GDPR documents. Re-run `eval/run_ir_metrics.py` to refresh them.

---

## Quick start

```bash
pip install -r requirements.txt
# create a .env file with GROQ_API_KEY (and optionally TAVILY_API_KEY)

# Seed the corpus and build the Qdrant index
python3 src/seeder.py
python3 src/embedder.py

# Start the API (web UI at http://localhost:8000/, API docs at /docs)
python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

> Embedded Qdrant allows one process per index directory. Stop the API before rebuilding the index, and rebuild whenever the corpus changes.

```bash
# Tests
pytest tests/

# Evals (need GROQ_API_KEY)
python3 evals/eval_retriever.py --limit 5
python3 evals/eval_pipeline.py --retrieval-only     # free: Hit@k / MRR only
python3 evals/eval_scope.py
python3 evals/eval_leakage.py
```

### Deployment

The `Dockerfile` seeds the corpus and builds the Qdrant index at image build time, so containers start with the index ready. `render.yaml` targets a small free instance: it uses `all-MiniLM-L6-v2`, disables the cross-encoder (`ENABLE_CROSS_ENCODER=0`), and runs embeddings on ONNX to fit in limited RAM. Locally the default is `intfloat/multilingual-e5-small` with the reranker enabled.

---

## Project layout

```
src/
  seeder.py  chunker.py  embedder.py     offline: corpus, chunks, Qdrant index
  retriever.py  reranker.py              hybrid retrieval, RRF, cross-encoder
  evaluator.py  refiner.py  searcher.py  CRAG gate, knowledge stripping, web fallback
  generator.py  crag_pipeline.py         prompt, LLM, orchestration, answerability check
  api.py  auth.py  rate_limiter.py       serving and security
  cache.py  db.py  config.py             semantic cache, SQLite logging, settings
evals/  goldens/                         DeepEval suites and golden datasets
eval/                                    benchmark and ablation scripts
tests/                                   unit, security and corrective-loop tests
static/                                  web UI
```

---

## Known limitations

- **Embedded Qdrant is single-process** and has no replication. At larger scale, move to Qdrant Server or Cloud (same client API), add payload indexes on `jurisdiction` and `effective_from`, and replace the in-memory BM25 with Qdrant's native sparse vectors and server-side fusion.
- **The index is rebuilt in full** on any corpus change; there is no incremental update. `content_hash` is stored to enable that later.
- **The evaluator is a heuristic** with hand-set thresholds. The answerability check catches its false positives, but calibrating the thresholds on labelled answerable and unanswerable queries is still open work.
- **The corpus is small and partly synthetic.** Provisions are condensed rather than scraped live, and the patient records are generated. Retrieval metrics on it are indicative only.
- **Eval datasets are small** (9 to 20 cases each), and the judge and the generator are from the same model family, which can inflate scores.
