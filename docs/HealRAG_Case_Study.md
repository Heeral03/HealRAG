# HealRAG: Project Case Study & Architectural Deep Dive

## 1. Project Overview
HealRAG is a specialized Retrieval-Augmented Generation (RAG) system engineered to answer complex regulatory questions regarding EU and UK digital health governance (GDPR, EHDS, UK NHS frameworks, and HL7 FHIR standards).

Unlike native, "naive" RAG systems that blindly trust whatever their vector database returns, HealRAG is built around a self-corrective intelligence loop. It evaluates its own retrieval quality *before* generating an answer, dynamically suppressing noise and fetching external web data when local documents fail.

## 2. Problems Faced: What Broke in Naive RAG?
When initially testing a standard Vanilla RAG baseline on healthcare regulations, four critical structural failures emerged:

1. **Semantic Drift:** Users asked about "break-glass emergency," but the legal documents strictly used "EHDS Chapter II Article 7 vital interests override". Vector similarity failed to link these concepts, resulting in low-quality retrieval.
2. **The "Yes-Man" Hallucination:** When no relevant documents existed in the corpus (e.g., questions about US HIPAA laws in a European corpus), the LLM still tried to answer using unrelated documents, hallucinating disastrously wrong compliance advice.
3. **Prompt Distraction:** Injecting huge 1000-word document chunks into the prompt caused the LLM to lose focus on the actual question, resulting in verbose, generalized answers rather than precise citations.
4. **The Metrics Lie:** Our standard evaluation scripts initially reported an artificially low 81.25% Hit Rate due to hidden ground-truth annotation bugs (e.g., expecting `doc_112_nhs_dspt_compliance.txt` when the engine correctly retrieved `doc_112_nhs_dspt_framework.txt`).

## 3. What We Solved & How We Tackled It
To solve these failures, we transitioned from a standard RAG to a Corrective RAG (CRAG) architecture, backed by extensive component profiling.

### 1. The CRAG Evaluator (Solving the "Yes-Man" Issue)
We introduced a custom **Retrieval Evaluator** between the DB and the LLM. It scores retrieved chunks using a composite heuristic (cosine similarity + keyword coverage ratio). If a chunk falls below a strict 0.45 threshold, the system flags the retrieval as `INCORRECT`. Instead of hallucinating, it automatically triggers a DuckDuckGo/Tavily web search to find the missing information externally.

### 2. Knowledge Refinement (Solving Prompt Distraction)
For `CORRECT` retrievals (score >= 0.60), we built a **Sentence-Level Refiner**. It breaks paragraphs into sentences and strips away any sentence that doesn't share keyword overlap with the original query. This reduced prompt sizes from ~1,200 tokens to ~450 tokens (a 62% reduction), making generation faster and much more precise.

### 3. Hybrid Search Selection (Solving Semantic Drift)
To catch both conceptual intent and exact statutory keyword matches, we replaced vanilla Dense Search with a **Hybrid Search Engine** combining FAISS (Dense Vectors) and BM25 (Sparse Keyword Match) using Reciprocal Rank Fusion (RRF). 

### 4. Semantic Response Cache (Solving LLM Latency Bottlenecks)
Through component-profiling, we proved the LLM generation caused ~88% of our latency. To combat this, we deployed a **Semantic Response Cache**. If an incoming query has high vector similarity (>= 0.95) to a previously answered question, the system returns the cached answer instantly (< 0.01ms), bypassing the LLM entirely.

## 4. Our Approach & Rationale
We prioritized **latency and determinism over black-box AI.**
The original CRAG academic paper mandated using a T5 LLM to evaluate retrieval chunks. We deliberately chose not to do this. Making multiple sub-LLM calls per query adds massive latency. Instead, we built a deterministic **heuristic evaluator** (cosine math + keyword overlap) that runs in `0.06 ms`. The trade-off was a minor loss in abstract semantic understanding, vastly outweighed by keeping the system lightweight and responsive.

When the web-search fallback (DuckDuckGo) proved to be slow (taking ~2,300ms due to scraping limits), we hot-swapped the backend to an official **Tavily API**, instantly shaving 1,000ms+ off the latency.

## 5. Final Results
After fully integrating all optimizations:
- **Accuracy Improvement:** We improved resolution on tough, out-of-corpus failure cases by **+50%** over standard RAG.
- **Search Robustness:** Hybrid Search achieved a flawless **100% Hit Rate @ k=3** on our IR Benchmark.
- **Latency Optimization:** 70% of standard queries run 2.5x faster and 45% cheaper than normal RAG due to our Noise Refiner reducing LLM prompt sizes.
- **Component Efficiency:** Because of semantic caching and API-swapping, the entire non-LLM portion of the pipeline executes in roughly ~20ms to ~1400ms depending on the exact route taken.

## 6. Scalability, Advantages, and Limitations

### Advantages & Why Someone Would Use This
1. **Uncompromising Accuracy:** In healthcare governance, a failed answer is better than a confident lie. A hospital compliance team would use HealRAG because of its provenance tracker: every answer comes tagged with a confidence grade (`HIGH_CONFIDENCE`, `LOW_CONFIDENCE_EXTERNAL_FALLBACK`) and exact document citations. 
2. **Cost Effective:** The pipeline dynamically scales its compute. Simple queries are cheap (noise stripped). Complex queries cost slightly more (web search triggered). Very common queries cost $0.00 (cache hit).
3. **High Throughput Safety:** It gracefully handles concurrent user loads, maintaining stable QPS throughput up to 128 queries per second at the retrieval layer.

### Limitations
1. **The Python GIL Bottleneck:** The retrieval engine currently peaks at ~125 QPS around 2-4 concurrent workers. Python's Global Interpreter Lock prevents true multi-core CPU scaling for the math-heavy Dense Search. (Fix: Move retrieval to a Rust/Go microservice).
2. **LLM Dependency:** No matter how fast our retrieval is, we are at the mercy of the Groq API provider for the final generation step. Network hiccups on their end degrade our perceived responsiveness. (Fix: Implement Token Streaming).
3. **BM25 In-Memory Loading:** The BM25 index currently builds completely in-memory on application start. While fine for 200 documents, this will OOM crash if scaled to 200,000 documents. (Fix: Migrate sparse search to an ElasticSearch cluster).
