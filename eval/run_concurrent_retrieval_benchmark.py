import json
import sys
import time
import concurrent.futures
from pathlib import Path
import numpy as np

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from retriever import Retriever

# Expanded 32-query suite for concurrent stress testing
CONCURRENT_QUERY_SUITE = [
    "What special categories of personal data are prohibited from processing under GDPR Article 9(1)?",
    "What three clinical sections are mandatory in the International Patient Summary (IPS) profile?",
    "What is the primary role of Health Data Access Bodies (HDABs) under EHDS Chapter IV Article 36?",
    "What are the required core attributes of a FHIR Patient Resource?",
    "What platform does the EU Commission establish under EHDS Chapter II Article 5 for cross-border health data exchange?",
    "What architectural principles define Fast Healthcare Interoperability Resources (FHIR)?",
    "How does EHDS secondary data use under Article 34 interact with GDPR Article 89 safeguards for scientific research?",
    "Compare the opt-out mechanism rights for natural persons under EHDS Chapter IV Article 38 against the restriction rights under Chapter II Article 7.",
    "In a cross-border IPS exchange, what coding standards are recommended for diagnostic results versus clinical procedure histories?",
    "How do academic evaluations of SMART on FHIR OAuth2 scope-based filters impact API performance and data leakage?",
    "What prohibited secondary uses under EHDS Article 35 safeguard individuals against financial discrimination?",
    "How can Zero-Knowledge Proofs (ZKPs) be integrated into EHDS Health Data Access Bodies (HDABs) to satisfy Chapter IV security guidelines?",
    "What are the core Caldicott Principles governing NHS patient data confidentiality?",
    "What are the key obligations under the UK NHS Data Security and Protection Toolkit (DSPT)?",
    "How does the NHS National Data Opt-out apply to secondary research?",
    "What are the required FHIR UK Core profiles for patient demographics?",
    "What are the consent rules for emergency break-glass data access under EHDS?",
    "How does GDPR Article 6 differ from Article 9 for health data processing?",
    "What are the technical requirements for MyHealth@EU national contact points?",
    "How are LOINC codes used in FHIR observation resources for lab results?",
    "What are SNOMED CT terminology requirements in international patient summaries?",
    "How does hyperledger blockchain consent auditing perform under high query loads?",
    "What are penalty fine tiers under EU GDPR for data security breaches?",
    "What are patient rights to data portability under GDPR Article 20?",
    "How does mTLS protect cross-border gateway exchanges in digital health?",
    "What are the 10 data security standards of the UK National Data Guardian?",
    "How do Caldicott Guardians handle requests for patient data disclosures?",
    "What is the function of OAuth2 scopes in FHIR API authorization?",
    "How are XML CDA documents transformed to FHIR resources?",
    "What is the role of Health Data Access Bodies in granting research permits?",
    "What security safeguards are required for EHDS secure processing environments?",
    "How are patient opt-outs propagated across regional EHR networks?"
]

def single_query_worker(retriever: Retriever, query: str, mode: str = "dense", top_k: int = 3) -> dict:
    t0 = time.perf_counter()
    if mode == "hybrid":
        res = retriever.hybrid_retrieve(query, top_k=top_k)
    else:
        res = retriever.retrieve(query, top_k=top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "query": query,
        "latency_ms": elapsed_ms,
        "results_count": len(res)
    }

def run_concurrent_pass(retriever: Retriever, mode: str, concurrency: int, top_k: int = 3) -> dict:
    queries = CONCURRENT_QUERY_SUITE
    total_queries = len(queries)
    
    t_start = time.perf_counter()
    latencies = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(single_query_worker, retriever, q, mode, top_k)
            for q in queries
        ]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            latencies.append(res["latency_ms"])
            
    wall_clock_sec = time.perf_counter() - t_start
    wall_clock_ms = wall_clock_sec * 1000.0
    qps = total_queries / wall_clock_sec if wall_clock_sec > 0 else 0.0
    
    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    avg_lat = float(np.mean(latencies))

    return {
        "concurrency": concurrency,
        "total_queries": total_queries,
        "wall_clock_ms": round(wall_clock_ms, 2),
        "qps": round(qps, 2),
        "avg_latency_ms": round(avg_lat, 2),
        "p50_ms": round(p50, 2),
        "p90_ms": round(p90, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2)
    }

def main():
    print("=======================================================================")
    print("      CONCURRENT MULTI-QUERY RETRIEVAL & THROUGHPUT BENCHMARK          ")
    print("=======================================================================")

    retriever = Retriever()
    concurrency_levels = [1, 2, 4, 8, 16, 32]
    
    results_matrix = {
        "dense": [],
        "hybrid": []
    }

    print("\n-----------------------------------------------------------------------")
    print(" 1. DENSE RETRIEVAL (FAISS FlatIP) CONCURRENCY MATRIX")
    print("-----------------------------------------------------------------------")
    print(f"{'Workers':<8} | {'QPS (Throughput)':<18} | {'Batch Time':<12} | {'p50 Latency':<12} | {'p90 Latency':<12} | {'p95 Latency':<12}")
    print("-" * 84)

    for c in concurrency_levels:
        stats = run_concurrent_pass(retriever, mode="dense", concurrency=c)
        results_matrix["dense"].append(stats)
        print(f"{c:<8} | {stats['qps']:<18.2f} | {stats['wall_clock_ms']:>8.1f} ms | {stats['p50_ms']:>8.2f} ms | {stats['p90_ms']:>8.2f} ms | {stats['p95_ms']:>8.2f} ms")

    print("\n-----------------------------------------------------------------------")
    print(" 2. HYBRID RETRIEVAL (BM25 + FAISS + RRF) CONCURRENCY MATRIX")
    print("-----------------------------------------------------------------------")
    print(f"{'Workers':<8} | {'QPS (Throughput)':<18} | {'Batch Time':<12} | {'p50 Latency':<12} | {'p90 Latency':<12} | {'p95 Latency':<12}")
    print("-" * 84)

    for c in concurrency_levels:
        stats = run_concurrent_pass(retriever, mode="hybrid", concurrency=c)
        results_matrix["hybrid"].append(stats)
        print(f"{c:<8} | {stats['qps']:<18.2f} | {stats['wall_clock_ms']:>8.1f} ms | {stats['p50_ms']:>8.2f} ms | {stats['p90_ms']:>8.2f} ms | {stats['p95_ms']:>8.2f} ms")

    output_path = Path(__file__).resolve().parent / "concurrent_benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_matrix, f, indent=2)

    print(f"\nSaved concurrent benchmark results to: {output_path}")

if __name__ == "__main__":
    main()
