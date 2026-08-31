import asyncio
import time
import numpy as np
import httpx
from typing import List, Dict

API_URL = "http://localhost:8000/query"
DEV_API_KEY = "sk_live_healrag_demo_2026"

QUERIES = [
    "What is GDPR Article 9?",
    "What are the consent requirements under GDPR Article 9(2)(a)?",
    "Explain the EHDS regulation for digital health data sharing.",
    "What are medical device software requirements under EU MDR 2017/745?",
    "What are the scientific research exemptions in GDPR Article 9(2)(j)?"
]

async def worker(worker_id: int, num_requests: int, results: List[Dict]):
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {
            "X-API-Key": DEV_API_KEY,
            "Content-Type": "application/json"
        }
        for i in range(num_requests):
            query = QUERIES[(worker_id + i) % len(QUERIES)]
            payload = {"query": query, "top_k": 3, "vanilla_mode": False}

            t0 = time.perf_counter()
            try:
                resp = await client.post(API_URL, json=payload, headers=headers)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                results.append({
                    "worker_id": worker_id,
                    "status_code": resp.status_code,
                    "latency_ms": elapsed_ms,
                    "success": resp.status_code == 200,
                    "rate_limited": resp.status_code == 429
                })
            except Exception as e:
                results.append({
                    "worker_id": worker_id,
                    "status_code": 500,
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                    "success": False,
                    "error": str(e)
                })

async def run_benchmark_level(concurrency: int, reqs_per_worker: int) -> Dict:
    print(f"\n🚀 Running Concurrency Benchmark Level: {concurrency} Workers ({concurrency * reqs_per_worker} Total Requests)...")
    results = []
    
    t_start = time.perf_counter()
    tasks = [
        worker(w_id, reqs_per_worker, results)
        for w_id in range(concurrency)
    ]
    await asyncio.gather(*tasks)
    total_time_sec = time.perf_counter() - t_start

    successful = [r for r in results if r["success"]]
    rate_limited = [r for r in results if r.get("rate_limited")]
    failed = [r for r in results if not r["success"] and not r.get("rate_limited")]

    latencies = [r["latency_ms"] for r in successful] if successful else [0]
    qps = len(successful) / total_time_sec if total_time_sec > 0 else 0.0

    return {
        "concurrency": concurrency,
        "total_requests": len(results),
        "successful": len(successful),
        "rate_limited": len(rate_limited),
        "failed": len(failed),
        "total_time_sec": round(total_time_sec, 2),
        "qps": round(qps, 2),
        "avg_latency_ms": round(float(np.mean(latencies)), 1),
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 1),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 1),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 1)
    }

async def main():
    print("=" * 70)
    print("  HealRAG Async Concurrency & Throughput Benchmark Suite  ")
    print("=" * 70)

    # Test levels: 2 workers, 4 workers, 8 workers
    summary_matrix = []
    for level in [2, 4, 8]:
        metrics = await run_benchmark_level(concurrency=level, reqs_per_worker=2)
        summary_matrix.append(metrics)

    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY MATRIX  ")
    print("=" * 70)
    print(f"{'Concurrency':<12} | {'QPS':<8} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'Success':<8} | {'Throttled (429)':<15}")
    print("-" * 75)
    for m in summary_matrix:
        print(f"{m['concurrency']:<12} | {m['qps']:<8} | {m['p50_latency_ms']:<10} | {m['p95_latency_ms']:<10} | {m['successful']:<8} | {m['rate_limited']:<15}")

if __name__ == "__main__":
    asyncio.run(main())
