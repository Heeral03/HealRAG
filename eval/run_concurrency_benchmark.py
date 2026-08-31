import asyncio
import time
import numpy as np
import httpx
from typing import List, Dict

API_URL = "http://localhost:8000/query"
DEV_API_KEY = "sk_live_healrag_demo_2026"

# Workload A: Fast-Path Only (Corpus Matches -> CORRECT Route, ~1.2s - 1.8s)
FAST_PATH_QUERIES = [
    "What is GDPR Article 9?",
    "What are the consent requirements under GDPR Article 9(2)(a)?",
    "Explain the EHDS regulation for digital health data sharing.",
    "What are medical device software requirements under EU MDR 2017/745?",
    "What are the scientific research exemptions in GDPR Article 9(2)(j)?"
]

# Workload C: Fallback Only (Triggers DuckDuckGo Web Search -> INCORRECT Route, ~6s - 12s)
FALLBACK_QUERIES = [
    "Who won the 2026 FIFA World Cup final in New Jersey?",
    "What is the latest stock price of Apple today?",
    "What were the exact revenue numbers for Microsoft in Q3 2026?",
    "What is the weather forecast in Tokyo tomorrow?",
    "Who is the current Prime Minister of Japan right now?"
]

# Workload B: Mixed Traffic (70% Fast-Path, 30% Fallback)
MIXED_QUERIES = FAST_PATH_QUERIES[:3] + FALLBACK_QUERIES[:2]

WORKLOAD_SUITES = {
    "Workload A: Fast-Path Only (Corpus Matches)": FAST_PATH_QUERIES,
    "Workload B: Mixed Traffic (70% Fast / 30% Fallback)": MIXED_QUERIES,
    "Workload C: Fallback Only (Web Search)": FALLBACK_QUERIES
}

async def worker(worker_id: int, num_requests: int, query_list: List[str], results: List[Dict]):
    async with httpx.AsyncClient(timeout=45.0) as client:
        headers = {
            "X-API-Key": DEV_API_KEY,
            "Content-Type": "application/json"
        }
        for i in range(num_requests):
            query = query_list[(worker_id + i) % len(query_list)]
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

async def run_benchmark_level(concurrency: int, reqs_per_worker: int, query_list: List[str]) -> Dict:
    results = []
    t_start = time.perf_counter()
    tasks = [
        worker(w_id, reqs_per_worker, query_list, results)
        for w_id in range(concurrency)
    ]
    await asyncio.gather(*tasks)
    total_time_sec = time.perf_counter() - t_start

    # STRICT METRIC SEPARATION:
    # Compute latencies EXCLUSIVELY for HTTP 200 (Success) responses!
    successful_responses = [r for r in results if r["success"]]
    rate_limited_responses = [r for r in results if r.get("rate_limited")]
    failed_responses = [r for r in results if not r["success"] and not r.get("rate_limited")]

    success_latencies = [r["latency_ms"] for r in successful_responses]
    qps = len(successful_responses) / total_time_sec if total_time_sec > 0 else 0.0

    if success_latencies:
        p50 = float(np.percentile(success_latencies, 50))
        p95 = float(np.percentile(success_latencies, 95))
        p99 = float(np.percentile(success_latencies, 99))
        avg_lat = float(np.mean(success_latencies))
    else:
        p50 = p95 = p99 = avg_lat = 0.0

    return {
        "concurrency": concurrency,
        "total_requests": len(results),
        "successful": len(successful_responses),
        "rate_limited": len(rate_limited_responses),
        "failed": len(failed_responses),
        "total_time_sec": round(total_time_sec, 2),
        "qps": round(qps, 2),
        "avg_latency_ms": round(avg_lat, 1),
        "p50_latency_ms": round(p50, 1),
        "p95_latency_ms": round(p95, 1),
        "p99_latency_ms": round(p99, 1)
    }

async def run_suite(suite_name: str, query_list: List[str]):
    print("\n" + "=" * 75)
    print(f"  {suite_name}  ")
    print("=" * 75)
    print(f"{'Concurrency':<12} | {'QPS (200 OK)':<14} | {'p50 Latency':<12} | {'p95 Latency':<12} | {'200 OK':<8} | {'429 Throttled':<14}")
    print("-" * 82)

    for concurrency in [2, 4, 8]:
        metrics = await run_benchmark_level(concurrency=concurrency, reqs_per_worker=2, query_list=query_list)
        p50_str = f"{metrics['p50_latency_ms']} ms" if metrics["successful"] > 0 else "—"
        p95_str = f"{metrics['p95_latency_ms']} ms" if metrics["successful"] > 0 else "—"
        
        print(f"{metrics['concurrency']:<12} | {metrics['qps']:<14} | {p50_str:<12} | {p95_str:<12} | {metrics['successful']:<8} | {metrics['rate_limited']:<14}")

async def main():
    print("=" * 80)
    print("  HealRAG Non-Blocking Async Concurrency & Workload Throughput Suite  ")
    print("=" * 80)

    for suite_name, queries in WORKLOAD_SUITES.items():
        await run_suite(suite_name, queries)

if __name__ == "__main__":
    asyncio.run(main())
