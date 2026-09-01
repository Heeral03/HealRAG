import asyncio
import time
import numpy as np
import httpx
from typing import List, Dict

API_URL = "http://localhost:8000/query"
DEV_API_KEY = "sk_live_healrag_demo_2026"

# Distinct queries to prevent synthetic cache hits during benchmark
DISTINCT_QUERIES = [
    "What is GDPR Article 9 rules on processing special category data?",
    "Explain EHDS regulations for European digital health record exchange.",
    "What are software medical device requirements under EU MDR 2017/745?",
    "What are scientific research exceptions under GDPR Article 9(2)(j)?",
    "Describe security safeguard specifications under HIPAA Security Rule.",
    "What are FHIR R4 Patient resource mandatory data element fields?",
    "Explain ISO 27001 ISMS certification process for medical software.",
    "What are consent withdrawal requirements under GDPR Article 7?"
]

async def worker(worker_id: int, num_requests: int, query_list: List[str], use_vanilla: bool, results: List[Dict]):
    async with httpx.AsyncClient(timeout=60.0) as client:
        headers = {
            "X-API-Key": DEV_API_KEY,
            "Content-Type": "application/json"
        }
        for i in range(num_requests):
            query = query_list[(worker_id + i) % len(query_list)]
            payload = {"query": query, "top_k": 3, "vanilla_mode": use_vanilla}

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

async def run_benchmark_pass(mode_name: str, use_vanilla: bool, total_samples: int = 16) -> List[Dict]:
    print("\n" + "=" * 90)
    print(f"  BENCHMARK PASS: POST /query -> {mode_name}  ")
    print("=" * 90)
    print(f"{'Concurrency':<12} | {'QPS (Throughput)':<18} | {'p50 Latency':<14} | {'p95 Latency':<14} | {'200 OK':<8} | {'429 Throttled':<14}")
    print("-" * 90)

    pass_results = []
    for concurrency in [2, 4, 8]:
        results = []
        reqs_per_worker = max(1, total_samples // concurrency)
        
        t_start = time.perf_counter()
        tasks = [
            worker(w_id, reqs_per_worker, DISTINCT_QUERIES, use_vanilla, results)
            for w_id in range(concurrency)
        ]
        await asyncio.gather(*tasks)
        total_time_sec = time.perf_counter() - t_start

        successful = [r for r in results if r["success"]]
        rate_limited = [r for r in results if r.get("rate_limited")]
        success_latencies = [r["latency_ms"] for r in successful]

        qps = len(successful) / total_time_sec if total_time_sec > 0 else 0.0
        p50 = float(np.percentile(success_latencies, 50)) if success_latencies else 0.0
        p95 = float(np.percentile(success_latencies, 95)) if success_latencies else 0.0

        p50_str = f"{p50:.1f} ms" if successful else "—"
        p95_str = f"{p95:.1f} ms" if successful else "—"

        print(f"{concurrency:<12} | {qps:<18.2f} | {p50_str:<14} | {p95_str:<14} | {len(successful):<8} | {len(rate_limited):<14}")
        
        pass_results.append({
            "concurrency": concurrency,
            "qps": round(qps, 2),
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "successful": len(successful),
            "rate_limited": len(rate_limited)
        })

    return pass_results

async def main():
    print("=" * 90)
    print("  HealRAG Official POST /query Concurrency & Throughput Benchmark Matrix  ")
    print("=" * 90)

    await run_benchmark_pass("CRAG Pipeline (Groq LLM Generation)", use_vanilla=False, total_samples=16)

if __name__ == "__main__":
    asyncio.run(main())
