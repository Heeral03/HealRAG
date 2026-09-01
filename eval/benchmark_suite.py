import asyncio
import time
import numpy as np
import httpx
from typing import List, Dict

API_URL = "http://localhost:8000/query"
DEV_API_KEY = "sk_live_healrag_demo_2026"

BENCHMARK_QUERIES = [
    "What is GDPR Article 9 rules on processing special category data?",
    "Explain EHDS regulations for European digital health record exchange.",
    "What are software medical device requirements under EU MDR 2017/745?",
    "What are scientific research exceptions under GDPR Article 9(2)(j)?",
    "Describe security safeguard specifications under HIPAA Security Rule.",
    "What are FHIR R4 Patient resource mandatory data element fields?"
]

async def send_request(client: httpx.AsyncClient, query: str, use_vanilla: bool) -> Dict:
    headers = {"X-API-Key": DEV_API_KEY, "Content-Type": "application/json"}
    payload = {"query": query, "top_k": 3, "vanilla_mode": use_vanilla}
    t0 = time.perf_counter()
    try:
        resp = await client.post(API_URL, json=payload, headers=headers)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        data = resp.json() if resp.status_code == 200 else {}
        is_mock_fallback = "LOCAL MOCK LLM MODE" in data.get("response", "")
        return {
            "status_code": resp.status_code,
            "latency_ms": elapsed_ms,
            "success": resp.status_code == 200,
            "rate_limited": resp.status_code == 429,
            "is_mock_fallback": is_mock_fallback
        }
    except Exception as e:
        return {
            "status_code": 500,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "success": False,
            "rate_limited": False,
            "is_mock_fallback": False,
            "error": str(e)
        }

async def run_pass(title: str, concurrency_levels: List[int], samples_per_level: int, use_vanilla: bool):
    print("\n" + "=" * 90)
    print(f"  BENCHMARK PASS: {title}  ")
    print("=" * 90)
    print(f"{'Concurrency':<12} | {'QPS':<12} | {'p50 Latency':<14} | {'p95 Latency':<14} | {'200 OK':<8} | {'429 Throttled':<14}")
    print("-" * 90)

    async with httpx.AsyncClient(timeout=60.0) as client:
        for conc in concurrency_levels:
            reqs_per_worker = max(1, samples_per_level // conc)
            tasks = []
            for w_id in range(conc):
                for i in range(reqs_per_worker):
                    q = BENCHMARK_QUERIES[(w_id + i) % len(BENCHMARK_QUERIES)]
                    tasks.append(send_request(client, q, use_vanilla))
            
            t_start = time.perf_counter()
            raw_results = await asyncio.gather(*tasks)
            total_sec = time.perf_counter() - t_start

            successful = [r for r in raw_results if r["success"]]
            throttled = [r for r in raw_results if r.get("rate_limited")]
            latencies = [r["latency_ms"] for r in successful]

            qps = len(successful) / total_sec if total_sec > 0 else 0.0
            p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
            p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

            p50_str = f"{p50:.1f} ms" if latencies else "—"
            p95_str = f"{p95:.1f} ms" if latencies else "—"

            print(f"{conc:<12} | {qps:<12.2f} | {p50_str:<14} | {p95_str:<14} | {len(successful):<8} | {len(throttled):<14}")

async def main():
    print("=" * 90)
    print("  HealRAG Clean & Unpolluted Benchmark Suite  ")
    print("=" * 90)

    # Pass 1: Local Direct Retrieval (Isolated Framework & Vector DB Throughput)
    await run_pass("Vanilla Vector Retrieval (Local Direct FAISS)", concurrency_levels=[2, 4, 8], samples_per_level=16, use_vanilla=True)

    # Pass 2: Full CRAG Pipeline with Real Groq LLM
    await run_pass("Full CRAG Pipeline (Real Groq LLM API)", concurrency_levels=[1, 2], samples_per_level=4, use_vanilla=False)

if __name__ == "__main__":
    asyncio.run(main())
