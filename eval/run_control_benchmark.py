import asyncio
import time
import numpy as np
import httpx
from typing import List, Dict

API_URL = "http://localhost:8000/health"

async def worker(worker_id: int, num_requests: int, results: List[Dict]):
    async with httpx.AsyncClient(timeout=10.0) as client:
        for _ in range(num_requests):
            t0 = time.perf_counter()
            try:
                resp = await client.get(API_URL)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                results.append({
                    "worker_id": worker_id,
                    "status_code": resp.status_code,
                    "latency_ms": elapsed_ms,
                    "success": resp.status_code == 200
                })
            except Exception as e:
                results.append({
                    "worker_id": worker_id,
                    "status_code": 500,
                    "latency_ms": (time.perf_counter() - t0) * 1000.0,
                    "success": False,
                    "error": str(e)
                })

async def run_control_benchmark(total_samples: int = 100):
    print("=" * 80)
    print(f"  CONTROL BENCHMARK: GET /health (Framework Baseline Concurrency Test)  ")
    print("=" * 80)
    print(f"{'Concurrency':<12} | {'QPS (Throughput)':<18} | {'p50 Latency':<14} | {'p95 Latency':<14} | {'200 OK':<8}")
    print("-" * 80)

    for concurrency in [2, 4, 8, 16, 32]:
        results = []
        reqs_per_worker = max(1, total_samples // concurrency)
        
        t_start = time.perf_counter()
        tasks = [
            worker(w_id, reqs_per_worker, results)
            for w_id in range(concurrency)
        ]
        await asyncio.gather(*tasks)
        total_time_sec = time.perf_counter() - t_start

        successful = [r for r in results if r["success"]]
        success_latencies = [r["latency_ms"] for r in successful]

        qps = len(successful) / total_time_sec if total_time_sec > 0 else 0.0
        p50 = float(np.percentile(success_latencies, 50)) if success_latencies else 0.0
        p95 = float(np.percentile(success_latencies, 95)) if success_latencies else 0.0

        print(f"{concurrency:<12} | {qps:<18.2f} | {p50:.2f} ms{'':<6} | {p95:.2f} ms{'':<6} | {len(successful):<8}")

if __name__ == "__main__":
    asyncio.run(run_control_benchmark(total_samples=200))
