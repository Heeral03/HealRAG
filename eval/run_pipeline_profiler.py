import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from retriever import Retriever
from evaluator import RetrievalEvaluator
from refiner import KnowledgeRefiner
from searcher import WebSearcher
from generator import Generator

PROFILER_QUERIES = [
    # CORRECT route queries (high-confidence local match)
    "What is GDPR Article 9?",
    "What three clinical sections are mandatory in the International Patient Summary?",
    "What are the core Caldicott Principles governing NHS patient data confidentiality?",
    "What architectural principles define FHIR?",
    "What are the key obligations under the UK NHS Data Security and Protection Toolkit?",
    # AMBIGUOUS route queries (jargon / terminology drift)
    "What is break-glass emergency?",
    "What are LOINC codes used for in FHIR observations?",
    "How does SMART on FHIR OAuth2 scope filtering work?",
    # INCORRECT route queries (out-of-corpus / web fallback)
    "What are the penalty fine tiers under US HIPAA Privacy Rule?",
    "What are software medical device requirements under EU MDR 2017/745?",
]

def time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result, elapsed_ms

def main():
    print("=======================================================================")
    print("      COMPONENT-LEVEL PIPELINE LATENCY PROFILER (v2)                   ")
    print("      Includes DuckDuckGo network verification & per-route breakdown   ")
    print("=======================================================================")

    retriever = Retriever()
    evaluator = RetrievalEvaluator()
    refiner = KnowledgeRefiner()
    searcher = WebSearcher()
    generator = Generator()

    print(f"\n[CONFIG] DuckDuckGo search available: {searcher._ddg_available}")
    if not searcher._ddg_available:
        print("[WARNING] Web search is using MOCK fallback — timings will NOT reflect real network latency!")

    per_query_results = []
    route_buckets = {"CORRECT": [], "AMBIGUOUS": [], "INCORRECT": []}

    for idx, query in enumerate(PROFILER_QUERIES):
        t_total_start = time.perf_counter()

        # Step 1: Retrieval
        chunks, t_retrieval = time_call(retriever.retrieve, query, 3)

        # Step 2: Evaluation
        (action, score, annotated, details), t_eval = time_call(evaluator.evaluate_chunks, query, chunks)

        # Step 3: Route-dependent stage
        t_refine = 0.0
        t_search = 0.0
        web_search_fired = False

        if action == "CORRECT":
            refined, t_refine = time_call(refiner.refine_chunks, query, annotated)
            final_chunks = refined
        elif action == "AMBIGUOUS":
            refined, t_refine = time_call(refiner.refine_chunks, query, annotated)
            web_res, t_search = time_call(searcher.search, query)
            web_search_fired = True
            final_chunks = refined + web_res
        else:  # INCORRECT
            web_res, t_search = time_call(searcher.search, query)
            web_search_fired = True
            final_chunks = web_res

        # Step 4: Generation (LLM)
        response, t_gen = time_call(generator.generate, query, final_chunks)

        t_total = (time.perf_counter() - t_total_start) * 1000.0

        row = {
            "query": query,
            "route": action,
            "web_search_fired": web_search_fired,
            "retrieval_ms": round(t_retrieval, 2),
            "evaluation_ms": round(t_eval, 2),
            "refinement_ms": round(t_refine, 2),
            "web_search_ms": round(t_search, 2),
            "generation_ms": round(t_gen, 2),
            "total_ms": round(t_total, 2)
        }
        per_query_results.append(row)
        route_buckets[action].append(row)

        ws_flag = " [WEB SEARCH]" if web_search_fired else ""
        print(f"[{idx+1:02d}/{len(PROFILER_QUERIES)}] {action:<10} | Retr: {t_retrieval:>8.2f}ms | Eval: {t_eval:>6.2f}ms | Refine: {t_refine:>6.2f}ms | WebSearch: {t_search:>8.2f}ms | Gen: {t_gen:>8.2f}ms | TOTAL: {t_total:>8.2f}ms{ws_flag}")

    # ========================== AGGREGATE REPORT ==========================
    print("\n=======================================================================")
    print("      SECTION 1: OVERALL AGGREGATE LATENCY BREAKDOWN                   ")
    print("=======================================================================")

    all_totals = [r["total_ms"] for r in per_query_results]
    avg_total = float(np.mean(all_totals))

    for comp in ["retrieval", "evaluation", "refinement", "web_search", "generation"]:
        vals = [r[f"{comp}_ms"] for r in per_query_results]
        mean_val = float(np.mean(vals))
        p50 = float(np.percentile(vals, 50))
        p95 = float(np.percentile(vals, 95))
        pct = (mean_val / avg_total * 100) if avg_total > 0 else 0.0
        print(f"  {comp:<16} | Mean: {mean_val:>8.2f}ms | p50: {p50:>8.2f}ms | p95: {p95:>8.2f}ms | Share: {pct:>5.1f}%")
    print(f"  {'TOTAL':<16} | Mean: {avg_total:>8.2f}ms")

    # ========================== PER-ROUTE REPORT ==========================
    print("\n=======================================================================")
    print("      SECTION 2: PER-ROUTE LATENCY BREAKDOWN (Conditional Averages)    ")
    print("=======================================================================")

    route_summary = {}
    for route_name in ["CORRECT", "AMBIGUOUS", "INCORRECT"]:
        bucket = route_buckets[route_name]
        n = len(bucket)
        if n == 0:
            continue

        print(f"\n  --- Route: {route_name} (n={n}) ---")
        route_data = {}
        for comp in ["retrieval", "evaluation", "refinement", "web_search", "generation", "total"]:
            vals = [r[f"{comp}_ms"] for r in bucket]
            mean_val = float(np.mean(vals))
            route_data[comp] = round(mean_val, 2)
            print(f"    {comp:<16} | Mean: {mean_val:>8.2f}ms")

        # Calculate per-route component shares
        route_total = route_data["total"]
        if route_total > 0:
            for comp in ["retrieval", "evaluation", "refinement", "web_search", "generation"]:
                pct = route_data[comp] / route_total * 100
                route_data[f"{comp}_pct"] = round(pct, 1)

        route_summary[route_name] = {"count": n, **route_data}

    # ========================== WEB SEARCH VERIFICATION ==========================
    print("\n=======================================================================")
    print("      SECTION 3: WEB SEARCH VERIFICATION                               ")
    print("=======================================================================")
    ws_fired_queries = [r for r in per_query_results if r["web_search_fired"]]
    ws_not_fired = [r for r in per_query_results if not r["web_search_fired"]]
    print(f"  Queries where web search fired: {len(ws_fired_queries)} / {len(per_query_results)}")
    print(f"  DuckDuckGo package available: {searcher._ddg_available}")
    if ws_fired_queries:
        ws_latencies = [r["web_search_ms"] for r in ws_fired_queries]
        print(f"  Web Search Mean Latency (when fired): {float(np.mean(ws_latencies)):.2f} ms")
        print(f"  Web Search p50 (when fired): {float(np.percentile(ws_latencies, 50)):.2f} ms")
        print(f"  Web Search Max (when fired): {max(ws_latencies):.2f} ms")
        for r in ws_fired_queries:
            print(f"    [{r['route']}] {r['query'][:50]}... -> WebSearch: {r['web_search_ms']:.2f}ms")

    # Save output
    output = {
        "config": {
            "ddg_available": searcher._ddg_available,
            "query_count": len(PROFILER_QUERIES)
        },
        "overall_avg_total_ms": round(avg_total, 2),
        "per_route_summary": route_summary,
        "per_query_breakdown": per_query_results
    }

    report_path = Path(__file__).resolve().parent / "pipeline_profiler_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved profiler results to: {report_path}")

if __name__ == "__main__":
    main()
