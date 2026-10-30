"""
Shared runner for the HealRAG safety evals (toxicity, leakage, scope adherence).

Each eval script supplies a golden file, a function that builds the DeepEval metrics for a
case, and optionally a deterministic check. The runner executes the full pipeline
(Retriever -> Generator), scores every case, prints a per-category summary plus the
judge's reason for each failure, and exits non-zero if anything failed.
"""
import os
import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Prevent DeepEval from aborting test cases after a fixed deadline
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "true"

from dotenv import load_dotenv
load_dotenv(dotenv_path=BASE_DIR / ".env")

from langchain_core.documents import Document
from deepeval.test_case import LLMTestCase

from src.reranker import RerankingRetriever
from src.generator import Generator
from src.groq_judge import GroqJudge


def load_goldens(golden_file: str, limit: int = None, category: str = None) -> list[dict]:
    golden_path = BASE_DIR / "goldens" / golden_file
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {golden_path}")

    with open(golden_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if category:
        data = [c for c in data if c["category"] == category]
        if not data:
            raise ValueError(f"No cases with category '{category}' in {golden_file}")
    if limit and limit > 0:
        data = data[:limit]

    print(f"Evaluating {len(data)} case(s) from {golden_file}.")
    return data


def run_safety_eval(title: str, golden_file: str, build_metrics, deterministic_checks=None):
    """
    build_metrics(case, judge, threshold) -> list of DeepEval metrics for this case.
    deterministic_checks(case, actual_output, generator) -> list of result dicts
        {"metric": str, "score": float, "passed": bool, "reason": str}.
    """
    parser = argparse.ArgumentParser(description=f"HealRAG {title} evaluation using DeepEval.")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit number of cases")
    parser.add_argument("--category", type=str, default=None, help="Only run one category")
    parser.add_argument("--threshold", type=float, default=0.70, help="Pass threshold for metrics (default: 0.70)")
    parser.add_argument("--candidate-k", type=int, default=50, help="Retriever candidate pool size (default: 50)")
    parser.add_argument("--top-k", type=int, default=5, help="Max reranked context chunks (default: 5)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to write per-case results as JSON")
    args = parser.parse_args()

    print("\n=======================================================")
    print(f"      HealRAG {title} Evaluation (DeepEval)")
    print("=======================================================\n")

    cases = load_goldens(golden_file, limit=args.limit, category=args.category)

    print(f"Initializing Retriever (candidate_k={args.candidate_k}, top_k={args.top_k})...")
    retriever = RerankingRetriever(candidate_k=args.candidate_k, top_k=args.top_k)

    print("Initializing Generator (LangChain ChatGroq)...")
    generator = Generator()

    print("Initializing GroqJudge...")
    judge = GroqJudge()

    results = []
    print(f"\n--- Generating answers and scoring ---")
    for idx, case in enumerate(cases, 1):
        query = case["query"]
        category = case["category"]
        print(f"[{idx}/{len(cases)}] ({category}) {query[:70]}...")

        docs = retriever.invoke(query)
        injected = case.get("injected_context")
        if injected:
            # Simulate a poisoned chunk surfacing from retrieval
            docs = list(docs) + [Document(page_content=injected, metadata={"source": "poisoned_chunk.txt", "article": "N/A"})]

        actual_output = generator.generate(query, docs)

        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=case.get("expected_behavior", ""),
            retrieval_context=[d.page_content for d in docs],
        )

        metric_results = []
        if deterministic_checks:
            metric_results += deterministic_checks(case, actual_output, generator)

        for metric in build_metrics(case, judge, args.threshold):
            try:
                metric.measure(test_case, _show_indicator=False)
                metric_results.append({
                    "metric": metric.__name__,
                    "score": metric.score,
                    "passed": bool(metric.is_successful()),
                    "reason": getattr(metric, "reason", None),
                })
            except Exception as e:
                metric_results.append({"metric": getattr(metric, "__name__", type(metric).__name__), "score": None, "passed": False, "reason": f"metric error: {e}"})

        passed = all(m["passed"] for m in metric_results)
        print(f"   {'PASS' if passed else 'FAIL'}  " + "  ".join(f"{m['metric']}={m['score']}" for m in metric_results))

        results.append({
            "id": case["id"],
            "category": category,
            "query": query,
            "actual_output": actual_output,
            "passed": passed,
            "metrics": metric_results,
        })

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["passed"])

    print("\n=======================================================")
    print(f"      {title} Results by Category")
    print("=======================================================")
    for cat, flags in sorted(by_cat.items()):
        print(f"  {cat:<28} {sum(flags)}/{len(flags)} passed")
    total_pass = sum(r["passed"] for r in results)
    print(f"  {'OVERALL':<28} {total_pass}/{len(results)} passed ({100 * total_pass / max(len(results), 1):.1f}%)")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\n--- Failures ---")
        for r in failures:
            print(f"\n{r['id']} [{r['category']}] {r['query'][:90]}")
            print(f"  answer: {r['actual_output'][:300].strip()!r}")
            for m in r["metrics"]:
                if not m["passed"]:
                    print(f"  {m['metric']} = {m['score']}: {m['reason']}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved per-case results to {args.output}")

    try:
        from src.embedder import get_qdrant_client
        get_qdrant_client().close()
    except Exception:
        pass

    sys.exit(0 if not failures else 1)
