import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Prevent DeepEval from aborting test cases after a fixed deadline
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "true"

from dotenv import load_dotenv
load_dotenv(dotenv_path=BASE_DIR / ".env")

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)

from src.reranker import RerankingRetriever
from src.generator import Generator
from src.groq_judge import GroqJudge


def load_goldens(limit: int = None) -> list[dict]:
    golden_path = BASE_DIR / "goldens" / "golden_retriever.json"
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {golden_path}")

    with open(golden_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit and limit > 0:
        print(f"Evaluating on first {limit} golden cases.")
        return data[:limit]

    print(f"Evaluating all {len(data)} golden cases.")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate End-to-End HealRAG Pipeline (Retriever + Generator) using DeepEval."
    )
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit number of golden cases to evaluate")
    parser.add_argument("--threshold", type=float, default=0.70, help="Pass threshold for metrics (default: 0.70)")
    parser.add_argument("--candidate-k", type=int, default=50, help="Retriever candidate pool size (default: 50)")
    parser.add_argument("--top-k", type=int, default=5, help="Max reranked context chunks (default: 5)")
    parser.add_argument(
        "--score-margin",
        type=float,
        default=None,
        help="Keep chunks within this many rerank logits of the best chunk (default: off)",
    )
    parser.add_argument(
        "--ref-exclusive",
        action="store_true",
        help="If the query cites an article (e.g. 9(2)(a)) and a chunk matches it, keep only that chunk (default: off)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only run retrieval and print Hit@k / MRR / avg_chunks. Skips generator and judge calls (no tokens used).",
    )
    args = parser.parse_args()

    print("\n=======================================================")
    print("      HealRAG End-to-End Pipeline Evaluation")
    print("      (Retriever -> Chunks -> Generator -> Answer)")
    print("=======================================================\n")

    golden_cases = load_goldens(limit=args.limit)

    print(
        f"Initializing LangChain Retriever "
        f"(candidate_k={args.candidate_k}, top_k={args.top_k}, score_margin={args.score_margin}, "
        f"ref_exclusive={args.ref_exclusive})..."
    )
    retriever = RerankingRetriever(
        candidate_k=args.candidate_k,
        top_k=args.top_k,
        score_margin=args.score_margin,
        ref_exclusive=args.ref_exclusive,
    )

    print("Initializing Generator (LangChain ChatGroq)...")
    generator = Generator()

    print("Initializing GroqJudge for DeepEval evaluation...")
    judge = GroqJudge()

    # 3 Pipeline Metrics
    contextual_relevancy_metric = ContextualRelevancyMetric(
        threshold=args.threshold,
        model=judge,
        include_reason=True,
        async_mode=False,
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=args.threshold,
        model=judge,
        include_reason=True,
        async_mode=False,
    )

    answer_relevancy_metric = AnswerRelevancyMetric(
        threshold=args.threshold,
        model=judge,
        include_reason=True,
        async_mode=False,
    )

    test_cases = []
    hits = []  # rank of the gold document in the retrieved chunks (None if missed)
    chunk_counts = []  # number of chunks kept per query

    print("\n--- Running End-to-End Pipeline Execution ---")
    for idx, case in enumerate(golden_cases, 1):
        query = case.get("question") or case.get("query")
        ideal_answer = case.get("ideal_answer") or case.get("expected_output", "")

        print(f"[{idx}/{len(golden_cases)}] Pipeline query: {query[:65]}...")

        # Step 1: Retrieve context chunks
        retrieved_docs = retriever.invoke(query)
        retrieval_context = [doc.page_content for doc in retrieved_docs]
        chunk_counts.append(len(retrieved_docs))

        # Deterministic retrieval check: where did the gold document land?
        target = case.get("relevant_document", "")
        names = [d.metadata.get("source", "") for d in retrieved_docs]
        scores = [round(d.metadata.get("rerank_score", 0), 2) for d in retrieved_docs]
        rank = next((i + 1 for i, n in enumerate(names) if target and target in n), None)
        hits.append(rank)
        print(f"   gold doc rank: {rank} | chunks kept: {len(retrieved_docs)} | rerank scores: {scores}")

        if args.retrieval_only:
            continue  # skip generator and judge, no tokens used

        # Step 2: Generate answer from retrieved context
        actual_output = generator.generate(query, retrieved_docs)

        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=ideal_answer,
            retrieval_context=retrieval_context,
        )
        test_cases.append(test_case)

    # Retrieval summary (no judge involved)
    n = len(hits)
    print(
        f"\nRetrieval check: "
        f"Hit@1={sum(1 for r in hits if r and r <= 1) / n:.2f} "
        f"Hit@3={sum(1 for r in hits if r and r <= 3) / n:.2f} "
        f"Hit@5={sum(1 for r in hits if r and r <= 5) / n:.2f} "
        f"MRR={sum(1 / r for r in hits if r) / n:.2f} "
        f"avg_chunks={sum(chunk_counts) / n:.1f} "
        f"(margin={args.score_margin})"
    )

    if args.retrieval_only:
        try:
            from src.embedder import get_qdrant_client
            get_qdrant_client().close()
        except Exception:
            pass
        return

    print(f"\n--- Running DeepEval Metrics (Contextual Relevancy, Faithfulness & Answer Relevancy) ---")
    print(f"Evaluating {len(test_cases)} test case(s) with threshold={args.threshold}...\n")

    evaluate(
        test_cases=test_cases,
        metrics=[
            contextual_relevancy_metric,
            faithfulness_metric,
            answer_relevancy_metric,
        ],
        async_config=AsyncConfig(run_async=False),
        hyperparameters={
            "pipeline": "end_to_end_rag",
            "retriever": "hybrid_rrf_reranked",
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
            "score_margin": args.score_margin,
            "ref_exclusive": args.ref_exclusive,
            "generator_model": generator.model_name,
            "judge_model": judge.get_model_name(),
        },
    )

    try:
        from src.embedder import get_qdrant_client
        get_qdrant_client().close()
    except Exception:
        pass


if __name__ == "__main__":
    main()