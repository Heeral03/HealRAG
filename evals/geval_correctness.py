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
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

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


def build_correctness_metric(judge: GroqJudge, threshold: float) -> GEval:
    """G-Eval metric judging HealRAG's actual answer against the expected answer."""
    return GEval(
        name="Correctness",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradict any facts in 'expected output'. Any contradiction should be penalized heavily.",
            "Check whether 'actual output' covers the key points of 'expected output' (e.g. article numbers, obligations, rights, time limits, thresholds). Missing key points should be penalized.",
            "Check that numbers, article/section references, and legal terms in 'actual output' match 'expected output'. Wrong or fabricated references should be penalized.",
            "Extra detail in 'actual output' is acceptable as long as it does not contradict 'expected output'; vague or generic answers lacking the specifics of 'expected output' should be penalized.",
            "Do not penalize differences in wording, ordering, or formatting if the meaning is the same.",
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=threshold,
        model=judge,
        async_mode=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="G-Eval Correctness: HealRAG actual output vs expected output (DeepEval)."
    )
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit number of golden cases to evaluate")
    parser.add_argument("--threshold", type=float, default=0.70, help="Pass threshold for correctness (default: 0.70)")
    parser.add_argument("--candidate-k", type=int, default=50, help="Retriever candidate pool size (default: 50)")
    parser.add_argument("--top-k", type=int, default=5, help="Max reranked context chunks (default: 5)")
    parser.add_argument(
        "--save-outputs",
        type=str,
        default=None,
        help="Optional path to dump generated (query, actual, expected) triples as JSON for later re-scoring",
    )
    args = parser.parse_args()

    print("\n=======================================================")
    print("      HealRAG G-Eval Correctness Evaluation")
    print("      (actual_output vs expected_output)")
    print("=======================================================\n")

    golden_cases = load_goldens(limit=args.limit)

    print(f"Initializing Retriever (candidate_k={args.candidate_k}, top_k={args.top_k})...")
    retriever = RerankingRetriever(candidate_k=args.candidate_k, top_k=args.top_k)

    print("Initializing Generator (LangChain ChatGroq)...")
    generator = Generator()

    print("Initializing GroqJudge for G-Eval...")
    judge = GroqJudge()

    correctness_metric = build_correctness_metric(judge, args.threshold)

    test_cases = []
    print("\n--- Generating HealRAG Answers ---")
    for idx, case in enumerate(golden_cases, 1):
        query = case.get("question") or case.get("query")
        expected_output = case.get("ideal_answer") or case.get("expected_output", "")

        print(f"[{idx}/{len(golden_cases)}] {query[:70]}...")
        retrieved_docs = retriever.invoke(query)
        actual_output = generator.generate(query, retrieved_docs)

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=actual_output,
                expected_output=expected_output,
            )
        )

    if args.save_outputs:
        with open(args.save_outputs, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"input": t.input, "actual_output": t.actual_output, "expected_output": t.expected_output}
                    for t in test_cases
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Saved generated outputs to {args.save_outputs}")

    print(f"\n--- Running G-Eval Correctness ---")
    print(f"Evaluating {len(test_cases)} test case(s) with threshold={args.threshold}...\n")

    evaluate(
        test_cases=test_cases,
        metrics=[correctness_metric],
        async_config=AsyncConfig(run_async=False),
        hyperparameters={
            "pipeline": "end_to_end_rag",
            "metric": "geval_correctness",
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
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
