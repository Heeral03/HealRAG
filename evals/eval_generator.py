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
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

from src.generator import Generator
from src.groq_judge import GroqJudge

def load_golden_generator(limit: int = None) -> list[dict]:
    golden_path = BASE_DIR / "goldens" / "golden_generator.json"
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden generator dataset not found at {golden_path}")

    with open(golden_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit and limit > 0:
        print(f"Evaluating on first {limit} golden cases.")
        return data[:limit]
    
    print(f"Evaluating all {len(data)} golden cases.")
    return data

def main():
    parser = argparse.ArgumentParser(description="Evaluate HealRAG Generator component using DeepEval.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test cases to evaluate")
    parser.add_argument("--threshold", type=float, default=0.70, help="Pass threshold for Faithfulness & Relevancy")
    args = parser.parse_args()

    print("\n=======================================================")
    print("      HealRAG Generator Evaluation (DeepEval)")
    print("=======================================================\n")

    golden_cases = load_golden_generator(limit=args.limit)

    print("Initializing Generator (LangChain ChatGroq)...")
    generator = Generator()

    print(f"Initializing GroqJudge with accelerated model for DeepEval evaluation...")
    judge = GroqJudge()

    faithfulness_metric = FaithfulnessMetric(
        threshold=args.threshold,
        model=judge,
        include_reason=True,
        async_mode=False
    )

    relevancy_metric = AnswerRelevancyMetric(
        threshold=args.threshold,
        model=judge,
        include_reason=True,
        async_mode=False
    )

    test_cases = []
    print("\n--- Generating Answers from Golden Context ---")
    for idx, case in enumerate(golden_cases, 1):
        query = case["query"]
        golden_context = case["golden_context"]
        expected_output = case.get("expected_output", "")

        print(f"[{idx}/{len(golden_cases)}] Generating answer for: {query[:65]}...")
        actual_output = generator.generate(query=query, retrieved_chunks=golden_context)

        # Context texts required by DeepEval metrics
        retrieval_context_texts = [c.get("text", "") for c in golden_context]

        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context_texts
        )
        test_cases.append(test_case)

    print(f"\n--- Running DeepEval Metrics (Faithfulness & Answer Relevancy) ---")
    print(f"Evaluating {len(test_cases)} test case(s) with threshold={args.threshold}...\n")

    evaluate(
        test_cases=test_cases,
        metrics=[faithfulness_metric, relevancy_metric],
        async_config=AsyncConfig(run_async=False)
    )

if __name__ == "__main__":
    main()
