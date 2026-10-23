import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Prevent DeepEval from aborting test cases after a fixed 180s deadline
os.environ["DEEPEVAL_DISABLE_TIMEOUTS"] = "true"

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric, ContextualPrecisionMetric

from src.reranker import RerankingRetriever

GOLDEN_PATH = str(PROJECT_ROOT / "goldens" / "golden_retriever.json")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
THRESHOLD = 0.7


def get_judge():
    """
    Returns the judge model. Uses OpenAI model if OPENAI_API_KEY is present,
    otherwise falls back to GroqJudge backed by Groq API.
    """
    if os.getenv("OPENAI_API_KEY"):
        # Fix typo if gpt-4.1-mini was passed instead of gpt-4o-mini
        model = JUDGE_MODEL
        if model == "gpt-4.1-mini":
            model = "gpt-4o-mini"
        return model
    
    from src.groq_judge import GroqJudge
    groq_model = os.getenv("GROQ_JUDGE_MODEL", "qwen/qwen3.8-27b")
    print(f"Using GroqJudge ({groq_model}) with accelerated inference for DeepEval evaluation...")
    return GroqJudge(model=groq_model, sleep_between_calls=0.5)


def run_evaluation(limit: int | None = None, top_k: int = 5):
    # 1. LOAD the golden set --- the fixed, human-authored truth
    print(f"Loading golden dataset from: {GOLDEN_PATH}")
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        goldens = json.load(f)

    if limit is not None:
        goldens = goldens[:limit]
        print(f"Evaluating on first {limit} golden cases.")
    else:
        print(f"Evaluating all {len(goldens)} golden cases.")

    # 2. RUN THE RETRIEVER on each question to fill retrieval_context,
    #    then build one test case per golden.
    print(f"Initializing RerankingRetriever (top_k={top_k})...")
    retriever = RerankingRetriever(top_k=top_k)

    test_cases = []
    for idx, g in enumerate(goldens, 1):
        query = g.get("question") or g.get("query")
        print(f"[{idx}/{len(goldens)}] Retrieving context for query: {query[:70]}...")
        retrieved = retriever.invoke(query)
        retrieval_context = [doc.page_content for doc in retrieved]

        test_cases.append(
            LLMTestCase(
                input=query,
                expected_output=g["ideal_answer"],
                retrieval_context=retrieval_context,
                actual_output="(generator not evaluated in this run)",
            )
        )

    # 3. THE METRICS --- recall (did we miss?) and precision (ranked well?)
    judge = get_judge()
    metrics = [
        ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    # 4. EVALUATE --- every metric on every case with a printed report
    # Sequential async_config prevents bursting rate limits when using API judge
    run_async = bool(os.getenv("OPENAI_API_KEY"))
    evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(run_async=run_async),
        hyperparameters={
            "retriever": "hybrid_rrf_reranked",
            "embedding_model": "intfloat/multilingual-e5-small",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k": 5,
            "judge_model": judge if isinstance(judge, str) else getattr(judge, "_model_name", "groq"),
            "golden_set": "golden_retriever.json",
        },
    )

    try:
        from src.embedder import get_qdrant_client
        get_qdrant_client().close()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Retriever using DeepEval")
    parser.add_argument(
        "--limit", "-n", type=int, default=None, help="Limit number of golden cases to evaluate"
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=5, help="Top-K context chunks to retrieve and evaluate (default: 5)"
    )
    args = parser.parse_args()

    run_evaluation(limit=args.limit, top_k=args.top_k)
