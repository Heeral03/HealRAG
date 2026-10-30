from deepeval.metrics import ToxicityMetric

from safety_runner import run_safety_eval


def build_metrics(case, judge, threshold):
    # In this DeepEval version a higher score is safer (success = score >= threshold).
    return [ToxicityMetric(threshold=threshold, model=judge, include_reason=True, async_mode=False)]


if __name__ == "__main__":
    run_safety_eval("Toxicity", "golden_toxicity.json", build_metrics)
