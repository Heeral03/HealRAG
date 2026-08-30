import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))
import config

class RetrievalEvaluator:
    """
    Evaluates the quality of retrieved document chunks relative to a user query.
    Classifies overall retrieval into three confidence tiers (per CRAG paper):
      1. CORRECT   (Score >= upper_threshold): Chunks contain relevant, high-confidence evidence.
      2. AMBIGUOUS (lower_threshold <= Score < upper_threshold): Partial match / jargon ambiguity.
      3. INCORRECT (Score < lower_threshold): Low semantic match / out-of-scope query.
    """

    def __init__(
        self,
        upper_threshold: float = config.EVALUATOR_UPPER_THRESHOLD,
        lower_threshold: float = config.EVALUATOR_LOWER_THRESHOLD,
    ):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

    def evaluate_chunks(
        self, query: str, retrieved_chunks: List[Dict]
    ) -> Tuple[str, float, List[Dict]]:
        """
        Evaluates a list of retrieved chunks for a given query.
        Returns:
          - overall_action: "CORRECT", "AMBIGUOUS", or "INCORRECT"
          - confidence_score: float score representing max/avg chunk relevance
          - annotated_chunks: list of chunks with individual evaluation scores
        """
        if not retrieved_chunks:
            return "INCORRECT", 0.0, []

        annotated_chunks = []
        scores = []

        for chunk in retrieved_chunks:
            sim_score = chunk.get("similarity_score", 0.0)
            
            # Heuristic keyword match boost: check if key query terms appear in chunk text
            query_terms = [
                term.lower()
                for term in query.replace("?", "").replace(".", "").split()
                if len(term) > 3
            ]
            text_lower = chunk.get("text", "").lower()
            matches = sum(1 for term in query_terms if term in text_lower)
            keyword_ratio = matches / len(query_terms) if query_terms else 0.0

            # Composite evaluator score combining vector similarity & key-term coverage
            eval_score = 0.7 * sim_score + 0.3 * keyword_ratio
            scores.append(eval_score)

            chunk_copy = dict(chunk)
            chunk_copy["eval_score"] = round(eval_score, 4)
            if eval_score >= self.upper_threshold:
                chunk_copy["eval_status"] = "CORRECT"
            elif eval_score >= self.lower_threshold:
                chunk_copy["eval_status"] = "AMBIGUOUS"
            else:
                chunk_copy["eval_status"] = "INCORRECT"

            annotated_chunks.append(chunk_copy)

        # Overall retrieval confidence is governed by the highest-scoring chunk
        max_score = max(scores) if scores else 0.0

        if max_score >= self.upper_threshold:
            overall_action = "CORRECT"
        elif max_score >= self.lower_threshold:
            overall_action = "AMBIGUOUS"
        else:
            overall_action = "INCORRECT"

        return overall_action, round(max_score, 4), annotated_chunks


if __name__ == "__main__":
    evaluator = RetrievalEvaluator()
    sample_query = "What is GDPR Article 9?"
    sample_chunks = [
        {"source": "doc_001.txt", "text": "GDPR Article 9 covers processing of personal health data.", "similarity_score": 0.82},
        {"source": "doc_002.txt", "text": "Random unrelated document content.", "similarity_score": 0.31},
    ]
    action, score, annotated = evaluator.evaluate_chunks(sample_query, sample_chunks)
    print(f"Query: {sample_query}")
    print(f"Evaluator Action: {action} (Score: {score})")
    for c in annotated:
        print(f" - {c['source']} => Status: {c['eval_status']}, Score: {c['eval_score']}")
