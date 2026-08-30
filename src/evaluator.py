import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add src directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent))
import config

class T5RetrievalEvaluator:
    """
    Evaluator backend based on the original Corrective RAG paper (Yan et al., 2024).
    Uses a T5 sequence-to-sequence model (e.g. google/flan-t5-base) to compute relevance
    scores for each (query, document_chunk) pair by predicting sequence probabilities.
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._loaded = False

    def _load_model(self):
        if not self._loaded:
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                import torch

                print(f"Loading T5 Evaluator Model ({self.model_name})...")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self._loaded = True
            except Exception as e:
                print(f"[T5 Evaluator Warning] Failed to load Transformers T5 model ({e}). Falling back to heuristic scoring.")
                self._loaded = False

    def score_pair(self, query: str, chunk_text: str) -> float:
        """
        Scores a single (query, chunk_text) pair using T5 sequence likelihood or classification prompt.
        Format: 'Query: {query} Document: {chunk_text} Relevant:' -> predict 'Yes'/'No'
        """
        self._load_model()
        if not self._loaded:
            return 0.50

        import torch
        prompt = f"Determine if the following document is relevant to answer the query.\nQuery: {query}\nDocument: {chunk_text}\nRelevant (Yes/No):"
        inputs = self._tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=3, return_dict_in_generate=True, output_scores=True)
            
        generated_text = self._tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip().lower()
        if "yes" in generated_text:
            return 0.85
        elif "no" in generated_text:
            return 0.20
        else:
            return 0.50


class RetrievalEvaluator:
    """
    Evaluates the quality of retrieved document chunks relative to a user query.
    Classifies overall retrieval into three confidence tiers (per CRAG paper):
      1. CORRECT   (Score >= upper_threshold): Chunks contain relevant, high-confidence evidence.
      2. AMBIGUOUS (lower_threshold <= Score < upper_threshold): Partial match / jargon ambiguity.
      3. INCORRECT (Score < lower_threshold): Low semantic match / out-of-scope query.

    Supports dual backends:
      - 'heuristic' (Default): Composite vector similarity + key-term coverage (Fast, 0 cold-start latency)
      - 't5' (Paper Baseline): T5 / Flan-T5 seq2seq relevance scoring engine
    """

    def __init__(
        self,
        upper_threshold: float = config.EVALUATOR_UPPER_THRESHOLD,
        lower_threshold: float = config.EVALUATOR_LOWER_THRESHOLD,
        backend: str = "heuristic"
    ):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.backend = backend.lower()
        self.t5_evaluator = T5RetrievalEvaluator() if self.backend == "t5" else None

    def evaluate_chunks(
        self, query: str, retrieved_chunks: List[Dict]
    ) -> Tuple[str, float, List[Dict], Dict]:
        """
        Evaluates a list of retrieved chunks for a given query.
        Returns:
          - overall_action: "CORRECT", "AMBIGUOUS", or "INCORRECT"
          - confidence_score: float score representing max chunk relevance
          - annotated_chunks: list of chunks with individual evaluation scores
          - eval_details: dict containing granular observability breakdowns and rationale
        """
        if not retrieved_chunks:
            eval_details = {
                "max_score": 0.0,
                "top_similarity_score": 0.0,
                "top_keyword_coverage": 0.0,
                "reasoning": "No chunks retrieved from vector database."
            }
            return "INCORRECT", 0.0, [], eval_details

        annotated_chunks = []
        scores = []
        chunk_metrics = []

        for chunk in retrieved_chunks:
            sim_score = chunk.get("similarity_score", 0.0)

            if self.backend == "t5" and self.t5_evaluator:
                t5_score = self.t5_evaluator.score_pair(query, chunk.get("text", ""))
                eval_score = 0.5 * sim_score + 0.5 * t5_score
                kw_ratio = 0.0
            else:
                query_terms = [
                    term.lower()
                    for term in query.replace("?", "").replace(".", "").split()
                    if len(term) > 3
                ]
                text_lower = chunk.get("text", "").lower()
                matches = sum(1 for term in query_terms if term in text_lower)
                kw_ratio = matches / len(query_terms) if query_terms else 0.0
                eval_score = 0.7 * sim_score + 0.3 * kw_ratio

            scores.append(eval_score)

            chunk_copy = dict(chunk)
            chunk_copy["eval_score"] = round(eval_score, 4)
            chunk_copy["similarity_score"] = round(sim_score, 4)
            chunk_copy["keyword_coverage_score"] = round(kw_ratio, 4)

            if eval_score >= self.upper_threshold:
                chunk_copy["eval_status"] = "CORRECT"
            elif eval_score >= self.lower_threshold:
                chunk_copy["eval_status"] = "AMBIGUOUS"
            else:
                chunk_copy["eval_status"] = "INCORRECT"

            annotated_chunks.append(chunk_copy)
            chunk_metrics.append({
                "source": chunk.get("source", "Unknown"),
                "composite_score": round(eval_score, 4),
                "similarity_score": round(sim_score, 4),
                "keyword_coverage": round(kw_ratio, 4),
                "eval_status": chunk_copy["eval_status"]
            })

        max_score = max(scores) if scores else 0.0
        best_chunk_idx = scores.index(max_score) if scores else 0
        best_metric = chunk_metrics[best_chunk_idx] if chunk_metrics else {}
        winning_chunk = annotated_chunks[best_chunk_idx] if annotated_chunks else {}

        if max_score >= self.upper_threshold:
            overall_action = "CORRECT"
            trust_grade = "HIGH_CONFIDENCE_VERIFIED"
            reasoning = f"Top chunk '{best_metric.get('source', 'N/A')}' score ({max_score:.4f}) >= upper threshold ({self.upper_threshold}). Local context is highly relevant."
        elif max_score >= self.lower_threshold:
            overall_action = "AMBIGUOUS"
            trust_grade = "MEDIUM_CONFIDENCE_HYBRID"
            reasoning = f"Top chunk '{best_metric.get('source', 'N/A')}' score ({max_score:.4f}) is between lower ({self.lower_threshold}) and upper ({self.upper_threshold}) thresholds. Triggering query expansion and hybrid search."
        else:
            overall_action = "INCORRECT"
            trust_grade = "LOW_CONFIDENCE_EXTERNAL_FALLBACK"
            reasoning = f"Top chunk '{best_metric.get('source', 'N/A')}' score ({max_score:.4f}) < lower threshold ({self.lower_threshold}). Local context is irrelevant; discarding local chunks and triggering web search."

        # Provenance Metadata Extension
        provenance = {
            "winning_chunk_index": best_chunk_idx,
            "provenance_source": winning_chunk.get("source", "External Web Search"),
            "provenance_confidence_score": round(max_score, 4),
            "trust_grade": trust_grade,
            "trust_rationale": reasoning
        }

        eval_details = {
            "max_score": round(max_score, 4),
            "top_similarity_score": best_metric.get("similarity_score", 0.0),
            "top_keyword_coverage": best_metric.get("keyword_coverage", 0.0),
            "upper_threshold": self.upper_threshold,
            "lower_threshold": self.lower_threshold,
            "reasoning": reasoning,
            "provenance": provenance,
            "chunk_metrics": chunk_metrics
        }

        return overall_action, round(max_score, 4), annotated_chunks, eval_details


if __name__ == "__main__":
    evaluator = RetrievalEvaluator(backend="heuristic")
    sample_query = "What is GDPR Article 9?"
    sample_chunks = [
        {"source": "doc_001.txt", "text": "GDPR Article 9 covers processing of personal health data.", "similarity_score": 0.82},
        {"source": "doc_002.txt", "text": "Random unrelated document content.", "similarity_score": 0.31},
    ]
    action, score, annotated, details = evaluator.evaluate_chunks(sample_query, sample_chunks)
    print(f"Backend: {evaluator.backend}")
    print(f"Query: {sample_query}")
    print(f"Evaluator Action: {action} (Score: {score})")
    print(f"Reasoning: {details['reasoning']}")
    for c in annotated:
        print(f" - {c['source']} => Status: {c['eval_status']}, Composite: {c['eval_score']} (Sim: {c['similarity_score']}, Key: {c['keyword_coverage_score']})")
