import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from retriever import Retriever
from evaluator import RetrievalEvaluator
from refiner import KnowledgeRefiner
from searcher import WebSearcher
from generator import Generator

class CRAGPipeline:
    """
    End-to-End Corrective Retrieval-Augmented Generation (CRAG) Pipeline.
    Orchestrates:
      1. Vector Retrieval (FAISS)
      2. Retrieval Evaluator (Confidence Classification: CORRECT, AMBIGUOUS, INCORRECT)
      3. Action Routing:
         - CORRECT: Apply Knowledge Refinement (strip noise) -> Generate
         - AMBIGUOUS: Apply Knowledge Refinement + External Web Search Fallback -> Merge & Generate
         - INCORRECT: Trigger External Web Search Fallback -> Generate
    """

    def __init__(self):
        self.retriever = Retriever()
        self.evaluator = RetrievalEvaluator()
        self.refiner = KnowledgeRefiner()
        self.searcher = WebSearcher()
        self.generator = Generator()

    def run(self, query: str, top_k: int = 3) -> Dict:
        """
        Executes the CRAG pipeline for a given query.
        Returns a dictionary with execution metrics, evaluation state, context chunks, and final response.
        """
        # Step 1: Initial Retrieval
        raw_chunks = self.retriever.retrieve(query, top_k=top_k)

        # Step 2: Evaluate Retrieval Confidence
        eval_action, confidence_score, annotated_chunks = self.evaluator.evaluate_chunks(
            query, raw_chunks
        )

        final_chunks = []
        pipeline_log = []

        pipeline_log.append(f"Evaluator Action: {eval_action} (Confidence Score: {confidence_score:.4f})")

        # Step 3: Route based on Evaluator Action
        if eval_action == "CORRECT":
            # Knowledge Refinement (Strip noise)
            refined_chunks = self.refiner.refine_chunks(query, annotated_chunks)
            final_chunks = refined_chunks
            pipeline_log.append("Action Taken: Knowledge Refinement (Stripped noise from retrieved local chunks).")

        elif eval_action == "AMBIGUOUS":
            # Hybrid: Refine local + Query Expansion & Web Search
            refined_local = self.refiner.refine_chunks(query, annotated_chunks)
            web_results = self.searcher.search(query)
            final_chunks = refined_local + web_results
            pipeline_log.append("Action Taken: Query Expansion & External Web Search Fallback (Merged with refined local chunks).")

        else: # INCORRECT
            # Discard poor local chunks -> Trigger Web Search Fallback
            web_results = self.searcher.search(query)
            final_chunks = web_results
            pipeline_log.append("Action Taken: Discarded irrelevant local chunks; executed External Web Search Fallback.")

        # Step 4: Generate Structured Answer using Final Chunks
        generated_response = self.generator.generate(query, final_chunks)

        return {
            "query": query,
            "eval_action": eval_action,
            "confidence_score": confidence_score,
            "pipeline_log": pipeline_log,
            "raw_chunks_count": len(raw_chunks),
            "final_chunks_count": len(final_chunks),
            "final_chunks": final_chunks,
            "response": generated_response,
        }


if __name__ == "__main__":
    crag = CRAGPipeline()
    sample_queries = [
        "What is GDPR Article 9?", # Should be CORRECT
        "How are break-glass emergency procedures handled?", # Should be AMBIGUOUS / Query Expansion
        "What are the penalty fine tiers under US HIPAA Privacy Rule?", # Should be INCORRECT / Web Search
    ]

    for q in sample_queries:
        print(f"\n=======================================================")
        print(f"CRAG Pipeline Query: '{q}'")
        res = crag.run(q)
        print(f"Evaluator Decision: {res['eval_action']} (Score: {res['confidence_score']})")
        for log_entry in res["pipeline_log"]:
            print(f" Log: {log_entry}")
        print("--- Final Response Preview ---")
        print(res["response"][:300].replace("\n", " "), "...")
        print("=======================================================")
