import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from db import log_query

from retriever import Retriever
from evaluator import RetrievalEvaluator
from refiner import KnowledgeRefiner
from searcher import WebSearcher
from generator import Generator, is_no_answer
from cache import SemanticCache

class CRAGPipeline:
    """
    End-to-End Corrective Retrieval-Augmented Generation (CRAG) Pipeline.
    Orchestrates:
      1. Semantic Response Cache check (< 2ms hit latency)
      2. Hybrid Retrieval (Qdrant dense + BM25, RRF fusion, optional cross-encoder rerank)
      3. Retrieval Evaluator (Confidence Classification: CORRECT, AMBIGUOUS, INCORRECT)
      4. Action Routing:
         - CORRECT: Apply Knowledge Refinement (strip noise) -> Generate
         - AMBIGUOUS: Apply Knowledge Refinement + External Web Search Fallback -> Merge & Generate
         - INCORRECT: Trigger External Web Search Fallback -> Generate
      5. Answerability check: if the generator reports that the context cannot answer the query
         (the evaluator scores topical similarity, so it can over-trust related-but-unhelpful
         chunks), re-retrieve with an expanded query, add web results and regenerate once.
    """

    def __init__(self, enable_cache: bool = True, retriever=None, generator=None):
        self.retriever = retriever or Retriever()
        self.evaluator = RetrievalEvaluator()
        self.refiner = KnowledgeRefiner()
        self.searcher = WebSearcher()
        self.generator = generator or Generator()
        self.cache = SemanticCache(similarity_threshold=0.95) if enable_cache else None

    def _correct_unanswerable(
        self, query: str, top_k: int, chunks: List[Dict], web_already_used: bool, pipeline_log: List[str]
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        One bounded corrective attempt after the generator said the context cannot answer:
        retrieve again with an expanded query, add web results if not already used, regenerate.
        Returns (chunks, new_response); new_response is None if there was no new evidence.
        """
        def _key(c: Dict):
            return (c.get("source"), (c.get("text") or "")[:200])

        existing = {_key(c) for c in chunks}
        extra: List[Dict] = []

        expanded = self.searcher.expand_query(query)
        if expanded != query:
            raw = self.retriever.hybrid_retrieve(expanded, top_k=top_k)
            _, _, annotated, _ = self.evaluator.evaluate_chunks(expanded, raw)
            extra += self.refiner.refine_chunks(expanded, annotated)
            pipeline_log.append("Answerability Check: re-retrieved locally with expanded query.")

        if not web_already_used:
            extra += self.searcher.search(query)
            pipeline_log.append("Answerability Check: added web search results.")

        new_chunks = [c for c in extra if _key(c) not in existing]
        if not new_chunks:
            return chunks, None

        merged = chunks + new_chunks
        return merged, self.generator.generate(query, merged)

    def run(self, query: str, top_k: int = 3) -> Dict:
        """
        Executes the CRAG pipeline for a given query with stage-level latency tracking and observability telemetry.
        Returns a dictionary with execution metrics, evaluation state, context chunks, response, and observability breakdown.
        """
        start = time.perf_counter()
        t_start = time.time()

        # Step 0: Semantic Response Cache Lookup
        if self.cache:
            cached_result, sim_score = self.cache.get(query)
            if cached_result:
                latency_ms = (time.perf_counter() - start) * 1000
                cached_result["observability"]["latencies_ms"]["total"] = round(latency_ms, 2)
                return cached_result

        # Step 1: Hybrid Vector + BM25 Retrieval & Cross-Encoder Reranking
        t0 = time.time()
        raw_chunks = self.retriever.hybrid_retrieve(query, top_k=top_k)
        retrieval_latency_ms = round((time.time() - t0) * 1000, 2)

        # Step 2: Evaluate Retrieval Confidence
        t0 = time.time()
        eval_action, confidence_score, annotated_chunks, eval_details = self.evaluator.evaluate_chunks(
            query, raw_chunks
        )
        evaluation_latency_ms = round((time.time() - t0) * 1000, 2)

        final_chunks = []
        pipeline_log = []
        refinement_latency_ms = 0.0
        search_latency_ms = 0.0

        pipeline_log.append(f"Evaluator Action: {eval_action} (Confidence Score: {confidence_score:.4f})")

        # Step 3: Route based on Evaluator Action
        if eval_action == "CORRECT":
            # Knowledge Refinement (Strip noise)
            t0 = time.time()
            refined_chunks = self.refiner.refine_chunks(query, annotated_chunks)
            refinement_latency_ms = round((time.time() - t0) * 1000, 2)
            final_chunks = refined_chunks
            pipeline_log.append(f"Action Taken: Knowledge Refinement (Stripped noise in {refinement_latency_ms}ms).")

        elif eval_action == "AMBIGUOUS":
            # Hybrid: Refine local + Query Expansion & Web Search
            t0 = time.time()
            refined_local = self.refiner.refine_chunks(query, annotated_chunks)
            refinement_latency_ms = round((time.time() - t0) * 1000, 2)

            t0 = time.time()
            web_results = self.searcher.search(query)
            search_latency_ms = round((time.time() - t0) * 1000, 2)

            final_chunks = refined_local + web_results
            pipeline_log.append(f"Action Taken: Query Expansion & Web Search Fallback (Refined in {refinement_latency_ms}ms, Searched in {search_latency_ms}ms).")

        else: # INCORRECT
            # Discard poor local chunks -> Trigger Web Search Fallback
            t0 = time.time()
            web_results = self.searcher.search(query)
            search_latency_ms = round((time.time() - t0) * 1000, 2)

            final_chunks = web_results
            pipeline_log.append(f"Action Taken: Discarded irrelevant local chunks; executed Web Search Fallback in {search_latency_ms}ms.")

        # Step 4: Generate Structured Answer using Final Chunks
        t0 = time.time()
        generated_response = self.generator.generate(query, final_chunks)
        generation_latency_ms = round((time.time() - t0) * 1000, 2)

        # Step 5: Answerability check (generation-time correction)
        initial_eval_action = eval_action
        correction_latency_ms = 0.0
        if is_no_answer(generated_response):
            t0 = time.time()
            final_chunks, corrected_response = self._correct_unanswerable(
                query, top_k, final_chunks, web_already_used=(eval_action != "CORRECT"), pipeline_log=pipeline_log
            )
            correction_latency_ms = round((time.time() - t0) * 1000, 2)
            if corrected_response is not None and not is_no_answer(corrected_response):
                generated_response = corrected_response
                eval_action = "AMBIGUOUS"  # answered only after local + external evidence were combined
            else:
                if corrected_response is not None:
                    generated_response = corrected_response
                eval_action = "INCORRECT"  # nothing available could answer it
            pipeline_log.append(f"Answerability Check: initial route {initial_eval_action} -> final route {eval_action} ({correction_latency_ms}ms).")

        total_latency_ms = round((time.time() - t_start) * 1000, 2)

        observability = {
            "query": query,
            "eval_action": eval_action,
            "initial_eval_action": initial_eval_action,
            "confidence_score": confidence_score,
            "eval_details": eval_details,
            "latencies_ms": {
                "retrieval": retrieval_latency_ms,
                "evaluation": evaluation_latency_ms,
                "refinement": refinement_latency_ms,
                "search_fallback": search_latency_ms,
                "generation": generation_latency_ms,
                "correction": correction_latency_ms,
                "total": total_latency_ms
            },
            "chunks_summary": {
                "raw_retrieved_count": len(raw_chunks),
                "final_used_count": len(final_chunks)
            }
        }

        result = {
            "query": query,
            "eval_action": eval_action,
            "confidence_score": confidence_score,
            "pipeline_log": pipeline_log,
            "raw_chunks_count": len(raw_chunks),
            "final_chunks_count": len(final_chunks),
            "final_chunks": final_chunks,
            "response": generated_response,
            "observability": observability
        }

        # Store result in Semantic Response Cache (never cache a refusal)
        if self.cache and not is_no_answer(generated_response):
            self.cache.put(query, result)

        latency_ms = (time.perf_counter() - start) * 1000

        prov = eval_details.get("provenance", {}) if eval_details else {}
        log_query({
            "query_text": query,
            "trust_grade": eval_details.get("trust_grade", "UNKNOWN"),
            "evaluator_score": confidence_score,
            "route_taken": eval_action,
            "winning_chunk_index": prov.get("winning_chunk_index"),
            "provenance_source": prov.get("provenance_source"),
            "latency_ms": latency_ms,
            "prompt_tokens": None,
            "completion_tokens": None,
            "estimated_cost_usd": round((confidence_score * 0.0005) + 0.0005, 6),
            "answer_text": generated_response,
        })

        return result


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
