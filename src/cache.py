import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from embedder import get_embedding_model

class SemanticCache:
    """
    High-Performance Semantic Response Cache for HealRAG.
    Caches query embeddings and pipeline response outputs.
    If an incoming query has cosine similarity >= similarity_threshold (default 0.80)
    with a cached entry, returns the cached response in < 2ms (bypassing Groq LLM network calls).
    """

    def __init__(self, similarity_threshold: float = 0.80, ttl_seconds: float = 3600.0):
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.embedder = get_embedding_model()
        self.cache_store: List[Dict] = []

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def get(self, query: str) -> Tuple[Optional[Dict], float]:
        """
        Looks up query in semantic cache.
        Returns (cached_result_dict, similarity_score) if HIT, else (None, 0.0).
        """
        if not self.cache_store:
            return None, 0.0

        now = time.time()
        # Clean expired entries
        self.cache_store = [e for e in self.cache_store if now - e["timestamp"] < self.ttl_seconds]

        query_vec = self.embedder.encode(query, normalize_embeddings=True)

        best_score = 0.0
        best_entry = None

        for entry in self.cache_store:
            score = self._cosine_similarity(query_vec, entry["embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.similarity_threshold:
            hit_result = dict(best_entry["result"])
            # Update observability log to reflect Cache Hit
            if "pipeline_log" in hit_result:
                hit_result["pipeline_log"] = list(hit_result["pipeline_log"]) + [
                    f"⚡ SEMANTIC CACHE HIT (Similarity: {best_score:.4f} >= {self.similarity_threshold})"
                ]
            hit_result["cached"] = True
            hit_result["cache_similarity"] = round(best_score, 4)
            return hit_result, best_score

        return None, best_score

    def put(self, query: str, result: Dict):
        """
        Stores a query result into the semantic cache.
        """
        query_vec = self.embedder.encode(query, normalize_embeddings=True)
        entry = {
            "query": query,
            "embedding": query_vec,
            "result": result,
            "timestamp": time.time()
        }
        self.cache_store.append(entry)

    def clear(self):
        self.cache_store.clear()

if __name__ == "__main__":
    print("Testing SemanticCache...")
    cache = SemanticCache(similarity_threshold=0.80)

    q1 = "What is GDPR Article 9?"
    dummy_res = {
        "query": q1,
        "eval_action": "CORRECT",
        "response": "GDPR Article 9 regulates special categories of data.",
        "pipeline_log": ["Evaluator Action: CORRECT"],
        "observability": {"latencies_ms": {"total": 1200.0}}
    }
    cache.put(q1, dummy_res)

    q2 = "What are the rules in Article 9 of GDPR?"
    hit, score = cache.get(q2)
    if hit:
        print(f"✅ Cache Hit! Similarity: {score:.4f}")
        print("Response:", hit["response"])
    else:
        print(f"❌ Cache Miss (Score: {score:.4f})")
