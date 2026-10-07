import sys
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from pathlib import Path
from typing import List, Dict

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

class Reranker:
    """
    Cross-Encoder Reranker Stage.
    Reads (query, document_chunk) pairs together using a Cross-Encoder model 
    (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2) to calculate fine-grained relevance logits.

    Re-orders candidate chunks (e.g. top 30 candidates from RRF hybrid retrieval) 
    and returns the top_k (e.g. top 3).

    Preserves the original `similarity_score` on each chunk so the downstream 
    CRAG Evaluator thresholds (0.60 / 0.45) remain un-broken.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._loaded = False

    def _load_model(self):
        if not self._loaded:
            try:
                from sentence_transformers import CrossEncoder
                print(f"Loading Cross-Encoder Reranker Model ({self.model_name})...")
                self._model = CrossEncoder(self.model_name, max_length=512)
                self._loaded = True
            except Exception as e:
                print(f"[Reranker Warning] Failed to load CrossEncoder ({e}). Fallback to identity ordering.")
                self._loaded = False

    def rerank(self, query: str, candidate_chunks: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Reranks a candidate pool of document chunks against the query.
        Returns the top_k reordered chunks with 'rerank_score' attached.
        """
        if not candidate_chunks:
            return []

        self._load_model()

        if not self._loaded or self._model is None:
            # Fallback: return original top_k candidates unaltered
            return candidate_chunks[:top_k]

        t0 = time.time()
        pairs = [(query, chunk.get("text", "")) for chunk in candidate_chunks]
        
        try:
            scores = self._model.predict(pairs)
            reranked_candidates = []
            for idx, chunk in enumerate(candidate_chunks):
                chunk_copy = dict(chunk)
                chunk_copy["rerank_score"] = round(float(scores[idx]), 4)
                reranked_candidates.append(chunk_copy)

            # Sort candidate chunks by rerank_score descending
            reranked_chunks = sorted(reranked_candidates, key=lambda c: c.get("rerank_score", -999.0), reverse=True)
            elapsed = time.time() - t0
            print(f"[Reranker] Reranked {len(candidate_chunks)} candidates in {elapsed*1000:.2f}ms. Keeping top {top_k}.")
            return reranked_chunks[:top_k]
        except Exception as e:
            print(f"[Reranker Warning] Reranking failed ({e}). Returning original candidates.")
            return candidate_chunks[:top_k]


if __name__ == "__main__":
    reranker = Reranker()
    sample_query = "What is GDPR Article 9?"
    sample_candidates = [
        {"source": "doc_001.txt", "text": "GDPR Article 9 covers processing of personal health data.", "similarity_score": 0.55},
        {"source": "doc_002.txt", "text": "GDPR Article 8 covers child consent for information society services.", "similarity_score": 0.70},
        {"source": "doc_003.txt", "text": "Random unrelated statutory overview text.", "similarity_score": 0.40},
    ]
    results = reranker.rerank(sample_query, sample_candidates, top_k=2)
    print("Reranked Top Chunks:")
    for r in results:
        print(f" - {r['source']} | Rerank Score: {r.get('rerank_score', 'N/A'):.4f} | Original Sim: {r['similarity_score']}")
