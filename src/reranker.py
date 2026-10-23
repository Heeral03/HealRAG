import sys
import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

import re
from pathlib import Path
from typing import List, Dict, Optional

# Article references like "9(2)(a)" or "89(1)". Used to detect exact-reference queries.
_REF = re.compile(r'\d+(?:\(\w+\))+')


def _refs(text: str) -> set:
    return set(_REF.findall(text.lower()))


# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

class Reranker:
    """
    Cross-Encoder Reranker Stage.
    Reads (query, document_chunk) pairs together using a Cross-Encoder model
    (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2) to calculate fine-grained relevance logits.

    Re-orders candidate chunks (e.g. top 30 candidates from RRF hybrid retrieval)
    and returns the top_k (e.g. top 3).

    If the query cites an article reference (e.g. "Article 9(2)(a)"), chunks whose
    heading carries that same reference are placed first (flagged `ref_match`),
    because a semantic cross-encoder cannot tell "9(2)(a)" from "9(2)(h)".

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

    def rerank(self, query: str, candidate_chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Reranks a candidate pool of document chunks against the query.
        Returns the top_k reordered chunks with 'rerank_score' and 'ref_match' attached.
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
            q_refs = _refs(query)
            reranked_candidates = []
            for idx, chunk in enumerate(candidate_chunks):
                chunk_copy = dict(chunk)
                chunk_copy["rerank_score"] = round(float(scores[idx]), 4)
                # Exact article reference in the query matches this chunk's heading
                chunk_copy["ref_match"] = bool(
                    q_refs and (q_refs & _refs(chunk.get("text", "")[:120]))
                )
                reranked_candidates.append(chunk_copy)

            # Reference matches first, then by rerank_score descending
            reranked_chunks = sorted(
                reranked_candidates,
                key=lambda c: (c["ref_match"], c.get("rerank_score", -999.0)),
                reverse=True,
            )
            elapsed = time.time() - t0
            print(f"[Reranker] Reranked {len(candidate_chunks)} candidates in {elapsed*1000:.2f}ms. Keeping top {top_k}.")
            return reranked_chunks[:top_k]
        except Exception as e:
            print(f"[Reranker Warning] Reranking failed ({e}). Returning original candidates.")
            return candidate_chunks[:top_k]


from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from pydantic import Field


class RetrievedDoc(Document):
    """LangChain Document with dict-like fallback support."""
    def __getitem__(self, item):
        if item == "text" or item == "page_content":
            return self.page_content
        return self.metadata.get(item)

    def get(self, item, default=None):
        if item == "text" or item == "page_content":
            return self.page_content
        return self.metadata.get(item, default)


class RerankingRetriever(BaseRetriever):
    """
    LangChain BaseRetriever combining Qdrant dense vector search,
    BM25 sparse keyword search (RRF), and Cross-Encoder reranking.
    """
    top_k: int = Field(default=5, description="Max number of documents to return")
    candidate_k: int = Field(default=50, description="Candidate pool size for reranking")
    score_margin: Optional[float] = Field(
        default=None,
        description="Keep chunks whose rerank_score is within this many logits of the best chunk. None disables the cutoff.",
    )
    min_k: int = Field(default=1, description="Always keep at least this many chunks")
    ref_exclusive: bool = Field(
        default=False,
        description="If the query cites an article reference and a chunk matches it, return only the matching chunk(s).",
    )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        global _CACHED_RETRIEVER
        if "_CACHED_RETRIEVER" not in globals() or _CACHED_RETRIEVER is None:
            from retriever import Retriever
            _CACHED_RETRIEVER = Retriever()

        raw_results = _CACHED_RETRIEVER.hybrid_retrieve(
            query, top_k=self.top_k, candidate_k=self.candidate_k, enable_rerank=True
        )

        # Exact-lookup mode: the query cites an article and we found it, so drop the rest
        if self.ref_exclusive and any(r.get("ref_match") for r in raw_results):
            raw_results = [r for r in raw_results if r.get("ref_match")]

        # Score cutoff. Measured from the highest score (the first chunk may be a
        # low-scoring exact-reference match). Reference matches are never dropped.
        if self.score_margin is not None and raw_results and "rerank_score" in raw_results[0]:
            best = max(r.get("rerank_score", -999.0) for r in raw_results)
            kept = [
                r for r in raw_results
                if r.get("ref_match") or r.get("rerank_score", -999.0) >= best - self.score_margin
            ]
            raw_results = kept if len(kept) >= self.min_k else raw_results[: self.min_k]

        documents = []
        for r in raw_results:
            text = r.get("text", "")
            meta = {k: v for k, v in r.items() if k != "text"}
            documents.append(RetrievedDoc(page_content=text, metadata=meta))
        return documents


_CACHED_RETRIEVER = None


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