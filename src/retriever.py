import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config
from embedder import load_index, get_embedding_model

def tokenize(text: str) -> list[str]:
    """Simple alphanumeric tokenizer for BM25 keyword matching."""
    return re.findall(r'\w+', text.lower())

class Retriever:
    def __init__(self):
        print("Loading Qdrant DB index and metadata...")
        self.client, self.metadata = load_index()
        print("Loading embedding model for retrieval...")
        self.model = get_embedding_model()

        # Build BM25 index over corpus chunks for sparse keyword retrieval
        print("Building BM25 index over metadata corpus...")
        self.doc_ids = list(self.metadata.keys())
        self.corpus_texts = [self.metadata[doc_id]["text"] for doc_id in self.doc_ids]
        tokenized_corpus = [tokenize(text) for text in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Dense vector retrieval using Qdrant (all-MiniLM-L6-v2 embeddings).
        """
        query_vector = self.model.encode(query, normalize_embeddings=True).tolist()

        search_results = self.client.query_points(
            collection_name=config.QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        ).points

        results = []
        for hit in search_results:
            chunk_info = dict(hit.payload or {})
            chunk_info["similarity_score"] = float(hit.score)
            results.append(chunk_info)

        return results

    def hybrid_retrieve(self, query: str, top_k: int = 5, candidate_k: int = 20, rrf_k: int = 60) -> list[dict]:
        """
        Hybrid retrieval combining Dense Vector Search (Qdrant) and Sparse Keyword Search (BM25)
        using Reciprocal Rank Fusion (RRF).
        RRF Score = 1 / (rrf_k + Dense_Rank) + 1 / (rrf_k + BM25_Rank)
        """
        # 1. Dense Retrieval (Qdrant)
        dense_results = self.retrieve(query, top_k=candidate_k)
        dense_ranks = {}
        for rank, item in enumerate(dense_results, start=1):
            key = (item["source"], item["chunk_index"])
            dense_ranks[key] = (rank, item)

        # 2. Sparse Retrieval (BM25)
        tokenized_query = tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:candidate_k]

        bm25_ranks = {}
        for rank, idx in enumerate(top_bm25_indices, start=1):
            doc_id = self.doc_ids[idx]
            chunk_info = self.metadata[doc_id]
            key = (chunk_info["source"], chunk_info["chunk_index"])
            bm25_ranks[key] = (rank, chunk_info, bm25_scores[idx])

        # 3. Reciprocal Rank Fusion (RRF)
        all_keys = set(dense_ranks.keys()).union(set(bm25_ranks.keys()))
        rrf_scored_items = []

        for key in all_keys:
            rrf_score = 0.0
            chunk_info = None
            dense_sim_score = 0.0

            if key in dense_ranks:
                rank_d, item_d = dense_ranks[key]
                rrf_score += 1.0 / (rrf_k + rank_d)
                chunk_info = item_d
                dense_sim_score = item_d["similarity_score"]

            if key in bm25_ranks:
                rank_b, chunk_b, _ = bm25_ranks[key]
                rrf_score += 1.0 / (rrf_k + rank_b)
                if chunk_info is None:
                    chunk_info = dict(chunk_b)
                    chunk_info["similarity_score"] = 0.0

            item_out = dict(chunk_info)
            item_out["rrf_score"] = float(rrf_score)
            item_out["similarity_score"] = float(dense_sim_score if dense_sim_score > 0 else rrf_score)
            rrf_scored_items.append(item_out)

        # Sort by RRF score descending
        rrf_scored_items.sort(key=lambda x: x["rrf_score"], reverse=True)
        return rrf_scored_items[:top_k]

if __name__ == "__main__":
    print("Testing Qdrant Hybrid Retriever...")
    retriever = Retriever()

    test_query = "What are the required FHIR UK Core profiles for patient demographics?"
    print(f"\nQuery: '{test_query}'")

    print("\n--- Dense Retrieval (Qdrant) ---")
    dense_res = retriever.retrieve(test_query, top_k=3)
    for idx, res in enumerate(dense_res):
        print(f"#{idx+1} [Score: {res['similarity_score']:.4f}] {res['source']}")

    print("\n--- Hybrid Retrieval (BM25 + Qdrant + RRF) ---")
    hybrid_res = retriever.hybrid_retrieve(test_query, top_k=3)
    for idx, res in enumerate(hybrid_res):
        print(f"#{idx+1} [RRF Score: {res['rrf_score']:.5f} | Dense Score: {res['similarity_score']:.4f}] {res['source']}")
