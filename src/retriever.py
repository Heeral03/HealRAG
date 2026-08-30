import numpy as np
from sentence_transformers import SentenceTransformer

import config
from embedder import load_index, get_embedding_model

class Retriever:
    def __init__(self):
        print("Loading FAISS DB index and metadata...")
        self.index, self.metadata = load_index()
        print("Loading embedding model for retrieval...")
        self.model = get_embedding_model()
        
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Embed the query and retrieve the top-k most similar document chunks.
        """
        # Embed and normalize to compute cosine similarity using FlatIP
        query_vector = self.model.encode([query], normalize_embeddings=True)
        query_vector = np.array(query_vector).astype("float32")
        
        # Search the index
        # index.search returns (distances, indices)
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        # scores[0] contains the similarity scores, indices[0] contains the index IDs
        for score, idx_val in zip(scores[0], indices[0]):
            # FAISS returns -1 for empty slots, or if index is out of bounds
            if idx_val == -1 or str(idx_val) not in self.metadata:
                continue
                
            chunk_info = self.metadata[str(idx_val)]
            results.append({
                "text": chunk_info["text"],
                "source": chunk_info["source"],
                "chunk_index": chunk_info["chunk_index"],
                "similarity_score": float(score)  # Cosine similarity score
            })
            
        return results

if __name__ == "__main__":
    print("Testing Retriever...")
    retriever = Retriever()
    
    # Test query
    test_query = "What rights do natural persons have to access health data?"
    print(f"\nQuery: '{test_query}'")
    
    results = retriever.retrieve(test_query, top_k=3)
    for idx, res in enumerate(results):
        print(f"\nResult #{idx+1} [Score: {res['similarity_score']:.4f}] from {res['source']}:")
        print(f"Content: {res['text'][:300]}...")
