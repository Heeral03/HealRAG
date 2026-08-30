import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

import config

def get_embedding_model() -> SentenceTransformer:
    """
    Load the SentenceTransformer model.
    """
    # The models are downloaded locally to ~/.cache/huggingface/hub
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)

def build_index(chunks: list[dict]):
    """
    Generate embeddings for chunks, build a FAISS index, and persist index & metadata.
    """
    if not chunks:
        print("No chunks provided to build index.")
        return
        
    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
    model = get_embedding_model()
    
    texts = [c["text"] for c in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    # Generate L2 normalized embeddings (so inner product in FAISS is equivalent to cosine similarity)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    
    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")
    
    # Create FAISS IndexFlatIP (Inner Product)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Ensure DB directory exists
    config.DB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the index to disk
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    print(f"Saved FAISS index to {config.FAISS_INDEX_PATH}")
    
    # Save the metadata (mappings from FAISS internal ID to original chunk/document data)
    metadata = {}
    for idx, chunk in enumerate(chunks):
        metadata[str(idx)] = {
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"]
        }
        
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {config.METADATA_PATH}")

def load_index():
    """
    Load the FAISS index and metadata.
    Returns: (index, metadata)
    """
    if not config.FAISS_INDEX_PATH.exists() or not config.METADATA_PATH.exists():
        raise FileNotFoundError("FAISS index or metadata file not found. Please build the index first.")
        
    index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    return index, metadata

if __name__ == "__main__":
    from chunker import chunk_directory
    
    print("Testing embedder/indexer...")
    # Gather chunks
    chunks = chunk_directory(config.CORPUS_DIR, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
    
    # Build/Re-build index
    build_index(chunks)
    
    # Test loading
    idx, meta = load_index()
    print(f"Loaded FAISS index containing {idx.ntotal} vectors.")
    print(f"Loaded metadata containing {len(meta)} entries.")
    if len(meta) > 0:
        print(f"First entry sample: {meta['0']}")
