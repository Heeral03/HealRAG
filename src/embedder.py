import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import config

def get_embedding_model() -> SentenceTransformer:
    """Load the SentenceTransformer model."""
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)

_QDRANT_CLIENT_INSTANCE = None

def get_qdrant_client() -> QdrantClient:
    """Returns local disk-persisted QdrantClient instance (singleton)."""
    global _QDRANT_CLIENT_INSTANCE
    if _QDRANT_CLIENT_INSTANCE is None:
        config.QDRANT_DIR.mkdir(parents=True, exist_ok=True)
        _QDRANT_CLIENT_INSTANCE = QdrantClient(path=str(config.QDRANT_DIR))
    return _QDRANT_CLIENT_INSTANCE

def build_index(chunks: list[dict]):
    """
    Generate embeddings for chunks, build a Qdrant collection, and persist points & metadata payload.
    """
    if not chunks:
        print("No chunks provided to build index.")
        return

    print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
    model = get_embedding_model()

    texts = [c["text"] for c in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")

    client = get_qdrant_client()

    # Re-create collection if exists
    if client.collection_exists(config.QDRANT_COLLECTION_NAME):
        client.delete_collection(config.QDRANT_COLLECTION_NAME)

    client.create_collection(
        collection_name=config.QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
    )

    points = []
    metadata = {}
    for idx, chunk in enumerate(chunks):
        chunk_info = dict(chunk)
        metadata[str(idx)] = chunk_info
        points.append(PointStruct(
            id=idx,
            vector=embeddings[idx].tolist(),
            payload=chunk_info
        ))

    print(f"Upserting {len(points)} points into Qdrant collection '{config.QDRANT_COLLECTION_NAME}'...")
    client.upsert(
        collection_name=config.QDRANT_COLLECTION_NAME,
        points=points
    )

    # Save local metadata backup JSON for fast BM25 index initialization
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved Qdrant collection to {config.QDRANT_DIR} and metadata to {config.METADATA_PATH}")

def load_index():
    """
    Load the Qdrant client and metadata.
    Returns: (qdrant_client, metadata)
    """
    client = get_qdrant_client()
    if not client.collection_exists(config.QDRANT_COLLECTION_NAME) or not config.METADATA_PATH.exists():
        raise FileNotFoundError("Qdrant collection or metadata file not found. Please build the index first.")

    with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return client, metadata

if __name__ == "__main__":
    from chunker import chunk_directory

    print("Testing Qdrant embedder/indexer...")
    chunks = chunk_directory(config.CORPUS_DIR, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
    build_index(chunks)

    client, meta = load_index()
    coll_info = client.get_collection(config.QDRANT_COLLECTION_NAME)
    print(f"Loaded Qdrant collection '{config.QDRANT_COLLECTION_NAME}' containing {coll_info.points_count} points.")
    print(f"Loaded metadata containing {len(meta)} entries.")
