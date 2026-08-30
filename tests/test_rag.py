import pytest
from pathlib import Path
import numpy as np

import sys
# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import config
from chunker import chunk_text, chunk_directory
from embedder import load_index
from retriever import Retriever
from generator import Generator

def test_chunk_text():
    sample_text = "This is a simple text document used to test our sliding-window chunker logic. It has some words."
    # With chunk size 5 and overlap 1:
    # Chunk 1: "This is a simple text"
    # Chunk 2: "text document used to test"
    # Chunk 3: "test our sliding-window chunker logic."
    # Chunk 4: "logic. It has some words."
    chunks = chunk_text(sample_text, chunk_size_words=5, overlap_words=1)
    assert len(chunks) > 0
    assert "This is a simple text" in chunks[0]
    
def test_chunk_directory():
    # Verify that chunking the seeded corpus directory produces chunks
    chunks = chunk_directory(config.CORPUS_DIR, chunk_size_words=100, overlap_words=10)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        assert "source" in chunk
        assert "chunk_index" in chunk

def test_embedder_faiss():
    index, metadata = load_index()
    assert index.ntotal > 0
    assert len(metadata) > 0
    
    # Retrieve embedding dimension
    dimension = index.d
    assert dimension == 384  # Dimension of all-MiniLM-L6-v2

def test_retriever():
    retriever = Retriever()
    query = "What is GDPR Article 9?"
    results = retriever.retrieve(query, top_k=2)
    assert len(results) == 2
    assert "text" in results[0]
    assert "source" in results[0]
    assert "similarity_score" in results[0]
    # Similarity scores should be numerical between -1.0 and 1.0 (normally > 0.0 for relevant queries)
    assert -1.0 <= results[0]["similarity_score"] <= 1.0

def test_generator_mock():
    gen = Generator()
    # Force use_mock True for testing mock response format
    gen.use_mock = True
    
    mock_chunks = [
        {
            "text": "GDPR Article 9 prohibits processing of health data with exceptions.",
            "source": "doc_001.txt",
            "chunk_index": 0
        }
    ]
    query = "Is special category data processing allowed under GDPR?"
    response = gen.generate(query, mock_chunks)
    assert "[LOCAL MOCK LLM MODE" in response
    assert "doc_001.txt" in response
    assert "prohibits processing of health data" in response
