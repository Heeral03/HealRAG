import argparse
import sys
from pathlib import Path

# Ensure src directory is in sys.path
sys.path.append(str(Path(__file__).resolve().parent))

import config
from seeder import seed_corpus
from chunker import chunk_directory
from embedder import build_index, load_index
from retriever import Retriever
from generator import Generator

def initialize_system(rebuild: bool = False):
    """
    Ensure the corpus is seeded, chunked, and index is loaded.
    If rebuild is True, recreate the vector database.
    """
    db_exists = config.FAISS_INDEX_PATH.exists() and config.METADATA_PATH.exists()
    
    if rebuild or not db_exists:
        print("\n=== Initializing Vector Database ===")
        # 1. Seed corpus
        # Check if corpus folder is empty or need to rebuild
        txt_files = list(config.CORPUS_DIR.glob("*.txt"))
        if rebuild or len(txt_files) == 0:
            print("Seeding corpus data...")
            seed_corpus()
            
        # 2. Chunk documents
        print("Chunking documents...")
        chunks = chunk_directory(config.CORPUS_DIR, config.CHUNK_SIZE_WORDS, config.CHUNK_OVERLAP_WORDS)
        print(f"Total chunks generated: {len(chunks)}")
        
        # 3. Create FAISS index
        print("Building FAISS index...")
        build_index(chunks)
        print("=== Database Initialization Complete ===\n")
    else:
        print("Vector database already exists. Loading index...")

from crag_pipeline import CRAGPipeline

def run_rag_pipeline(query: str, top_k: int = 3, use_vanilla: bool = False):
    """
    Run retrieve-then-generate pipeline (CRAG by default, or Vanilla RAG if flag set).
    """
    if use_vanilla:
        print("\n=== Mode: Vanilla RAG Baseline ===")
        retriever = Retriever()
        print(f"Retrieving top-{top_k} matching chunks for: '{query}'...")
        results = retriever.retrieve(query, top_k=top_k)
        
        print("\n--- Retrieved Chunks ---")
        for i, res in enumerate(results):
            print(f"[{i+1}] {res['source']} (Score: {res['similarity_score']:.4f})")
            preview = res['text'][:250].replace('\n', ' ')
            print(f"    Excerpt: {preview}...")
            
        print("\nInitializing Generator...")
        gen = Generator()
        print("\n--- Generating Response ---")
        answer = gen.generate(query, results)
        print(answer)
        print("---------------------------\n")
    else:
        print("\n=== Mode: Corrective RAG (CRAG) Pipeline ===")
        crag = CRAGPipeline()
        res = crag.run(query, top_k=top_k)
        print(f"\nEvaluator Decision: {res['eval_action']} (Confidence Score: {res['confidence_score']:.4f})")
        for log_entry in res["pipeline_log"]:
            print(f" Log: {log_entry}")
        print("\n--- Generating CRAG Response ---")
        print(res["response"])
        print("---------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Digital Health Corrective RAG (CRAG) CLI")
    parser.add_argument("--query", type=str, help="Question to ask the RAG pipeline")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of corpus seeder and FAISS index")
    parser.add_argument("--vanilla", action="store_true", help="Run Vanilla RAG instead of CRAG pipeline")
    
    args = parser.parse_args()
    
    # Initialize DB (creates index if needed)
    initialize_system(rebuild=args.rebuild)
    
    if args.query:
        run_rag_pipeline(args.query, top_k=args.top_k, use_vanilla=args.vanilla)
    else:
        print("No query provided. Run with --query 'YOUR QUESTION' to execute CRAG.")
        print("Example: python3 src/main.py --query 'What is GDPR Article 9?'")
        print("For Vanilla RAG baseline, add --vanilla flag.")

if __name__ == "__main__":
    main()
