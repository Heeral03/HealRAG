import json
import sys
import time
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from retriever import Retriever
from generator import Generator

def run_stress_benchmark():
    dataset_path = Path(__file__).resolve().parent / "stress_test_dataset.json"
    results_path = Path(__file__).resolve().parent / "stress_results.json"
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        stress_cases = json.load(f)
        
    print(f"Loaded {len(stress_cases)} stress test cases targeting Vanilla RAG failure modes.")
    retriever = Retriever()
    generator = Generator()
    
    benchmark_results = []
    
    for idx, case in enumerate(stress_cases):
        q_id = case["id"]
        failure_mode = case["failure_mode"]
        question = case["question"]
        desc = case["description"]
        expected_crag = case["expected_behavior_crag"]
        
        print(f"\n=======================================================")
        print(f"STRESS TEST #{idx+1} [{q_id}]")
        print(f"Failure Mode: {failure_mode}")
        print(f"Query: '{question}'")
        print(f"=======================================================")
        
        start_time = time.time()
        retrieved_chunks = retriever.retrieve(question, top_k=3)
        retrieval_time = time.time() - start_time
        
        print("\n--- Top-3 Retrieved Chunks ---")
        for rank, c in enumerate(retrieved_chunks):
            print(f" [{rank+1}] Score: {c['similarity_score']:.4f} | Source: {c['source']}")
            print(f"      Excerpt: {c['text'][:180].replace('\n', ' ')}...")
            
        gen_start = time.time()
        response_text = generator.generate(question, retrieved_chunks)
        gen_time = time.time() - gen_start
        
        print("\n--- LLM Output (Vanilla RAG) ---")
        print(response_text)
        print("-------------------------------------------------------\n")
        
        benchmark_results.append({
            "id": q_id,
            "failure_mode": failure_mode,
            "question": question,
            "description": desc,
            "expected_behavior_crag": expected_crag,
            "top_retrieved_sources": [c["source"] for c in retrieved_chunks],
            "top_similarity_scores": [round(c["similarity_score"], 4) for c in retrieved_chunks],
            "llm_response": response_text,
            "timing_sec": {
                "retrieval": round(retrieval_time, 3),
                "generation": round(gen_time, 3)
            }
        })
        
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
        
    print(f"\nStress test benchmark completed. Full log saved to: {results_path}")

if __name__ == "__main__":
    run_stress_benchmark()
