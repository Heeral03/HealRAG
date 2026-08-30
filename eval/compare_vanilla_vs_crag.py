import json
import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from retriever import Retriever
from generator import Generator
from crag_pipeline import CRAGPipeline

def run_side_by_side_comparison():
    test_queries = [
        {
            "id": "q1_high_confidence",
            "title": "High Confidence / Single Chunk Lookup",
            "query": "What is GDPR Article 9?"
        },
        {
            "id": "q2_jargon_ambiguity",
            "title": "Industry Jargon / Terminology Drift",
            "query": "What is break-glass emergency?"
        },
        {
            "id": "q3_out_of_corpus",
            "title": "Out-of-Corpus Query (Missing Local Data)",
            "query": "What are the penalty fine tiers under US HIPAA Privacy Rule?"
        },
        {
            "id": "q4_noisy_document",
            "title": "Noisy Document / Feature Filtering",
            "query": "In EHR FHIR Export for Patient PAT-0015, what was the exact LOINC code and numerical value recorded for body weight?"
        }
    ]

    print("Initializing Vanilla RAG and CRAG Pipeline...")
    retriever = Retriever()
    generator = Generator()
    crag = CRAGPipeline()

    comparison_results = []

    for idx, item in enumerate(test_queries):
        title = item["title"]
        query = item["query"]

        print(f"\n=======================================================")
        print(f"TEST [{idx+1}/{len(test_queries)}]: {title}")
        print(f"QUERY: '{query}'")
        print(f"=======================================================")

        # 1. Run Vanilla RAG (WITHOUT Evaluator)
        raw_chunks = retriever.retrieve(query, top_k=3)
        vanilla_response = generator.generate(query, raw_chunks)

        # 2. Run CRAG (WITH Evaluator & Refinement/Fallback)
        crag_result = crag.run(query, top_k=3)

        print("\n--- [WITHOUT EVALUATOR] Vanilla RAG ---")
        print(f"Top Retrieved Score: {raw_chunks[0]['similarity_score']:.4f} ({raw_chunks[0]['source']})")
        print("Response Excerpt:")
        print(vanilla_response[:280].replace("\n", " ") + "...")

        print("\n--- [WITH EVALUATOR] Corrective RAG (CRAG) ---")
        print(f"Evaluator Decision: {crag_result['eval_action']} (Score: {crag_result['confidence_score']:.4f})")
        print(f"Action Taken: {crag_result['pipeline_log'][-1]}")
        print("Response Excerpt:")
        print(crag_result["response"][:280].replace("\n", " ") + "...")
        print("-------------------------------------------------------\n")

        comparison_results.append({
            "test_title": title,
            "query": query,
            "vanilla_rag": {
                "top_score": round(raw_chunks[0]["similarity_score"], 4),
                "retrieved_sources": [c["source"] for c in raw_chunks],
                "response": vanilla_response
            },
            "crag": {
                "eval_action": crag_result["eval_action"],
                "confidence_score": crag_result["confidence_score"],
                "pipeline_log": crag_result["pipeline_log"],
                "response": crag_result["response"]
            }
        })

    # Save output log
    output_path = Path(__file__).resolve().parent / "comparison_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    print(f"Comparison log saved to: {output_path}")

if __name__ == "__main__":
    run_side_by_side_comparison()
