import json
import sys
import time
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from retriever import Retriever
from generator import Generator

def run_evaluation():
    eval_dataset_path = Path(__file__).resolve().parent / "eval_dataset.json"
    results_path = Path(__file__).resolve().parent / "baseline_results.json"
    
    with open(eval_dataset_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)
        
    print(f"Loaded {len(eval_cases)} evaluation test cases from {eval_dataset_path}")
    print("Initializing Retriever and Generator...")
    retriever = Retriever()
    generator = Generator()
    
    evaluation_output = []
    
    category_counts = {"Easy": 0, "Hard": 0, "Adversarial": 0}
    category_success = {"Easy": 0, "Hard": 0, "Adversarial": 0}
    
    for idx, case in enumerate(eval_cases):
        q_id = case["id"]
        cat = case["category"]
        question = case["question"]
        expected_keywords = case["expected_answer_keywords"]
        is_out_of_scope = case["out_of_scope"]
        
        category_counts[cat] += 1
        print(f"\n[{idx+1}/{len(eval_cases)}] Evaluating {q_id} ({cat}): '{question}'")
        
        # 1. Retrieve top-3 chunks
        start_time = time.time()
        retrieved_chunks = retriever.retrieve(question, top_k=3)
        retrieval_time = time.time() - start_time
        
        # Extract metadata
        retrieved_sources = [c["source"] for c in retrieved_chunks]
        top_score = retrieved_chunks[0]["similarity_score"] if retrieved_chunks else 0.0
        
        # 2. Generate response
        gen_start = time.time()
        response_text = generator.generate(question, retrieved_chunks)
        gen_time = time.time() - gen_start
        
        # 3. Automatic classification & check
        passed = False
        notes = ""
        
        if is_out_of_scope:
            # For adversarial queries, success means correctly refusing or noting missing info
            if "does not contain" in response_text.lower() or "not present" in response_text.lower() or "no information" in response_text.lower():
                passed = True
                notes = "Correct Refusal (No Hallucination)"
            else:
                passed = False
                notes = "FAILURE: Hallucination or False Acceptance on Out-of-Scope Query"
        else:
            # For Easy / Hard queries, check if expected keywords are present in response
            matched_keywords = [kw for kw in expected_keywords if kw.lower() in response_text.lower()]
            match_ratio = len(matched_keywords) / len(expected_keywords) if expected_keywords else 1.0
            
            if match_ratio >= 0.5:
                passed = True
                notes = f"Success ({len(matched_keywords)}/{len(expected_keywords)} keywords matched)"
            else:
                passed = False
                notes = f"FAILURE: Incomplete or Missing Synthesis ({len(matched_keywords)}/{len(expected_keywords)} keywords matched)"
                
        if passed:
            category_success[cat] += 1
            
        print(f" -> Top Similarity Score: {top_score:.4f}")
        print(f" -> Result: {'PASSED' if passed else 'FAILED'} ({notes})")
        
        evaluation_output.append({
            "id": q_id,
            "category": cat,
            "question": question,
            "top_similarity_score": top_score,
            "retrieved_sources": retrieved_sources,
            "passed": passed,
            "evaluation_note": notes,
            "response_preview": response_text[:300].replace("\n", " "),
            "full_response": response_text,
            "timing_sec": {
                "retrieval": round(retrieval_time, 3),
                "generation": round(gen_time, 3)
            }
        })
        
    # Save baseline results
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_questions": len(eval_cases),
            "overall_accuracy": sum(category_success.values()) / len(eval_cases),
            "category_breakdown": {
                c: {
                    "total": category_counts[c],
                    "passed": category_success[c],
                    "accuracy": category_success[c] / category_counts[c] if category_counts[c] > 0 else 0
                } for c in category_counts
            },
            "results": evaluation_output
        }, f, indent=2)
        
    print("\n=======================================================")
    print("             BASELINE EVALUATION SUMMARY               ")
    print("=======================================================")
    print(f"Total Test Questions: {len(eval_cases)}")
    print(f"Overall Accuracy:     {sum(category_success.values()) / len(eval_cases) * 100:.1f}%")
    for cat in ["Easy", "Hard", "Adversarial"]:
        acc = category_success[cat] / category_counts[cat] * 100 if category_counts[cat] > 0 else 0
        print(f" - {cat:12s}: {category_success[cat]}/{category_counts[cat]} ({acc:.1f}%)")
    print("=======================================================")
    print(f"Full benchmark log saved to: {results_path}")

if __name__ == "__main__":
    run_evaluation()
