import json
import sys
import time
from pathlib import Path
import numpy as np

# Add src to python path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from retriever import Retriever
import config

# Ground-truth document mappings for standard IR evaluation matching FAISS metadata sources
IR_EVAL_DATASET = [
    {
        "query": "What special categories of personal data are prohibited from processing under GDPR Article 9(1)?",
        "relevant_docs": ["doc_001_gdpr_art9_para1.txt", "doc_002_gdpr_art9_para2_a.txt"]
    },
    {
        "query": "What three clinical sections are mandatory in the International Patient Summary (IPS) profile?",
        "relevant_docs": ["doc_023_ips_profile_summary.txt", "doc_025_ips_profile_summary.txt"]
    },
    {
        "query": "What is the primary role of Health Data Access Bodies (HDABs) under EHDS Chapter IV Article 36?",
        "relevant_docs": ["doc_015_ehds_ch4_art36_data_access_bodies.txt"]
    },
    {
        "query": "What are the required core attributes of a FHIR Patient Resource?",
        "relevant_docs": ["doc_019_fhir_patient_resource.txt"]
    },
    {
        "query": "What platform does the EU Commission establish under EHDS Chapter II Article 5 for cross-border health data exchange?",
        "relevant_docs": ["doc_010_ehds_ch2_art5_myhealth_eu.txt"]
    },
    {
        "query": "What architectural principles define Fast Healthcare Interoperability Resources (FHIR)?",
        "relevant_docs": ["doc_024_fhir_architecture_principles.txt"]
    },
    {
        "query": "How does EHDS secondary data use under Article 34 interact with GDPR Article 89 safeguards for scientific research?",
        "relevant_docs": ["doc_013_ehds_ch4_art34_purposes_secondary_use.txt", "doc_006_gdpr_art89_para1.txt", "doc_007_gdpr_art89_para2.txt"]
    },
    {
        "query": "Compare the opt-out mechanism rights for natural persons under EHDS Chapter IV Article 38 against the restriction rights under Chapter II Article 7.",
        "relevant_docs": ["doc_016_ehds_ch4_art38_opt_out_mechanism.txt", "doc_011_ehds_ch2_art7_right_to_restrict_access.txt"]
    },
    {
        "query": "In a cross-border IPS exchange, what coding standards are recommended for diagnostic results versus clinical procedure histories?",
        "relevant_docs": ["doc_024_ips_diagnostic_results.txt", "doc_025_ips_procedures_history.txt", "doc_027_ips_procedures_history.txt", "doc_026_ips_diagnostic_results.txt"]
    },
    {
        "query": "How do academic evaluations of SMART on FHIR OAuth2 scope-based filters impact API performance and data leakage?",
        "relevant_docs": ["doc_027_paper_fhir_oauth2_interop.txt", "doc_029_paper_fhir_oauth2_interop.txt"]
    },
    {
        "query": "What prohibited secondary uses under EHDS Article 35 safeguard individuals against financial discrimination?",
        "relevant_docs": ["doc_014_ehds_ch4_art35_prohibited_purposes.txt"]
    },
    {
        "query": "How can Zero-Knowledge Proofs (ZKPs) be integrated into EHDS Health Data Access Bodies (HDABs) to satisfy Chapter IV security guidelines?",
        "relevant_docs": ["doc_028_paper_zkp_ehds_privacy.txt", "doc_030_paper_zkp_ehds_privacy.txt"]
    },
    {
        "query": "What are the core Caldicott Principles governing NHS patient data confidentiality?",
        "relevant_docs": ["doc_111_nhs_caldicott_principles.txt"]
    },
    {
        "query": "What are the key obligations under the UK NHS Data Security and Protection Toolkit (DSPT)?",
        "relevant_docs": ["doc_112_nhs_dspt_framework.txt"]
    },
    {
        "query": "How does the NHS National Data Opt-out apply to secondary research?",
        "relevant_docs": ["doc_114_nhs_national_data_opt_out.txt"]
    },
    {
        "query": "What are the required FHIR UK Core profiles for patient demographics?",
        "relevant_docs": ["doc_113_nhs_fhir_uk_core.txt"]
    }
]

def evaluate_retriever_mode(retriever, mode="dense", k_list=[1, 3, 5, 10]):
    metrics_results = {k: {"hits": 0, "reciprocal_ranks": [], "recalls": []} for k in k_list}
    total_queries = len(IR_EVAL_DATASET)
    start_time = time.time()

    for item in IR_EVAL_DATASET:
        query = item["query"]
        relevant_docs = set(item["relevant_docs"])
        max_k = max(k_list)
        
        if mode == "hybrid":
            retrieved_chunks = retriever.hybrid_retrieve(query, top_k=max_k)
        else:
            retrieved_chunks = retriever.retrieve(query, top_k=max_k)
            
        retrieved_sources = [c["source"] for c in retrieved_chunks]

        for k in k_list:
            top_k_sources = retrieved_sources[:k]
            hits = [doc for doc in top_k_sources if doc in relevant_docs]
            hit = 1.0 if len(hits) > 0 else 0.0
            recall = len(set(hits)) / len(relevant_docs) if relevant_docs else 0.0
            
            rr = 0.0
            for rank, doc in enumerate(top_k_sources, start=1):
                if doc in relevant_docs:
                    rr = 1.0 / rank
                    break

            metrics_results[k]["hits"] += hit
            metrics_results[k]["recalls"].append(recall)
            metrics_results[k]["reciprocal_ranks"].append(rr)

    total_time_ms = round((time.time() - start_time) * 1000, 2)
    avg_latency_ms = round(total_time_ms / total_queries, 2)

    summary = {}
    for k in k_list:
        hit_rate = round(metrics_results[k]["hits"] / total_queries * 100, 2)
        mean_recall = round(float(np.mean(metrics_results[k]["recalls"])) * 100, 2)
        mrr = round(float(np.mean(metrics_results[k]["reciprocal_ranks"])), 4)
        summary[f"k={k}"] = {
            "Hit Rate (%)": f"{hit_rate}%",
            "Recall (%)": f"{mean_recall}%",
            "MRR": mrr
        }

    return {
        "summary": summary,
        "avg_latency_ms": avg_latency_ms
    }

def calculate_ir_metrics(k_list=[1, 3, 5, 10]):
    print("=======================================================================")
    print("      STANDARD & HYBRID INFORMATION RETRIEVAL BENCHMARK                ")
    print("=======================================================================")

    retriever = Retriever()

    print("\n--- Running Evaluation: Dense Only (FAISS FlatIP) ---")
    dense_res = evaluate_retriever_mode(retriever, mode="dense", k_list=k_list)
    for k, val in dense_res["summary"].items():
        print(f" - {k:<4s} | Hit Rate: {val['Hit Rate (%)']:>7s} | Recall: {val['Recall (%)']:>7s} | MRR: {val['MRR']:.4f}")
    print(f"   Avg Latency: {dense_res['avg_latency_ms']} ms")

    print("\n--- Running Evaluation: Hybrid Retrieval (BM25 + FAISS + RRF) ---")
    hybrid_res = evaluate_retriever_mode(retriever, mode="hybrid", k_list=k_list)
    for k, val in hybrid_res["summary"].items():
        print(f" - {k:<4s} | Hit Rate: {val['Hit Rate (%)']:>7s} | Recall: {val['Recall (%)']:>7s} | MRR: {val['MRR']:.4f}")
    print(f"   Avg Latency: {hybrid_res['avg_latency_ms']} ms")

    output_payload = {
        "benchmark_metadata": {
            "query_count": len(IR_EVAL_DATASET),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_index_type": "FAISS IndexFlatIP + BM25 RRF"
        },
        "dense_retrieval_summary": dense_res["summary"],
        "hybrid_retrieval_summary": hybrid_res["summary"],
        "latency_ms": {
            "dense": dense_res["avg_latency_ms"],
            "hybrid": hybrid_res["avg_latency_ms"]
        }
    }

    report_path = Path(__file__).resolve().parent / "ir_metrics_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved IR benchmark report to: {report_path}")

if __name__ == "__main__":
    calculate_ir_metrics()
