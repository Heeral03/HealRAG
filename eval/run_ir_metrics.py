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
    {"query": "What special categories of personal data are prohibited from processing under GDPR Article 9(1)?", "relevant_docs": ["doc_001_gdpr_art9_para1.txt"]},
    {"query": "Under what explicit consent conditions can special category data be processed per GDPR Article 9(2)(a)?", "relevant_docs": ["doc_002_gdpr_art9_para2_a.txt"]},
    {"query": "How does GDPR Article 9(2)(h) handle data processing for medical diagnosis and health care provision?", "relevant_docs": ["doc_003_gdpr_art9_para2_h.txt"]},
    {"query": "What public health threat exception exists under GDPR Article 9(2)(i)?", "relevant_docs": ["doc_004_gdpr_art9_para2_i.txt"]},
    {"query": "What safeguards for scientific research and statistics are required under GDPR Article 89(1)?", "relevant_docs": ["doc_005_gdpr_art89_para1.txt"]},
    {"query": "What access rights do natural persons have to their electronic health data under EHDS Chapter II Article 3?", "relevant_docs": ["doc_006_ehds_ch2_art3.txt"]},
    {"query": "How are health professionals across EU Member States granted access to patient data under EHDS Article 4?", "relevant_docs": ["doc_007_ehds_ch2_art4.txt"]},
    {"query": "What platform does the EU Commission establish under EHDS Chapter II Article 5 for cross-border health data exchange?", "relevant_docs": ["doc_008_ehds_ch2_art5.txt"]},
    {"query": "How does EHDS Article 7 regulate patient restriction rights and break-glass emergency overrides?", "relevant_docs": ["doc_009_ehds_ch2_art7.txt"]},
    {"query": "What categories of health data are listed for secondary use under EHDS Chapter IV Article 33?", "relevant_docs": ["doc_010_ehds_ch4_art33.txt"]},
    {"query": "What permitted purposes for secondary health data use are established under EHDS Article 34?", "relevant_docs": ["doc_011_ehds_ch4_art34.txt"]},
    {"query": "What prohibited secondary uses safeguard individuals under EHDS Chapter IV Article 35?", "relevant_docs": ["doc_012_ehds_ch4_art35.txt"]},
    {"query": "What is the primary role of Health Data Access Bodies (HDABs) under EHDS Chapter IV Article 36?", "relevant_docs": ["doc_013_ehds_ch4_art36.txt"]},
    {"query": "How does natural person opt-out right operate under EHDS Chapter IV Article 38?", "relevant_docs": ["doc_014_ehds_ch4_art38.txt"]},
    {"query": "What requirements define Secure Processing Environments (SPE) under EHDS Article 39?", "relevant_docs": ["doc_015_ehds_ch4_art39.txt"]},
    {"query": "What guidance does the European Commission provide on MyHealth@EU architecture and NCPH integration?", "relevant_docs": ["doc_016_ec_ehds_guidance_myhealth.txt"]},
    {"query": "What guidelines has the EC issued for technical implementation of Secure Processing Environments?", "relevant_docs": ["doc_017_ec_ehds_guidance_spe_standards.txt"]},
    {"query": "What are the core Caldicott Principles governing NHS patient data confidentiality?", "relevant_docs": ["doc_018_nhs_caldicott_principles.txt"]},
    {"query": "How does the NHS National Data Opt-out policy and COPI statutory scope operate for secondary research?", "relevant_docs": ["doc_019_nhs_national_data_opt_out.txt"]},
    {"query": "What are the key obligations and submission deadlines for the NHS Data Security and Protection Toolkit (DSPT) 2025-26?", "relevant_docs": ["doc_020_nhs_dspt_2025_26.txt"]},
    {"query": "What are the core required attributes of an HL7 FHIR R4 Patient Resource?", "relevant_docs": ["doc_021_fhir_r4_patient.txt"]},
    {"query": "What are the required FHIR UK Core profiles and NHS Number extension rules for patient demographics?", "relevant_docs": ["doc_022_fhir_uk_core_patient.txt"]},
    {"query": "What status codes and value sets are required for an HL7 FHIR R4 Observation Resource?", "relevant_docs": ["doc_023_fhir_r4_observation.txt"]},
    {"query": "How does an HL7 FHIR R4 Bundle container structure search set and transaction payloads?", "relevant_docs": ["doc_024_fhir_r4_bundle.txt"]},
    {"query": "How are opt-out and consent policies modeled within an HL7 FHIR R4 Consent resource?", "relevant_docs": ["doc_025_fhir_r4_consent.txt"]},
    {"query": "How do ICO guidelines handle health data processing, special category definitions, and anonymisation?", "relevant_docs": ["doc_026_ico_health_data_guidance.txt"]},
    {"query": "What government cyber security strategy standards apply to UK health data protection?", "relevant_docs": ["doc_027_govuk_data_security_standards.txt"]},
    {"query": "How do smart contract consent engines enable cross-border consent under EHDS networks?", "relevant_docs": ["doc_028_paper_blockchain_consent.txt"]},
    {"query": "How can Zero-Knowledge Proofs (ZKPs) be integrated into EHDS Health Data Access Bodies (HDABs)?", "relevant_docs": ["doc_029_paper_zkp_ehds_privacy.txt"]},
    {"query": "What demographics and NHS number extension are recorded in EHR FHIR Patient PAT-0001?", "relevant_docs": ["doc_030_fhir_patient_1.txt"]},
    {"query": "What patient record payload corresponds to PAT-0005 in EHR exports?", "relevant_docs": ["doc_034_fhir_patient_5.txt"]},
    {"query": "Which EHR patient payload contains PAT-0010?", "relevant_docs": ["doc_039_fhir_patient_10.txt"]},
    {"query": "Find EHR FHIR export record for PAT-0015.", "relevant_docs": ["doc_044_fhir_patient_15.txt"]},
    {"query": "Find EHR FHIR export record for PAT-0020.", "relevant_docs": ["doc_049_fhir_patient_20.txt"]},
    {"query": "Find EHR FHIR export record for PAT-0025.", "relevant_docs": ["doc_054_fhir_patient_25.txt"]},
    {"query": "Find EHR FHIR export record for PAT-0030.", "relevant_docs": ["doc_059_fhir_patient_30.txt"]},
    {"query": "Find EHR FHIR export record for PAT-0035.", "relevant_docs": ["doc_064_fhir_patient_35.txt"]},
    {"query": "Find EHR FHIR export record for PAT-0040.", "relevant_docs": ["doc_069_fhir_patient_40.txt"]},
    {"query": "Find EHR FHIR export record for PAT-0045.", "relevant_docs": ["doc_074_fhir_patient_45.txt"]},
    {"query": "Find EHR FHIR export record for PAT-0050.", "relevant_docs": ["doc_079_fhir_patient_50.txt"]},
    {"query": "Find EHR FHIR export record for PAT-0055.", "relevant_docs": ["doc_084_fhir_patient_55.txt"]},
    {"query": "Find EHR FHIR export record for PAT-0060.", "relevant_docs": ["doc_089_fhir_patient_60.txt"]},
    {"query": "Find EHR FHIR export record for PAT-0065.", "relevant_docs": ["doc_094_fhir_patient_65.txt"]},
    {"query": "Find EHR FHIR export record for PAT-0070.", "relevant_docs": ["doc_099_fhir_patient_70.txt"]},
    {"query": "Find EHR FHIR export record for PAT-0075.", "relevant_docs": ["doc_104_fhir_patient_75.txt"]},
    {"query": "What prohibited purposes under EHDS Article 35 prevent commercial insurance re-underwriting?", "relevant_docs": ["doc_012_ehds_ch4_art35.txt"]},
    {"query": "What cross-border health data access rights exist for clinicians under EHDS Chapter II?", "relevant_docs": ["doc_007_ehds_ch2_art4.txt"]},
    {"query": "How is patient explicit consent defined under GDPR Article 9?", "relevant_docs": ["doc_002_gdpr_art9_para2_a.txt"]},
    {"query": "What is the role of National Contact Points for eHealth (NCPH) in MyHealth@EU?", "relevant_docs": ["doc_016_ec_ehds_guidance_myhealth.txt"]},
    {"query": "What anonymisation techniques are recommended by the ICO Code of Practice for health research?", "relevant_docs": ["doc_026_ico_health_data_guidance.txt"]}
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
