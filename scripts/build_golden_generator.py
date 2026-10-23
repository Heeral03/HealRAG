import json
from pathlib import Path

def build_golden_generator():
    base_dir = Path(__file__).resolve().parent.parent
    db_meta_path = base_dir / "db" / "metadata.json"
    golden_retriever_path = base_dir / "goldens" / "golden_retriever.json"
    output_path = base_dir / "goldens" / "golden_generator.json"

    with open(db_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(golden_retriever_path, "r", encoding="utf-8") as f:
        retriever_cases = json.load(f)

    # Group chunks by source filename
    chunks_by_source = {}
    for doc_id, chunk in meta.items():
        src = chunk.get("source")
        if src not in chunks_by_source:
            chunks_by_source[src] = []
        chunks_by_source[src].append(chunk)

    # Secondary / supporting documents mapped for multi-chunk context
    supporting_sources = {
        "gold_001": ["doc_000_gdpr_art6.txt"],
        "gold_002": ["doc_002_gdpr_art9_para2_a.txt"],
        "gold_003": ["doc_001_gdpr_art9_para1.txt"],
        "gold_004": ["doc_001_gdpr_art9_para1.txt"],
        "gold_005": ["doc_001_gdpr_art9_para1.txt"],
        "gold_006": ["doc_000_gdpr_art5.txt"],
        "gold_007": ["doc_000_gdpr_art5.txt"],
        "gold_008": ["doc_007_ehds_ch2_art4.txt"],
        "gold_009": ["doc_016_ec_ehds_guidance_myhealth.txt"],
        "gold_010": ["doc_006_ehds_ch2_art3.txt"],
        "gold_011": ["doc_011_ehds_ch4_art34.txt"],
        "gold_012": ["doc_010_ehds_ch4_art33.txt"],
        "gold_013": ["doc_011_ehds_ch4_art34.txt"],
        "gold_014": ["doc_015_ehds_ch4_art39.txt"],
        "gold_015": ["doc_019_nhs_national_data_opt_out.txt"],
        "gold_016": ["doc_017_ec_ehds_guidance_spe_standards.txt"],
        "gold_017": ["doc_019_nhs_national_data_opt_out.txt"],
        "gold_018": ["doc_018_nhs_caldicott_principles.txt"],
        "gold_019": [],
        "gold_020": ["doc_021_fhir_r4_patient.txt"]
    }

    generator_goldens = []

    for item in retriever_cases:
        case_id = item["id"]
        primary_src = item["relevant_document"]
        sec_srcs = supporting_sources.get(case_id, [])

        mapped_chunks = []
        # Add primary source chunks
        if primary_src in chunks_by_source:
            mapped_chunks.extend(chunks_by_source[primary_src])

        # Add supporting source chunks
        for s in sec_srcs:
            if s in chunks_by_source:
                mapped_chunks.extend(chunks_by_source[s])

        # Format context objects
        golden_context = []
        for c in mapped_chunks:
            golden_context.append({
                "source": c.get("source", ""),
                "article": c.get("article", ""),
                "publisher": c.get("publisher", ""),
                "jurisdiction": c.get("jurisdiction", ""),
                "text": c.get("text", "")
            })

        gen_case = {
            "id": f"gen_{case_id}",
            "query": item["question"],
            "expected_output": item["ideal_answer"],
            "primary_source": primary_src,
            "golden_context": golden_context
        }
        generator_goldens.append(gen_case)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(generator_goldens, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {len(generator_goldens)} golden generator cases -> {output_path}")

if __name__ == "__main__":
    build_golden_generator()
