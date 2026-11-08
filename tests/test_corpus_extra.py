import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from chunker import chunk_text
from corpus_extra import EXTRA_CORPUS, US_CORPUS
from seeder import AUTHORITATIVE_CORPUS, FHIR_START_IDX, HIERARCHY_MAP, generate_patient_fhir_records

REQUIRED_KEYS = {"id", "title", "publisher", "jurisdiction", "document_type", "publication_date",
                 "effective_from", "version", "article", "source_url", "content"}


# ---- Corpus integrity (no index needed) -------------------------------------------------

def test_extra_corpus_schema():
    assert len(EXTRA_CORPUS) >= 60
    for doc in EXTRA_CORPUS:
        assert REQUIRED_KEYS <= doc.keys(), doc["id"]
        assert doc["document_type"] in HIERARCHY_MAP, doc["id"]
        assert doc["source_url"].startswith("http"), doc["id"]
        assert doc["content"].strip(), doc["id"]


def test_ids_unique_and_do_not_disturb_existing_filenames():
    existing = [d["id"] for d in AUTHORITATIVE_CORPUS] + \
               [d["id"] for d in generate_patient_fhir_records(start_idx=FHIR_START_IDX, count=75)]
    extra = [d["id"] for d in EXTRA_CORPUS]
    assert len(set(extra)) == len(extra)
    assert not set(extra) & set(existing)
    last_existing = max(int(i.split("_")[1]) for i in existing)
    assert min(int(i.split("_")[1]) for i in extra) > last_existing


def test_us_documents_are_tagged_us():
    us = [d for d in US_CORPUS if d["jurisdiction"] == "US"]
    assert len(us) >= 35
    assert any("hipaa" in d["id"] for d in us)


@pytest.mark.parametrize("doc", EXTRA_CORPUS, ids=lambda d: d["id"])
def test_documents_are_not_fragmented_by_chunker(doc):
    """The chunker splits before every 'Article N' / 'Section N' / 'Chapter X'. A short document must stay one chunk."""
    words = len(doc["content"].split())
    expected = max(1, -(-words // 400))
    assert len(chunk_text(doc["content"], 400, 55)) == expected


# ---- Retrieval (needs the built index, same as test_rag.py) -----------------------------

def _sources(results):
    return [r["source"] for r in results]


def test_hipaa_breach_deadline_is_answerable_locally():
    from retriever import Retriever
    results = Retriever().hybrid_retrieve("What are the HIPAA breach notification deadlines for US hospitals?", top_k=3)
    assert "doc_122_hipaa_breach_164_404_408.txt" in _sources(results)


def test_what_is_hipaa_finds_overview():
    from retriever import Retriever
    results = Retriever().hybrid_retrieve("What is HIPAA?", top_k=3)
    assert "doc_108_hipaa_overview.txt" in _sources(results)


def test_us_jurisdiction_filter():
    from retriever import Retriever
    results = Retriever().hybrid_retrieve("breach notification", top_k=5, jurisdiction="US", enable_rerank=False)
    assert results and all(r["jurisdiction"] == "US" for r in results)


def test_existing_core_lookups_still_work():
    from retriever import Retriever
    r = Retriever()
    assert "doc_000_gdpr_art5.txt" in _sources(r.hybrid_retrieve(
        "What core principles relating to the processing of personal data are established under GDPR Article 5?", top_k=3))
    assert "doc_012_ehds_ch4_art35.txt" in _sources(r.hybrid_retrieve(
        "What prohibited purposes under EHDS Article 35 prevent commercial insurance re-underwriting?", top_k=3))


def test_exact_patient_record_lookup():
    """Near-identical synthetic records are indistinguishable to the embedder; the exact-ID boost must find them."""
    from retriever import Retriever
    r = Retriever()
    for n in (4, 26, 48, 75):
        want = f"doc_{FHIR_START_IDX + n - 1:03d}_fhir_patient_{n}.txt"
        got = _sources(r.hybrid_retrieve(f"Find EHR FHIR export record for PAT-{n:04d}.", top_k=3))
        assert got[0] == want


def test_record_id_helpers():
    from retriever import extract_record_ids, has_record_id
    assert extract_record_ids("Find EHR FHIR export record for PAT-0026.") == {"pat-0026"}
    assert extract_record_ids("What is GDPR Article 9(2)(a)?") == set()
    assert has_record_id("Patient ID PAT-0026 active", {"pat-0026"})
    assert not has_record_id("Patient ID PAT-00261 active", {"pat-0026"})
