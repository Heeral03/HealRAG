import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import crag_pipeline
from crag_pipeline import CRAGPipeline
from generator import NO_ANSWER_SENTENCE, is_no_answer

DEFINITION = "The General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, is the EU data protection law."


def _chunk(source, text, score):
    return {"source": source, "text": text, "similarity_score": score, "chunk_index": 0}


class StubRetriever:
    """Plain query only finds topical chunks; the expanded query also finds the definition."""

    def hybrid_retrieve(self, query, top_k=3):
        topical = [_chunk("doc_026_ico.txt", "The ICO publishes UK GDPR health data guidance for providers.", 0.67)]
        if "General Data Protection Regulation" in query:
            return [_chunk("doc_000_gdpr_overview.txt", DEFINITION, 0.80)] + topical
        return topical


class StubSearcher:
    def __init__(self, results=None):
        self.calls = 0
        self.results = results or []

    def expand_query(self, query):
        return query + " (General Data Protection Regulation (EU) 2016/679 overview)" if "gdpr" in query.lower() else query

    def search(self, query):
        self.calls += 1
        return self.results


class RefusingGenerator:
    """Mimics the LLM rule: refuse unless the context actually contains the definition."""
    model_name = "stub"

    def __init__(self):
        self.calls = 0

    def generate(self, query, chunks):
        self.calls += 1
        if any(DEFINITION in c.get("text", "") for c in chunks):
            return "GDPR is Regulation (EU) 2016/679. [Source: doc_000_gdpr_overview.txt]"
        return NO_ANSWER_SENTENCE


def _pipeline(monkeypatch, searcher):
    monkeypatch.setattr(crag_pipeline, "log_query", lambda *a, **k: None)
    gen = RefusingGenerator()
    pipe = CRAGPipeline(enable_cache=False, retriever=StubRetriever(), generator=gen)
    pipe.searcher = searcher
    return pipe, gen


def test_is_no_answer():
    assert is_no_answer(NO_ANSWER_SENTENCE + "\n\nThe context mentions GDPR only in passing.")
    assert not is_no_answer("GDPR is Regulation (EU) 2016/679.")


def test_refusal_triggers_corrective_reretrieval(monkeypatch):
    pipe, gen = _pipeline(monkeypatch, StubSearcher())
    res = pipe.run("What is GDPR?")

    # The evaluator trusted the topical chunk (CORRECT), but the answer was a refusal -> corrected
    assert res["observability"]["initial_eval_action"] == "CORRECT"
    assert res["eval_action"] == "AMBIGUOUS"
    assert not is_no_answer(res["response"])
    assert gen.calls == 2  # one corrective regeneration, no loops
    assert any(c["source"] == "doc_000_gdpr_overview.txt" for c in res["final_chunks"])
    assert any("Answerability Check" in line for line in res["pipeline_log"])


def test_unanswerable_query_stays_honest(monkeypatch):
    pipe, gen = _pipeline(monkeypatch, StubSearcher())
    res = pipe.run("What colour is the sky?")  # nothing new to retrieve, no web results

    assert is_no_answer(res["response"])
    assert res["eval_action"] == "INCORRECT"
    assert gen.calls == 1  # no new evidence -> no pointless regeneration


def test_no_second_web_search_when_already_used(monkeypatch):
    searcher = StubSearcher()
    pipe, _ = _pipeline(monkeypatch, searcher)
    pipe.retriever = type("R", (), {"hybrid_retrieve": lambda self, q, top_k=3: [_chunk("x.txt", "unrelated", 0.30)]})()
    pipe.run("What is the weather?")
    assert searcher.calls == 1  # INCORRECT route already searched; the correction must not search again
