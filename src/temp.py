from retriever import Retriever
r = Retriever()
q = "Under what condition does GDPR Article 9(2)(a) permit the processing of special category health data?"
for d in r.metadata.values():
    if "9_para2_a" in d["source"]:
        print(d["source"], "| contains 9(2)(a):", "9(2)(a)" in d["text"])
        print(d["text"][:400])
pool = r.hybrid_retrieve(q, top_k=50, candidate_k=50, enable_rerank=False)
print([i for i, x in enumerate(pool, 1) if "9_para2_a" in x["source"]])