"""Policy RAG: chunking, offline retrieval, queue citations, and the embedding/LLM paths with the API mocked."""
import numpy as np

from closepilot import graph, llm, rag
from eval.gold_policy import KIND_CLAUSE, QUESTIONS


def test_chunks_have_unique_clause_ids_and_gold_refers_to_real_clauses():
    ids = [c["clause"] for c in rag.chunks()]
    assert len(ids) == len(set(ids)) >= 25 and all(c["text"] for c in rag.chunks())
    gold = set().union(*[g for _, g in QUESTIONS], *KIND_CLAUSE.values())
    assert gold <= set(ids)


def test_lexical_search_finds_the_clause():
    assert rag.search("three-way match goods receipt note before payment", k=1)[0]["clause"] == "AP-2"
    hit = rag.search("vendor bank account change call back", k=1)[0]
    assert hit["clause"] == "VM-2" and hit["retrieval"] == "lexical"


def test_close_run_cites_a_clause_for_every_queue_item(con):
    s = graph.run_close(con, use_llm=False)
    n_queue = con.execute("SELECT count(*) FROM approval_queue").fetchone()[0]
    assert s["cited"] == n_queue > 0
    dup = con.execute("SELECT DISTINCT c.clause FROM approval_queue q JOIN policy_citations c USING (item_id) "
                      "WHERE q.kind = 'duplicate_invoice'").fetchall()
    assert dup == [("AP-5",)]


def _fake_embed(calls):
    """Deterministic bag-of-words vectors, so the hybrid path runs without the API."""
    def embed(texts, retries=3):
        calls.append(len(texts))
        out = []
        for t in texts:
            v = np.zeros(64)
            for w in t.lower().split():
                v[hash(w) % 64] += 1
            out.append(v.tolist())
        return out
    return embed


def test_hybrid_stores_embeddings_in_duckdb_and_reuses_them(con, monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "embed", _fake_embed(calls))
    hit = rag.search("duplicate invoices from the same vendor", k=1, con=con, hybrid=True)[0]
    assert hit["retrieval"] == "hybrid"
    assert con.execute("SELECT count(*) FROM policy_chunks").fetchone()[0] == len(rag.chunks())
    rag.search("segregation of duties", k=1, con=con, hybrid=True)
    assert calls == [len(rag.chunks()), 1, 1]  # chunks embedded once, then one call per query


def test_hybrid_falls_back_to_lexical_when_embedding_fails(con, monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "embed", lambda texts, retries=3: None)
    assert rag.search("bank charges", k=1, con=con, hybrid=True)[0]["retrieval"] == "lexical"


def test_answer_keeps_only_retrieved_clause_ids(con, monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "embed", lambda texts, retries=3: None)
    seen = {}

    def structured(prompt, schema, retries=3):
        seen["prompt"] = prompt
        return schema(answer="No, a goods receipt note is needed.", clauses=["AP-2", "ZZ-99"])

    monkeypatch.setattr(llm, "structured", structured)
    r = rag.answer(con, "Can we pay before the goods receipt note is issued?")
    assert r["citations"] == ["AP-2"] and "[AP-2]" in seen["prompt"]


def test_answer_without_llm_returns_sources_only(con):
    r = rag.answer(con, "bank charges journal entry")
    assert r["answer"] is None and r["sources"][0]["clause"] == "BR-4"
