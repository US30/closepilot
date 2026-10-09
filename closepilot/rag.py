"""Policy RAG over docs/policies (accounting policy, delegation of authority, vendor contract).

One chunk per numbered clause. Retrieval is TF-IDF (always on, offline) fused by reciprocal
rank with Gemini embeddings when a key is set. Embeddings are stored in DuckDB and reused
across runs. An LLM answer may cite only clauses that were actually retrieved.
"""
from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

from closepilot import llm

DOCS = Path(__file__).resolve().parent.parent / "docs" / "policies"
CLAUSE = re.compile(r"^### (\S+) (.+)$", re.M)
RRF_K = 60


class PolicyAnswer(BaseModel):
    answer: str
    clauses: list[str]  # clause ids the answer relies on


@lru_cache
def chunks() -> tuple[dict, ...]:
    out = []
    for f in sorted(DOCS.glob("*.md")):
        md = f.read_text()
        doc = md.splitlines()[0].lstrip("# ").strip()
        parts = CLAUSE.split(md)[1:]  # clause, title, body, clause, title, body, ...
        for clause, title, body in zip(parts[::3], parts[1::3], parts[2::3]):
            out.append(dict(clause=clause, doc=doc, title=title.strip(), text=body.split("\n## ")[0].strip()))
    return tuple(out)


def _text(c: dict) -> str:
    return f"{c['title']}. {c['text']}"


@lru_cache
def _tfidf():
    v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    return v, v.fit_transform([_text(c) for c in chunks()])


def _lexical_scores(query: str) -> np.ndarray:
    v, m = _tfidf()
    return (m @ v.transform([query]).T).toarray().ravel()


def _vector_scores(con, query: str) -> np.ndarray | None:
    """Cosine similarity per chunk, computed in DuckDB. Embeds only new or changed chunks. None if the API fails."""
    con.execute("CREATE TABLE IF NOT EXISTS policy_chunks (clause VARCHAR PRIMARY KEY, text_hash VARCHAR, embedding FLOAT[])")
    model = os.getenv("GEMINI_EMBED_MODEL", llm.DEFAULT_EMBED_MODEL)
    want = {c["clause"]: hashlib.sha1(f"{model}|{_text(c)}".encode()).hexdigest() for c in chunks()}
    have = dict(con.execute("SELECT clause, text_hash FROM policy_chunks").fetchall())
    todo = [c for c in chunks() if have.get(c["clause"]) != want[c["clause"]]]
    if todo:
        vecs = llm.embed([_text(c) for c in todo])
        if vecs is None:
            return None
        con.executemany("INSERT OR REPLACE INTO policy_chunks VALUES (?,?,?)",
                        [(c["clause"], want[c["clause"]], v) for c, v in zip(todo, vecs)])
    q = llm.embed([query])
    if q is None:
        return None
    sims = dict(con.execute("SELECT clause, list_cosine_similarity(embedding, ?::FLOAT[]) FROM policy_chunks", [q[0]]).fetchall())
    return np.array([sims[c["clause"]] for c in chunks()])


def search(query: str, k: int = 3, con=None, hybrid: bool = False) -> list[dict]:
    """Top-k clauses. hybrid=True adds the embedding ranking; it falls back to TF-IDF alone if embeddings are unavailable."""
    rankings = [np.argsort(-_lexical_scores(query), kind="stable")]
    if hybrid and con is not None and llm.available():
        vs = _vector_scores(con, query)
        if vs is not None:
            rankings.append(np.argsort(-vs, kind="stable"))
    score = np.zeros(len(chunks()))
    for r in rankings:
        score[r] += 1 / (RRF_K + 1 + np.arange(len(r)))
    mode = "hybrid" if len(rankings) == 2 else "lexical"
    return [chunks()[i] | dict(score=round(float(score[i]), 4), retrieval=mode)
            for i in np.argsort(-score, kind="stable")[:k]]


def cite_queue(con, hybrid: bool = False) -> int:
    """Attach the best-matching policy clause to every approval-queue item. Retrieval runs once per (agent, kind)."""
    q = con.execute("SELECT item_id, agent, kind, summary FROM approval_queue ORDER BY item_id").df()
    rows = []
    for (_, kind), g in q.groupby(["agent", "kind"]):
        hit = search(f"{kind.replace('_', ' ')}. {g.summary.iloc[0]}", k=1, con=con, hybrid=hybrid)[0]
        rows += [(i, hit["clause"], hit["doc"], hit["title"], hit["retrieval"]) for i in g.item_id]
    con.execute("CREATE OR REPLACE TABLE policy_citations "
                "(item_id VARCHAR, clause VARCHAR, doc VARCHAR, title VARCHAR, retrieval VARCHAR)")
    if rows:
        con.executemany("INSERT INTO policy_citations VALUES (?,?,?,?,?)", rows)
    return len(rows)


def answer(con, question: str, k: int = 4, use_llm: bool = True) -> dict:
    """Retrieve clauses, then let the LLM answer from them. Without an LLM only the sources are returned."""
    src = search(question, k, con, hybrid=use_llm)
    out = dict(answer=None, citations=[], sources=src)
    if not (use_llm and llm.available()):
        return out
    ctx = "\n".join(f"[{s['clause']}] {s['title']}: {s['text']}" for s in src)
    a = llm.structured("Answer the question using only the policy clauses below. If they do not answer it, say so. "
                       f"In `clauses` list the ids you relied on.\n{ctx}\nQ: {question}", PolicyAnswer)
    if a:
        allowed = {s["clause"] for s in src}
        out |= dict(answer=a.answer, citations=[c for c in a.clauses if c in allowed])  # drop invented clause ids
    return out
