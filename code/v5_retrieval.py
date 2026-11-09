"""v5 retrieval for table/list evidence (the pre-registration log, "Retrieval fix for table/list evidence - v5 retrieval").

Runtime patches on top of v3r retrieval; frozen files untouched. V5 env var selects the variant:
  R0  v3r as frozen (baseline)
  R1  deeper candidate pool: top-k per sub-query 8 -> 20
  R2  section expansion ("small-to-big"): for every sub-query, the 2 sections (doc + H2 heading) whose clause-
      embedding centroid is closest to the sub-query add all their clauses (max 25 each) to the candidate pool
      before the cross-encoder
  R3  R2 + per_sub_top 2 -> 3
  R4  R1 + HyDE evidence-vocabulary queries (hyde_queries_v5.json) added to the retrieval / cross-encoder sub-queries
  R5  R4 + R2 section expansion + per_sub_top 3

"""
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

import pipeline_v3
from main import MultiQueryEnsembleRetriever

VARIANT = os.environ.get("V5", "R0")
_V3_RETRIEVE = pipeline_v3._v3_retrieve
_ORIG_INIT = MultiQueryEnsembleRetriever.__init__
_ORIG_EXTRA = {k: list(v) for k, v in pipeline_v3.EXTRA_QUERIES.items()}
HYDE = json.load(open("synthetic_iso27001_uupdp_dataset/hyde_queries_v5.json", encoding="utf-8"))["queries"]
sec_of = lambda cid: cid.rsplit("_Sent_", 1)[0]


def _init(self, top_k_per_subquery=5, **kw):
    if VARIANT in ("R1", "R4", "R5") and top_k_per_subquery == 8:
        top_k_per_subquery = 20
    _ORIG_INIT(self, top_k_per_subquery=top_k_per_subquery, **kw)


def _section_expand(self, expanded_queries, regulations, bi_retriever, company_embeddings, encoder,
                    kw_retriever, corpus_clauses):
    merged = _V3_RETRIEVE(self, expanded_queries=expanded_queries, regulations=regulations,
                          bi_retriever=bi_retriever, company_embeddings=company_embeddings, encoder=encoder,
                          kw_retriever=kw_retriever, corpus_clauses=corpus_clauses)
    emb = company_embeddings if torch.is_tensor(company_embeddings) else torch.tensor(company_embeddings)
    emb = torch.nn.functional.normalize(emb.float(), dim=-1)
    groups = defaultdict(list)
    for i, c in enumerate(corpus_clauses):
        groups[sec_of(c["id"])].append(i)
    keys = list(groups)
    cent = torch.nn.functional.normalize(torch.stack([emb[groups[k]].mean(0) for k in keys]), dim=-1)
    q = dict(expanded_queries)
    for r in regulations:                                    # same query set as the v3 retrieval
        if r["id"] in pipeline_v3.EXTRA_QUERIES and r["id"] not in q:
            q[r["id"]] = [r["text"]] + pipeline_v3.EXTRA_QUERIES[r["id"]]
    out = []
    for reg, cands in zip(regulations, merged):
        subs = q.get(reg["id"], [reg["text"]])
        qe = encoder.encode_clauses([{"text": s} for s in subs])
        qe = torch.nn.functional.normalize((qe if torch.is_tensor(qe) else torch.tensor(qe)).float(), dim=-1)
        sims = qe.to(cent.device) @ cent.T
        chosen = {keys[j] for row in sims for j in torch.topk(row, k=min(2, len(keys))).indices.tolist()}
        have = {c["internal_clause_id"] for c in cands}
        extra = []
        for k in sorted(chosen):
            for i in groups[k][:25]:
                cl = corpus_clauses[i]
                if cl["id"] not in have:
                    extra.append({"internal_clause_id": cl["id"], "text": cl["text"], "similarity_score": 0.0,
                                  "rrf_score": 0.0, "section_expanded": True})
                    have.add(cl["id"])
        out.append(list(cands) + extra)
    return out


def apply(variant=None):
    """Installs the variant's patches (call after pipeline_v3.retrieval(True))."""
    global VARIANT
    VARIANT = variant or VARIANT
    MultiQueryEnsembleRetriever.__init__ = _init
    pipeline_v3.PER_SUB_TOP = 3 if VARIANT in ("R3", "R5") else 2
    if VARIANT in ("R2", "R3", "R5"):
        MultiQueryEnsembleRetriever.retrieve = _section_expand
    hyde = VARIANT in ("R4", "R5")
    for cid, base in _ORIG_EXTRA.items():          # same list objects back both EXTRA_QUERIES and _BY_TEXT
        pipeline_v3.EXTRA_QUERIES[cid][:] = base + (HYDE.get(cid, []) if hyde else [])

