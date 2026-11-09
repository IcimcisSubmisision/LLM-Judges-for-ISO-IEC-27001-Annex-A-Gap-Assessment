"""
Quantitative evaluation of the clause-level semantic model against golden_eval.json.

Measures the two things the thesis actually claims:
  1. Clause-level retrieval recall — did the reranked evidence pool contain at
     least one of the human-labeled expected_clause_ids?
  2. Compliance-gap-detection accuracy — does the end-to-end LLM verdict match
     the human ground_truth_label?
"""
import collections
import json
import os
import re
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

from main import (
    RegulatoryDocumentParser, SemanticEncoder, CandidateRetriever, KeywordRetriever,
    CrossEncoderReranker, MultiQueryEnsembleRetriever,
    RoleKeywordForcer, LLMAnalyzer, build_clause_context, annotate_candidates,
)

# golden_eval.json was hand-labeled against an older paragraph-level segmentation
# (e.g. "DOC-01_Sec_4_Para_1"); the current pipeline segments at sentence level
# (e.g. "DOC-01_Sec_4_Sent_2"). Exact clause-ID matching is therefore meaningless —
# it compares two different ID schemes. Normalize both sides down to the section
# they belong to, which is the granularity the original annotations actually support.
_CLAUSE_SUFFIX_RE = re.compile(r'_(?:Sent|Para)_\d+[a-z]?$')


def section_key(clause_id: str) -> str:
    return _CLAUSE_SUFFIX_RE.sub('', clause_id)


ORG_CLIENT = "HACK23-ORG"
_EMB_CACHE_DIR = Path(".cache_embeddings")


def load_corpus(client_id: str, policy_dir: Path):
    """Returns the clause corpus a golden entry should be evaluated against.

    'HACK23-ORG' pools every Hack23 policy document into one ~8.5k-clause corpus,
    because Hack23 is ONE organisation with many policy files and an auditor asks
    "is the organisation compliant", not "does this particular file mention the
    control". Per-document scoring treated correct document scoping as a
    nonconformity: A.8.8 was Non-Compliant against the Access Control Policy
    while a 372-clause Vulnerability Management policy existed.

    The synthetic DOC-* corpus is deliberately NOT pooled - those are 30
    different companies, so pooling would let one company's manual 'evidence' a
    control for another.
    """
    if client_id == ORG_CLIENT:
        corpus = []
        for p in sorted(policy_dir.glob("HACK23-*.json")):
            with open(p, "r", encoding="utf-8") as f:
                corpus.extend(json.load(f))
        return corpus
    path = policy_dir / f"{client_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_cached(encoder, corpus, key):
    """Encoding the organisation corpus costs ~350s on CPU and is identical
    across ablation configurations, so cache it on disk keyed by content."""
    import hashlib
    import torch
    # Keyed by id AND text. An id-only key silently reused stale embeddings
    # whenever a corpus was re-segmented or cleaned without renumbering (OCR
    # fixes, decoration stripping) - the retrieval would then run on text that
    # no longer exists.
    digest = hashlib.sha1("".join(c["id"] + "\x1f" + c["text"] for c in corpus)
                          .encode("utf-8")).hexdigest()[:16]
    _EMB_CACHE_DIR.mkdir(exist_ok=True)
    path = _EMB_CACHE_DIR / f"{key}_{digest}.pt"
    if path.exists():
        print(f"[cache] reusing embeddings for {key} ({len(corpus)} clauses)")
        return torch.load(path, weights_only=False)
    emb = encoder.encode_clauses(corpus)
    torch.save(emb, path)
    return emb


BINARY_LABELS = os.environ.get("BINARY_LABELS", "1") not in ("0", "false", "False")


def to_scheme(label: str) -> str:
    """Normalises any label into the active scheme.

    golden_eval.json stores three-class ground truth. Under the binary scheme
    the mapping is mechanical, not a fresh judgement: 'Partially Compliant'
    means at least one required element is unevidenced, which is a
    nonconformity under ISO audit practice, so it maps to 'Not Compliant'
    alongside 'Non-Compliant'. No entry is re-assessed, so collapsing the
    scheme cannot introduce labelling bias.
    """
    if not BINARY_LABELS:
        return label
    return "Compliant" if label == "Compliant" else "Not Compliant"


def label_match(predicted: str, ground_truth: str) -> str:
    """Returns 'exact', 'soft', or 'miss'.

    'soft' existed because the pipeline used to emit a 4th label,
    'Insufficient Evidence', that the ground truth never uses - making every
    grounding-gated control an automatic miss. The gate now returns
    'Non-Compliant' with evidence_status='insufficient_evidence' instead, so
    those cases score as exact when correct. The soft branch is retained only
    so older result files remain comparable; it should now be 0.
    """
    if predicted == ground_truth:
        return "exact"
    if predicted == "Insufficient Evidence" and ground_truth == "Non-Compliant":
        return "soft"
    return "miss"


def run_retrieval():
    """Runs the clause-level semantic model (retrieval + rerank) once for every
    golden_eval.json entry. This is independent of which LLM does the final
    compliance judgment, so callers comparing multiple models should call this
    once and reuse the result rather than recomputing retrieval per model."""
    # GOLDEN_PATH selects the evaluation set (default: the original 69-entry set).
    # LIMIT_CLIENTS / LIMIT_PER_CLIENT exist so a full-scale run can be SMOKE
    # TESTED on a handful of entries first: the ISO sector set is 1,583 entries
    # and costs hours, and discovering a wiring bug after that is expensive.
    golden_path = Path(os.environ.get(
        "GOLDEN_PATH", "synthetic_iso27001_uupdp_dataset/golden_eval.json"))
    print(f"[config] golden set: {golden_path}")
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    parser = RegulatoryDocumentParser()
    with open("iso.txt", "r", encoding="utf-8") as f:
        iso_regulations = parser.parse_document(f.readlines(), doc_type="ISO27001")
    # UUPDP_LANG selects the regulatory source language:
    #   id  - legally binding Indonesian text only (cross-lingual vs English corpus)
    #   en  - English translation only (monolingual; used for the language ablation)
    #   bi  - BOTH, paired per clause (default). Retrieval benefits from the
    #         Indonesian legal terms and the English surface forms that actually
    #         appear in the policy corpus, and the assessor is no longer asked to
    #         bridge Indonesian obligations to English evidence unaided. Clause IDs
    #         are identical across UUPDP_ID.txt and UUPDP_EN.txt, so pairing is exact.
    uupdp_lang = os.environ.get("UUPDP_LANG", "bi").lower()
    if uupdp_lang not in ("id", "en", "bi"):
        raise SystemExit(f"UUPDP_LANG must be id, en or bi (got {uupdp_lang!r})")

    def load_uupdp(path):
        with open(path, "r", encoding="utf-8") as fh:
            return parser.parse_document(fh.readlines(), doc_type="UUPDP")

    if uupdp_lang == "bi":
        id_regs = {r["id"]: r["text"] for r in load_uupdp("UUPDP_ID.txt")}
        en_regs = {r["id"]: r["text"] for r in load_uupdp("UUPDP_EN.txt")}
        uupdp_regulations = [
            {"id": rid, "text": f"{txt}\n[EN] {en_regs[rid]}" if rid in en_regs else txt}
            for rid, txt in id_regs.items()
        ]
        uupdp_queries_file = "expanded_queries_uupdp_bi.json"
        print(f"[config] UU PDP source: bilingual (UUPDP_ID.txt + UUPDP_EN.txt)")
    else:
        uupdp_file = "UUPDP_ID.txt" if uupdp_lang == "id" else "UUPDP_EN.txt"
        uupdp_regulations = load_uupdp(uupdp_file)
        uupdp_queries_file = ("expanded_queries_uupdp.json" if uupdp_lang == "id"
                              else "expanded_queries_uupdp_EN.json.bak")
        print(f"[config] UU PDP source: {uupdp_file}")
    regs_by_id = {r["id"]: r for r in iso_regulations}
    regs_by_id.update({r["id"]: r for r in uupdp_regulations})

    with open("synthetic_iso27001_uupdp_dataset/expanded_queries-v1.1.json", "r", encoding="utf-8") as f:
        expanded_queries = json.load(f)
    uupdp_cache_path = Path("synthetic_iso27001_uupdp_dataset") / uupdp_queries_file
    if uupdp_cache_path.exists():
        with open(uupdp_cache_path, "r", encoding="utf-8") as f:
            expanded_queries.update(json.load(f))
    else:
        print(f"[warn] {uupdp_cache_path} missing - falling back to unexpanded Pasal text")

    # EXPANSION=off judges every ISO control as ONE obligation (retrieval and
    # assessment both use the control text), exactly as ablation.py's
    # 'ablation-no-expansion' configuration does. That configuration scored
    # 90.2% against 87.2% for the decomposed pipeline on dev (McNemar p = 1e-3):
    # the machine decomposition splits enumerations an auditor reads as one
    # requirement (A.5.4: 2/9 -> 8/9 correct), drops parts of some controls and
    # emits fragments (A.8.11), and the all-elements conjunction turns each such
    # defect into a nonconformity. The ablation finished after the v1 freeze,
    # which is why v1 still decomposes. UU PDP expansions are untouched - the
    # evidence is ISO-only.
    expansion = os.environ.get("EXPANSION", "on").lower()
    if expansion not in ("on", "off"):
        raise SystemExit(f"EXPANSION must be on or off (got {expansion!r})")
    if expansion == "off":
        expanded_queries = {k: v for k, v in expanded_queries.items() if not k.startswith("ISO_A_")}
        print("[config] EXPANSION=off - ISO controls retrieved and judged as one obligation each")

    by_client = defaultdict(list)
    for entry in golden:
        # golden_eval.json carries "_section" documentation markers between
        # provenance groups (synthetic vs real-world); they are not test cases.
        if "control_id" not in entry or "client_id" not in entry:
            continue
        by_client[entry["client_id"]].append(entry)

    encoder = SemanticEncoder()
    bi_retriever = CandidateRetriever(k_candidates=8)
    multi_retriever = MultiQueryEnsembleRetriever(top_k_per_subquery=8)
    reranker = CrossEncoderReranker(top_n=6)
    forcer = RoleKeywordForcer()
    policy_dir = Path("synthetic_iso27001_uupdp_dataset/segmented_policies")

    prepared = []  # list of dicts: reg_clause, reranked, sub_queries, entry, retrieval_hit
    retrieval_hits, retrieval_total = 0, 0

    # SPLIT restricts evaluation to the development or the held-out test split.
    # Tuning decisions must only ever be informed by 'dev'; 'test' is run once,
    # after they are frozen, or the reported accuracy is optimistically biased.
    split = os.environ.get("SPLIT", "").strip().lower()
    if split:
        for k in list(by_client):
            by_client[k] = [e for e in by_client[k] if e.get("_split") == split]
        by_client = {k: v for k, v in by_client.items() if v}
        print(f"[config] SPLIT={split} -> {len(by_client)} clients, "
              f"{sum(len(v) for v in by_client.values())} entries")

    only = {c.strip() for c in os.environ.get("LIMIT_CLIENTS", "").split(",") if c.strip()}
    per = int(os.environ.get("LIMIT_PER_CLIENT", "0") or 0)
    if only:
        by_client = {k: v for k, v in by_client.items() if k in only}
        print(f"[config] LIMIT_CLIENTS -> {sorted(by_client)}")
    if per:
        by_client = {k: v[:per] for k, v in by_client.items()}
        print(f"[config] LIMIT_PER_CLIENT -> {per} entries per client")

    for client_id, entries in by_client.items():
        company_policies = load_corpus(client_id, policy_dir)
        if company_policies is None:
            print(f"[skip] no corpus for {client_id}")
            continue
        print(f"[corpus] {client_id}: {len(company_policies)} clauses")
        # Section and list-introducer context per clause. Always attached (it is
        # cheap); only read by the assessor under CONTEXT_MODE=lead.
        clause_context = build_clause_context(company_policies)
        company_embeddings = encode_cached(encoder, company_policies, client_id)
        kw_retriever = KeywordRetriever(k_candidates=multi_retriever.top_k_per_subquery)
        kw_retriever.fit(company_policies)

        valid_entries = [e for e in entries if e["control_id"] in regs_by_id]
        for cid in [e["control_id"] for e in entries if e["control_id"] not in regs_by_id]:
            print(f"[skip] {cid} not found in iso.txt/UUPDP_ID.txt parse output")

        regulations = [regs_by_id[e["control_id"]] for e in valid_entries]
        merged_results = multi_retriever.retrieve(
            expanded_queries=expanded_queries, regulations=regulations,
            bi_retriever=bi_retriever, company_embeddings=company_embeddings,
            encoder=encoder, kw_retriever=kw_retriever, corpus_clauses=company_policies,
        )

        for reg_clause, candidates, entry in zip(regulations, merged_results, valid_entries):
            sub_queries = expanded_queries.get(reg_clause["id"], [reg_clause["text"]])
            if forcer:
                candidates = forcer.force_include(reg_clause["id"], candidates, company_policies)
            guaranteed = {c["internal_clause_id"] for c in candidates if "forced_by_keyword" in c}
            reranked = reranker.rerank(query_text=reg_clause["text"], candidates=candidates,
                                       guaranteed_ids=guaranteed, sub_queries=sub_queries)
            annotate_candidates(reranked, clause_context)

            retrieved_ids = {c["internal_clause_id"] for c in reranked}
            expected_sections = {section_key(cid) for cid in entry["expected_clause_ids"]}
            retrieved_sections = {section_key(cid) for cid in retrieved_ids}

            retrieval_hit = None
            if expected_sections:
                retrieval_total += 1
                retrieval_hit = bool(expected_sections & retrieved_sections)
                if retrieval_hit:
                    retrieval_hits += 1

            prepared.append({
                "client": client_id,
                "reg_clause": reg_clause,
                "reranked": reranked,
                "sub_queries": sub_queries,
                "entry": entry,
                "retrieval_hit": retrieval_hit,
                "retrieved_ids": sorted(retrieved_ids),
            })

    return prepared, retrieval_hits, retrieval_total


def _analyse_one(analyzer, p):
    return p, analyzer.analyze_gap(p["reg_clause"], p["reranked"],
                                   sub_queries=p["sub_queries"])


def run_analysis(analyzer, prepared):
    """Runs the given LLMAnalyzer over pre-computed retrieval results.

    EVAL_WORKERS>1 evaluates controls concurrently. Each control is an
    independent judgement over its own retrieved evidence - nothing is shared
    between them - so concurrency changes throughput, not semantics. It is off
    by default because it must be validated against the sequential path before
    being trusted: a silent divergence would be indistinguishable from a real
    accuracy change. Results are always reassembled in input order.
    """
    workers = max(1, int(os.environ.get("EVAL_WORKERS", "1") or 1))
    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        print(f"[config] EVAL_WORKERS={workers} (concurrent judgement)")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            done = list(ex.map(lambda p: _analyse_one(analyzer, p), prepared))
    else:
        done = [_analyse_one(analyzer, p) for p in prepared]

    rows = []
    label_counts = {"exact": 0, "soft": 0, "miss": 0}
    for p, report in done:
        entry = p["entry"]
        truth = to_scheme(entry["ground_truth_label"])
        match = label_match(report.compliance_level, truth)
        label_counts[match] += 1
        rows.append({
            "client": p["client"],
            "control_id": entry["control_id"],
            "retrieved_ids": p["retrieved_ids"],
            "retrieval_hit": p["retrieval_hit"],
            "predicted_label": report.compliance_level,
            "evidence_status": getattr(report, "evidence_status", "assessed"),
            "low_retrieval_confidence": bool(getattr(report, "low_retrieval_confidence", False)),
            "subject_check": getattr(report, "subject_check", None),
            "ground_truth_label": truth,
            "label_match": match,
            # Kept so a run can be turned into an auditor-facing SoA or a paper
            # table without re-running the LLM: the printed summary alone throws
            # away the reasoning and the coverage that produced the label.
            "coverage_ratio": round(getattr(report, "coverage_ratio", 0.0), 3),
            "confidence": round(getattr(report, "confidence_score", 0.0), 3),
            "matched_obligations": list(getattr(report, "matched_obligations", [])),
            "partial_obligations": list(getattr(report, "partial_obligations", [])),
            "missing_obligations": list(getattr(report, "missing_obligations", [])),
            "reasoning": getattr(report, "reasoning", ""),
            "suggestion": getattr(report, "suggestion", ""),
            # Full decision trail, so any verdict can be audited without re-running.
            "sub_assessments": list(getattr(report, "sub_assessments", [])),
            "evidence_considered": list(getattr(report, "evidence_considered", [])),
            "soa_status": entry.get("_soa_status", ""),
            "sector": entry.get("_sector", ""),
            "tier": entry.get("_tier", ""),
        })
    return rows, label_counts


def print_report(rows, label_counts, retrieval_hits, retrieval_total):
    print("\n" + "=" * 100)
    print(f"{'Client':<8} | {'Control':<16} | {'Retrieval':<10} | {'Predicted':<20} | {'Ground Truth':<18} | {'Match':<6}")
    print("-" * 100)
    for r in rows:
        retr = {True: "HIT", False: "MISS", None: "n/a"}[r["retrieval_hit"]]
        print(f"{r['client']:<8} | {r['control_id']:<16} | {retr:<10} | "
              f"{r['predicted_label']:<20} | {r['ground_truth_label']:<18} | {r['label_match']:<6}")
    print("=" * 100)

    label_total = len(rows)
    print(f"\nSection-level retrieval recall: {retrieval_hits}/{retrieval_total} "
          f"({retrieval_hits / retrieval_total:.0%})" if retrieval_total else
          "\nSection-level retrieval recall: n/a (no golden entries with expected_clause_ids)")
    print(f"  NOTE: golden_eval.json's expected_clause_ids use an older paragraph-level ID scheme")
    print(f"  (e.g. '..._Para_1'); segmented_policies/ now uses sentence-level IDs ('..._Sent_k').")
    print(f"  Recall is computed at section granularity (e.g. 'DOC-01_Sec_4') since that is the")
    print(f"  precision the original annotations support. Re-annotate golden_eval.json against the")
    print(f"  current sentence-level segmentation to measure true clause-level (not section-level) recall.")
    print(f"\nCompliance-label accuracy (exact):        {label_counts['exact']}/{label_total} "
          f"({label_counts['exact'] / label_total:.0%})")
    print(f"Compliance-label accuracy (exact + soft):  {label_counts['exact'] + label_counts['soft']}/{label_total} "
          f"({(label_counts['exact'] + label_counts['soft']) / label_total:.0%})")
    print(f"  soft matches = predicted 'Insufficient Evidence' where ground truth is 'Non-Compliant'")

    # --- breakdown by provenance and framework -------------------------------
    # Synthetic and real-world results must be reported separately: they are
    # different populations, and collapsing them hides which one the model
    # actually performs well on.
    def bucket(rs, name):
        if not rs:
            return
        e = sum(r["label_match"] == "exact" for r in rs)
        s = sum(r["label_match"] == "soft" for r in rs)
        n = len(rs)
        print(f"  {name:34s} exact {e}/{n} ({e/n:3.0%})   +soft {e+s}/{n} ({(e+s)/n:3.0%})")

    print("\nBreakdown by corpus provenance:")
    bucket([r for r in rows if r["client"].startswith("DOC-")], "synthetic (DOC-*)")
    bucket([r for r in rows if r["client"].startswith("HACK23")], "real-world (HACK23-*)")

    print("\nBreakdown by regulatory framework:")
    bucket([r for r in rows if r["control_id"].startswith("ISO_A_")], "ISO/IEC 27001 Annex A")
    bucket([r for r in rows if r["control_id"].startswith("Pasal_")], "UU PDP")

    gated = [r for r in rows if r.get("evidence_status") == "insufficient_evidence"]
    if gated:
        gated_ok = sum(r["label_match"] == "exact" for r in gated)
        print(f"\nGrounding-gated controls (no evidence cleared the threshold): {len(gated)}")
        print(f"  of these, ground truth agreed it is Non-Compliant: {gated_ok}/{len(gated)}")
        print(f"  NOTE: these are reported as Non-Compliant but flagged "
              f"evidence_status=insufficient_evidence - they may be retrieval")
        print(f"  failures rather than genuine gaps and warrant manual triage.")

    low = [r for r in rows if r.get("low_retrieval_confidence")
           and r.get("evidence_status") == "assessed"]
    if low:
        low_ok = sum(r["label_match"] == "exact" for r in low)
        print(f"\nLow-retrieval-confidence controls ASSESSED anyway (GATE_MODE=assess): {len(low)}")
        print(f"  correct: {low_ok}/{len(low)}  |  predicted Compliant: "
              f"{sum(r['predicted_label'] == 'Compliant' for r in low)}")

    # --- balanced accuracy -------------------------------------------------
    # A mature ISMS genuinely covers most controls, so the label set is ~2:1
    # Compliant. Raw accuracy therefore rewards a model that simply leans
    # Compliant. Balanced accuracy (mean of per-class recall) is immune to that
    # and is the honest headline for an imbalanced set. Rebalancing the labels
    # instead would mean gerrymandering the data to flatter the metric.
    print("\nPer-class recall (balanced accuracy):")
    recalls = []
    for lab in sorted({r["ground_truth_label"] for r in rows}):
        sub = [r for r in rows if r["ground_truth_label"] == lab]
        hit = sum(r["label_match"] == "exact" for r in sub)
        recalls.append(hit / len(sub))
        print(f"  {lab:<24} {hit}/{len(sub)} ({hit/len(sub):.0%})")
    if recalls:
        print(f"  {'BALANCED ACCURACY':<24} {sum(recalls)/len(recalls):.0%}")
        maj = max(collections.Counter(r["ground_truth_label"] for r in rows).values())
        print(f"  {'(majority baseline)':<24} {maj}/{len(rows)} ({maj/len(rows):.0%}) "
              f"-> balanced 50%")

    print("\nError direction (systematic bias check):")
    rank = {"Non-Compliant": 0, "Not Compliant": 0, "Insufficient Evidence": 0,
            "Partially Compliant": 1, "Compliant": 2}
    strict = sum(1 for r in rows if r["label_match"] == "miss"
                 and rank[r["predicted_label"]] < rank[r["ground_truth_label"]])
    lenient = sum(1 for r in rows if r["label_match"] == "miss"
                  and rank[r["predicted_label"]] > rank[r["ground_truth_label"]])
    print(f"  too strict (under-claims compliance): {strict}")
    print(f"  too lenient (over-claims compliance): {lenient}")
    print(f"  hard misses = {label_counts['miss']}/{label_total}")

    # Persist the full per-control result. Without this every downstream
    # artefact - the SoA comparison, the paper's tables, the error analysis -
    # requires paying for the whole run again.
    out = os.environ.get("RESULTS_JSON")
    if out:
        json.dump({"golden": os.environ.get(
                       "GOLDEN_PATH",
                       "synthetic_iso27001_uupdp_dataset/golden_eval.json"),
                   "reranker": os.environ.get(
                       "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
                   # Recorded so a results file can never be mistaken for one
                   # produced by a different judge.
                   "judge_model": os.environ.get("JUDGE_MODEL", "gemma4:31b-cloud"),
                   "judge_base_url": os.environ.get("JUDGE_BASE_URL",
                                                    "http://localhost:11434/v1"),
                   # v1 = every option at its default (the frozen configuration).
                   "pipeline": {**{env: os.environ.get(env, default)
                                   for env, _, default in LLMAnalyzer._OPTIONS.values()},
                                "EXPANSION": os.environ.get("EXPANSION", "on").lower()},
                   "label_scheme": "binary" if BINARY_LABELS else "three-class",
                   "rows": rows},
                  open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"\nper-control results written to {out}")


def main():
    prepared, retrieval_hits, retrieval_total = run_retrieval()
    # JUDGE_MODEL / JUDGE_BASE_URL select the assessor. The default is the frozen
    # configuration (FROZEN_CONFIG.json); anything else is a DIFFERENT
    # configuration and must be reported as such, not as the frozen result.
    # Retrieval and the sub-obligation decomposition are cached and unaffected,
    # so a model swap changes only the judgement stage.
    model = os.environ.get("JUDGE_MODEL", "gemma4:31b-cloud")
    base_url = os.environ.get("JUDGE_BASE_URL", "http://localhost:11434/v1")
    analyzer = LLMAnalyzer(model_name=model, base_url=base_url,
                           binary_labels=BINARY_LABELS)
    print(f"[config] judge: {model} @ {base_url}")
    print("[config] pipeline: " + ", ".join(
        f"{env}={getattr(analyzer, name)}" for name, (env, _, _) in LLMAnalyzer._OPTIONS.items())
        + f", EXPANSION={os.environ.get('EXPANSION', 'on').lower()}")
    print(f"[config] label scheme: {'binary' if BINARY_LABELS else 'three-class'}")
    rows, label_counts = run_analysis(analyzer, prepared)
    print_report(rows, label_counts, retrieval_hits, retrieval_total)


if __name__ == "__main__":
    main()
