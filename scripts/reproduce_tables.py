"""Reproduce the paper's result tables from the released labels and per-control verdicts.

    python scripts/reproduce_tables.py

Standard library only. Reads labels/*.csv and results/verdicts.csv and prints:
  - Table III (constructed corpus, both judges; tuning vs. held-out companies)
  - Tables IV-V (government documents and public corpora)
  - Table VI  (safeguard ablation: assessor only, gate only, verifier only, both)
  - the own-query run of gpt-oss:120b on the four external sets
  - inter-rater agreement of the three practitioners on the government documents

Verdict rule (frozen): a control is Compliant only if evidence was assessed, every assessor vote
is YES (coverage = (YES + 0.5 * PARTIAL) / votes must reach 1.0), the evidence gate did not fire,
and the contrastive verifier did not reject the target control.
"""
import csv
import random
from collections import Counter, defaultdict
from math import comb, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEMMA, OSS = "gemma4:31b", "gpt-oss:120b"
B, SEED = 2000, 20261107


def read(path):
    with open(ROOT / path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- data
TRUTH = {}            # set -> {control: status}
SPLIT = {}            # company -> tuning | held_out
for r in read("labels/constructed_panel.csv"):
    TRUTH.setdefault(r["company"], {})[r["control"]] = r["panel_status"]
    SPLIT[r["company"]] = r["split"]
for r in read("labels/public_panel.csv"):
    if r["applicable"] == "1":
        TRUTH.setdefault(r["set"], {})[r["control"]] = r["panel_status"]
GOV = defaultdict(list)
for r in read("labels/government_raters.csv"):
    GOV[r["document"]].append(r)
    maj = Counter(r[k] for k in ("rater_A", "rater_B", "rater_C")).most_common(1)[0]
    status = maj[0] if maj[1] >= 2 else "Minor Gap"
    if (r["majority_binary"] == "Compliant") != (status == "No Gap"):
        status = "No Gap" if r["majority_binary"] == "Compliant" else "Minor Gap"
    TRUTH.setdefault(r["document"], {})[r["control"]] = status

RUNS = defaultdict(dict)   # (set, judge, queries) -> {control: row}; reported runs only
REPEATS = defaultdict(dict)  # (set, judge, run) -> {control: row}; government documents, same cached evidence
for r in read("results/verdicts.csv"):
    if r["run"] == "reported":
        RUNS[(r["set"], r["judge"], r["evidence_queries_by"])][r["control"]] = r
    if r["evidence_queries_by"] == GEMMA and r["set"] in ("Bappenas", "DJPb"):
        REPEATS[(r["set"], r["judge"], r["run"])][r["control"]] = r

COMPANIES = sorted(SPLIT)
EXTERNAL = ["SecureNusantara", "Hack23", "Bappenas", "DJPb"]


def verdict(r, gate=True, verify=True):
    if r["evidence_status"] != "assessed":
        return False
    if gate and r["gate_fired"] == "1":
        return False
    if verify and r["verifier_addresses_subject"] == "0":
        return False
    votes = [v for v in r["assessor_votes"].split(";") if v]
    if not votes:
        return r["fallback_label"] == "Compliant"
    return (votes.count("YES") + 0.5 * votes.count("PARTIAL")) / len(votes) >= 1.0


def pairs(sets, judge, queries=GEMMA, gate=True, verify=True):
    out = []
    for s in sets:
        R, T = RUNS[(s, judge, queries)], TRUTH[s]
        for c in sorted(T):
            if c in R:
                out.append((s, c, verdict(R[c], gate, verify), T[c]))
    return out


def kappa(pred, truth):
    n = len(pred)
    acc = sum(p == t for p, t in zip(pred, truth)) / n
    pp, pt = sum(pred) / n, sum(truth) / n
    pe = pp * pt + (1 - pp) * (1 - pt)
    return acc, (acc - pe) / (1 - pe) if pe < 1 else float("nan")


def summary(P):
    pred = [p for _, _, p, _ in P]
    truth = [t == "No Gap" for *_, t in P]
    acc, k = kappa(pred, truth)
    fc = sum(p and not t for p, t in zip(pred, truth))
    fnc = sum(t and not p for p, t in zip(pred, truth))
    fc_major = sum(p and st == "Major Gap" for _, _, p, st in P)
    fc_minor = sum(p and st == "Minor Gap" for _, _, p, st in P)
    nc_true = sum(not t for t in truth); nc_pred = sum(not p for p in pred)
    tp = nc_true - fc
    return dict(n=len(P), acc=acc, kappa=k, fc=fc, fnc=fnc, fc_major=fc_major, fc_minor=fc_minor,
                prec=tp / nc_pred if nc_pred else float("nan"), rec=tp / nc_true if nc_true else float("nan"),
                majority=max(sum(truth), len(truth) - sum(truth)) / len(truth),
                n_major=sum(st == "Major Gap" for *_, st in P), n_minor=sum(st == "Minor Gap" for *_, st in P))


def mcnemar(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


def wilson(x, n, z=1.959964):
    p = x / n; d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return m - h, m + h


def cluster_ci(sets, judge):
    rng = random.Random(SEED); ks = []
    for _ in range(B):
        P = []
        for s in (rng.choice(sets) for _ in sets):
            P += pairs([s], judge)
        ks.append(summary(P)["kappa"])
    ks.sort(); return ks[int(.025 * B)], ks[int(.975 * B) - 1]


def cluster_ci_diff(sets):
    rng = random.Random(SEED); ds = []
    for _ in range(B):
        samp = [rng.choice(sets) for _ in sets]
        ks = []
        for j in (GEMMA, OSS):
            P = []
            for s in samp:
                P += pairs([s], j)
            ks.append(summary(P)["kappa"])
        ds.append(ks[0] - ks[1])
    ds.sort(); return ds[int(.025 * B)], ds[int(.975 * B) - 1]


def control_ci(P):
    rng = random.Random(SEED); ks = []
    for _ in range(B):
        S = [rng.choice(P) for _ in P]
        ks.append(summary(S)["kappa"])
    ks.sort(); return ks[int(.025 * B)], ks[int(.975 * B) - 1]


def judges_mcnemar(sets, qa=GEMMA, qb=GEMMA, ja=GEMMA, jb=OSS):
    A = {(s, c): p == (t == "No Gap") for s, c, p, t in pairs(sets, ja, qa)}
    Bm = {(s, c): p == (t == "No Gap") for s, c, p, t in pairs(sets, jb, qb)}
    ks = A.keys() & Bm.keys()
    b = sum(A[k] and not Bm[k] for k in ks); c = sum(Bm[k] and not A[k] for k in ks)
    return b, c, mcnemar(b, c)


def vs_baseline(P):
    truth = [t == "No Gap" for *_, t in P]
    maj = sum(truth) * 2 >= len(truth)
    right = [p == t for _, _, p, t in [(a, b, p, tt == "No Gap") for a, b, p, tt in P]]
    base = [t == maj for t in truth]
    b = sum(r and not q for r, q in zip(right, base)); c = sum(q and not r for r, q in zip(right, base))
    return mcnemar(b, c)


def line(label, s):
    return (f"  {label:<28} n={s['n']:<5} acc {s['acc']:.3f}  kappa {s['kappa']:.3f}  FC {s['fc']:>3} "
            f"(Major {s['fc_major']}, Minor {s['fc_minor']})  FNC {s['fnc']:>3}")


# ---------------------------------------------------------------- Table III
print("TABLE III - constructed corpus (18 companies)")
for j in (GEMMA, OSS):
    s = summary(pairs(COMPANIES, j)); lo, hi = cluster_ci(COMPANIES, j)
    print(line(j, s) + f"  kappa CI [{lo:.2f}, {hi:.2f}]")
    print(f"      NC precision/recall {s['prec']:.3f}/{s['rec']:.3f}; Major/Minor flagged "
          f"{1 - s['fc_major'] / s['n_major']:.1%}/{1 - s['fc_minor'] / s['n_minor']:.1%}; majority-class acc {s['majority']:.3f}")
b, c, p = judges_mcnemar(COMPANIES)
print(f"  uniquely correct: gemma {b}, gpt-oss {c}, exact McNemar p = {p:.2g}")
per_co = [judges_mcnemar([co])[:2] for co in COMPANIES]
wins = sum(b > c for b, c in per_co); ties = sum(b == c for b, c in per_co)
print(f"  companies where gemma had more uniquely correct verdicts: {wins} of {len(COMPANIES) - ties} "
      f"(sign test p = {mcnemar(wins, len(COMPANIES) - ties - wins):.2g})")
for split in ("tuning", "held_out"):
    sets = [co for co in COMPANIES if SPLIT[co] == split]
    for j in (GEMMA, OSS):
        s = summary(pairs(sets, j)); lo, hi = cluster_ci(sets, j)
        print(line(f"{split} ({len(sets)}) {j}", s) + f"  kappa CI [{lo:.2f}, {hi:.2f}]")
    lo, hi = cluster_ci_diff(sets)
    d = summary(pairs(sets, GEMMA))["kappa"] - summary(pairs(sets, OSS))["kappa"]
    print(f"    kappa difference {d:.3f}, cluster-bootstrap 95% CI [{lo:.2f}, {hi:.2f}]")

# ---------------------------------------------------------------- Tables IV-V
print("\nTABLES IV-V - government documents and public corpora (final configuration)")
for e in EXTERNAL:
    for j in (GEMMA, OSS):
        P = pairs([e], j); s = summary(P)
        w = wilson(round(s["acc"] * s["n"]), s["n"]); lo, hi = control_ci(P)
        off = summary(pairs([e], j, gate=False, verify=False))["kappa"]
        print(line(f"{e} {j}", s) + f"  Wilson [{w[0]:.2f}, {w[1]:.2f}]  kappa CI [{lo:.2f}, {hi:.2f}]  "
              f"safeguards off {off:.3f}  p vs. baseline {vs_baseline(P):.3f}  (majority {s['majority']:.3f})")
    b, c, p = judges_mcnemar([e])
    print(f"    uniquely correct gemma {b} vs gpt-oss {c}, p = {p:.2f}")
b, c, p = judges_mcnemar(EXTERNAL)
print(f"  pooled over the four sets: uniquely correct {b} vs {c}, p = {p:.2f}")

# ---------------------------------------------------------------- Table VI
print("\nTABLE VI - safeguard ablation (constructed corpus; external kappa SN/Hack23/Bappenas/DJPb)")
for j in (GEMMA, OSS):
    for name, g, v in (("assessor only", False, False), ("gate only", True, False),
                       ("verifier only", False, True), ("both (final)", True, True)):
        s = summary(pairs(COMPANIES, j, gate=g, verify=v))
        ext = "/".join(f"{summary(pairs([e], j, gate=g, verify=v))['kappa']:.3f}" for e in EXTERNAL)
        print(f"  {j:<13} {name:<14} kappa {s['kappa']:.3f}  missed Major/Minor {s['fc_major']}/{s['fc_minor']}  "
              f"false gaps {s['fnc']}  external {ext}")

# ---------------------------------------------------------------- gate threshold sweep
# Post hoc, no new model calls: the gate blocks a Compliant verdict when the top reranker score among
# the evidence passed to the judge is below t. Development (tuning) companies only.
def verdict_at(r, t):
    if r["evidence_status"] != "assessed":
        return False
    if t is not None and (r["top_reranker_score"] == "" or float(r["top_reranker_score"]) < t):
        return False
    return verdict(r, gate=False, verify=True)

print("\nGATE THRESHOLD SWEEP - final pipeline (gate + verifier), nine development companies")
TUNING = [co for co in COMPANIES if SPLIT[co] == "tuning"]
for j in (GEMMA, OSS):
    for t in (None, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        P = [(s, c, verdict_at(RUNS[(s, j, GEMMA)][c], t), TRUTH[s][c])
             for s in TUNING for c in sorted(TRUTH[s]) if c in RUNS[(s, j, GEMMA)]]
        s_ = summary(P)
        print(f"  {j:<13} threshold {'none' if t is None else f'{t:.2f}':<5} kappa {s_['kappa']:.3f}  "
              f"missed Major/Minor {s_['fc_major']}/{s_['fc_minor']}  false gaps {s_['fnc']}")

# ---------------------------------------------------------------- own queries
print("\nOWN-QUERY RUN - gpt-oss:120b with evidence queries written by gpt-oss:120b")
for e in EXTERNAL:
    a = summary(pairs([e], OSS, GEMMA))["kappa"]; o = summary(pairs([e], OSS, OSS))["kappa"]
    b, c, p = judges_mcnemar([e], GEMMA, OSS, OSS, OSS)
    print(f"  {e:<16} kappa {a:.3f} -> {o:.3f}  (fixed {c}, broken {b}, p = {p:.2f})")
right_g = sum(p == (t == "No Gap") for *_, p, t in pairs(EXTERNAL, OSS, GEMMA))
right_o = sum(p == (t == "No Gap") for *_, p, t in pairs(EXTERNAL, OSS, OSS))
b, c, p = judges_mcnemar(EXTERNAL, GEMMA, OSS, OSS, OSS)
print(f"  pooled: {right_o} vs {right_g} correct of {len(pairs(EXTERNAL, OSS, OSS))} (fixed {c}, broken {b}, p = {p:.2f})")

# ---------------------------------------------------------------- inter-rater
print("\nINTER-RATER AGREEMENT - government documents, three practitioners labeling independently")


def fleiss(table):
    n, k = len(table), len(table[0]); cats = sorted({x for r in table for x in r})
    P = [(sum(r.count(c) ** 2 for c in cats) - k) / (k * (k - 1)) for r in table]
    p = [sum(r.count(c) for r in table) / (n * k) for c in cats]
    return (sum(P) / n - sum(q * q for q in p)) / (1 - sum(q * q for q in p))


for doc, rows in GOV.items():
    t3 = [[r["rater_A"], r["rater_B"], r["rater_C"]] for r in rows]
    t2 = [[x == "No Gap" for x in r] for r in t3]
    spec = {}
    for cat in ("No Gap", "Minor Gap", "Major Gap"):
        den = sum(2 * r.count(cat) for r in t3)
        spec[cat] = sum(r.count(cat) * (r.count(cat) - 1) for r in t3) / den if den else float("nan")
    print(f"  {doc:<9} Fleiss kappa binary {fleiss(t2):.3f}, three-level {fleiss(t3):.3f}; category-specific agreement "
          + ", ".join(f"{k} {v:.2f}" for k, v in spec.items()))

# ---------------------------------------------------------------- repeatability
print("\nREPEATABILITY - government documents, reported run + two repeats on the same cached evidence")
for j in (GEMMA, OSS):
    n_all = changed_all = pairs_all = 0
    for doc in ("Bappenas", "DJPb"):
        runs = [REPEATS[(doc, j, r)] for r in ("reported", "repeat_1", "repeat_2")]
        ks = sorted(set.intersection(*(set(r) for r in runs)) & set(TRUTH[doc]))
        V = [{k: verdict(r[k]) for k in ks} for r in runs]
        changed = sum(len({v[k] for v in V}) > 1 for k in ks)
        both = sum(V[1][k] != V[0][k] and V[2][k] != V[0][k] for k in ks)
        pw = [sum(V[a][k] != V[b][k] for k in ks) for a, b in ((0, 1), (0, 2), (1, 2))]
        truth = [TRUTH[doc][k] == "No Gap" for k in ks]
        kap = [kappa([v[k] for k in ks], truth)[1] for v in V]
        k_major = kappa([sum(v[k] for v in V) >= 2 for k in ks], truth)[1]
        print(f"  {doc:<9} {j:<13} changed in >=1 run {changed}/{len(ks)}; changed in both repeats {both}; "
              f"changes reported-vs-repeat1, reported-vs-repeat2, repeat1-vs-repeat2 {pw}; kappa per run "
              + "/".join(f"{x:.3f}" for x in kap) + f"; majority-of-three kappa {k_major:.3f}")
        n_all += len(ks); changed_all += changed; pairs_all += sum(pw)
    lo, hi = wilson(changed_all, n_all)
    print(f"  pooled    {j:<13} {changed_all}/{n_all} controls changed ({changed_all / n_all:.1%}, Wilson [{lo:.1%}, {hi:.1%}]); "
          f"{pairs_all / (3 * n_all):.1%} of run pairs disagree")

# ---------------------------------------------------------------- sources of error (Section IV-E)
print("\nSOURCES OF ERROR - first failing stage of each false Not Compliant verdict (final configuration)")


def stage(r):
    # retrieval fails first when the section the constructed policy was designed to evidence the control with was
    # not among the retrieved sentences (designed_evidence_retrieved = 0; recorded for the constructed corpus only)
    if r["evidence_status"] != "assessed" or r.get("designed_evidence_retrieved") == "0":
        return "retrieval"
    if r["gate_fired"] == "1":
        return "gate"
    if r["verifier_addresses_subject"] == "0":
        return "verifier"
    return "assessor"


for label, sets in (("constructed corpus", COMPANIES), ("SecureNusantara", ["SecureNusantara"]), ("Hack23", ["Hack23"])):
    for j in (GEMMA, OSS):
        cnt = Counter()
        for s in sets:
            R, T = RUNS[(s, j, GEMMA)], TRUTH[s]
            for c in T:
                if c in R and T[c] == "No Gap" and not verdict(R[c]):
                    cnt[stage(R[c])] += 1
        print(f"  {label:<18} {j:<13} false Not Compliant {sum(cnt.values()):3d}: "
              + ", ".join(f"{k} {cnt[k]}" for k in ("assessor", "retrieval", "gate", "verifier")))

# ---------------------------------------------------------------- gate firing rates (Section IV-D)
print("\nGATE FIRING RATE - share of assessed controls whose top reranker score is below the gate threshold")
for label, sets in (("constructed corpus", COMPANIES), ("SecureNusantara", ["SecureNusantara"]), ("Hack23", ["Hack23"]),
                    ("Bappenas", ["Bappenas"]), ("DJPb", ["DJPb"])):
    fired = n = 0
    for s in sets:
        R = RUNS[(s, GEMMA, GEMMA)]
        for c in TRUTH[s]:
            if c in R:
                n += 1; fired += R[c]["gate_fired"] == "1"
    print(f"  {label:<18} {fired}/{n} ({fired / n:.0%})")

# ---------------------------------------------------------------- design vs. panel, run times (Sections III-B, IV-F)
print("\nDESIGN VS. PANEL - constructed corpus, binary (design 'Fully Implemented' = Compliant)")
pairs_dp = [(r["design_status"] == "Fully Implemented", r["panel_status"] == "No Gap")
            for r in read("labels/constructed_panel.csv")]
acc_dp, k_dp = kappa([a for a, _ in pairs_dp], [b for _, b in pairs_dp])
print(f"  agreement {acc_dp:.3f}, kappa {k_dp:.3f} (n = {len(pairs_dp)})")

print("\nAUTOMATED JUDGING TIME - mean minutes per constructed company (results/run_times.csv)")
per_company = defaultdict(float)          # resumed runs are logged in several segments: sum them per company
disturbed = set()                         # runs with LLM errors or a quota stop include retry waits: left out
for r in read("results/run_times.csv"):
    if r["set"] in SPLIT and r["judging_min"]:
        key = (r["judge"].replace("-cloud", ""), r["set"])
        per_company[key] += float(r["judging_min"])
        if int(r["llm_errors"] or 0) or r["stopped"] == "1":
            disturbed.add(key)
for j in (GEMMA, OSS):
    v = [m for k, m in per_company.items() if k[0] == j and k not in disturbed]
    left = sorted(k[1] for k in disturbed if k[0] == j)
    print(f"  {j:<13} {sum(v) / len(v):.1f} min (n = {len(v)} companies"
          + (f"; left out for LLM errors/retries: {', '.join(left)})" if left else ")"))
