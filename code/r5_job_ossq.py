"""gpt-oss own-query variant of r5_job.py (the pre-registration log, "Judge-specific evidence queries"). One R5 job: one document set judged by one judge model (the pre-registration log, "R5 full rerun").

  python r5_job_ossq.py --set NAME --client CLIENT --golden GOLDEN --judge MODEL --outdir DIR [--test]

Retrieval (v3r + R5 patch: top-k 20, HyDE queries, section expansion, per_sub_top 3) runs once per set and is cached
in DIR/retrieval/<set>.json, so both judges see exactly the same evidence. Judging uses the frozen v2 analyzer
through v3_eval (resumable; [STOP] on quota). Every segment appends one line to DIR/timing.jsonl with the columns
of the run table (company, controls, retrieval min, judging min, s/control) plus context.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

ap = argparse.ArgumentParser()
for a in ("--set", "--client", "--golden", "--judge", "--outdir"):
    ap.add_argument(a, required=True)
ap.add_argument("--package", default="local")
ap.add_argument("--test", action="store_true")
A = ap.parse_args()
OUTDIR = Path(A.outdir); (OUTDIR / "retrieval").mkdir(parents=True, exist_ok=True)
TAG = ("TEST_" if A.test else "") + A.set
PREP = OUTDIR / "retrieval" / f"{TAG}.json"
OUT = OUTDIR / f"{TAG}__{A.judge.split(':')[0]}.jsonl"
os.environ["JUDGE_MODEL"] = A.judge
os.environ["GOLDEN_PATH"] = A.golden
if A.test:
    os.environ["LIMIT_PER_CLIENT"] = "2"
sys.stdout.reconfigure(encoding="utf-8")

import pipeline_v3  # noqa: E402
import v5_retrieval  # noqa: E402
v5_retrieval.HYDE = json.load(open("synthetic_iso27001_uupdp_dataset/hyde_queries_v5_gptoss.json", encoding="utf-8"))["queries"]  # own queries

_orig = pipeline_v3.retrieval


def _retrieval(v3):
    _orig(v3)
    if v3:
        v5_retrieval.apply("R5")
    else:
        v5_retrieval.MultiQueryEnsembleRetriever.__init__ = v5_retrieval._ORIG_INIT
        pipeline_v3.PER_SUB_TOP = 2
        for cid, base in v5_retrieval._ORIG_EXTRA.items():
            pipeline_v3.EXTRA_QUERIES[cid][:] = base


pipeline_v3.retrieval = _retrieval
sys.argv = ["v3_eval.py", "company", "--client", A.client, "--modes", "base"]
import v3_eval  # noqa: E402

os.environ["GOLDEN_PATH"] = A.golden                     # v3_eval's 'company' branch reset it at import
HYDE_SHA = hashlib.sha256(Path("synthetic_iso27001_uupdp_dataset/hyde_queries_v5_gptoss.json").read_bytes()).hexdigest()[:12]
v3_eval.PIPELINE.update(GOLDEN=A.golden, CONFIG=f"R5 retrieval, judge {A.judge}",
                        V5=f"R5: top-k 20 + gpt-oss-written HyDE queries ({HYDE_SHA}) + section expansion + per_sub_top 3")
v3_eval.NAME, v3_eval.OUT = OUT.stem, OUT

# ---- retrieval, once per set (cached) --------------------------------------------------------------------------
if PREP.exists():
    cache = json.load(open(PREP, encoding="utf-8"))
    if cache.get("queries_sha", HYDE_SHA) != HYDE_SHA:      # released-code guard: never judge evidence retrieved
        raise SystemExit(f"{PREP} was retrieved with other evidence queries ({cache['queries_sha']}); "
                         "use a separate --outdir for each query file")
    prepared, retrieval_s, cached = cache["prepared"], cache["retrieval_s"], True
else:
    t0 = time.time()
    pipeline_v3.retrieval(True)
    try:
        prepared = v3_eval.eg.run_retrieval()[0]
    finally:
        pipeline_v3.retrieval(False)
    retrieval_s, cached = time.time() - t0, False
    tmp = PREP.with_suffix(".tmp")
    json.dump({"set": A.set, "client": A.client, "queries_sha": HYDE_SHA, "retrieval_s": retrieval_s, "created": datetime.now().isoformat(" ", "seconds"),
               "machine": platform.platform(), "prepared": prepared}, open(tmp, "w", encoding="utf-8"),
              ensure_ascii=False, default=float)
    tmp.replace(PREP)
    prepared = json.load(open(PREP, encoding="utf-8"))["prepared"]       # judge exactly what is cached
v3_eval.prepare = lambda mode: prepared

# ---- judging ------------------------------------------------------------------------------------------------------
def rows_ok():
    if not OUT.exists():
        return 0
    return sum(1 for l in open(OUT, encoding="utf-8") if l.strip() and not json.loads(l).get("llm_error"))


before, start = rows_ok(), datetime.now()
t0 = time.time()
lines = []
_print = print


class Tee:
    def write(self, s):
        sys.__stdout__.write(s); lines.append(s)

    def flush(self):
        sys.__stdout__.flush()


sys.stdout = Tee()
try:
    v3_eval.main()
finally:
    sys.stdout = sys.__stdout__
judge_s = time.time() - t0
after = rows_ok()
text = "".join(lines)
judged_now = after - before
rec = {"package": A.package, "set": A.set, "client": A.client, "judge": A.judge, "test": A.test,
       "controls": len(prepared), "done_controls": after,
       "retrieval_min": round(retrieval_s / 60, 2), "retrieval_cached": cached,
       "judging_min": round(judge_s / 60, 2), "judged_this_segment": judged_now,
       "s_per_control": round(judge_s / judged_now, 1) if judged_now else None,
       "llm_errors": text.count("LLM ERROR"), "stopped": "[STOP]" in text,
       "mean_quotes": round(sum(len(p["reranked"]) for p in prepared) / max(1, len(prepared)), 1),
       "start": start.isoformat(" ", "seconds"), "end": datetime.now().isoformat(" ", "seconds"),
       "workers": v3_eval.WORKERS, "machine": platform.platform(), "python": platform.python_version()}
with open(OUTDIR / "timing.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"[timing] {json.dumps(rec, ensure_ascii=False)}", flush=True)
sys.exit(2 if rec["stopped"] or after < len(prepared) else 0)
