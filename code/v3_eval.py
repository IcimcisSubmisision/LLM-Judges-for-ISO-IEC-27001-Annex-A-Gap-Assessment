"""Company-mode runner used by r5_job.py and r5_job_ossq.py: retrieval and judging of one document set
with the frozen analyzer (resumable; stops itself when LLM errors pile up).

  python v3_eval.py company --client CLIENT_ID [--modes base] [--retrieval]

--retrieval prints the offline retrieval report only, without judging. Output: v3_eval/<set>.jsonl.
"""
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SET = sys.argv[1] if len(sys.argv) > 1 else ""
if SET != "company":
    raise SystemExit("usage: python v3_eval.py company --client ID [--modes base] [--retrieval]")
FULL, OFFLINE = "--full" in sys.argv, "--retrieval" in sys.argv
_arg = lambda flag, default: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
MODES = tuple(_arg("--modes", "base,v3").split(","))   # candidate modes: v3, v3b
KEYS = _arg("--keys", None)             # JSON list of [client, control_id] restricting the dev set
CLIENT = _arg("--client", None)         # SET company: one dataset company (e.g. a test company with GRC labels)
if SET == "company" and not CLIENT:
    raise SystemExit("company needs --client <client_id>")
NAME = (SET + ("_" + CLIENT if CLIENT else "") + ("_full" if FULL else "") + ("_" + Path(KEYS).stem if KEYS else "")
        + ("" if MODES == ("base", "v3") else "_" + "-".join(MODES)))
OUTDIR = Path("v3_eval"); OUTDIR.mkdir(exist_ok=True)
OUT = OUTDIR / f"{NAME}.jsonl"
WORKERS = max(1, int(os.environ.get("EVAL_WORKERS", "4") or 4))
os.environ["EVAL_WORKERS"] = "1"          # eg.run_analysis only builds single rows here


if SET == "company":                  # one company, whichever split it is in (reporting only for test companies)
    os.environ["GOLDEN_PATH"] = "synthetic_iso27001_uupdp_dataset/golden_iso_sectors.json"
    os.environ.pop("SPLIT", None)
    os.environ["LIMIT_CLIENTS"] = CLIENT

import evaluate_golden as eg  # noqa: E402
import pipeline_v3  # noqa: E402
from main import LLMAnalyzer  # noqa: E402

MODEL = os.environ.get("JUDGE_MODEL", "gemma4:31b-cloud")
BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://localhost:11434/v1")
PIPELINE = {**{env: os.environ.get(env, default) for env, _, default in LLMAnalyzer._OPTIONS.values()},
            "EXPANSION": os.environ.get("EXPANSION", "on").lower(), "JUDGE_MODEL": MODEL, "SET": NAME,
            "V3": "element-check + extractor line + ID/element retrieval queries, per_sub_top=2",
            **({"V3B": "strict same-activity 2b"} if "v3b" in MODES else {})}


def prepare(mode):
    v3 = mode != "base"                 # every candidate mode uses the v3 retrieval queries
    pipeline_v3.retrieval(v3)
    try:
        prepared, _, _ = eg.run_retrieval()
    finally:
        pipeline_v3.retrieval(False)
    return prepared


class _Fixed:
    def __init__(self, report):
        self.report = report

    def analyze_gap(self, *a, **k):
        return self.report


def main():
    prep = {}
    for m in MODES:
        t0 = time.time()
        prep[m] = prepare(m)
        print(f"[retrieval] {NAME} {m}: {len(prep[m])} controls, mean evidence "
              f"{sum(len(p['reranked']) for p in prep[m]) / max(1, len(prep[m])):.1f} clauses "
              f"({time.time() - t0:.0f}s)", flush=True)
    key = lambda p: (p["client"], p["entry"]["control_id"])
    by = {m: {key(p): p for p in prep[m]} for m in MODES}
    for m in MODES:
        hs = [p["retrieval_hit"] for p in prep[m] if p["retrieval_hit"] is not None]
        print(f"[offline] {m}: section recall {sum(hs)}/{len(hs)} ({sum(hs) / max(1, len(hs)):.1%})", flush=True)
    if OFFLINE:
        return

    done = {}
    if OUT.exists():
        for line in open(OUT, encoding="utf-8"):
            r = json.loads(line)
            if (r.get("pipeline") == PIPELINE and not r.get("llm_error")
                    and not (r.get("subject_check") or {}).get("error")):
                done[(r["client"], r["control_id"], r["mode"])] = r
    todo = [k for k in by["base"] if any((k[0], k[1], m) not in done for m in MODES)]
    print(f"[run] {len(by['base'])} controls; {len(todo)} to judge x {len(MODES)} modes with {WORKERS} workers "
          f"({len(done)} rows reusable)", flush=True)
    analyzers = {m: (LLMAnalyzer(model_name=MODEL, base_url=BASE_URL, binary_labels=eg.BINARY_LABELS)
                     if m == "base" else
                     pipeline_v3.make_analyzer(variant=m, model_name=MODEL, base_url=BASE_URL,
                                               binary_labels=eg.BINARY_LABELS)) for m in MODES}
    stop, lock, stats = threading.Event(), threading.Lock(), {"n": 0, "err": 0}

    def judge(k):
        rows = []
        for m in MODES:
            if (k[0], k[1], m) in done or stop.is_set():
                continue
            p = by[m][k]
            for _ in range(2):
                rep = analyzers[m].analyze_gap(p["reg_clause"], p["reranked"], sub_queries=p["sub_queries"])
                err = (rep is None or getattr(rep, "evidence_status", "") == "llm_error"
                       or bool((getattr(rep, "subject_check", None) or {}).get("error")))
                if not err:
                    break
                time.sleep(20)
            row = eg.run_analysis(_Fixed(rep), [p])[0][0] if not err else {
                "client": k[0], "control_id": k[1], "predicted_label": "ERROR"}
            row.update(mode=m, pipeline=PIPELINE, llm_error=err)
            rows.append(row)
        return k, rows

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed([ex.submit(judge, k) for k in todo]):
            k, rows = fut.result()
            with lock:
                with open(OUT, "a", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                stats["n"] += 1
                stats["err"] += sum(r["llm_error"] for r in rows)
                if stats["err"] >= 6 and stats["err"] > 0.15 * stats["n"] * len(MODES) and not stop.is_set():
                    stop.set()
                    print(f"[STOP] {stats['err']} LLM errors - quota or rate limit; re-run later to resume", flush=True)
                if stats["n"] % 20 == 0 or any(r["llm_error"] for r in rows):
                    el = time.time() - t0
                    print(f"[{stats['n']}/{len(todo)}] errors={stats['err']} ({el / 60:.1f} min, "
                          f"~{el / stats['n'] * (len(todo) - stats['n']) / 60:.0f} left)", flush=True)
    print("[done]", NAME, stats, flush=True)


if __name__ == "__main__":
    main()
