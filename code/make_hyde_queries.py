"""HyDE-style evidence-vocabulary queries for the 93 Annex A controls (the pre-registration log, v5 retrieval addendum R4).

  python make_hyde_queries.py [--model M --out F]   -> synthetic_iso27001_uupdp_dataset/hyde_queries_v5.json

Written from the control text only; no company document is shown to the model. Retrieval only: these sentences
are added to the retrieval and cross-encoder sub-queries and are never given to the assessor.
"""
import json
import sys
import time
import urllib.error
import urllib.request

import pipeline_v3

sys.stdout.reconfigure(encoding="utf-8")
MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "gemma4:31b-cloud"
OUT = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "synthetic_iso27001_uupdp_dataset/hyde_queries_v5.json"
PROMPT = ("You write example text for a search index. ISO/IEC 27001:2022 Annex A control {cid} '{title}': {text}\n\n"
          "Write 4 short, different sentences (max 30 words each) as they would appear in a company's information "
          "security policy, procedure, register or table and that would show this control is implemented. Use the "
          "concrete words organisations use for it (roles, systems, records, steps, frequencies), not the wording "
          "of the control. Do not mention ISO, Annex A or the control number. Output only a JSON list of 4 strings.")
titles = json.load(open("synthetic_iso27001_uupdp_dataset/iso_annex_a_titles.json", encoding="utf-8"))["titles"]
out = {"_model": MODEL, "_temperature": 0, "_prompt": PROMPT, "queries": {}}
for reg in pipeline_v3.REGS:
    cid = reg["id"]
    if not cid.startswith("ISO_A_"):
        continue
    body = json.dumps({"model": MODEL, "stream": False, "options": {"temperature": 0},
                       "messages": [{"role": "user", "content": PROMPT.format(cid=cid[6:], title=titles[cid],
                                                                               text=reg["text"])}]}).encode()
    for attempt in range(5):
        try:
            raw = json.load(urllib.request.urlopen(urllib.request.Request(
                "http://localhost:11434/api/chat", body, {"Content-Type": "application/json"}), timeout=180))
        except (urllib.error.URLError, TimeoutError) as e:          # transient 502 / timeouts from the cloud model
            print(f"{cid}: {e}; retrying", flush=True)
            time.sleep(10 * (attempt + 1))
            continue
        txt = raw["message"]["content"].strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        try:
            qs = [q.strip() for q in json.loads(txt) if isinstance(q, str) and q.strip()]
            if len(qs) >= 3:
                break
        except json.JSONDecodeError:
            pass
    else:
        sys.exit(f"{cid}: no valid list after 3 attempts: {txt[:200]}")
    out["queries"][cid] = qs[:4]
    print(cid, "|", qs[0][:100], flush=True)
assert len(out["queries"]) == 93, len(out["queries"])
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1,
          ensure_ascii=False)
print("93 controls written")
