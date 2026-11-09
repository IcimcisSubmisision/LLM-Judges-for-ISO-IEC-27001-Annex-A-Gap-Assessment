"""Indonesian element queries for the 93 ISO/IEC 27001:2022 Annex A controls
(pipeline v3, the pre-registration log "Pipeline v3 candidate").

  python make_iso_elements.py   -> synthetic_iso27001_uupdp_dataset/iso_annex_a_elements_id.json

One short Indonesian search query per element the control names (action, object,
scope, timing, condition). Used for RETRIEVAL only - the assessor still judges
the whole control as one obligation. Made by the judge model from the control
text ONLY (no company document is shown), written once, then frozen.
"""
import json
import re
import sys
from pathlib import Path

from openai import OpenAI

from main import RegulatoryDocumentParser

OUT = Path("synthetic_iso27001_uupdp_dataset/iso_annex_a_elements_id.json")
MODEL = "gemma4:31b-cloud"
SYSTEM = (
    "You split an ISO/IEC 27001:2022 Annex A control into the distinct requirements (elements) it names: "
    "each required action, object, scope, timing or condition. Evaluative words such as 'appropriate', "
    "'relevant' or 'adequate' are not separate elements. Do not add requirements the control does not name. "
    "Write each element as a short, self-contained search query in formal Indonesian (the vocabulary of "
    "Indonesian information-security policies), at most 8 elements. "
    'Output ONLY JSON: {"elements": ["...", "..."]}'
)

if OUT.exists() and "--force" not in sys.argv:
    raise SystemExit(f"{OUT} exists and is frozen; pass --force only to rebuild it deliberately")
regs = RegulatoryDocumentParser().parse_document(open("iso.txt", encoding="utf-8").readlines(), doc_type="ISO27001")
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
out = {}
for r in regs:
    els = []
    for attempt in range(3):
        try:
            raw = client.chat.completions.create(
                model=MODEL, temperature=0.0, timeout=120, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["text"]}],
            ).choices[0].message.content
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
            els = [e.strip() for e in json.loads(raw).get("elements", []) if isinstance(e, str) and e.strip()][:8]
            if els:
                break
        except Exception as e:
            print("retry", r["id"], e)
    out[r["id"]] = els
    print(r["id"], len(els), "|", " || ".join(els)[:150], flush=True)
missing = [k for k, v in out.items() if not v]
if missing:
    raise SystemExit(f"no elements for {missing}; nothing written")
json.dump({"_note": "Indonesian element queries for retrieval only; made from the control text only.",
           "model": MODEL, "system_prompt": SYSTEM, "temperature": 0.0, "elements": out},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {OUT} ({len(out)} controls, {sum(map(len, out.values()))} element queries)")
