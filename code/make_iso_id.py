"""Indonesian retrieval queries for the 93 ISO/IEC 27001:2022 Annex A controls
(pre-specified in the pre-registration log, "bilingual retrieval queries").

  python make_iso_id.py            -> synthetic_iso27001_uupdp_dataset/iso_annex_a_id.json

The translation is made by the judge model from the control text ONLY - no
company document is shown - so the queries cannot be fitted to any evaluation
document. The file is written once and then frozen; re-running refuses to
overwrite it.
"""
import json
import sys
from pathlib import Path

from openai import OpenAI

from main import RegulatoryDocumentParser

OUT = Path("synthetic_iso27001_uupdp_dataset/iso_annex_a_id.json")
MODEL = "gemma4:31b-cloud"
SYSTEM = ("You translate ISO/IEC 27001:2022 Annex A controls into formal Indonesian, using the terminology "
          "of Indonesian information-security standards and regulations (SNI ISO/IEC 27001). "
          "Return only the Indonesian translation of the control, as one sentence, with no notes.")

if OUT.exists() and "--force" not in sys.argv:
    raise SystemExit(f"{OUT} exists and is frozen; pass --force only to rebuild it deliberately")
regs = RegulatoryDocumentParser().parse_document(open("iso.txt", encoding="utf-8").readlines(), doc_type="ISO27001")
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
out = {}
for r in regs:
    for attempt in range(3):
        try:
            t = client.chat.completions.create(
                model=MODEL, temperature=0.0, timeout=120,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["text"]}],
            ).choices[0].message.content.strip()
            if t:
                break
        except Exception as e:
            print("retry", r["id"], e)
    out[r["id"]] = t
    print(r["id"], "|", t[:110], flush=True)
json.dump({"_note": "Indonesian retrieval queries; translated from the control text only.",
           "model": MODEL, "system_prompt": SYSTEM, "temperature": 0.0, "translations": out},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {OUT} ({len(out)} controls)")
