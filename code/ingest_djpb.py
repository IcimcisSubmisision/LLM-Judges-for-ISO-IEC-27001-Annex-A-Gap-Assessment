"""DJPb regulation PER-1/PB/2021 -> clause corpus.

  python ingest_djpb.py [inputs/djpb_text.txt]
      -> synthetic_iso27001_uupdp_dataset/segmented_policies/DJPB-PER1.json
Structure of the regulation: BAB > Bagian > Pasal > ayat (1) > huruf a. > angka 1). One corpus section per Pasal, so
the provision context groups its ayat and items; every sentence carries '<n>. <Bagian or BAB title> - Pasal <k>'.
Source text: inputs/djpb_text.txt (OCR, reading-order lines from ocr_lines.py; same text the annotators saw).
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "inputs/djpb_text.txt"
CLIENT = "DJPB-PER1"
_SENT = re.compile(r'(?<=[a-z0-9)\]"])\.\s+(?=[A-Z])')
ROMAN = r"[0-9IVXl]+"

lines = [l.strip() for l in SRC.read_text(encoding="utf-8").split("\n")]
start = [i for i, l in enumerate(lines) if re.fullmatch(rf"BAB\s+{ROMAN}", l)][0]       # skip Menimbang / Mengingat
lines = [l for l in lines[start:] if l and not l.startswith("[Halaman PDF")]
end = [i for i, l in enumerate(lines) if l.startswith("Ditetapkan di")]
lines = lines[:end[0]] if end else lines                                                 # drop signature block

ITEM = re.compile(r"^(\(\d+\)|[a-z]\.|\d+\)|\d+\.)\s+")
bab = bagian = pasal = None
sections, cur, unit = [], None, []


def flush_unit():
    global unit
    if unit and cur is not None:
        cur["units"].append(" ".join(unit))
    unit = []


i = 0
while i < len(lines):
    l = lines[i]
    if re.fullmatch(rf"BAB\s+{ROMAN}", l):
        flush_unit(); bab = lines[i + 1] if i + 1 < len(lines) else ""; bagian = None; cur = None; i += 2; continue
    if re.fullmatch(r"Bagian\s+Ke\w+", l):                                           # title may wrap onto 2 lines
        flush_unit(); j = i + 1
        while j < len(lines) and not re.fullmatch(r"Pasal\s+[0-9IlO]+", lines[j]) and j - i <= 2:
            j += 1
        bagian = " ".join(lines[i + 1:j]); cur = None; i = j; continue
    m = re.fullmatch(r"Pasal\s+([0-9IlO]+)", l)
    if m:
        flush_unit()
        pasal = m.group(1).replace("I", "1").replace("l", "1").replace("O", "0")
        cur = {"title": f"{(bagian or bab or '').strip()} - Pasal {pasal}", "units": []}
        sections.append(cur); i += 1; continue
    if cur is None:
        i += 1; continue
    if ITEM.match(l):
        flush_unit()
    unit.append(l)
    i += 1
flush_unit()

corpus = []
n = 0
for s in sections:
    sents = []
    for u in s["units"]:
        u = re.sub(r"\s+", " ", u).strip()
        for sent in _SENT.split(u):
            if len(sent.split()) >= 5:
                sents.append(sent.strip())
    if not sents:
        continue
    n += 1
    for k, sent in enumerate(sents, 1):
        corpus.append({"id": f"{CLIENT}_Sec_{n}_Sent_{k}", "text": f"{n}. {s['title']} - {sent}"})
out = HERE / "synthetic_iso27001_uupdp_dataset/segmented_policies" / f"{CLIENT}.json"
json.dump(corpus, open(out, "w", encoding="utf-8", newline="\r\n"), indent=1, ensure_ascii=False)
print(f"{CLIENT}: {n} sections, {len(corpus)} clauses -> {out}")
for c in corpus[:2] + corpus[150:153] + corpus[-2:]:
    print("  ", c["id"], "|", c["text"][:160])
