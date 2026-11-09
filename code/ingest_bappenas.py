"""Bappenas guideline No. 2/2024 (information security policy) -> clause corpus.

  python ingest_bappenas.py [inputs/bappenas_text.txt]
      -> synthetic_iso27001_uupdp_dataset/segmented_policies/BAPPENAS-PPN.json
Structure of the regulation: BAB > huruf section (A. ...) > numbered provision (1. Title) > items (a., 1), -).
One corpus section per numbered provision (or per huruf section where it has none), so the provision context
groups the items of one provision; every sentence carries '<n>. <huruf title> - <provision title>'.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "inputs/bappenas_text.txt"   # from extract_pdf_text.py
CLIENT = "BAPPENAS-PPN"
SMALL = {"dan", "atau", "untuk", "dalam", "di", "yang", "terhadap", "ke", "dari", "pada", "&", "dengan", "serta", "oleh"}
_SENT = re.compile(r'(?<=[a-z0-9)\]"])\.\s+(?=[A-Z])')

lines = [l.strip() for l in SRC.read_text(encoding="utf-8").split("\n")]
start = [i for i, l in enumerate(lines) if l == "BAB I"][0]                     # skip cover, preface, table of contents
lines = [l for l in lines[start:] if l and not l.startswith("[Halaman PDF") and not re.fullmatch(r"[ivx]+|\d+", l)]


def titleish(s):
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-/&()]*", s) if w.lower() not in SMALL]
    return bool(words) and sum(w[0].isupper() for w in words) / len(words) >= 0.6


def is_num_heading(l):
    m = re.match(r"^(\d+)\.\s+(.*)$", l)
    return bool(m) and len(m.group(2).split()) <= 12 and not m.group(2).endswith((";", ",", ".", ":")) and titleish(m.group(2))


ITEM = re.compile(r"^([a-z]\.|\d+\)|\d+\.|[-•])\s+")
bab = huruf = prov = None
sections, cur, unit = [], None, []


def flush_unit():
    global unit
    if unit and cur is not None:
        cur["units"].append(" ".join(unit))
    unit = []


def new_section():
    global cur
    flush_unit()
    cur = {"title": " - ".join(t for t in (huruf, prov) if t), "units": []}
    sections.append(cur)


i = 0
while i < len(lines):
    l = lines[i]
    if re.fullmatch(r"BAB\s+([IVX]+|\d+)", l):
        flush_unit(); bab = lines[i + 1] if i + 1 < len(lines) else ""; huruf = prov = None
        cur = None; i += 2; continue
    m = re.match(r"^([A-Z])\.\s+(.+)$", l)
    if m and titleish(m.group(2)) and len(m.group(2).split()) <= 10:
        huruf, prov = m.group(2).strip(), None
        if i + 1 < len(lines) and lines[i + 1] and not ITEM.match(lines[i + 1]) and not is_num_heading(lines[i + 1]) \
                and len(lines[i + 1].split()) <= 4 and titleish(lines[i + 1]):          # wrapped title ('... Continuity' / 'Management')
            huruf += " " + lines[i + 1]; i += 1
        new_section(); i += 1; continue
    if is_num_heading(l) and huruf:
        prov = re.sub(r"^\d+\.\s+", "", l); new_section(); i += 1; continue
    if cur is None:
        huruf = huruf or bab; new_section()
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
for c in corpus[:3] + corpus[200:203]:
    print("  ", c["id"], "|", c["text"][:150])
