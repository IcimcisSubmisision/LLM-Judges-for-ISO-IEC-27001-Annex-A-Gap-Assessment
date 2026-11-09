"""Set up the input layout that the code in code/ expects, from the files in this repository.

  python scripts/prepare_inputs.py

Creates, inside code/:
  synthetic_iso27001_uupdp_dataset/            query files from queries/, plus two stand-in files (below)
  synthetic_iso27001_uupdp_dataset/segmented_policies/   empty; filled by the ingest_*.py scripts
  golden/<set>.json                             the controls to judge per document set (from labels/)
  UUPDP_ID.txt, UUPDP_EN.txt                    empty stand-ins (below)

Stand-in files. evaluate_golden.py loads two inputs that do not affect Annex A verdicts in the reported runs:
  - expanded_queries-v1.1.json (English control elements). With EXPANSION=off, as in every reported run, all ISO
    entries of this file are dropped right after loading (evaluate_golden.py, "EXPANSION=off"), so an empty JSON
    object gives identical verdicts. The original file was derived from the copyrighted Annex A text.
  - UUPDP_ID.txt / UUPDP_EN.txt (Indonesian Personal Data Protection Law, from an earlier experiment). Empty files
    parse to no clauses; only ISO controls are judged.
Still needed from you: iso.txt and the two Indonesian files derived from it (see README), and the public and
government documents.
"""
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
DS = CODE / "synthetic_iso27001_uupdp_dataset"
GOLD = CODE / "golden"
(DS / "segmented_policies").mkdir(parents=True, exist_ok=True)
GOLD.mkdir(exist_ok=True)

for q in sorted((ROOT / "queries").glob("*.json")):
    shutil.copy2(q, DS / q.name)

stub = DS / "expanded_queries-v1.1.json"
if not stub.exists():
    stub.write_text("{}\n", encoding="utf-8")
for name in ("UUPDP_ID.txt", "UUPDP_EN.txt"):
    if not (CODE / name).exists():
        (CODE / name).write_text("", encoding="utf-8")


def rows(path):
    with open(ROOT / path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def entry(control, client, label):
    return {"control_id": "ISO_A_" + control[2:], "client_id": client, "expected_clause_ids": [],
            "ground_truth_label": label}


binary = lambda status: "Compliant" if status == "No Gap" else "Not Compliant"
golden = {}
golden["golden_iso_sectors.json"] = [entry(r["control"], r["company"], binary(r["panel_status"]))
                                     for r in rows("labels/constructed_panel.csv")]
for r in rows("labels/public_panel.csv"):                        # all 93 controls are judged; Hack23's nine
    client, fname = {"Hack23": ("HACK23-ORG", "golden_hack23.json"),            # Not Applicable ones are
                     "SecureNusantara": ("MSSP21-SecureNusantara", "golden_mssp21.json")}[r["set"]]  # dropped when scoring
    golden.setdefault(fname, []).append(entry(r["control"], client, binary(r["panel_status"])))
for r in rows("labels/government_raters.csv"):
    client, fname = {"Bappenas": ("BAPPENAS-PPN", "golden_bappenas.json"),
                     "DJPb": ("DJPB-PER1", "golden_djpb.json")}[r["document"]]
    golden.setdefault(fname, []).append(entry(r["control"], client, r["majority_binary"]))
for fname, entries in golden.items():
    (GOLD / fname).write_text(json.dumps(entries, indent=1), encoding="utf-8")
    print(f"golden/{fname}: {len(entries)} controls")
print(f"queries and stand-in files written to {DS.relative_to(ROOT)} and {CODE.relative_to(ROOT)}")
