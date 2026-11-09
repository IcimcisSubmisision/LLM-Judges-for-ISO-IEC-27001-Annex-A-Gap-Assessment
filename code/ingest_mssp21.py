"""Ingests all 21 SecureNusantara (MSSP) documents into one organization corpus.

  python -I ingest_mssp21.py --src <checkout of hazwanjaafar/mssp-isms-iso27001-simulation at 8d2def4> [--exclude a.md,b.md]

Same segmentation as ingest_hack23.py (parse_document / to_clauses), pooled into one organization corpus
MSSP21-SecureNusantara. README.md is not a document. Use the checkout unmodified: the reported corpus keeps the
Statement of Applicability's implementation-status column (see "Data notes" in the repository README). Run with -I:
the source files are a download.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ingest_hack23 import parse_document, to_clauses  # noqa: E402

CLIENT = sys.argv[sys.argv.index("--client") + 1] if "--client" in sys.argv else "MSSP21-SecureNusantara"
EXCLUDE = set(sys.argv[sys.argv.index("--exclude") + 1].split(",")) if "--exclude" in sys.argv else set()
SRC = Path(sys.argv[sys.argv.index("--src") + 1])
corpus = []
for path in sorted(SRC.rglob("*.md")):
    if path.name.lower() == "readme.md" or path.name in EXCLUDE:
        continue
    doc_id = "MSSP21-" + "_".join(w.capitalize() for w in path.stem.split("-"))
    clauses = to_clauses(parse_document(path.read_text(encoding="utf-8")), doc_id)
    print(f"  {str(path.relative_to(SRC)):<55} {len(clauses):4d} clauses")
    corpus += clauses
out = HERE / "synthetic_iso27001_uupdp_dataset" / "segmented_policies" / f"{CLIENT}.json"
out.write_text(json.dumps(corpus, indent=4, ensure_ascii=False), encoding="utf-8", newline="\r\n")
print(f"{CLIENT}: {len(corpus)} clauses -> {out}")

