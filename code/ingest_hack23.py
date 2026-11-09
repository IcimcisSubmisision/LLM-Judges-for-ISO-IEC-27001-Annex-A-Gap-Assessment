"""
Ingests the Hack23 AB public ISMS (https://github.com/Hack23/ISMS-PUBLIC, Apache-2.0)
into the same segmented-clause JSON format the pipeline already consumes.

Why this matters for the thesis: the internal-policy corpus was previously 100%
synthetic. These are 40 policy documents from a real operating company that
publishes its actual ISMS openly, which is what Section 3.1(2) claims as a data
source. Provenance stays visible through the HACK23- id prefix.

These files are NOT in the '1. Section' numbered form that
util.segment_by_section_and_sentence expects - they are heavily decorated
Markdown (HTML blocks, shields.io badges, Mermaid diagrams, emoji, tables).
This module handles that shape specifically; it does not replace the existing
segmenter for the synthetic corpus.

Usage:  python ingest_hack23.py [--src DIR] [--out DIR]
"""
import argparse
import json
import re
from pathlib import Path

# --- Markdown / decoration stripping -------------------------------------

FENCE_RE = re.compile(r'^\s*```')
HTML_TAG_RE = re.compile(r'<[^>]+>')
# [![alt](img)](link)  and  ![alt](img)  -> badges and images, pure decoration
BADGE_RE = re.compile(r'\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)')
IMAGE_RE = re.compile(r'!\[[^\]]*\]\([^)]*\)')
# [text](url) -> text
LINK_RE = re.compile(r'\[([^\]]*)\]\([^)]*\)')
EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002190-\U000021FF" "\U00002300-\U000023FF"
    "\U000024C2-\U0001F251" "\U00002600-\U000027BF" "\U0000FE00-\U0000FE0F"
    "\U0001F1E6-\U0001F1FF" "✅❌⭐⚠ℹ©®™"
    "]+", flags=re.UNICODE)
EMPHASIS_RE = re.compile(r'(\*\*|__|\*|`)')
HRULE_RE = re.compile(r'^\s*([-*_]\s*){3,}$')
H2_RE = re.compile(r'^##\s+(.*)$')
H3_RE = re.compile(r'^###+\s+(.*)$')
H1_RE = re.compile(r'^#\s+(.*)$')
BULLET_RE = re.compile(r'^\s*[-*+]\s+(.*)$')
ORDERED_RE = re.compile(r'^\s*\d+[.)]\s+(.*)$')
TABLE_SEP_RE = re.compile(r'^\s*\|[\s:|-]+\|\s*$')
KV_LINE_RE = re.compile(r'^\s*\*\*[^*]+:\*\*')

_SENT_BOUNDARY = re.compile(r'(?<=[a-z0-9)\]"])\.\s+(?=[A-Z])')

# Not every file in the repo is an ISMS policy. These are documentation-about-
# the-documentation, business/marketing material, or a different standard
# entirely (ISO 5230 is OpenChain licence compliance, not 27001). Including them
# in the retrieval corpus invites false-positive "evidence" - e.g. STYLE_GUIDE's
# "Security Badge Standards" section matching a control about security standards.
# Excluded only when --policies-only is passed, so the default stays faithful to
# the published repository and the choice is visible rather than silent.
NON_POLICY_FILES = {
    "STYLE_GUIDE.md",              # how to format the ISMS docs
    "ISMS_QA_CHECKLIST.md",        # QA of the documentation itself
    "ISMS_METRICS_DASHBOARD.md",   # dashboard/reporting artefact
    "Security_Metrics.md",         # metric definitions, not control statements
    "Partnership_Framework.md",    # commercial partnership material
    "NIS2_Compliance_Service.md",  # a consulting service offering
    "ISO_5230_Self_Certification.md",  # OpenChain (licence compliance), not 27001
    "ISMS_Transparency_Plan.md",   # publication plan for the ISMS
    # NOT excluded: External_Stakeholder_Registry.md. It looks like a register
    # rather than a policy, but Compliance_Checklist.md names it as the evidence
    # for BOTH A.5.5 (contact with authorities) and A.5.6 (contact with special
    # interest groups). Excluding it made those controls unevidenced by
    # construction - an artefact of curation, not a real gap.
}


def clean_inline(text: str) -> str:
    """Strip decoration from a single line, preserving its wording."""
    text = BADGE_RE.sub(' ', text)
    text = IMAGE_RE.sub(' ', text)
    text = LINK_RE.sub(r'\1', text)
    text = HTML_TAG_RE.sub(' ', text)
    text = EMOJI_RE.sub(' ', text)
    text = EMPHASIS_RE.sub('', text)
    text = text.replace('&nbsp;', ' ')
    return re.sub(r'\s+', ' ', text).strip()


def split_table_row(line: str):
    cells = [clean_inline(c) for c in line.strip().strip('|').split('|')]
    return [c for c in cells]


def parse_document(md: str):
    """Returns a list of (section_no, section_title, subsection_title, content)."""
    lines = md.split('\n')
    units = []

    sec_no = 0
    sec_title = None
    sub_title = None
    in_fence = False
    para = []
    table_header = None

    def flush_para():
        nonlocal para
        if not para:
            return
        joined = clean_inline(' '.join(para))
        para = []
        if not joined:
            return
        for sent in _SENT_BOUNDARY.split(joined):
            sent = sent.strip()
            if len(sent.split()) >= 5:
                units.append((sec_no, sec_title, sub_title, sent))

    for raw in lines:
        if FENCE_RE.match(raw):           # ``` ... ``` (incl. mermaid) - drop wholesale
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if HRULE_RE.match(raw):
            flush_para()
            continue

        m = H2_RE.match(raw)
        if m:
            flush_para()
            table_header = None
            sec_no += 1
            sec_title = clean_inline(m.group(1))
            sub_title = None
            continue

        m = H3_RE.match(raw)
        if m:
            flush_para()
            table_header = None
            sub_title = clean_inline(m.group(1))
            continue

        m = H1_RE.match(raw)
        if m:
            flush_para()
            continue

        stripped = raw.strip()
        if not stripped:
            flush_para()
            continue

        # --- tables: keep them. A row like
        #     | Cloud Core Infrastructure | Extreme | SSO | 4 hours |
        # carries real control detail (MFA, timeouts, review cadence), so it is
        # re-expressed as "Header: value; Header: value" rather than discarded.
        if stripped.startswith('|'):
            flush_para()
            if TABLE_SEP_RE.match(stripped):
                continue
            cells = split_table_row(stripped)
            if not any(cells):
                continue
            if table_header is None:
                table_header = cells
                continue
            pairs = []
            for i, c in enumerate(cells):
                if not c:
                    continue
                head = table_header[i] if i < len(table_header) else ''
                pairs.append(f"{head}: {c}" if head else c)
            row = '; '.join(pairs)
            if len(row.split()) >= 5:
                units.append((sec_no, sec_title, sub_title, row))
            continue
        table_header = None

        m = BULLET_RE.match(stripped) or ORDERED_RE.match(stripped)
        if m:
            flush_para()
            item = clean_inline(m.group(1))
            if len(item.split()) >= 5:
                units.append((sec_no, sec_title, sub_title, item))
            continue

        # "**Document Owner:** CEO | **Version:** 2.6" metadata lines
        if KV_LINE_RE.match(stripped):
            flush_para()
            item = clean_inline(stripped)
            if len(item.split()) >= 5:
                units.append((sec_no, sec_title, sub_title, item))
            continue

        para.append(stripped)

    flush_para()
    return units


def to_clauses(units, doc_id):
    """Mirrors util.segment_by_section_and_sentence's id/text contract:
    id = <doc>_Sec_<n>_Sent_<k>, text = '<n>. <section> - <content>' so the
    section title travels with every chunk for semantic density."""
    clauses = []
    counter = {}
    for sec_no, sec_title, sub_title, content in units:
        if sec_no == 0:
            continue                      # preamble before the first H2
        counter[sec_no] = counter.get(sec_no, 0) + 1
        header = f"{sec_no}. {sec_title}" if sec_title else str(sec_no)
        if sub_title:
            header = f"{header} - {sub_title}"
        clauses.append({
            "id": f"{doc_id}_Sec_{sec_no}_Sent_{counter[sec_no]}",
            "text": f"{header} - {content}",
        })
    return clauses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory holding the Hack23 .md files")
    ap.add_argument("--out", default="synthetic_iso27001_uupdp_dataset/segmented_policies")
    ap.add_argument("--policies-only", action="store_true",
                    help="exclude non-policy files (style guide, metrics dashboards, "
                         "marketing material, ISO 5230) - see NON_POLICY_FILES")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    md_files = sorted(p for p in src.glob("*.md") if p.name.lower() != "readme.md")
    if args.policies_only:
        skipped = [p.name for p in md_files if p.name in NON_POLICY_FILES]
        md_files = [p for p in md_files if p.name not in NON_POLICY_FILES]
        print(f"--policies-only: excluding {len(skipped)} non-policy files: "
              f"{', '.join(sorted(skipped))}\n")
    manifest = []
    total = 0

    for i, path in enumerate(md_files, start=1):
        # Derived from the filename, NOT a sequence number. Sequential ids
        # silently renumber when a document is added or excluded, which
        # invalidates every golden_eval reference without any error - adding
        # External_Stakeholder_Registry.md shifted HACK23-15 from
        # Information_Security_Policy to a different document. Filename-derived
        # ids are stable under curation changes and self-describing.
        doc_id = "HACK23-" + re.sub(r'[^A-Za-z0-9]+', '_', path.stem).strip('_')
        units = parse_document(path.read_text(encoding="utf-8"))
        clauses = to_clauses(units, doc_id)
        if not clauses:
            print(f"  [skip] {path.name}: no clauses extracted")
            continue
        (out / f"{doc_id}.json").write_text(
            json.dumps(clauses, indent=4, ensure_ascii=False), encoding="utf-8", newline="\r\n")
        manifest.append({"doc_id": doc_id, "source_file": path.name,
                         "clauses": len(clauses),
                         "sections": len({c["id"].rsplit("_Sent_", 1)[0] for c in clauses})})
        total += len(clauses)
        print(f"  {doc_id}  {len(clauses):4d} clauses  <- {path.name}")

    (out / "HACK23_manifest.json").write_text(
        json.dumps({
            "source": "https://github.com/Hack23/ISMS-PUBLIC",
            "license": "Apache-2.0",
            "provenance": "real-world published ISMS of Hack23 AB (operating company)",
            "policies_only": args.policies_only,
            "excluded_non_policy": sorted(NON_POLICY_FILES) if args.policies_only else [],
            "documents": manifest,
        }, indent=2, ensure_ascii=False), encoding="utf-8", newline="\r\n")

    print(f"\n{len(manifest)} documents, {total} clauses -> {out}")


if __name__ == "__main__":
    main()
