"""Text of a PDF with a text layer, page by page, as used for the Bappenas guideline.

  python extract_pdf_text.py <pdf> inputs/bappenas_text.txt

Page-number lines ('-18-') are dropped and each page starts with '[Halaman PDF n]'. For the Bappenas guideline
No. 2/2024 the PDF and the resulting text should have the SHA-256 values listed in the README.
"""
import hashlib
import re
import sys
from pathlib import Path

from pypdf import PdfReader

pdf, out = Path(sys.argv[1]), Path(sys.argv[2])
pages = []
for i, p in enumerate(PdfReader(pdf).pages, 1):
    t = (p.extract_text() or "").replace("\r", "")
    t = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", t, flags=re.M)                 # page-number lines like ' -18- '
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    pages.append(f"[Halaman PDF {i}]\n{t}")
text = "\n\n".join(pages) + "\n"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text, encoding="utf-8", newline="\r\n")
print(f"{len(pages)} pages, {len(text)} chars")
print("pdf  sha256", hashlib.sha256(pdf.read_bytes()).hexdigest())
print("text sha256", hashlib.sha256(out.read_bytes()).hexdigest())
