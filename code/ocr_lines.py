"""Rebuild reading-order lines from OCR word boxes (<dir>/img/p<n>.tsv: x, y, w, h, text) -> <dir>/djpb_text.txt.
The word boxes of the reported run came from the Windows OCR engine on the 37 page images of the scanned PDF.
Words are grouped into a line when their vertical centres are within 0.6 x median word height, then sorted by x.
Page-number lines ('-12-') are dropped; '[Halaman PDF n]' marks each page."""
import re, statistics, sys
from pathlib import Path
D = Path(sys.argv[1])
pages = []
for f in sorted(D.glob("img/p*.tsv"), key=lambda p: int(p.stem[1:])):
    words = []
    for l in f.read_text(encoding="utf-8-sig").splitlines():
        p = l.split("\t")
        if len(p) == 5:
            x, y, w, h = map(int, p[:4]); words.append((y + h / 2, x, h, p[4]))
    if not words:
        pages.append(f"[Halaman PDF {f.stem[1:]}]"); continue
    tol = 0.6 * statistics.median(w[2] for w in words)
    words.sort()
    lines, cur = [], [words[0]]
    for wd in words[1:]:
        if abs(wd[0] - statistics.mean(c[0] for c in cur)) <= tol:
            cur.append(wd)
        else:
            lines.append(cur); cur = [wd]
    lines.append(cur)
    out = []
    for ln in lines:
        t = " ".join(w[3] for w in sorted(ln, key=lambda w: w[1]))
        if re.fullmatch(r"-\s*\d+\s*-", t):
            continue
        out.append(t)
    pages.append(f"[Halaman PDF {f.stem[1:]}]\n" + "\n".join(out))
text = "\n\n".join(pages) + "\n"
(D / "djpb_text.txt").write_text(text, encoding="utf-8", newline="\r\n")
print(len(pages), "pages", len(text), "chars")
