"""Build reports/final_report.pdf from reports/final_report.md (Markdown -> HTML -> PDF via headless Chrome).

    python scripts/build_report.py                 # uses the REPO_URL placeholder in the markdown
    REPO_URL=https://github.com/<user>/<repo> python scripts/build_report.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "reports" / "final_report.md"
HTML = ROOT / "reports" / "final_report.html"
PDF = ROOT / "reports" / "final_report.pdf"
CHROME = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.38; color: #111; max-width: 180mm; margin: 0 auto; }
h1 { font-size: 20pt; margin: 0 0 4pt 0; }
h2 { font-size: 14pt; margin: 16pt 0 6pt 0; border-bottom: 1px solid #bbb; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt 0; }
p { margin: 5pt 0; text-align: justify; }
ul, ol { margin: 4pt 0 4pt 18pt; padding: 0; } li { margin: 2pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 9.2pt; margin: 6pt 0; page-break-inside: avoid; table-layout: auto; }
.stages td:nth-child(1) { width: 3%; } .stages td:nth-child(2) { width: 16%; } .stages td:nth-child(3) { width: 53%; } .stages td:nth-child(4) { width: 28%; }
td code { word-break: break-all; white-space: normal; }
th, td { border: 1px solid #c9c9c9; padding: 3pt 5pt; vertical-align: top; } th { background: #f1f3f5; text-align: left; }
code { font-family: Menlo, 'DejaVu Sans Mono', monospace; font-size: 8.8pt; background: #f3f4f6; padding: 0 2px; border-radius: 2px; }
pre { background: #f3f4f6; padding: 6pt 8pt; font-size: 8.5pt; overflow-x: auto; border-radius: 3px; white-space: pre-wrap; }
img { max-width: 100%; display: block; margin: 5pt auto; border: 1px solid #ddd; }
img[alt=architecture] { max-width: 88%; } img[alt=metrics] { max-width: 82%; }
figure { margin: 8pt 0; page-break-inside: avoid; } figcaption { font-size: 9pt; color: #444; text-align: center; margin-top: 3pt; }
.meta { color: #444; font-size: 10pt; margin-bottom: 10pt; }
.pagebreak { page-break-before: always; }
a { color: #1d4ed8; text-decoration: none; }
"""


def main() -> None:
    text = MD.read_text(encoding="utf-8")
    repo = os.environ.get("REPO_URL", "")
    if repo:
        text = text.replace("<REPO_URL>", repo)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "attr_list", "md_in_html", "toc"])
    HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Final report</title><style>{CSS}</style></head>"
                    f"<body>{body}</body></html>", encoding="utf-8")
    if not CHROME:
        sys.exit("Chrome/Chromium is needed to print the PDF; the HTML was written instead: " + str(HTML))
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True, timeout=180)
    print(f"wrote {PDF} ({PDF.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
