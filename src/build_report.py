"""Render report/e1304439_report.md to report/e1304439.pdf (A4) with headless Google Chrome.

Markdown is converted to HTML with a compact print stylesheet, then printed by Chrome. The
script reports the page count so the 4-page limit can be checked.

Run from the project root: ``python -m src.build_report``.
"""

import re
import subprocess
from pathlib import Path

import markdown

REPORT_DIR = Path("report")
SOURCE = REPORT_DIR / "e1304439_report.md"
HTML = REPORT_DIR / "e1304439_report.html"
PDF = REPORT_DIR / "e1304439.pdf"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

STYLE = """
@page { size: A4; margin: 14mm 15mm 14mm 15mm; }
body { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 9.1pt; line-height: 1.32;
       color: #0b0b0b; margin: 0; }
h1 { font-size: 13.5pt; margin: 0 0 2pt 0; }
h2 { font-size: 11pt; margin: 8pt 0 2pt 0; border-bottom: 0.6pt solid #c9c8c2; padding-bottom: 1pt; }
h3 { font-size: 9.8pt; margin: 6pt 0 1pt 0; }
p { margin: 2.5pt 0; text-align: justify; }
ul, ol { margin: 2pt 0 2pt 0; padding-left: 14pt; }
li { margin: 0.5pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 7.9pt; margin: 3pt 0 4pt 0; }
th, td { border-bottom: 0.4pt solid #d6d5cf; padding: 1.4pt 3.5pt; text-align: left; vertical-align: top; }
th { border-bottom: 0.8pt solid #52514e; }
figure { margin: 4pt 0; text-align: center; break-inside: avoid; }
figcaption { font-size: 7.9pt; color: #52514e; text-align: left; margin-top: 1pt; }
img { max-width: 100%; }
a { color: #1f5fa8; text-decoration: none; }
code { font-size: 8.3pt; }
"""


def render():
    body = markdown.markdown(SOURCE.read_text(), extensions=["tables", "sane_lists"])
    HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>e1304439</title>"
                    f"<style>{STYLE}</style></head><body>{body}</body></html>")
    subprocess.run(
        [str(CHROME), "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={PDF.resolve()}", HTML.resolve().as_uri()],
        check=True, capture_output=True,
    )
    pages = len(re.findall(rb"/Type\s*/Page[^s]", PDF.read_bytes()))
    return pages


def main():
    pages = render()
    print(f"Wrote {PDF} ({pages} pages)")
    if pages > 4:
        print("WARNING: over the 4-page limit")


if __name__ == "__main__":
    main()
