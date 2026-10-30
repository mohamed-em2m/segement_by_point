#!/usr/bin/env python
"""Render ``report/technical_report.md`` to HTML and PDF.

The markdown is converted with pandoc (GitHub-flavoured markdown, embedded
images, syntax-free code blocks) and the PDF is produced with WeasyPrint. The
script refuses to build while the report still contains an ``AUTO_*``
placeholder, so a half-written report cannot be exported by accident.

Usage::

    uv run python scripts/build_report.py
    uv run python scripts/build_report.py --html-only
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

CSS = """
@page {
  size: A4;
  margin: 18mm 16mm 20mm 16mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-size: 9pt;
    color: #666;
  }
}
body {
  font-family: "DejaVu Serif", Georgia, serif;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #1a1a1a;
}
h1, h2, h3, h4 {
  font-family: "DejaVu Sans", Helvetica, Arial, sans-serif;
  color: #10314f;
  line-height: 1.2;
}
h1 { font-size: 20pt; margin: 0 0 4pt 0; }
h2 { font-size: 14pt; margin-top: 18pt; border-bottom: 1px solid #d5dde5; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin-top: 14pt; }
p { text-align: justify; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 10pt 0;
  font-size: 9pt;
  font-family: "DejaVu Sans", Helvetica, sans-serif;
}
th, td { border: 1px solid #c9d3dd; padding: 4pt 6pt; text-align: left; }
th { background: #eef2f6; }
tr:nth-child(even) td { background: #fafbfc; }
code, pre {
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 8.8pt;
  background: #f4f6f8;
}
pre { padding: 8pt; border-left: 3px solid #9fb3c8; overflow-x: auto; }
img { max-width: 100%; display: block; margin: 10pt auto; }
figure { margin: 12pt 0; text-align: center; }
figcaption { font-size: 9pt; color: #555; }
blockquote { border-left: 3px solid #9fb3c8; margin-left: 0; padding-left: 10pt; color: #333; }
"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def check_placeholders(text: str) -> list[str]:
    return re.findall(r"<!--\s*AUTO_[A-Z_]+\s*-->", text)


def pandoc(markdown: Path, output: Path) -> None:
    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc is required to build the report")
    subprocess.run(
        [
            "pandoc",
            str(markdown),
            "--from",
            "gfm+tex_math_dollars+pipe_tables",
            "--to",
            "html5",
            "--standalone",
            "--metadata",
            f"title={markdown.stem.replace('_', ' ').title()}",
            "--resource-path",
            str(markdown.parent),
            "--output",
            str(output),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="report/technical_report.md")
    parser.add_argument("--html-only", action="store_true")
    parser.add_argument("--allow-placeholders", action="store_true")
    args = parser.parse_args()

    report = Path(args.report)
    if not report.is_file():
        raise SystemExit(f"report not found: {report}")
    text = report.read_text(encoding="utf-8")
    missing = check_placeholders(text)
    if missing and not args.allow_placeholders:
        raise SystemExit(f"report still contains placeholders: {', '.join(missing)}")

    out_dir = report.parent
    title = text.splitlines()[0].lstrip("# ").strip() or "Technical report"

    body_html = out_dir / "technical_report.body.html"
    pandoc(report, body_html)
    body = body_html.read_text(encoding="utf-8")
    # Keep only the body of pandoc's standalone document.
    body = re.sub(r"(?s)^.*?<body[^>]*>", "", body)
    body = re.sub(r"(?s)</body>.*$", "", body)
    html_path = out_dir / "technical_report.html"
    html_path.write_text(HTML_TEMPLATE.format(title=title, css=CSS, body=body), encoding="utf-8")
    body_html.unlink()
    print(f"wrote {html_path}")

    if args.html_only:
        return 0

    try:
        from weasyprint import HTML
    except ImportError:
        raise SystemExit(
            "weasyprint is not installed; run `uv pip install --group docs` or pass --html-only"
        ) from None
    pdf_path = out_dir / "technical_report.pdf"
    HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    print(f"wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
