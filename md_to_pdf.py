#!/usr/bin/env python3
"""Convert a Markdown file to PDF using markdown + weasyprint."""
import sys
from pathlib import Path

import markdown
from weasyprint import HTML, CSS

def main():
    md_path = Path(__file__).parent / "Milestone1_First_Baseline_Report.md"
    if len(sys.argv) > 1:
        md_path = Path(sys.argv[1])
    pdf_path = md_path.with_suffix(".pdf")

    text = md_path.read_text(encoding="utf-8")
    html_content = markdown.markdown(text, extensions=["extra", "tables"])
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{md_path.stem}</title>
  <style>
    @page {{ size: A4; margin: 0.5in; }}
    body {{ font-family: Georgia, serif; margin: 0; line-height: 1.5; color: #222; width: 100%; }}
    h1 {{ font-size: 1.6em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }}
    h2 {{ font-size: 1.3em; margin-top: 1.2em; }}
    h3 {{ font-size: 1.1em; }}
    table {{ border-collapse: collapse; margin: 1em 0; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    code {{ background: #f5f5f5; padding: 2px 6px; font-size: 0.9em; }}
    pre {{ background: #f5f5f5; padding: 12px; overflow-x: auto; font-size: 0.85em; }}
    hr {{ border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }}
    a {{ color: #1967d2; }}
  </style>
</head>
<body>
{html_content}
</body>
</html>
"""
    HTML(string=html_doc).write_pdf(pdf_path)
    print(f"Wrote {pdf_path}")

if __name__ == "__main__":
    main()
