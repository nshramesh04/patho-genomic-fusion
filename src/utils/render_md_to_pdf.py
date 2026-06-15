"""
Render reports/TECHNICAL_REPORT.md → reports/TECHNICAL_REPORT.pdf via WeasyPrint.

Style: MICCAI / Springer LNCS approximation.
  - A4, single column, 10pt Times New Roman
  - Section headings bold; subsections bold-italic (numbered in Markdown)
  - Abstract indented with bold keyword, separated by rules
  - Tables: booktabs-style (horizontal rules only, no shading)
  - Figures: full-colour, centred, caption below in 9pt italic
  - Figure callouts of the form:
        *Figure N: Title. Description. Source: reports/figures/FILE.png.*
    are replaced with the embedded colour image + caption.
  - Running headers suppressed on page 1; page numbers centred in footer

Usage
-----
    python src/utils/render_md_to_pdf.py
"""

import base64
import os
import re
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT    = Path(__file__).resolve().parents[2]
MD_IN   = ROOT / "reports" / "TECHNICAL_REPORT.md"
PDF_OUT = ROOT / "reports" / "TECHNICAL_REPORT.pdf"
FIG_DIR = ROOT / "reports" / "figures"

# ── CSS — MICCAI / Springer LNCS approximation ────────────────────────────────
CSS = """
/* ── Page geometry ─────────────────────────────────── */
@page {
    size: A4;
    margin: 27mm 28mm 30mm 28mm;

    @top-left {
        content: "Multimodal Pathogenomic Fusion";
        font-family: 'Times New Roman', Times, serif;
        font-size: 8pt;
        color: #333;
        border-bottom: 0.4pt solid #aaa;
        padding-bottom: 2mm;
    }
    @top-right {
        content: "Architecture & Interpretability";
        font-family: 'Times New Roman', Times, serif;
        font-size: 8pt;
        color: #333;
        border-bottom: 0.4pt solid #aaa;
        padding-bottom: 2mm;
    }
    @bottom-center {
        content: counter(page);
        font-family: 'Times New Roman', Times, serif;
        font-size: 9pt;
        color: #333;
    }
}
@page :first {
    @top-left  { content: normal; border: none; }
    @top-right { content: normal; border: none; }
}

/* ── Reset ──────────────────────────────────────────── */
* { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Body ───────────────────────────────────────────── */
body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 10pt;
    color: #000;
    background: #fff;
    line-height: 1.5;
    text-align: justify;
    hyphens: auto;
}

/* ── Title ──────────────────────────────────────────── */
h1 {
    font-size: 14pt;
    font-weight: bold;
    color: #000;
    text-align: center;
    line-height: 1.25;
    margin-bottom: 4pt;
    padding-bottom: 0;
    border: none;
}

/* Subtitle / affiliation paragraphs directly after h1 */
h1 + p,
h1 + p + p {
    text-align: center;
    font-size: 10pt;
    margin-bottom: 3pt;
}

/* ── Horizontal rule (used as section divider after title block) ─── */
hr {
    border: none;
    border-top: 0.5pt solid #000;
    margin: 8pt 0;
}

/* ── Abstract block ─────────────────────────────────── */
/* Detected as the first paragraph whose text begins with "Abstract." */
p.abstract {
    font-size: 9pt;
    margin: 0 10mm 8pt 10mm;
    line-height: 1.4;
    text-align: justify;
}

/* ── Section headings (LNCS style) ─────────────────── */
h2 {
    font-size: 10pt;
    font-weight: bold;
    color: #000;
    margin-top: 14pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
    border: none;
    text-transform: none;
    letter-spacing: 0;
}
h3 {
    font-size: 10pt;
    font-weight: bold;
    font-style: italic;
    color: #000;
    margin-top: 10pt;
    margin-bottom: 3pt;
    page-break-after: avoid;
}
h4 {
    font-size: 10pt;
    font-weight: bold;
    color: #000;
    margin-top: 8pt;
    margin-bottom: 2pt;
    page-break-after: avoid;
}

/* ── Body text ──────────────────────────────────────── */
p   { margin-bottom: 5pt; orphans: 3; widows: 3; }
ul, ol { padding-left: 14pt; margin-bottom: 5pt; }
li  { margin-bottom: 2pt; }
strong { font-weight: bold; }
em     { font-style: italic; }

/* ── Code ───────────────────────────────────────────── */
code {
    font-family: 'Courier New', Courier, monospace;
    font-size: 8pt;
    background: #f5f5f5;
    padding: 0 2pt;
}
pre {
    font-family: 'Courier New', Courier, monospace;
    font-size: 8pt;
    background: #f7f7f7;
    border-left: 2pt solid #ccc;
    padding: 5pt 8pt;
    margin: 6pt 0;
    white-space: pre;
    line-height: 1.35;
    page-break-inside: avoid;
}
pre code { background: none; padding: 0; }

/* ── Tables — booktabs (rules only, no row shading) ─── */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9pt;
    margin: 8pt 0 10pt 0;
    page-break-inside: avoid;
}
/* Caption above table is handled as a <p> immediately before the table */
thead tr {
    border-top: 1pt solid #000;
    border-bottom: 0.5pt solid #000;
}
thead th {
    padding: 3pt 5pt;
    font-weight: bold;
    text-align: left;
}
tbody td {
    padding: 2pt 5pt;
    vertical-align: top;
}
tbody tr:last-child {
    border-bottom: 1pt solid #000;
}

/* ── Figures ────────────────────────────────────────── */
figure {
    margin: 12pt 0;
    page-break-inside: avoid;
    text-align: center;
}
figure img {
    max-width: 92%;
    height: auto;
    display: block;
    margin: 0 auto;
}
figcaption {
    font-size: 9pt;
    font-style: italic;
    margin-top: 4pt;
    text-align: center;
    line-height: 1.35;
    color: #000;
}

/* ── Blockquotes ────────────────────────────────────── */
blockquote {
    border-left: 1.5pt solid #999;
    margin: 6pt 0 6pt 4pt;
    padding: 3pt 10pt;
    font-size: 9.5pt;
    color: #111;
}
blockquote p { margin-bottom: 2pt; }
"""


# ── Figure embedding ──────────────────────────────────────────────────────────

def _color_b64(fig_path: Path) -> str:
    """Return a base64 PNG data-URI for a figure (full colour, no conversion)."""
    with open(fig_path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _embed_figures(html: str) -> str:
    """
    Replace every figure-callout paragraph:
        <p><em>Figure N: Title. Description. Source: reports/figures/FILE.png.</em></p>
    with a <figure> element containing the full-colour embedded image and caption.
    """
    pattern = re.compile(
        r'<p><em>\s*'
        r'(Figure\s+(\d+):\s*'
        r'(.*?)'
        r'Source:\s*reports/figures/([^\s<]+\.png)'
        r'[^<]*?)'
        r'(?:<br\s*/?>)?\s*</em></p>',
        re.DOTALL | re.IGNORECASE,
    )

    def _replace(m: re.Match) -> str:
        num       = m.group(2)
        body_text = m.group(3).strip().rstrip(". ")
        filename  = m.group(4)
        fig_path  = FIG_DIR / filename

        parts   = body_text.split(". ", 1)
        title   = parts[0].strip()
        detail  = parts[1].strip() if len(parts) > 1 else ""
        caption = f"Fig. {num}. {title}." + (f" {detail}." if detail else "")

        if not fig_path.exists():
            return f'<p><em>[Fig. {num}: {title} — file not found: {filename}]</em></p>'

        src = _color_b64(fig_path)
        return (
            f'<figure>'
            f'<img src="{src}" alt="Fig. {num}: {title}"/>'
            f'<figcaption>{caption}</figcaption>'
            f'</figure>'
        )

    return pattern.sub(_replace, html)


def _mark_abstract(html: str) -> str:
    """
    Wrap the Abstract paragraph in <p class="abstract"> so it receives
    the LNCS-style indented formatting.
    """
    return re.sub(
        r'<p>(<strong>Abstract\.</strong>)',
        r'<p class="abstract">\1',
        html, count=1,
    )


def convert(md_path: Path, pdf_path: Path) -> None:
    source = md_path.read_text(encoding="utf-8").replace("\r\n", "\n")

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "nl2br", "attr_list", "sane_lists"],
    )
    body_html = md.convert(source)
    body_html = _embed_figures(body_html)
    body_html = _mark_abstract(body_html)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Multimodal Pathogenomic Fusion — Technical Report</title>
  <style>{CSS}</style>
</head>
<body>
{body_html}
</body>
</html>"""

    os.makedirs(pdf_path.parent, exist_ok=True)
    HTML(string=full_html, base_url=str(ROOT)).write_pdf(str(pdf_path))
    kb = pdf_path.stat().st_size / 1024
    print(f"PDF saved → {pdf_path}  ({kb:.0f} KB)")


if __name__ == "__main__":
    convert(MD_IN, PDF_OUT)
