"""
render_ieee_pdf.py
Convert reports/TECHNICAL_REPORT.md → reports/TECHNICAL_REPORT_IEEE.pdf.

Layout
------
- Two-column IEEE-style, US Letter page (8.5 × 11 in)
- 11pt Times New Roman body font
- Standard IEEE margins: 19 mm top/bottom, 16 mm left/right
- Title, subtitle, abstract, and keywords span both columns
- Page numbers centred in footer

Figure embedding
----------------
[DIAGRAM: ...] placeholders in the Markdown are replaced with
base64-encoded images from reports/figures/, matched by keyword
heuristics against the placeholder description text.

Mapping (first match wins, case-insensitive):
  "flowchart" / "q = genomic"          → architecture.png
  "line chart" / "training epoch"       → training_curves.png
  "top-1%-mass" / "pr status (x-axis"  → attention_distributions.png
  "grouped bar chart" / "regularised"  → tutorial_attention_metrics.png
  "attention operations" / "n=39,052"  → attention_distributions.png

Usage
-----
    python src/utils/render_ieee_pdf.py
"""

import base64
import os
import re
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT    = Path(__file__).resolve().parents[2]
MD_IN   = ROOT / "reports" / "TECHNICAL_REPORT.md"
PDF_OUT = ROOT / "reports" / "TECHNICAL_REPORT_IEEE.pdf"
FIG_DIR = ROOT / "reports" / "figures"

# ── Figure keyword routing ─────────────────────────────────────────────────────
# Each tuple: (list-of-lowercase-keywords, figure-filename).
# ANY keyword match → use that figure. First matching rule wins.
FIGURE_MAP = [
    (
        ["architecture flowchart", "flowchart", "q = genomic", "(q=genes)"],
        "architecture.png",
    ),
    (
        ["line chart", "training epoch", "epoch (x-axis", "val auc (y-axis"],
        "training_curves.png",
    ),
    (
        ["top-1%-mass", "pr status (x-axis", "pr+, pr−", "top-1% concentration"],
        "attention_distributions.png",
    ),
    (
        ["grouped bar chart", "cross-attention regularised", "late fusion regularised"],
        "tutorial_attention_metrics.png",
    ),
    (
        ["attention operations", "operations at n=", "architecture vs. attention"],
        "attention_distributions.png",
    ),
]

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
/* ── Page geometry ─────────────────────────────────────────────────────────── */
@page {
    size: letter;                            /* 8.5 × 11 in — IEEE standard */
    margin: 19mm 16mm 19mm 16mm;

    @bottom-center {
        content: counter(page);
        font-family: 'Times New Roman', Times, serif;
        font-size: 9pt;
        color: #333;
        padding-top: 3mm;
    }
}

/* ── Reset ──────────────────────────────────────────────────────────────────── */
* { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Two-column body ─────────────────────────────────────────────────────────  */
body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 11pt;
    color: #000;
    line-height: 1.42;
    column-count: 2;
    column-gap: 5.5mm;
    text-align: justify;
    hyphens: auto;
}

/* ── Title block — spans both columns ──────────────────────────────────────── */
.title-block {
    column-span: all;
    text-align: center;
    padding-bottom: 7pt;
}
.title-block h1 {
    font-size: 20pt;
    font-weight: bold;
    line-height: 1.18;
    margin-bottom: 5pt;
    text-align: center;
    text-transform: none;
    border-bottom: none;
}
.title-block p { text-align: center; }
.title-block .subtitle {
    font-size: 10pt;
    font-weight: bold;
    margin-bottom: 2pt;
}
.title-block .affiliation {
    font-size: 10pt;
    font-style: italic;
    margin-bottom: 0;
}

/* ── Divider below title block ──────────────────────────────────────────────── */
.col-rule {
    column-span: all;
    border: none;
    border-top: 1pt solid #000;
    margin: 0 0 0 0;
}

/* ── Abstract — spans both columns, indented ────────────────────────────────── */
.abstract-block {
    column-span: all;
    margin: 6pt 16mm 0 16mm;
    padding: 5pt 0;
    font-size: 9pt;
    line-height: 1.35;
    text-align: justify;
    border-top: 0.5pt solid #000;
    border-bottom: 0.5pt solid #000;
}
.abstract-block p { font-size: 9pt; margin-bottom: 3pt; }
.abstract-label { font-weight: bold; font-variant: small-caps; }

/* ── Column-span divider after abstract ─────────────────────────────────────── */
.col-rule-light {
    column-span: all;
    border: none;
    border-top: 0.5pt solid #ccc;
    margin: 4pt 0 0 0;
}

/* ── Section headings ───────────────────────────────────────────────────────── */
h2 {
    font-size: 11pt;
    font-weight: bold;
    text-align: center;
    margin-top: 9pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
}
h3 {
    font-size: 11pt;
    font-weight: bold;
    font-style: italic;
    margin-top: 7pt;
    margin-bottom: 3pt;
    page-break-after: avoid;
}
h4 {
    font-size: 11pt;
    font-weight: bold;
    margin-top: 5pt;
    margin-bottom: 2pt;
    page-break-after: avoid;
}

/* ── Body text ──────────────────────────────────────────────────────────────── */
p { margin-bottom: 5pt; orphans: 3; widows: 3; }
strong { font-weight: bold; }
em     { font-style: italic; }
ul, ol { padding-left: 13pt; margin-bottom: 4pt; }
li { margin-bottom: 2pt; }

/* ── Inline code & preformatted (equations, data) ───────────────────────────── */
code {
    font-family: 'Courier New', Courier, monospace;
    font-size: 8pt;
    background: #f4f4f4;
    padding: 0 2pt;
}
pre {
    font-family: 'Courier New', Courier, monospace;
    font-size: 7.5pt;
    background: #f8f8f8;
    border: 0.5pt solid #ccc;
    padding: 4pt 7pt;
    margin: 5pt 0;
    white-space: pre;
    line-height: 1.3;
    page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 7.5pt; }

/* ── Tables — IEEE booktabs style ───────────────────────────────────────────── */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 8pt;
    margin: 5pt 0 7pt 0;
    page-break-inside: avoid;
}
thead tr {
    border-top: 1pt solid #000;
    border-bottom: 0.5pt solid #000;
}
thead th {
    padding: 2.5pt 4pt;
    font-weight: bold;
    text-align: left;
    font-size: 8pt;
}
tbody td  { padding: 2pt 4pt; vertical-align: top; font-size: 8pt; }
tfoot tr,
tbody tr:last-child { border-bottom: 1pt solid #000; }

/* ── Figures ─────────────────────────────────────────────────────────────────── */
figure {
    margin: 7pt 0;
    page-break-inside: avoid;
    text-align: center;
}
figure img   { max-width: 100%; height: auto; }
figcaption {
    font-size: 8pt;
    font-style: italic;
    margin-top: 3pt;
    text-align: center;
    line-height: 1.3;
}
.fig-placeholder {
    border: 0.75pt dashed #bbb;
    padding: 12pt 6pt;
    text-align: center;
    font-size: 8pt;
    color: #777;
    font-style: italic;
}

/* ── Horizontal rules ───────────────────────────────────────────────────────── */
hr { border: none; border-top: 0.5pt solid #ccc; margin: 6pt 0; }

/* ── Blockquotes ────────────────────────────────────────────────────────────── */
blockquote {
    margin: 5pt 8pt;
    padding-left: 6pt;
    border-left: 1.5pt solid #555;
    font-size: 10pt;
}
blockquote p { color: #222; }
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_b64(filename: str) -> str:
    """Return a base64 PNG data-URI for a figure file, or '' if missing."""
    path = FIG_DIR / filename
    if not path.exists():
        return ""
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _match_figure(diagram_text: str) -> str:
    """Return base64 data-URI for the best-matching figure; '' if none match."""
    lower = diagram_text.lower()
    for keywords, fname in FIGURE_MAP:
        if any(kw in lower for kw in keywords):
            return _load_b64(fname)
    return ""


def _short_caption(diagram_text: str) -> str:
    """Extract a clean caption from the DIAGRAM description."""
    text = re.sub(r'\s+', ' ', diagram_text)
    text = text.split('.')[0]
    text = re.sub(r'^Insert\s+', '', text, flags=re.IGNORECASE).strip()
    return text


def _replace_diagrams(html: str) -> str:
    """
    Replace every <p>[DIAGRAM: ...]</p> block in the rendered HTML with a
    <figure> element containing the matched base64-embedded image.
    Unmatched placeholders render as dashed placeholder boxes.
    """
    # nl2br may add a trailing <br/> before </p>; handle both forms
    pattern = re.compile(
        r'<p>\[DIAGRAM:\s*(.*?)\](?:<br\s*/?>)?\s*</p>',
        re.DOTALL | re.IGNORECASE,
    )
    counter = [0]

    def _sub(m: re.Match) -> str:
        counter[0] += 1
        n   = counter[0]
        txt = m.group(1).strip()
        src = _match_figure(txt)
        cap = _short_caption(txt)

        if src:
            return (
                f'<figure>'
                f'<img src="{src}" alt="Fig. {n}"/>'
                f'<figcaption>Fig. {n}. {cap}.</figcaption>'
                f'</figure>'
            )
        return (
            f'<figure>'
            f'<div class="fig-placeholder">[Fig. {n} — {cap[:110]}]</div>'
            f'<figcaption>Fig. {n}. {cap}.</figcaption>'
            f'</figure>'
        )

    return pattern.sub(_sub, html)


def _restructure(html: str) -> str:
    """
    Post-process the rendered HTML:

    1. Wrap everything before the first <h2> in <div class="title-block">
       so the title / subtitle / abstract span both columns.
    2. Insert a full-width rule between the title block and body columns.
    3. Apply .abstract-block class to the Abstract paragraph.
    4. Apply .subtitle / .affiliation classes to the two descriptor paragraphs.
    """
    # ── Split at the first section heading ──────────────────────────────────
    m = re.search(r'<h2\b', html)
    if m:
        header_html = html[: m.start()]
        body_html   = html[m.start() :]
    else:
        header_html, body_html = html, ""

    # ── Abstract paragraph ───────────────────────────────────────────────────
    header_html = re.sub(
        r'<p>\s*<strong>(Abstract)</strong>\s*—\s*',
        (r'<div class="abstract-block">'
         r'<p><span class="abstract-label">\1</span> — '),
        header_html, count=1,
    )
    # Close the abstract div before the Keywords paragraph
    header_html = re.sub(
        r'(<p>\s*<strong>Keywords</strong>)',
        r'</p></div>\1',
        header_html, count=1,
    )

    # ── Subtitle and affiliation ─────────────────────────────────────────────
    header_html = re.sub(
        r'<p><strong>(Technical Report[^<]*)</strong></p>',
        r'<p class="subtitle"><strong>\1</strong></p>',
        header_html, count=1,
    )
    header_html = re.sub(
        r'<p><em>([^<]+)</em></p>',
        r'<p class="affiliation"><em>\1</em></p>',
        header_html, count=1,
    )

    # ── Wrap header content in .title-block ─────────────────────────────────
    wrapped = (
        f'<div class="title-block">\n{header_html}\n</div>'
        f'<hr class="col-rule"/>\n'
        + body_html
    )

    return wrapped


# ── Main conversion ────────────────────────────────────────────────────────────

def convert(md_path: Path, pdf_path: Path) -> None:
    source = md_path.read_text(encoding="utf-8").replace("\r\n", "\n")

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "nl2br", "attr_list", "sane_lists"],
    )
    body_html = md.convert(source)
    body_html = _replace_diagrams(body_html)
    body_html = _restructure(body_html)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Multimodal Pathogenomic Fusion — IEEE Technical Report</title>
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
