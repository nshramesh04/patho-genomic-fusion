"""
Render reports/PR_Status_Fusion_CaseStudy.md → reports/PR_Status_Fusion_CaseStudy.pdf

Pre-processing pipeline (applied before markdown conversion):
  1. YAML frontmatter stripped; key metrics rendered as a sub-title block
  2. Display math  $$...$$  → Unicode approximation in a centred italic block
  3. Inline math   $...$   → Unicode approximation wrapped in <em>
  4. Obsidian callouts  > [!type] Title  → styled <div class="callout"> elements
  5. Wikilink images  ![[file.png]] + *caption*  → <figure> with embedded base64 image

Usage
-----
    python src/utils/render_case_study_pdf.py
"""

import base64
import os
import re
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT    = Path(__file__).resolve().parents[2]
MD_IN   = ROOT / "reports" / "PR_Status_Fusion_CaseStudy.md"
PDF_OUT = ROOT / "reports" / "PR_Status_Fusion_CaseStudy_8.pdf"
FIG_DIR = ROOT / "reports" / "figures"

CALLOUT_META: dict[str, tuple[str, str]] = {
    "warning":  ("Warning",      "#CC3311"),
    "info":     ("Info",         "#0077BB"),
    "note":     ("Note",         "#444444"),
    "tip":      ("Key Takeaway", "#229944"),
    "abstract": ("Context",      "#7755BB"),
    "example":  ("Question",     "#EE7733"),
    "important":("Important",    "#CC3311"),
}

# ── CSS — MICCAI / Springer LNCS approximation ────────────────────────────────
CSS = """
@page {
    size: A4;
    margin: 27mm 28mm 30mm 28mm;
    @top-left  { content: normal; }
    @top-right { content: normal; }
    @bottom-center {
        content: counter(page);
        font-family: 'Times New Roman', Times, serif;
        font-size: 9pt; color: #333;
    }
}
@page :first {
    @bottom-center { content: normal; }
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 10pt; color: #000; background: #fff;
    line-height: 1.45; text-align: justify; hyphens: auto;
}
h1 {
    font-size: 14pt; font-weight: bold; color: #000;
    text-align: center; line-height: 1.25;
    margin-bottom: 4pt; padding-bottom: 0; border: none;
}
h1 + p, h1 + p + p {
    text-align: center; font-size: 10pt; margin-bottom: 3pt;
}
hr { border: none; border-top: 0.5pt solid #000; margin: 6pt 0; }
h2 {
    font-size: 10pt; font-weight: bold; color: #000;
    margin-top: 11pt; margin-bottom: 3pt;
    page-break-after: avoid; border: none;
}
h3 {
    font-size: 10pt; font-weight: bold; font-style: italic; color: #000;
    margin-top: 8pt; margin-bottom: 2pt; page-break-after: avoid;
}
h4 {
    font-size: 10pt; font-weight: bold; color: #000;
    margin-top: 6pt; margin-bottom: 2pt; page-break-after: avoid;
}
p   { margin-bottom: 4pt; orphans: 3; widows: 3; }
ul, ol { padding-left: 14pt; margin-bottom: 4pt; }
li  { margin-bottom: 2pt; }
strong { font-weight: bold; }
em     { font-style: italic; }
code {
    font-family: 'Courier New', Courier, monospace;
    font-size: 8pt; background: #f5f5f5; padding: 0 2pt;
}
pre {
    font-family: 'Courier New', Courier, monospace;
    font-size: 8pt; background: #f7f7f7;
    border-left: 2pt solid #ccc;
    padding: 5pt 8pt; margin: 6pt 0;
    white-space: pre; line-height: 1.35; page-break-inside: avoid;
}
pre code { background: none; padding: 0; }
table {
    width: 100%; border-collapse: collapse;
    font-size: 9pt; margin: 8pt 0 10pt 0; page-break-inside: avoid;
}
thead tr { border-top: 1pt solid #000; border-bottom: 0.5pt solid #000; }
thead th { padding: 3pt 5pt; font-weight: bold; text-align: left; }
tbody td { padding: 2pt 5pt; vertical-align: top; }
tbody tr:last-child { border-bottom: 1pt solid #000; }
figure { margin: 6pt 0; page-break-inside: avoid; text-align: center; }
figure img { max-width: 92%; height: auto; display: block; margin: 0 auto; }
figcaption {
    font-size: 9pt; font-style: italic; margin-top: 4pt;
    text-align: center; line-height: 1.35; color: #000;
}
/* Figure 1 (architecture): margins reduced 10% — 6pt → 5.4pt */
figure.fig-arch { margin: 5pt 0; }
figure.fig-arch img { max-width: 83%; }
/* ── Figure 2 two-column flexbox layout ─────────────────────────────── */
.fig-wrap-table {
    display: flex;
    flex-direction: row;
    align-items: flex-start;
    width: 100%;
    gap: 10pt;
    margin: 4pt 0 6pt 0;
}
.fig-wrap-text {
    flex: 0 0 57%;
    max-width: 57%;
}
.fig-wrap-text p { margin-bottom: 4pt; }
.fig-wrap-fig {
    flex: 0 0 41%;
    max-width: 41%;
}
.fig-wrap-fig figure {
    float: none;
    width: 100%;
    margin: 0 0 3pt 0;
    text-align: center;
}
.fig-wrap-fig figure img  { max-width: 100%; display: block; margin: 0 auto; }
.fig-wrap-fig figcaption  {
    font-size: 7.8pt; text-align: justify;
    margin-top: 3pt; line-height: 1.3; color: #000;
}
blockquote {
    border-left: 1.5pt solid #999;
    margin: 6pt 0 6pt 4pt; padding: 3pt 10pt;
    font-size: 9.5pt; color: #111;
}
blockquote p { margin-bottom: 2pt; }
/* Metadata sub-title block (rendered from YAML frontmatter) */
p.meta-block {
    font-size: 9pt; color: #444; text-align: center;
    margin: 0 0 6pt 0; font-style: italic;
    border-bottom: 0.5pt solid #ccc; padding-bottom: 6pt;
}
/* Callout blocks (converted from Obsidian > [!type] syntax) */
div.callout {
    margin: 5pt 0; padding: 4pt 8pt;
    font-size: 9pt; line-height: 1.35;
    background: #fafafa; page-break-inside: avoid;
}
span.callout-label {
    font-weight: bold; font-size: 8.5pt;
    text-transform: uppercase; letter-spacing: 0.3pt;
}
/* Display math blocks (converted from $$...$$) */
p.display-math {
    font-style: italic; text-align: center;
    margin: 8pt 20pt; font-size: 9.5pt; line-height: 1.6; color: #111;
}
"""

# ── LaTeX → Unicode substitution table ───────────────────────────────────────
_LATEX_SUBS: list[tuple[str, str]] = [
    (r'\\text\{([^{}]+)\}',             r'\1'),
    (r'\\mathbb\{R\}',                  'ℝ'),
    (r'\\mathbb\{Z\}',                  'ℤ'),
    (r'\\mathbb\{N\}',                  'ℕ'),
    (r'\\operatorname\{([^{}]+)\}',     r'\1'),
    (r'\\mathrm\{([^{}]+)\}',           r'\1'),
    (r'\\displaystyle',                 ''),
    (r'\\left\(',                       '('),
    (r'\\right\)',                      ')'),
    (r'\\left\[',                       '['),
    (r'\\right\]',                      ']'),
    (r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1) / (\2)'),
    (r'\\sqrt\{([^{}]+)\}',             r'√\1'),
    (r'\\sqrt',                         '√'),
    (r'\\lfloor',                       '⌊'),
    (r'\\rfloor',                       '⌋'),
    (r'\\lceil',                        '⌈'),
    (r'\\rceil',                        '⌉'),
    (r'\\sum',                          'Σ'),
    (r'\\prod',                         'Π'),
    (r'\\int',                          '∫'),
    (r'\\max',                          'max'),
    (r'\\min',                          'min'),
    (r'\\log',                          'log'),
    (r'\\exp',                          'exp'),
    (r'\\in\b',                         '∈'),
    (r'\\notin\b',                      '∉'),
    (r'\\approx',                       '≈'),
    (r'\\leq',                          '≤'),
    (r'\\geq',                          '≥'),
    (r'\\neq',                          '≠'),
    (r'\\times',                        '×'),
    (r'\\cdot',                         '·'),
    (r'\\pm',                           '±'),
    (r'\\top',                          'ᵀ'),
    (r'\\infty',                        '∞'),
    (r'\\to\b',                         '→'),
    (r'\\rightarrow',                   '→'),
    (r'\\sigma',                        'σ'),
    (r'\\alpha',                        'α'),
    (r'\\beta',                         'β'),
    (r'\\gamma',                        'γ'),
    (r'\\delta',                        'δ'),
    (r'\\epsilon',                      'ε'),
    (r'\\mu',                           'μ'),
    (r'\\pi\b',                         'π'),
    (r'\\phi',                          'φ'),
    (r'\\Phi',                          'Φ'),
    (r'\\lambda',                       'λ'),
    (r'\\Sigma',                        'Σ'),
    (r'\\[,;:!]',                       ''),
    (r'\\quad',                         '  '),
    (r'\^\{([^{}]*)\}',                 r'^\1'),
    (r'_\{([^{}]*)\}',                  r'_\1'),
    (r'\\%',                            '%'),
    (r'\\\{',                           '{'),
    (r'\\\}',                           '}'),
    (r'\{,\}',                          ','),
    (r'\\[a-zA-Z]+',                    ''),
]


def _latex_to_unicode(s: str) -> str:
    """Apply substitution table repeatedly to resolve nested patterns."""
    for _ in range(4):
        for pat, rep in _LATEX_SUBS:
            s = re.sub(pat, rep, s)
    return s.strip()


# ── Pre-processing stages ─────────────────────────────────────────────────────

def _strip_frontmatter(source: str) -> tuple[dict, str]:
    """Remove YAML frontmatter block and return (metadata_dict, body)."""
    if not source.startswith('---'):
        return {}, source
    end = source.find('\n---', 3)
    if end == -1:
        return {}, source
    fm   = source[3:end].strip()
    body = source[end + 4:].lstrip('\n')
    meta: dict[str, str] = {}
    for line in fm.split('\n'):
        if ':' in line and not line.startswith(' ') and not line.startswith('-'):
            k, _, v = line.partition(':')
            meta[k.strip()] = v.strip().strip('"')
    return meta, body


def _render_meta_header(meta: dict) -> str:
    return ''


def _convert_display_math(source: str) -> str:
    """Replace $$...$$ with a centred italic Unicode paragraph."""
    def _replace(m: re.Match) -> str:
        content = _latex_to_unicode(m.group(1).strip())
        return f'\n\n<p class="display-math">{content}</p>\n\n'
    return re.sub(r'\$\$(.+?)\$\$', _replace, source, flags=re.DOTALL)


def _convert_inline_math(source: str) -> str:
    """Replace $...$ with <em>unicode</em> (display math already removed)."""
    def _replace(m: re.Match) -> str:
        return f'<em>{_latex_to_unicode(m.group(1))}</em>'
    return re.sub(r'\$([^$\n]+?)\$', _replace, source)


def _inline_md(text: str) -> str:
    """Minimal inline markdown → HTML for use inside callout div bodies."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'`([^`]+)`',     r'<code>\1</code>',     text)
    return text


def _convert_callouts(source: str) -> str:
    """
    Convert Obsidian callout blocks to styled HTML divs.

    Input:
        > [!warning] Title text
        > Body line one
        > Body line two

    Output:
        <div class="callout" style="border-left: 3pt solid #CC3311;">
          <span class="callout-label" style="color:#CC3311;">Warning</span>
          — <strong>Title text</strong><br/>Body line one Body line two
        </div>
    """
    lines = source.split('\n')
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = re.match(r'^> \[!(\w+)\](?:\s+(.*))?$', lines[i])
        if m:
            ctype = m.group(1).lower()
            title = (m.group(2) or '').strip()
            label, color = CALLOUT_META.get(ctype, ('Note', '#555555'))

            body_parts: list[str] = []
            i += 1
            while i < len(lines) and re.match(r'^>[ ]?', lines[i]):
                body_parts.append(re.sub(r'^>[ ]?', '', lines[i]))
                i += 1

            body       = _inline_md(' '.join(body_parts).strip())
            title_html = f' — <strong>{_inline_md(title)}</strong>' if title else ''
            out.append(
                f'<div class="callout" style="border-left: 3pt solid {color};">'
                f'<span class="callout-label" style="color:{color};">{label}</span>'
                f'{title_html}<br/>{body}</div>'
            )
        else:
            out.append(lines[i])
            i += 1
    return '\n'.join(out)


def _b64_png(path: Path) -> str:
    with open(path, 'rb') as fh:
        return 'data:image/png;base64,' + base64.b64encode(fh.read()).decode('ascii')


_CLASS_MAP = {
    "float-right": "fig-float-right",
    "fig-arch":    "fig-arch",
}


def _embed_wikilink_images(source: str) -> str:
    """
    Replace Obsidian wikilink image syntax with embedded <figure> HTML.

    Supported forms:
      ![[file.png]]                 ← standalone
      ![[file.png|class-hint]]      ← with CSS class hint
      ![[file.png]]
      *Figure N — caption.*         ← wikilink + italic caption on next line
      ![[file.png|class-hint]]
      *Figure N — caption.*         ← with both class hint and caption
    """
    def _parse_wikilink(raw: str) -> tuple[str, str]:
        """Return (filename, css_class) from 'file.png' or 'file.png|hint'."""
        if '|' in raw:
            fname, hint = raw.split('|', 1)
            return fname.strip(), _CLASS_MAP.get(hint.strip(), '')
        return raw.strip(), ''

    def _with_caption(m: re.Match) -> str:
        fname, css  = _parse_wikilink(m.group(1))
        caption     = m.group(2).strip()
        fig_path    = FIG_DIR / fname
        if not fig_path.exists():
            return f'<p><em>[Figure: {fname} — file not found]</em></p>'
        src        = _b64_png(fig_path)
        class_attr = f' class="{css}"' if css else ''
        return (
            f'<figure{class_attr}>'
            f'<img src="{src}" alt="{fname}"/>'
            f'<figcaption>{caption}</figcaption>'
            f'</figure>'
        )

    def _standalone(m: re.Match) -> str:
        fname, css = _parse_wikilink(m.group(1))
        fig_path   = FIG_DIR / fname
        if not fig_path.exists():
            return f'<p><em>[Figure: {fname} — file not found]</em></p>'
        src        = _b64_png(fig_path)
        class_attr = f' class="{css}"' if css else ''
        return f'<figure{class_attr}><img src="{src}" alt="{fname}"/></figure>'

    # Match wikilink (with optional |hint) + caption on next line — more specific first
    source = re.sub(
        r'!\[\[([^\]]+\.png(?:\|[^\]]*)?)\]\]\s*\n\s*\*([^*]+)\*',
        _with_caption,
        source,
    )
    # Then remaining standalone wikilinks
    source = re.sub(r'!\[\[([^\]]+\.png(?:\|[^\]]*)?)\]\]', _standalone, source)
    return source


# ── Post-processing: two-column table wrapper for floated figures ─────────────

def _wrap_float_figures(html: str) -> str:
    """
    Replace each  <figure class="fig-float-right">…</figure>  and all
    immediately following <p> sibling blocks with a two-column display-table
    layout (fig on the right, text on the left).  Stops wrapping at the next
    heading, figure, div, table, ul, or ol element.

    CSS display:table is reliably supported by WeasyPrint in paged output,
    unlike CSS float which can stall at block-formatting-context boundaries.
    """
    pattern = (
        r'(<figure class="fig-float-right">.*?</figure>)'   # the figure
        r'(\s*(?:<p>.*?</p>\s*)+)'                          # ≥1 paragraphs
        r'(?=\s*<(?:h[1-6]|figure|div|table|ul|ol)\b|\Z)'  # stop before next block
    )

    def _replace(m: re.Match) -> str:
        fig_html  = m.group(1)
        text_html = m.group(2).strip()
        return (
            '<div class="fig-wrap-table">'
            f'<div class="fig-wrap-text">{text_html}</div>'
            f'<div class="fig-wrap-fig">{fig_html}</div>'
            '</div>'
        )

    return re.sub(pattern, _replace, html, flags=re.DOTALL)


# ── Main conversion ───────────────────────────────────────────────────────────

def convert(md_path: Path, pdf_path: Path) -> None:
    source = md_path.read_text(encoding='utf-8').replace('\r\n', '\n')

    meta, source = _strip_frontmatter(source)
    source = _convert_display_math(source)
    source = _convert_inline_math(source)
    source = _convert_callouts(source)
    source = _embed_wikilink_images(source)

    # Collapse triple+ blank lines so block-level HTML elements (figure, div)
    # don't generate empty <p> artefacts when the markdown parser runs.
    source = re.sub(r'\n{3,}', '\n\n', source)

    md        = markdown.Markdown(extensions=['tables', 'fenced_code', 'sane_lists'])
    body_html = md.convert(source)

    # Remove any residual empty paragraphs that still slip through
    body_html = re.sub(r'<p[^>]*>\s*</p>', '', body_html)

    # Wrap floated figures into display-table two-column layout
    body_html = _wrap_float_figures(body_html)

    meta_html = _render_meta_header(meta)

    title = meta.get('title', 'Case Study — Multimodal PR-Status Classification')
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>{title}</title>
  <style>{CSS}</style>
</head>
<body>
{meta_html}
{body_html}
</body>
</html>"""

    os.makedirs(pdf_path.parent, exist_ok=True)
    HTML(string=full_html, base_url=str(ROOT)).write_pdf(str(pdf_path))
    kb = pdf_path.stat().st_size / 1024
    print(f"PDF saved → {pdf_path}  ({kb:.0f} KB)")


if __name__ == '__main__':
    convert(MD_IN, PDF_OUT)
