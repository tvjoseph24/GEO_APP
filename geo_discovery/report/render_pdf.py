"""PDF generation. Three routes, best first.

1. docx2pdf  — drives Microsoft Word (macOS/Windows). Highest fidelity: Word renders its
               own format, so the PDF is byte-faithful to the .docx.
2. LibreOffice headless — no Word required. Very close, occasional font substitution.
3. reportlab — text-only fallback. Loses charts and table styling; use only to unblock.

Rationale for converting rather than re-rendering: the Word renderer is the reference
format. A second full implementation would drift within two revisions.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from report.render_docx import render_docx

SOFFICE_CANDIDATES = [
    "soffice", "libreoffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "/usr/bin/soffice", "/usr/local/bin/soffice", "/opt/homebrew/bin/soffice",
    r"C:\Program Files\LibreOffice\program\soffice.exe",
]

def find_soffice():
    for c in SOFFICE_CANDIDATES:
        p = shutil.which(c) or (c if Path(c).exists() else None)
        if p:
            return p
    return None

def has_docx2pdf():
    try:
        import docx2pdf              # noqa: F401
        return True
    except Exception:
        return False

def available_routes():
    """For the UI — tell the analyst what will happen before they click."""
    out = []
    if has_docx2pdf():
        out.append(("docx2pdf", "Microsoft Word (highest fidelity)"))
    s = find_soffice()
    if s:
        out.append(("libreoffice", f"LibreOffice at {s}"))
    try:
        import reportlab             # noqa: F401
        out.append(("reportlab", "text-only fallback (no charts)"))
    except Exception:
        pass
    return out

# ------------------------------------------------------------------ routes
def _via_docx2pdf(docx_bytes, timeout=300):
    try:
        from docx2pdf import convert
    except Exception as e:
        return None, f"docx2pdf unavailable: {e}"
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "report.docx", Path(td) / "report.pdf"
        src.write_bytes(docx_bytes)
        try:
            convert(str(src), str(dst))
        except Exception as e:
            return None, (f"Word conversion failed: {str(e)[:200]}. Word may not be "
                          f"installed, or macOS automation permission was refused "
                          f"(System Settings → Privacy & Security → Automation).")
        if not dst.exists():
            return None, "Word produced no file"
        return dst.read_bytes(), "Microsoft Word (docx2pdf)"

def _via_soffice(docx_bytes, timeout=240):
    soffice = find_soffice()
    if not soffice:
        return None, "LibreOffice not found"
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "report.docx"
        src.write_bytes(docx_bytes)
        try:
            proc = subprocess.run(
                [soffice, "--headless", "--norestore", "--convert-to", "pdf",
                 "--outdir", td, str(src)],
                capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, f"LibreOffice timed out after {timeout}s"
        except Exception as e:
            return None, f"LibreOffice failed: {str(e)[:180]}"
        out = Path(td) / "report.pdf"
        if not out.exists():
            return None, f"no output. {(proc.stderr or b'').decode()[:200]}"
        return out.read_bytes(), f"LibreOffice ({Path(soffice).name})"

def _via_reportlab(rm, audience="both", profile=None):
    """Last resort: walk the IR directly. Text and tables only — charts are dropped,
    which is why this is a fallback and not a supported output."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                        Spacer, Table, TableStyle)
    except Exception as e:
        return None, f"reportlab unavailable: {e}"
    import io as _io

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9, leading=12,
                          alignment=TA_LEFT)
    cap = ParagraphStyle("cap", parent=body, fontSize=7.5,
                         textColor=colors.HexColor("#6B7280"))
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=17,
                        textColor=colors.HexColor("#0B5FFF"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12)

    def clean(t):
        return (str(t or "").replace("**", "<b>", 1).replace("**", "</b>", 1)
                .replace("&", "&amp;").replace("<b>", "<b>").replace("</b>", "</b>"))

    story = [Paragraph(f"GEO Intelligence Report — {rm.meta.get('brand','').title()}", h1),
             Paragraph(f"{rm.meta.get('parent','')} · {rm.meta.get('category','')} · "
                       f"{rm.meta.get('generated_utc','')[:10]}", cap),
             Spacer(1, 12)]
    if rm.provisional:
        story.append(Paragraph("<b>PROVISIONAL ANALYSIS</b> — "
                               + " ".join(rm.provisional_reasons), body))
    story.append(PageBreak())

    for i, s in enumerate(rm.filtered(audience, profile or "internal"), 1):
        story += [Paragraph(f"{i}. {s.title}", h1), Spacer(1, 6)]
        for b in s.blocks:
            try:
                if b.kind == "heading":
                    story.append(Paragraph(clean(b.payload), h2))
                elif b.kind in ("prose", "callout"):
                    story.append(Paragraph(clean(b.payload), body))
                elif b.kind == "bullets":
                    for it in (b.payload or []):
                        story.append(Paragraph(f"• {clean(it)}", body))
                elif b.kind == "kpi_row":
                    for kp in b.payload:
                        story.append(Paragraph(
                            f"<b>{kp.get('label')}:</b> {kp.get('value')} "
                            f"<font size=7>{kp.get('note','')}</font>", body))
                elif b.kind in ("table", "coverage", "evidence_key"):
                    rows = ([{"Parameter": k, "Value": v}
                             for k, v in (b.payload or {}).items()]
                            if b.kind == "coverage" else (b.payload or []))
                    rows = [r for r in rows if isinstance(r, dict)][:30]
                    if not rows:
                        continue
                    cols = list(rows[0].keys())[:6]
                    data = [[Paragraph(f"<b>{c}</b>", cap) for c in cols]]
                    for r in rows:
                        data.append([Paragraph(str(r.get(c, ""))[:170], cap)
                                     for c in cols])
                    tb = Table(data, hAlign="LEFT",
                               colWidths=[(doc.width / len(cols))] * len(cols))
                    tb.setStyle(TableStyle([
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#DDE1E6")),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F4F7")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP")]))
                    story += [tb, Spacer(1, 4)]
                    if b.caption:
                        story.append(Paragraph(clean(b.caption), cap))
                elif b.kind == "chart" and b.payload:
                    story.append(Image(_io.BytesIO(b.payload), width=doc.width,
                                       height=doc.width * 0.42,
                                       kind="proportional"))
                    if b.caption:
                        story.append(Paragraph(clean(b.caption), cap))
                story.append(Spacer(1, 3))
            except Exception:
                continue          # never let one block abort the whole document
        story.append(PageBreak())

    doc.build(story)
    return buf.getvalue(), "reportlab (text-only fallback)"

# ------------------------------------------------------------------ entry point
def render_pdf(rm, audience="both", profile=None, prefer=None):
    """Returns (pdf_bytes, method) or (None, reason).
    prefer: 'docx2pdf' | 'libreoffice' | 'reportlab' — otherwise best available."""
    tried = []
    order = ([prefer] if prefer else []) + ["docx2pdf", "libreoffice", "reportlab"]
    seen = set()
    order = [r for r in order if r and not (r in seen or seen.add(r))]

    docx = None
    for route in order:
        if route == "reportlab":
            data, how = _via_reportlab(rm, audience, profile)
        else:
            if docx is None:
                docx = render_docx(rm, audience=audience, profile=profile)
            data, how = (_via_docx2pdf(docx) if route == "docx2pdf"
                         else _via_soffice(docx))
        if data:
            return data, how
        tried.append(f"{route}: {how}")

    return None, ("No PDF route available.\n" + "\n".join(f"• {t}" for t in tried)
                  + "\n\nFix by any one of:\n"
                    "• pip install docx2pdf   (uses Microsoft Word if installed)\n"
                    "• Install LibreOffice from libreoffice.org/download (no Homebrew "
                    "needed)\n"
                    "• pip install reportlab   (text-only, charts included but styling "
                    "simplified)\n"
                    "• Or download the .docx and convert it manually.")