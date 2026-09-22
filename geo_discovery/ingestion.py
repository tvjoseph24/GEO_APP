"""Pluggable ingestion adapters → normalized RawSource[].
  Doc + Web (with thin-content warning + Playwright headless fallback).
  Image/Video stubs for next build.
"""
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

MIN_USEFUL_CHARS = 500          # below this → almost certainly blocked / JS-walled
JS_WALL_CUES = ["enable javascript", "requires javascript", "we use cookies",
                "accept cookies", "cookie policy", "please enable", "turn on javascript",
                "your browser", "loading..."]
# Phrases that mark boilerplate/consent/bot-check content (not brand positioning)
BOILERPLATE_MARKERS = [
    "accept all cookies", "cookie policy", "cookie settings", "cookie preferences",
    "strictly necessary cookies", "performance cookies", "functional cookies",
    "targeting cookies", "we use cookies", "storing of cookies",
    "privacy policy", "notice at collection", "privacy notice",
    "opt out", "opt-out", "global privacy control", "confirm my choices",
    "security check", "verifying the security", "verification successful",
    "enable javascript", "your browser",
    "terms of use", "terms and conditions", "targeted advertising",
    "personal information", "third-party providers", "manage consent",
]

# ---- W-01: does the corpus contain a product SPECIFICATION, not just brand narrative? ----
# If the engines assert pack-level facts ("fragrance-free", "net 12 oz", a retail price)
# that are absent from the register, the first question is whether a specification source
# was ingested at all. If it was not, the finding is an extraction/ingestion gap, not a
# substantiation gap — and the compliance conclusion is inverted.
SPEC_MARKERS = [
    "ingredients:", "ingredient list", "full ingredient", "inci",
    "active ingredient", "inactive ingredient", "other ingredients",
    "net wt", "net weight", "fl oz", "pack size", "available sizes", "size options",
    "directions for use", "how to use", "usage instructions", "application",
    "storage instructions", "shelf life", "expiry", "batch code",
    "nutrition information", "nutritional information", "per 100", "per serving",
    "contains:", "allergen", "warnings:", "caution:", "precautions",
    "specification", "technical data", "product code", "sku", "barcode", "ean", "gtin",
    "rrp", "recommended retail", "list price",
    "manufactured by", "distributed by", "country of origin",
]

def has_spec_content(text, min_hits=2):
    """(is_spec_like, markers_found). Two markers, because one is often incidental."""
    low = (text or "").lower()
    hits = [m for m in SPEC_MARKERS if m in low]
    return len(hits) >= min_hits, hits


def spec_coverage(sources):
    """Accepts list[RawSource] or list[{'origin','text'}] (the persisted form)."""
    rows, n_spec = [], 0
    for s in (sources or []):
        if isinstance(s, dict):
            origin, text = s.get("origin", ""), s.get("text") or ""
        else:
            origin, text = getattr(s, "origin", ""), getattr(s, "text", "") or ""
        ok, hits = has_spec_content(text)
        n_spec += int(ok)
        rows.append({"origin": origin, "chars": len(text), "spec_like": ok,
                     "markers": hits[:6]})
    return {"n_sources": len(rows), "n_spec_like": n_spec,
            "has_specification": n_spec > 0,
            "spec_origins": [r["origin"] for r in rows if r["spec_like"]],
            "detail": rows}

HEADERS = {"User-Agent": "Mozilla/5.0 (GEO brand-audit; research)"}

# ---- add near top ----
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm"}



@dataclass
class RawSource:
    source_type: str
    origin: str
    text: str = ""
    media: list = field(default_factory=list)
    structured: dict = field(default_factory=dict)   # ← NEW
    meta: dict = field(default_factory=dict)

# ======================================================================
#  DOC ADAPTER
# ======================================================================
def _read_pdf(path):
    from pypdf import PdfReader
    return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)

def _read_docx(path):
    import docx
    return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)

def _read_pptx(path):
    from pptx import Presentation
    out = []
    for s in Presentation(str(path)).slides:
        for sh in s.shapes:
            if sh.has_text_frame:
                out.append(sh.text_frame.text)
    return "\n".join(out)

def _strip_boilerplate(text):
    """Remove sentences/lines dominated by cookie/privacy/bot-check boilerplate."""
    # split into sentence-ish chunks, drop any chunk containing a boilerplate marker
    import re
    chunks = re.split(r'(?<=[.!?])\s+|\n+', text)
    clean = []
    dropped = 0
    for ch in chunks:
        low = ch.lower()
        if any(mark in low for mark in BOILERPLATE_MARKERS):
            dropped += 1
            continue
        clean.append(ch)
    return " ".join(clean).strip(), dropped

def ingest_doc(path_or_bytes, name=None):
    """Accepts a filepath OR (bytes, name) for the upload path."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        tmp = Path(f"/tmp/{name}"); tmp.write_bytes(path_or_bytes); path = tmp
    else:
        path = Path(path_or_bytes)
    ext = path.suffix.lower()
    reader = {".pdf": _read_pdf, ".docx": _read_docx, ".pptx": _read_pptx}.get(ext)
    text = reader(path) if reader else path.read_text(errors="ignore")
    return RawSource("doc", path.name, text=text, meta={"chars": len(text)})

def ingest_folder(folder, exts=(".pdf", ".docx", ".pptx", ".txt", ".md")):
    out = []
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() in exts:
            try: out.append(ingest_doc(p))
            except Exception as e: out.append(RawSource("doc", p.name, meta={"error": str(e)}))
    return out

# ======================================================================
#  WEB ADAPTER  (Fix 1: thin-content warning · Fix 2: headless fallback)
# ======================================================================
def _clean_html(html, max_chars):
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        t.decompose()
    # NEW: also drop common consent/policy containers by id/class
    for sel in ["#onetrust-banner-sdk", "#onetrust-consent-sdk",
                "[id*='cookie']", "[class*='cookie']", "[id*='consent']",
                "[class*='consent']", "[class*='privacy']", "[id*='gdpr']"]:
        for el in soup.select(sel):
            el.decompose()
    text = " ".join(soup.get_text(" ").split())[:max_chars]
    text, dropped = _strip_boilerplate(text)          # NEW: sentence-level scrub
    title = (soup.title.string if soup.title else "") or ""
    return text, title.strip(), dropped

def ingest_url(url, timeout=15, max_chars=20000):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        text, title, dropped = _clean_html(r.text, max_chars)
        thin, js_wall = _assess_content(text)
        warning = ("⚠️ Thin/JS-walled — needs headless render" if (thin or js_wall) else "")
        return RawSource("web", url, text=text,
                         meta={"title": title, "domain": urlparse(url).netloc,
                               "chars": len(text), "status": r.status_code,
                               "thin_content": thin, "likely_js_wall": js_wall,
                               "boilerplate_dropped": int(dropped), "rendered": False,
                               "warning": warning})
    except Exception as e:
        return RawSource("web", url, meta={"error": str(e)})

def ingest_url_rendered(url, timeout_ms=25000, max_chars=20000):
    """Headless scrape that runs page JS. Uses domcontentloaded (not networkidle)
    so heavy tracker-laden sites don't time out."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return RawSource("web", url, meta={"error": "playwright not installed"})
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=HEADERS["User-Agent"])
            # block heavy resources to speed up + avoid hangs
            page.route("**/*.{png,jpg,jpeg,gif,svg,woff,woff2,mp4,css}",
                       lambda route: route.abort())
            try:
                # domcontentloaded = DOM ready (fast, reliable) — NOT networkidle
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            except Exception:
                # last resort: 'commit' just waits for response to start
                page.goto(url, wait_until="commit", timeout=timeout_ms)
            # cookie-banner dismissal
            for sel in ["button:has-text('Accept')", "button:has-text('Agree')",
                        "button:has-text('I Accept')", "#onetrust-accept-btn-handler",
                        "[aria-label*='accept' i]"]:
                try: page.click(sel, timeout=1500)
                except Exception: pass
            page.wait_for_timeout(2500)          # let JS hydrate the content
            html = page.content()
            browser.close()
        text, title, dropped = _clean_html(html, max_chars)
        thin, js_wall = _assess_content(text)
        return RawSource("web", url, text=text,
                         meta={"title": title, "domain": urlparse(url).netloc,
                               "chars": len(text), "rendered": True,
                               "boilerplate_dropped": int(dropped),
                               "thin_content": thin, "likely_js_wall": js_wall,
                               "warning": ("⚠️ Still thin after render — likely bot-blocked"
                                           if thin else "")})
    except Exception as e:
        return RawSource("web", url, meta={"error": f"render_failed: {e}", "rendered": True})

def ingest_url_smart(url, allow_render=True):
    """Try fast requests; fall back to headless render if thin/JS-walled (Fix 2)."""
    src = ingest_url(url)
    needs_render = src.meta.get("thin_content") or src.meta.get("likely_js_wall")
    if allow_render and needs_render and "error" not in src.meta:
        rendered = ingest_url_rendered(url)
        # keep the render only if it genuinely improved
        if (not rendered.meta.get("error")
                and rendered.meta.get("chars", 0) > src.meta.get("chars", 0)):
            rendered.meta["fallback_from_requests"] = True
            return rendered
    return src

# ======================================================================
#  IMAGE / VIDEO (stubs for next build — vision path already in gateway)
# ======================================================================

def _mime_for(name):
    n = (name or "").lower()
    if n.endswith(".png"):  return "image/png"
    if n.endswith(".webp"): return "image/webp"
    if n.endswith((".mp4",)):  return "video/mp4"
    if n.endswith((".mov",)):  return "video/quicktime"
    if n.endswith((".webm",)): return "video/webm"
    return "image/jpeg"

# ---------------- IMAGE ADAPTER ----------------
def ingest_image(path_or_bytes, name=None):
    data = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) \
        else Path(path_or_bytes).read_bytes()
    nm = name or (Path(path_or_bytes).name if not isinstance(path_or_bytes, (bytes, bytearray)) else "image")
    return RawSource("image", nm, media=[(bytes(data), _mime_for(nm))],
                     meta={"bytes": len(data)})

# ---------------- VIDEO ADAPTER (cost-gated) ----------------
# Rough Gemini video cost proxy: ~300 tokens/second of video (adjust to live rates)
VIDEO_TOKENS_PER_SEC = 300

def estimate_video_cost(duration_sec, model, pricing):
    """Return (est_input_tokens, est_cost_usd) BEFORE processing — for the confirm gate."""
    tokens = int(duration_sec * VIDEO_TOKENS_PER_SEC)
    pin, _ = pricing.get(model, (0, 0))
    return tokens, round(tokens * pin, 4)

def ingest_video(path_or_bytes, name=None, max_duration_sec=60):
    """Bytes-upload video (PoC). Caps at max_duration for cost control.
    NOTE: actual duration check requires probing; here we tag the cap for the UI."""
    data = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) \
        else Path(path_or_bytes).read_bytes()
    nm = name or "video"
    return RawSource("video", nm, media=[(bytes(data), _mime_for(nm))],
                     meta={"bytes": len(data), "max_duration_sec": max_duration_sec,
                           "note": "cost-gated; confirm before extraction"})

def _assess_content(text):
    """Distinguish genuinely-small pages from blocked/JS-walled ones."""
    low = text.lower()
    js_wall = any(k in low for k in JS_WALL_CUES)
    # 'thin' only flags if BOTH short AND looks like a wall — not just short
    thin = len(text) < MIN_USEFUL_CHARS and (js_wall or len(text) < 100)
    return thin, js_wall