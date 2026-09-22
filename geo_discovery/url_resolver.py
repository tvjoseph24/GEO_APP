"""Concurrent Gemini-redirect resolution with graceful fallback.

W-02: when a redirect cannot be followed, Gemini's `title` was used as the `domain`.
Downstream that value is rendered as a source to pitch, so a page TITLE could appear in a
client-facing placement table. The fallback is now tagged (`domain_is_title`) and, where
the title contains a real domain, that domain is recovered instead.
"""
import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

HEADERS = {"User-Agent": "Mozilla/5.0 (GEO audit; research)"}
IS_REDIRECT = "vertexaisearch.cloud.google.com/grounding-api-redirect"

# A bare hostname, anchored. Rejects anything containing whitespace.
_DOMAIN_RX = re.compile(
    r"^(?=.{4,253}$)(?!-)[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?)*"
    r"\.(?:com|org|net|gov|edu|io|co|ai|app|store|info|biz|me|tv"
    r"|uk|de|fr|es|it|nl|se|no|dk|fi|ie|ch|at|be|pl|pt|cz|gr"
    r"|au|nz|ca|us|in|jp|cn|kr|sg|my|id|th|ph|vn|hk|tw"
    r"|br|mx|ar|cl|co\.uk|com\.au|co\.nz|co\.in|com\.br|co\.za|za|ae|sa|il|tr|ru)$")

# A domain appearing INSIDE free text ("Healthline — healthline.com").
_EMBEDDED_RX = re.compile(
    r"(?<![a-z0-9@.\-])((?:[a-z0-9][a-z0-9-]{0,62}\.)+"
    r"(?:com|org|net|gov|edu|io|co|ai|app|store|info|co\.uk|com\.au|co\.nz|co\.in"
    r"|com\.br|uk|de|fr|es|it|nl|se|no|dk|fi|ie|ch|au|nz|ca|us|in|jp|sg|my|za))"
    r"(?![a-z0-9\-])")

def looks_like_domain(value) -> bool:
    """True only for a bare, plausible hostname. A page title must never pass."""
    v = (value or "").strip().lower().rstrip(".")
    if not v or " " in v or len(v) > 90 or "/" in v:
        return False
    return bool(_DOMAIN_RX.match(v))

def domain_from_title(title) -> str:
    """Gemini frequently stores the real domain in `title`. Recover it when present."""
    t = (title or "").strip().lower()
    if looks_like_domain(t):
        return t.rstrip(".")
    m = _EMBEDDED_RX.search(t)
    return m.group(1).rstrip(".") if m else ""

def _resolve_one(url, timeout=8):
    """Follow a redirect to its true destination. Returns (resolved_url, ok)."""
    if IS_REDIRECT not in url:
        return url, True                       # already a real URL (ChatGPT)
    try:
        r = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=timeout)
        real = r.url
        if IS_REDIRECT in real:                # HEAD didn't resolve → try GET
            r = requests.get(url, headers=HEADERS, allow_redirects=True,
                             timeout=timeout, stream=True); real = r.url; r.close()
        ok = IS_REDIRECT not in real
        return (real if ok else url), ok
    except Exception:
        return url, False                      # graceful fallback to redirect

def attach_domain(s):
    """Derive `domain` for one source dict, recording HOW it was derived.

    domain_source ∈ url | title_domain | title_text
    domain_is_title is True only for title_text — the case that must never be
    presented to a client as a placement target.
    """
    real = s.get("resolved_url") or s.get("url") or ""
    dom = ""
    if s.get("resolved_ok") and str(real).startswith("http"):
        dom = urlparse(real).netloc.replace("www.", "").lower().rstrip(".")
    if dom and "vertexaisearch" not in dom:
        s.update({"domain": dom, "domain_source": "url", "domain_is_title": False})
        return s
    recovered = domain_from_title(s.get("title") or s.get("domain"))
    if recovered:
        s.update({"domain": recovered, "domain_source": "title_domain",
                  "domain_is_title": False})
        return s
    raw = (s.get("title") or s.get("domain") or "").strip().lower()
    s.update({"domain": raw, "domain_source": "title_text",
              "domain_is_title": not looks_like_domain(raw)})
    return s

def resolve_sources(sources, max_workers=10, timeout=8):
    """Resolve a list of {title,url,...} concurrently. Adds resolved_url + domain."""
    def work(s):
        real, ok = _resolve_one(s.get("url", ""), timeout)
        s["resolved_url"] = real
        s["resolved_ok"] = ok
        return attach_domain(s)
    if not sources:
        return sources
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(work, s) for s in sources]
        return [f.result() for f in as_completed(futs)]
