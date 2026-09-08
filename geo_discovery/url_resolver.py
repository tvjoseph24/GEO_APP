"""Concurrent Gemini-redirect resolution with graceful fallback."""
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

HEADERS = {"User-Agent": "Mozilla/5.0 (GEO audit; research)"}
IS_REDIRECT = "vertexaisearch.cloud.google.com/grounding-api-redirect"

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

def resolve_sources(sources, max_workers=10, timeout=8):
    """Resolve a list of {title,url,...} concurrently. Adds resolved_url + domain."""
    def work(s):
        real, ok = _resolve_one(s.get("url", ""), timeout)
        s["resolved_url"] = real
        s["resolved_ok"] = ok
        # prefer resolved domain; fall back to title (Gemini stores real domain there)
        dom = urlparse(real).netloc.replace("www.", "") if ok else ""
        if not dom or "vertexaisearch" in dom:
            dom = (s.get("title", "") or "").strip().lower()
        s["domain"] = dom
        return s
    if not sources:
        return sources
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(work, s) for s in sources]
        return [f.result() for f in as_completed(futs)]