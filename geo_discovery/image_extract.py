"""image_extract.py — turn a returned PAGE url into an image url. W-57.

`_extract_image_urls` required a literal file extension, so every page url an engine
returned went to `visual_pages` and was counted and discarded. Observed: 331 visual pages,
0 images interrogated — and the VISUAL_UNASSESSED gap correctly reported that visual
legibility was unmeasured rather than confirmed good.

Scoped deliberately narrow. Instagram and Facebook return a login wall to a plain fetch,
and YouTube's og:image is a video thumbnail — a frame, which says nothing about whether a
claim is legible on a pack. Chasing those is a scraping arms race with no analytical
payoff, so they are declared UNFETCHABLE and reported as such. The yield is from retailer
and editorial pages, which is where a product photograph actually lives.
"""
from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

import requests

from url_resolver import HEADERS

# ---------------------------------------------------------------- platform policy
# Platforms that will not serve an image to an unauthenticated fetch, or whose image is
# not a product photograph. Declared rather than attempted, so the coverage statement is
# "structurally unfetchable" rather than "we tried and failed".
UNFETCHABLE = {
    "instagram.com": "requires authentication; returns a login wall to any "
                     "unauthenticated fetch",
    "facebook.com": "requires authentication",
    "fb.com": "requires authentication",
    "tiktok.com": "requires authentication and serves video, not a product still",
    "x.com": "requires authentication",
    "twitter.com": "requires authentication",
    "linkedin.com": "requires authentication",
    "pinterest.": "serves a board thumbnail rather than the source image",
}
# Video platforms: an og:image is a frame of a moving asset. Fetchable, but a frame tells
# you nothing about pack legibility, so it is excluded by default and available on request.
VIDEO_PLATFORM = {
    "youtube.com": "og:image is a video thumbnail, not a product photograph",
    "youtu.be": "og:image is a video thumbnail, not a product photograph",
    "vimeo.com": "og:image is a video thumbnail",
}

IMG_EXT = re.compile(r"\.(?:png|jpe?g|webp|gif|avif)(?:\?|$|#)", re.I)
DIRECT_IMG = re.compile(r"https?://[^\s\"'<>)\]]+?\.(?:png|jpe?g|webp|gif|avif)"
                        r"(?:\?[^\s\"'<>)\]]*)?", re.I)

# Meta tags carrying a page's primary image, in order of reliability.
META_KEYS = ("og:image:secure_url", "og:image:url", "og:image",
             "twitter:image:src", "twitter:image")

# Hostname fragments that mark a tracking pixel, sprite, logo or placeholder rather than a
# product photograph. Checked on the PATH, so a legitimate CDN-hosted product shot passes.
JUNK_PATH = ("sprite", "placeholder", "pixel", "tracking", "1x1", "spacer",
             "icon-", "/icons/", "favicon", "logo-", "/logos/", "badge",
             "loading", "blank.", "transparent.")

def is_direct_image(url: str) -> bool:
    return bool(IMG_EXT.search(str(url or "")))

def host(url: str) -> str:
    try:
        return urlparse(str(url)).netloc.replace("www.", "").lower()
    except Exception:
        return ""

def fetch_policy(url: str, allow_video_frames=False) -> tuple[bool, str]:
    """Should this page be fetched? Returns (fetchable, reason_if_not)."""
    h = host(url)
    if not h:
        return False, "no resolvable host"
    for frag, why in UNFETCHABLE.items():
        if frag in h:
            return False, why
    for frag, why in VIDEO_PLATFORM.items():
        if frag in h:
            if allow_video_frames:
                return True, ""
            return False, why
    return True, ""

def looks_like_junk(url: str) -> bool:
    p = str(url or "").lower()
    return any(j in p for j in JUNK_PATH)

# ---------------------------------------------------------------- extraction
def _meta_images(html: str, base_url: str) -> list:
    """og:image and twitter:image, in declared order, absolutised."""
    out = []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for key in META_KEYS:
            for tag in soup.find_all("meta", attrs={"property": key}) + \
                       soup.find_all("meta", attrs={"name": key}):
                c = (tag.get("content") or "").strip()
                if c:
                    out.append((urljoin(base_url, c), f"meta:{key}"))
        # JSON-LD product images are frequently the cleanest pack shot on a retail page.
        import json as _j
        for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
            try:
                data = _j.loads(s.string or "{}")
            except Exception:
                continue
            for node in (data if isinstance(data, list) else [data]):
                if not isinstance(node, dict):
                    continue
                img = node.get("image")
                for u in ([img] if isinstance(img, str) else
                          (img if isinstance(img, list) else [])):
                    if isinstance(u, str) and u.strip():
                        out.append((urljoin(base_url, u.strip()), "jsonld:image"))
    except Exception:
        pass
    return out

def _largest_inline(html: str, base_url: str, limit=6) -> list:
    """Inline <img> candidates, biggest declared dimensions first. A fallback for pages
    with no meta image — common on editorial pages."""
    out = []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        scored = []
        for tag in soup.find_all("img"):
            src = (tag.get("src") or tag.get("data-src")
                   or tag.get("data-lazy-src") or "")
            if not src or src.startswith("data:"):
                continue
            try:
                area = int(tag.get("width") or 0) * int(tag.get("height") or 0)
            except Exception:
                area = 0
            scored.append((area, urljoin(base_url, src.strip())))
        scored.sort(key=lambda x: -x[0])
        out = [(u, "inline:img") for _, u in scored[:limit]]
    except Exception:
        pass
    return out

def extract_images_from_page(url, timeout=10, max_html=600_000, max_candidates=4,
                             allow_video_frames=False):
    """Fetch a page and return image candidates. Never raises.

    Returns {"url", "fetchable", "reason", "fetched", "status", "candidates": [...]}
    """
    ok, why = fetch_policy(url, allow_video_frames)
    if not ok:
        return {"url": url, "host": host(url), "fetchable": False, "reason": why,
                "fetched": False, "candidates": []}
    if is_direct_image(url):
        return {"url": url, "host": host(url), "fetchable": True, "reason": "",
                "fetched": True, "status": None,
                "candidates": [{"image_url": url, "source": "direct"}]}
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        status = r.status_code
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].lower()
        if status != 200:
            return {"url": url, "host": host(url), "fetchable": True,
                    "reason": f"HTTP {status}", "fetched": False, "status": status,
                    "candidates": []}
        if ctype.startswith("image/"):
            r.close()
            return {"url": url, "host": host(url), "fetchable": True, "reason": "",
                    "fetched": True, "status": status,
                    "candidates": [{"image_url": url, "source": "content_type"}]}
        html = r.raw.read(max_html, decode_content=True).decode("utf-8", "ignore")
        r.close()
    except Exception as e:
        return {"url": url, "host": host(url), "fetchable": True,
                "reason": f"{type(e).__name__}: {str(e)[:100]}", "fetched": False,
                "candidates": []}

    cands, seen = [], set()
    for u, src in (_meta_images(html, url)
                   + [(m, "body:regex") for m in DIRECT_IMG.findall(html)]
                   + _largest_inline(html, url)):
        u = u.strip()
        if not u or u in seen or looks_like_junk(u):
            continue
        seen.add(u)
        cands.append({"image_url": u, "source": src})
        if len(cands) >= max_candidates:
            break
    return {"url": url, "host": host(url), "fetchable": True,
            "reason": "" if cands else "page fetched but carried no usable image",
            "fetched": True, "status": status, "candidates": cands}

def extract_images(page_urls, *, limit=None, allow_video_frames=False,
                   progress_cb=None, per_host_cap=3):
    """Walk page urls and return (image_urls, report).

    `per_host_cap` because one retailer domain can account for most of a corpus, and
    interrogating thirty shots of the same pack from one site buys nothing.
    """
    from collections import Counter
    urls = list(dict.fromkeys(page_urls or []))
    if limit:
        urls = urls[:limit]
    found, rows, by_host = [], [], Counter()
    skipped_host_cap = 0

    for i, u in enumerate(urls):
        h = host(u)
        if by_host[h] >= per_host_cap:
            skipped_host_cap += 1
            rows.append({"url": u, "host": h, "fetchable": True,
                         "reason": f"per-host cap of {per_host_cap} reached",
                         "fetched": False, "candidates": []})
            continue
        res = extract_images_from_page(u, allow_video_frames=allow_video_frames)
        rows.append(res)
        if res["candidates"]:
            by_host[h] += 1
            for c in res["candidates"]:
                found.append({**c, "page_url": u, "host": h})
        if progress_cb:
            progress_cb((i + 1) / max(len(urls), 1))

    unfetchable = [r for r in rows if not r["fetchable"]]
    report = {
        "pages_seen": len(urls),
        "pages_fetched": sum(1 for r in rows if r.get("fetched")),
        "pages_with_an_image": sum(1 for r in rows if r["candidates"]),
        "images_found": len(found),
        "distinct_images": len({f["image_url"] for f in found}),
        "skipped_per_host_cap": skipped_host_cap,
        "unfetchable_pages": len(unfetchable),
        "unfetchable_by_reason": dict(Counter(r["reason"] for r in unfetchable)),
        "unfetchable_by_host": dict(Counter(r["host"] for r in unfetchable).most_common(12)),
        "failed_by_reason": dict(Counter(
            r["reason"] for r in rows
            if r["fetchable"] and not r["candidates"] and r["reason"]).most_common(12)),
        "yield_by_host": dict(Counter(f["host"] for f in found).most_common(15)),
        "detail": rows,
    }
    report["note"] = (
        f"{report['images_found']} image(s) recovered from {report['pages_seen']} returned "
        f"page(s). {report['unfetchable_pages']} page(s) are STRUCTURALLY unfetchable — "
        f"social platforms require authentication, and a video platform's preview image is "
        f"a frame rather than a product photograph. That is a limit of the medium, not a "
        f"gap in this analysis, and it is declared rather than retried."
        if report["unfetchable_pages"] else
        f"{report['images_found']} image(s) recovered from {report['pages_seen']} returned "
        f"page(s).")
    return found, report