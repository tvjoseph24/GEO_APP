"""visual_enrich.py — resolve stored visual_pages to images and interrogate them. W-57.

Separate from backfill, which is deliberately network-free and therefore cannot do this.
Separate from the audit run, because page fetching in the hot loop would slow every run
for a yield concentrated in a minority of pages.

The cache is PERSISTED. Engines return the same product shot repeatedly across hundreds of
runs, so a corpus-wide cache is the difference between ~300 vision calls and perhaps 40.
"""
from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CACHE = "image_insights"          # cache/image_insights.json
MANIFEST = "visual_enrichment"    # artifact

def collect_pages(raw_path, limit=None):
    """Every distinct visual page url in the log, with how often it was returned."""
    p = Path(raw_path)
    if not p.exists():
        return [], {}
    counts, by_query = Counter(), {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("success"):
                continue
            for u in (r.get("visual_pages") or []):
                counts[u] += 1
                by_query.setdefault(u, set()).add(r.get("query"))
    rows = [{"url": u, "times_returned": n, "n_queries": len(by_query.get(u, ()))}
            for u, n in counts.most_common(limit)]
    return rows, {"distinct_pages": len(counts),
                  "total_returns": sum(counts.values())}

def plan(parent, target, raw_path, *, page_limit=60, allow_video_frames=False,
         per_host_cap=3):
    """What a run would cost, before committing. Page fetches are free; vision is not."""
    from image_extract import fetch_policy, host
    from store import load_cache
    pages, meta = collect_pages(raw_path, limit=page_limit)
    cache = load_cache(parent, target, CACHE) or {}
    fetchable, unfetchable = [], []
    for r in pages:
        ok, why = fetch_policy(r["url"], allow_video_frames)
        (fetchable if ok else unfetchable).append({**r, "host": host(r["url"]),
                                                   "reason": why})
    return {"distinct_pages": meta.get("distinct_pages", 0),
            "total_returns": meta.get("total_returns", 0),
            "pages_in_plan": len(pages),
            "fetchable": len(fetchable),
            "unfetchable": len(unfetchable),
            "unfetchable_by_reason": dict(Counter(r["reason"] for r in unfetchable)),
            "unfetchable_by_host": dict(Counter(r["host"] for r in unfetchable)
                                        .most_common(10)),
            "cached_images": len(cache),
            "per_host_cap": per_host_cap,
            "note": (
                f"{len(fetchable)} of {len(pages)} page(s) can be fetched. "
                f"{len(unfetchable)} cannot: social platforms require authentication, and "
                f"a video platform's preview is a frame rather than a product photograph. "
                f"Page fetches cost nothing; only the vision calls on recovered images do, "
                f"and {len(cache)} image(s) are already cached.")}

def run(gateway, parent, target, raw_path, vision_model, *, page_limit=60,
        max_images=40, allow_video_frames=False, per_host_cap=3,
        write_back=True, progress_cb=None, budget_ok=None):
    """Extract, interrogate, and optionally write image_insights back onto the rows."""
    from audit_runner import interrogate_returned_images
    from image_extract import extract_images
    from store import load_cache, save_cache

    pages, _ = collect_pages(raw_path, limit=page_limit)
    page_urls = [r["url"] for r in pages]
    if progress_cb:
        progress_cb(0.02, "Fetching pages…")
    found, ext_report = extract_images(
        page_urls, allow_video_frames=allow_video_frames, per_host_cap=per_host_cap,
        progress_cb=lambda x: progress_cb(0.02 + 0.38 * x, "Fetching pages…")
        if progress_cb else None)

    cache = load_cache(parent, target, CACHE) or {}
    # Most-returned first: an image surfaced ninety times matters more than one surfaced
    # twice, and the budget may not cover both.
    ret = {r["url"]: r["times_returned"] for r in pages}
    images = sorted({f["image_url"]: f for f in found}.values(),
                    key=lambda f: -ret.get(f["page_url"], 0))
    to_do = [f["image_url"] for f in images][:max_images]

    if progress_cb:
        progress_cb(0.42, f"Reading {len(to_do)} image(s)…")
    insights = interrogate_returned_images(
        gateway, to_do, vision_model, max_images=max_images, cache=cache,
        budget_ok=budget_ok)
    save_cache(parent, target, CACHE, cache)
    if progress_cb:
        progress_cb(0.80, "Writing back…")

    page_of = {f["image_url"]: f["page_url"] for f in found}
    for ins in insights:
        if ins.get("url"):
            ins["page_url"] = page_of.get(ins["url"])

    written = 0
    if write_back:
        written = _write_back(raw_path, insights, page_of)
    if progress_cb:
        progress_cb(1.0, "Done")

    ok = [i for i in insights if i.get("fetched") and not i.get("error")]
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "vision_model": vision_model,
        "pages_considered": len(page_urls),
        "extraction": {k: ext_report[k] for k in
                       ("pages_seen", "pages_fetched", "pages_with_an_image",
                        "images_found", "distinct_images", "unfetchable_pages",
                        "unfetchable_by_reason", "unfetchable_by_host",
                        "failed_by_reason", "yield_by_host",
                        "skipped_per_host_cap", "note")},
        "images_interrogated": len(insights),
        "images_read_successfully": len(ok),
        "cache_hits": sum(1 for i in insights if i.get("cache_hit")),
        "cache_size": len(cache),
        "rows_written": written,
        "insights": insights,
        "payload_delivered": any(i.get("image_tokens") for i in ok),
        "coverage_note": _coverage_note(ext_report, insights, ok),
    }
    if not report["payload_delivered"] and ok:
        report["warning"] = (
            "Images were fetched and judged but NO image tokens were reported. Either the "
            "payload is not reaching the model, or the engine publishes no media-token "
            "breakdown. Verify with the vision self-test before trusting any visual "
            "conclusion drawn from this.")
    from store import save_artifact
    save_artifact(parent, target, MANIFEST,
                  {k: v for k, v in report.items() if k != "insights"},
                  {"vision_model": vision_model,
                   "images_read": len(ok), "rows_written": written})
    return report

def _coverage_note(ext, insights, ok):
    seen = ext.get("pages_seen") or 0
    unf = ext.get("unfetchable_pages") or 0
    if not seen:
        return ("No visual pages were returned by any engine, so there is nothing to "
                "interrogate. Visual legibility is unmeasured in this corpus.")
    if not ok:
        return (f"{seen} visual page(s) were returned and no readable image was recovered "
                f"from any of them. {unf} are structurally unfetchable; the remainder "
                f"carried no usable image. Visual legibility remains UNMEASURED — not "
                f"confirmed good.")
    return (f"{len(ok)} image(s) read from {seen} returned page(s). "
            f"{unf} page(s) are structurally unfetchable — social platforms require "
            f"authentication, and a video platform's preview image is a frame rather than "
            f"a product photograph. That is a limit of the medium rather than a gap in "
            f"this analysis, and the figures below cover only what could be read.")

def _write_back(raw_path, insights, page_of):
    """Attach insights to the rows whose visual_pages produced them. Atomic, with a
    timestamped backup — the raw log is the authority for everything downstream."""
    p = Path(raw_path)
    if not p.exists():
        return 0
    by_page = {}
    for i in insights:
        if i.get("page_url"):
            by_page.setdefault(i["page_url"], []).append(i)
    if not by_page:
        return 0

    tmp = p.with_suffix(".jsonl.tmp")
    written = 0
    with open(p, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                dst.write(line + "\n")      # never drop an unparseable line
                continue
            add = [i for u in (r.get("visual_pages") or [])
                   for i in by_page.get(u, [])]
            if add:
                have = {i.get("url") for i in (r.get("image_insights") or [])}
                new = [i for i in add if i.get("url") not in have]
                if new:
                    r["image_insights"] = (r.get("image_insights") or []) + new
                    r["num_images"] = len({i.get("url") for i in r["image_insights"]
                                           if i.get("url")})
                    r["_visual_enriched_utc"] = datetime.now(
                        timezone.utc).isoformat()
                    written += 1
            dst.write(json.dumps(r, ensure_ascii=False) + "\n")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    shutil.copy2(p, p.with_name(f"{p.name}.bak.{ts}"))
    os.replace(tmp, p)
    return written