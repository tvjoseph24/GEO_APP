"""backfill.py — recompute metrics_spec fields on EXISTING audit/experiment JSONL.

Pure function of stored `response_text` + `sources`. No LLM calls, no network, no cost.

Fixes retroactively:
  * mention detection      → word-boundary + alias folding + overlap resolution
  * polarity / hedging     → negated "avoid X" no longer counts as a clean mention
  * rank                   → ordinal among brands present (not char offset alone)
  * refusal taxonomy       → policy / safety / insufficient_info, + advisory_referral
  * num_images             → B2: was hardcoded 2 on every row (tuple length)
  * source_type_tally      → brand_owned / competitor_owned now actually populated
  * model                  → backfills from engine name so RESUME KEYS still match

Cannot backfill (needs fresh calls):
  * image_insights   — requires vision calls
  * sources          — web retrieval is not replayable
  * web_search_queries

Usage
-----
    python backfill.py --parent danone --target aptamil --dry-run
    python backfill.py --parent danone --target aptamil
    python backfill.py --all --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from audit_runner import _extract_image_urls, build_source_classifier, domains_from_sources
from metrics_spec import build_registry, fold, measure
from url_resolver import attach_domain, domain_from_title, looks_like_domain
from store import STORE_ROOT, load_artifact, project_dir

SCHEMA_VERSION = 4   # 1 = Batch-1 initial · 2 = clause-scoped polarity · 3 = W-02 domain
                     # provenance · 4 = W-56 source-pack classification + practitioner

# ---------------------------------------------------------------- legacy readers
def _legacy_mentioned(row, target_brand):
    bm = row.get("brand_mentions") or {}
    if target_brand in bm:
        return bool(bm[target_brand].get("mentioned"))
    tf = fold(target_brand)
    for k, v in bm.items():
        if fold(k) == tf:
            return bool(v.get("mentioned"))
    return None

def _tally_existing(rpt, r, tb):
    """Count an already-current row so the report reflects FILE STATE, not just changes."""
    m = _row_mentioned_row(r, tb)
    rpt["mention_before"] += int(bool(m))
    rpt["mention_after"] += int(bool(m))
    f = bool(r.get("is_first"))
    rpt["first_before"] += int(f)
    rpt["first_after"] += int(f)
    rpt["refusal_before"] += int(bool(r.get("refusal")))
    rpt["refusal_after"] += int(bool(r.get("refusal")))
    rpt["polarity"][r.get("polarity") or "absent"] += 1
    rpt["refusal_kind"][r.get("refusal_kind") or "none"] += 1
    rpt["advisory_referral"] += int(bool(r.get("advisory_referral")))
    rpt["hedged"] += int(bool(r.get("hedged")))
    rpt["images_before"] += int(r.get("num_images") or 0)
    rpt["images_after"] += int(r.get("num_images") or 0)
    rpt["visual_pages_after"] += len(r.get("visual_pages") or [])
    if r.get("visual_pages"):
        rpt["rows_with_visual_pages"] += 1
    for st_, n in (r.get("source_type_tally") or {}).items():
        rpt["source_type_before"][st_] += n
        rpt["source_type_after"][st_] += n

def _row_mentioned_row(r, tb):
    if r.get("mentioned") is not None:
        return bool(r["mentioned"])
    return _legacy_mentioned(r, tb)

def _legacy_first(row, target_brand):
    bm = row.get("brand_mentions") or {}
    present = [(b, v.get("first_index", -1)) for b, v in bm.items()
               if v.get("first_index", -1) >= 0]
    if not present:
        return None
    return fold(min(present, key=lambda x: x[1])[0]) == fold(target_brand)

# ---------------------------------------------------------------- context
def resolve_context(parent, target, target_brand=None, competitors=None):
    """Pull brand/competitor/domain context from saved artifacts.
    The role:'self' entry feeds BRAND domains — never competitor domains."""
    tb = target_brand or target
    cs = load_artifact(parent, target, "competitive_set") or []
    tf = fold(tb)

    def _is_self(c):
        return isinstance(c, dict) and (c.get("role") == "self"
                                        or fold(c.get("brand", "")) == tf)

    comps = competitors
    if comps is None:
        comps = [c for c in cs
                 if not _is_self(c) and (c.get("brand") if isinstance(c, dict) else c)]

    # aliases from ALL entries incl. self (build_registry routes them correctly)
    alias_overrides = {c.get("brand"): c.get("aliases", [])
                       for c in cs if isinstance(c, dict) and c.get("aliases")}

    brand_domains = set(domains_from_sources(
        load_artifact(parent, target, "ingested_sources")))
    for c in cs:
        if _is_self(c):
            brand_domains |= {str(d).lower() for d in (c.get("domains") or [])}

    competitor_domains = []
    for c in cs:
        if isinstance(c, dict) and not _is_self(c):
            competitor_domains += [str(d).lower() for d in (c.get("domains") or [])]
    # a brand domain must never be classified as competitor
    competitor_domains = [d for d in competitor_domains if d not in brand_domains]

    from source_profile import load_source_pack
    return {"target_brand": tb, "competitors": comps,
            "alias_overrides": alias_overrides,
            "brand_domains": sorted(brand_domains),
            "competitor_domains": sorted(set(competitor_domains)),
            "source_pack": load_source_pack(parent, target)}

# ---------------------------------------------------------------- audit
# ---------------------------------------------------------------- domain provenance
def redomain(srcs):
    """W-02: re-derive domain from the stored resolved_url/title. No network.

    Returns (recovered_from_title, still_unresolvable). A source whose `domain` is a page
    TITLE is not a placement target, and must not reach the client-facing hit-list.
    """
    recovered = unresolved = 0
    for s in (srcs or []):
        before = (s.get("domain") or "").strip().lower()
        attach_domain(s)
        if s.get("domain_source") == "title_domain" and s["domain"] != before:
            recovered += 1
        if s.get("domain_is_title"):
            unresolved += 1
    return recovered, unresolved

def backfill_audit(parent, target, *, dry_run=False, target_brand=None,
                   competitors=None, keep_llm_outcome=True, progress_cb=None,
                   force=False):
    path = project_dir(parent, target) / "audit_raw.jsonl"
    if not path.exists():
        return {"error": f"no audit_raw.jsonl at {path}"}

    ctx = resolve_context(parent, target, target_brand, competitors)
    registry = build_registry(ctx["target_brand"], ctx["competitors"],
                              ctx["alias_overrides"])
    stype_fn = build_source_classifier(ctx["brand_domains"], ctx["competitor_domains"],
                                       ctx.get("source_pack"))
    tb = registry.target.canonical

    rpt = {
        "parent": parent, "target": target, "target_brand": tb,
        "path": str(path), "dry_run": dry_run,
        "aliases": {b.canonical: list(b.aliases) for b in registry.all},
        "brand_domains": ctx["brand_domains"],
        "rows_total": 0, "rows_success": 0, "rows_skipped_no_text": 0,
        "already_backfilled": 0,
        "mention_before": 0, "mention_after": 0, "mention_unknown_before": 0,
        "mention_gained": 0, "mention_lost": 0,
        "first_before": 0, "first_after": 0, "first_changed": 0,
        "polarity": Counter(), "refusal_kind": Counter(),
        "advisory_referral": 0, "hedged": 0,
        "refusal_before": 0, "refusal_after": 0,
        "images_before": 0, "images_after": 0, "rows_with_images_fixed": 0,
        "source_type_before": Counter(), "source_type_after": Counter(),
        "model_backfilled": 0, "modality_backfilled": 0,
        "brands_in_registry": registry.names(),
        "unknown_brands_in_legacy": set(),
        "visual_pages_after": 0, "rows_with_visual_pages": 0,
        "polarity_low_confidence": 0, "multi_brand_sentences": 0,
        "neg_cues_seen": Counter(),
        # ---- W-02 domain provenance ----
        "domains_recovered_from_title": 0, "citations_still_titles": 0,
        "domain_source": Counter(),
        "examples_gained": [], "examples_lost": [], "examples_negated": [],
    }


    tmp = path.with_suffix(".jsonl.tmp")
    out = None if dry_run else open(tmp, "w", encoding="utf-8")
    try:
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    if out:
                        out.write(line + "\n")     # never drop unparseable lines
                    continue
                rpt["rows_total"] += 1

                if not r.get("success"):
                    if out:
                        out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    continue
                rpt["rows_success"] += 1

                if (r.get("metrics_spec_version") == SCHEMA_VERSION
                        and not dry_run and not force):
                    rpt["already_backfilled"] += 1
                    _tally_existing(rpt, r, tb)
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    continue

                text = r.get("response_text") or ""
                if not text:
                    rpt["rows_skipped_no_text"] += 1
                    if out:
                        out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    continue

                # ---------- BEFORE ----------
                mb = _legacy_mentioned(r, tb)
                fb = _legacy_first(r, tb)
                if mb is None:
                    rpt["mention_unknown_before"] += 1
                else:
                    rpt["mention_before"] += int(mb)
                if fb:
                    rpt["first_before"] += 1
                if r.get("outcome") == "refusal":
                    rpt["refusal_before"] += 1
                rpt["images_before"] += int(r.get("num_images") or 0)
                for st_, n in (r.get("source_type_tally") or {}).items():
                    rpt["source_type_before"][st_] += n
                for k in (r.get("brand_mentions") or {}):
                    if fold(k) not in {fold(x) for x in registry.names()}:
                        rpt["unknown_brands_in_legacy"].add(k)

                # ---------- RECOMPUTE ----------
                m = measure(text, registry)

                srcs = r.get("sources") or []
                _rec, _unres = redomain(srcs)  # W-02, must precede classification
                rpt["domains_recovered_from_title"] += _rec
                rpt["citations_still_titles"] += _unres
                for s in srcs:
                    rpt["domain_source"][s.get("domain_source", "unknown")] += 1
                    s["source_type"] = stype_fn(s.get("domain", ""))
                tally = Counter(s.get("source_type", "other") for s in srcs)

                img_urls, visual_pages = _extract_image_urls(text, srcs)

                legacy_outcome = r.get("outcome")
                cls = r.get("classification") or {}
                llm_judged = isinstance(cls, dict) and cls.get("method") == "llm"
                new_outcome = (legacy_outcome if (keep_llm_outcome and llm_judged)
                               else m["outcome"])

                # ---------- WRITE ----------
                r.update({
                    "brand_mentions": m["brand_mentions"],
                    "mentioned": m["mentioned"], "first_index": m["first_index"],
                    "rank": m["rank"], "is_first": m["is_first"],
                    "first_brand": m["first_brand"],
                    "positive_first_brand": m["positive_first_brand"],
                    "polarity": m["polarity"], "hedged": m["hedged"],
                    "refusal": m["refusal"], "refusal_kind": m["refusal_kind"],
                    "advisory_referral": m["advisory_referral"],
                    "n_brands_present": m["n_brands_present"],
                    "outcome": new_outcome,
                    "sources": srcs, "num_sources": len(srcs),
                    "source_type_tally": dict(tally),
                    "image_urls": img_urls, "num_images": len(img_urls),
                    "visual_pages": visual_pages,
                    "num_visual_pages": len(visual_pages),
                    "response_chars": len(text),
                    "metrics_spec_version": SCHEMA_VERSION,
                    "_backfilled_utc": datetime.now(timezone.utc).isoformat(),
                    "visual_pages_after": 0, "rows_with_visual_pages": 0,
                    "polarity_low_confidence": 0, "multi_brand_sentences": 0,
                    "neg_cues_seen": Counter(),
                })
                if legacy_outcome != new_outcome:
                    r.setdefault("outcome_legacy", legacy_outcome)
                # CRITICAL: resume key is (query, engine, model, web_on, run_index).
                # Legacy rows lacking `model` would look "not done" → full re-run.
                if not r.get("model"):
                    r["model"] = r.get("engine")
                    rpt["model_backfilled"] += 1
                if not r.get("modality"):
                    r["modality"] = "text"
                    rpt["modality_backfilled"] += 1

                # ---------- AFTER ----------
                rpt["mention_after"] += int(m["mentioned"])
                rpt["first_after"] += int(bool(m["is_first"]))
                rpt["refusal_after"] += int(bool(m["refusal"]))
                rpt["polarity"][m["polarity"]] += 1
                rpt["refusal_kind"][m["refusal_kind"]] += 1
                rpt["advisory_referral"] += int(bool(m["advisory_referral"]))
                rpt["hedged"] += int(bool(m["hedged"]))
                rpt["images_after"] += len(img_urls)
                rpt["visual_pages_after"] += len(visual_pages)
                if visual_pages:
                    rpt["rows_with_visual_pages"] += 1
                _tinfo = m["brand_mentions"].get(tb, {})
                for ev in (_tinfo.get("neg_evidence") or []):
                    if ev.get("low_confidence"):
                        rpt["polarity_low_confidence"] += 1
                    if (ev.get("n_brands_in_sentence") or 1) > 1:
                        rpt["multi_brand_sentences"] += 1
                    for c in (ev.get("cues") or []):
                        rpt["neg_cues_seen"][c] += 1

                for st_, n in tally.items():
                    rpt["source_type_after"][st_] += n

                if mb is False and m["mentioned"]:
                    rpt["mention_gained"] += 1
                    if len(rpt["examples_gained"]) < 5:
                        rpt["examples_gained"].append(
                            {"engine": r.get("engine"), "query": (r.get("query") or "")[:70],
                             "alias": m["brand_mentions"].get(tb, {}).get("alias_matched"),
                             "sentence": m["brand_mentions"].get(tb, {})
                                          .get("first_sentence", "")[:200]})
                if mb is True and not m["mentioned"]:
                    rpt["mention_lost"] += 1
                    if len(rpt["examples_lost"]) < 5:
                        rpt["examples_lost"].append(
                            {"engine": r.get("engine"), "query": (r.get("query") or "")[:70],
                             "why": "substring match without word boundary"})
                if fb is not None and bool(fb) != bool(m["is_first"]):
                    rpt["first_changed"] += 1
                if m["polarity"] == "neg" and len(rpt["examples_negated"]) < 8:
                    ev = (_tinfo.get("neg_evidence") or [{}])[0]
                    rpt["examples_negated"].append({
                        "engine": r.get("engine"),
                        "query": (r.get("query") or "")[:70],
                        "sentence": ev.get("sentence", "")[:240],
                        "cues": ev.get("cues", []),
                        "n_brands_in_sentence": ev.get("n_brands_in_sentence"),
                        "low_confidence": ev.get("low_confidence"),
                        "mix": _tinfo.get("polarity_mix"),
                        "n_mentions": _tinfo.get("n_mentions")})

                if out:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                if progress_cb and rpt["rows_total"] % 200 == 0:
                    progress_cb(rpt["rows_total"])
    finally:
        if out:
            out.close()

    if not dry_run:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        bak = path.with_name(f"audit_raw.jsonl.bak.{ts}")
        shutil.copy2(path, bak)
        os.replace(tmp, path)              # atomic
        rpt["backup"] = str(bak)
    elif tmp.exists():
        tmp.unlink()

    rpt["unknown_brands_in_legacy"] = sorted(rpt["unknown_brands_in_legacy"])
    for k in ("polarity", "refusal_kind", "source_type_before", "source_type_after",
              "neg_cues_seen", "domain_source"):
        rpt[k] = dict(rpt[k])

    n = max(rpt["rows_success"], 1)
    rpt["mention_rate_before"] = round(rpt["mention_before"] / n, 3)
    rpt["mention_rate_after"] = round(rpt["mention_after"] / n, 3)
    rpt["first_rate_before"] = round(rpt["first_before"] / n, 3)
    rpt["first_rate_after"] = round(rpt["first_after"] / n, 3)
    return rpt

# ---------------------------------------------------------------- experiment
def backfill_experiment(parent, target, *, dry_run=False, target_brand=None,
                        competitors=None):
    """Recompute experiment_raw.jsonl. NOTE: does NOT update the saved
    experiment_results / validated_findings aggregates — re-run the analysis
    (or wait for Batch 5, which recomputes aggregates from raw)."""
    path = project_dir(parent, target) / "experiment_raw.jsonl"
    if not path.exists():
        return {"error": f"no experiment_raw.jsonl at {path}"}

    ctx = resolve_context(parent, target, target_brand, competitors)
    registry = build_registry(ctx["target_brand"], ctx["competitors"],
                              ctx["alias_overrides"])
    tb = registry.target.canonical
    rpt = {"path": str(path), "dry_run": dry_run, "rows": 0,
           "mention_before": 0, "mention_after": 0,
           "first_before": 0, "first_after": 0,
           "negative_framing": 0, "no_text": 0,
           "arms": Counter(), "aggregates_stale": True}

    tmp = path.with_suffix(".jsonl.tmp")
    out = None if dry_run else open(tmp, "w", encoding="utf-8")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    if out:
                        out.write(line + "\n")
                    continue
                rpt["rows"] += 1
                rpt["arms"][f"{r.get('design_id')}/{r.get('arm')}"] += 1
                rpt["mention_before"] += int(bool(r.get("mentioned")))
                rpt["first_before"] += int(bool(r.get("is_first")))
                text = r.get("response_text") or ""
                if not text:
                    rpt["no_text"] += 1
                    if out:
                        out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    continue
                m = measure(text, registry)
                r.update({"mentioned": m["mentioned"], "is_first": m["is_first"],
                          "rank": m["rank"], "polarity": m["polarity"],
                          "hedged": m["hedged"], "refusal": m["refusal"],
                          "advisory_referral": m["advisory_referral"],
                          "first_brand": m["first_brand"],
                          "positive_first_brand": m["positive_first_brand"],
                          "outcome": m["outcome"],
                          "metrics_spec_version": SCHEMA_VERSION})
                rpt["mention_after"] += int(m["mentioned"])
                rpt["first_after"] += int(bool(m["is_first"]))
                if m["polarity"] == "neg":
                    rpt["negative_framing"] += 1
                if out:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
    finally:
        if out:
            out.close()
    if not dry_run:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        shutil.copy2(path, path.with_name(f"experiment_raw.jsonl.bak.{ts}"))
        os.replace(tmp, path)
    elif tmp.exists():
        tmp.unlink()
    rpt["arms"] = dict(rpt["arms"])
    return rpt

# ---------------------------------------------------------------- discovery
def find_projects():
    out = []
    if not STORE_ROOT.exists():
        return out
    for pdir in sorted(STORE_ROOT.iterdir()):
        if not pdir.is_dir() or pdir.name.startswith("."):
            continue
        for tdir in sorted(pdir.iterdir()):
            if tdir.is_dir() and (tdir / "audit_raw.jsonl").exists():
                out.append((pdir.name, tdir.name))
    return out

# ---------------------------------------------------------------- report
def print_report(rpt):
    if rpt.get("error"):
        print(f"  ✗ {rpt['error']}")
        return
    g = rpt.get
    tag = "DRY-RUN" if g("dry_run") else "WRITTEN"
    print(f"\n=== {g('parent')} → {g('target')}  [{tag}]  schema v{SCHEMA_VERSION} ===")
    print(f"  rows: {g('rows_total', 0)} total · {g('rows_success', 0)} success"
          f" · {g('already_backfilled', 0)} already current"
          f" · {g('rows_skipped_no_text', 0)} no text")
    if g("already_backfilled", 0) and g("already_backfilled", 0) == g("rows_success", 0):
        print("  ⚠ every row was SKIPPED as already-current — nothing recomputed.")
        print("    Bump SCHEMA_VERSION or pass --force to recompute.")
    print(f"  aliases for {g('target_brand')}: "
          f"{(g('aliases') or {}).get(g('target_brand'))}")
    print(f"  brand domains: {g('brand_domains') or '(none — ingest brand URLs in Discover)'}")
    print(f"  mention rate : {g('mention_rate_before', 0):.3f} → "
          f"{g('mention_rate_after', 0):.3f}   "
          f"(+{g('mention_gained', 0)} gained / -{g('mention_lost', 0)} lost"
          f" / {g('mention_unknown_before', 0)} previously unknown)")
    print(f"  first-mention: {g('first_rate_before', 0):.3f} → "
          f"{g('first_rate_after', 0):.3f}   ({g('first_changed', 0)} changed)")
    print(f"  polarity     : {g('polarity') or {}}")
    print(f"  refusal      : {g('refusal_before', 0)} → {g('refusal_after', 0)}"
          f"   kinds={g('refusal_kind') or {}}")
    print(f"  advisory ref : {g('advisory_referral', 0)}  · hedged: {g('hedged', 0)}")
    print(f"  images       : {g('images_before', 0)} → {g('images_after', 0)}")
    print(f"  visual pages : {g('visual_pages_after', 0)} across "
          f"{g('rows_with_visual_pages', 0)} row(s) (social/video pages, not images)")
    print(f"  source types : {g('source_type_before') or {} }")
    print(f"              →  {g('source_type_after') or {} }")
    if g("domain_source"):
        _ds = g("domain_source")
        _tt = _ds.get("title_text", 0)
        _tot = sum(_ds.values()) or 1
        print(f"  domain origin: {_ds}")
        print(f"    recovered from title: {g('domains_recovered_from_title', 0)} · "
              f"still a TITLE not a domain: {_tt} ({_tt / _tot:.1%})")
        if _tt / _tot > 0.15:
            print(f"    ⚠ {_tt / _tot:.0%} of citations carry a page TITLE as their "
                  f"'domain'. Provenance percentages and the source hit-list are "
                  f"unreliable until url_resolver follows these redirects.")
    if g("neg_cues_seen"):
        print(f"  neg cues fired: {g('neg_cues_seen')}")
        print(f"    low-confidence attributions: {g('polarity_low_confidence', 0)} · "
              f"multi-brand sentences: {g('multi_brand_sentences', 0)}")
    if g("model_backfilled", 0):
        print(f"  model field  : backfilled on {g('model_backfilled')} row(s)")
    if g("unknown_brands_in_legacy"):
        print(f"  ⚠ brands in old data but NOT in current competitive_set: "
              f"{g('unknown_brands_in_legacy')}")
    for label, key in (("GAINED", "examples_gained"), ("LOST", "examples_lost"),
                       ("NEGATED", "examples_negated")):
        if g(key):
            print(f"  --- {label} examples ---")
            for e in g(key):
                print(f"    [{e.get('engine')}] {e.get('query')}")
                if e.get("sentence"):
                    print(f"        \"{e['sentence']}\"")
                if e.get("cues"):
                    print(f"        cues: {e['cues']}  brands_in_sentence="
                          f"{e.get('n_brands_in_sentence')}  "
                          f"low_conf={e.get('low_confidence')}  mix={e.get('mix')}")
                if e.get("alias"):
                    print(f"        matched alias: {e['alias']}")
                if e.get("why"):
                    print(f"        reason: {e['why']}")
    if g("backup"):
        print(f"  backup: {g('backup')}")

def main():
    ap = argparse.ArgumentParser(description="Backfill metrics_spec fields on raw JSONL.")
    ap.add_argument("--parent")
    ap.add_argument("--target")
    ap.add_argument("--target-brand", help="override brand name (default: target folder)")
    ap.add_argument("--all", action="store_true", help="every project in the store")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="recompute even rows already at the current SCHEMA_VERSION")
    ap.add_argument("--domains", action="store_true",
                    help="REPORT ONLY: unclassified ('other') source domains, then exit")
    ap.add_argument("--experiments", action="store_true",
                    help="also backfill experiment_raw.jsonl")
    ap.add_argument("--json", action="store_true", help="emit report as JSON")
    a = ap.parse_args()

    if a.all:
        projects = find_projects()
    elif a.parent and a.target:
        projects = [(a.parent, a.target)]
    else:
        ap.error("supply --parent and --target, or --all")
    if not projects:
        print("No projects with audit_raw.jsonl found.")
        return

    # ---------- READ-ONLY DOMAIN REPORT — must short-circuit before any write ----------
    if a.domains:
        from source_profile import unclassified_domains
        for parent, target in projects:
            ctx = resolve_context(parent, target, a.target_brand)
            path = project_dir(parent, target) / "audit_raw.jsonl"
            _fn = build_source_classifier(ctx["brand_domains"],
                                          ctx["competitor_domains"],
                                          ctx.get("source_pack"))
            rows, meta = unclassified_domains(str(path), _fn, top=400)
            _pk = ctx.get("source_pack")
            print(f"\n=== {parent} → {target} · unclassified ('other') sources ===")
            print(f"  pack: "
                  + (f"v{_pk.get('pack_version')}, {_pk.get('n_domains')} domain(s)"
                     if _pk else "NONE — generate one in the Audit tab"))
            print(f"  {meta['unclassified_citations']:,} of "
                  f"{meta['total_citations']:,} citations unclassified "
                  f"({meta['unclassified_citations'] / max(meta['total_citations'], 1):.1%})"
                  f" across {meta['distinct_unclassified']:,} distinct domains")
            print(f"  {meta['excluded_titles']:,} citation(s) excluded: they carry an "
                  f"unresolved page TITLE, which is a redirect failure and not "
                  f"classifiable by any pack")
            print(f"  the {len(rows)} domains below cover "
                  f"{meta['covered_by_top_n']:,} of those citations "
                  f"({meta['covered_by_top_n'] / max(meta['unclassified_citations'], 1):.0%})")
            print(f"\n  {'cites':>6}  {'share':>7}  domain")
            for r in rows:
                print(f"  {r['citations']:6d}  {r['share']:7.2%}  {r['domain'][:68]}")
        return
    # ---------------------------------------------------------------------------------

    reports = []
    for parent, target in projects:
        rpt = backfill_audit(parent, target, dry_run=a.dry_run,
                             target_brand=a.target_brand, force=a.force)
        reports.append(rpt)
        if not a.json:
            print_report(rpt)
        if a.experiments:
            er = backfill_experiment(parent, target, dry_run=a.dry_run,
                                     target_brand=a.target_brand)
            reports.append(er)
            if not a.json:
                if er.get("error"):
                    print(f"  (experiments) {er['error']}")
                else:
                    print(f"  (experiments) {er['rows']} rows · "
                          f"mention {er['mention_before']}→{er['mention_after']} · "
                          f"first {er['first_before']}→{er['first_after']} · "
                          f"negative-framed {er['negative_framing']}")
                    print("  ⚠ saved experiment_results aggregates are now STALE — "
                          "re-run the analysis in the Experiment tab.")
    if a.json:
        print(json.dumps(reports, indent=2, default=str))


if __name__ == "__main__":
    main()