"""source_profile.py — per-category source classification, generated not curated.

W-56. A curated domain list cannot scale: 1,409 unmapped domains were observed in a
single engagement, and every category brings its own long tail. Three layers, cheapest
first:

  1. STRUCTURAL   — TLD and hostname rules. Category-independent, free, deterministic.
  2. SOURCE PACK  — LLM-classified, human-gated, saved as a versioned artifact.
                    Regenerate freely and re-apply with `backfill.py`.
  3. BASE MAP     — the legacy curated map in audit_runner, as a fallback only.

Brand-owned and competitor-owned always win, because a misclassified brand domain
inflates self-reference and a misclassified competitor domain inflates it further.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ARTIFACT = "source_pack"
PACK_VERSION = 1

# ---------------------------------------------------------------- canonical buckets
SOURCE_TYPES = {
    "editorial_authority": "Independent publisher, clinical body, regulator, academic "
                           "institution or established consumer-testing organisation. "
                           "Has an editorial or scientific standard it can be held to.",
    "practitioner": "A practising clinician, clinic, practice or professional individual "
                    "publishing under their own name. Professional standing but no "
                    "editorial standard, and frequently commercially interested.",
    "trade_press": "Industry or B2B publication serving the trade, not consumers.",
    "retailer": "Authorised commerce: the brand or its competitors are legitimately sold "
                "here. Marketplaces, pharmacies, supermarkets, specialist retail.",
    "grey_market_retailer": "Parallel-import or unauthorised reseller. Carries no "
                            "compliance obligation and publishes uncontrolled claims.",
    "content_affiliate": "SEO or affiliate content. Reads like editorial, has no "
                         "editorial standard. Review-roundup sites, product-comparison "
                         "sites, personal blogs monetised by commission.",
    "community_reviews": "User-generated discussion or reviews: forums, Q&A, "
                         "review platforms.",
    "video_social": "Video or social platform.",
    "reference": "Encyclopaedic or structured reference.",
    "cdn_asset": "Asset or CDN host. Not content — excluded from provenance maths.",
    "other": "Cannot be classified with confidence.",
}

# Buckets that corroborate a brand claim in a way an engine treats as independent.
CORROBORATING = {"editorial_authority", "trade_press", "reference"}
# Buckets representing voice the brand does not control and cannot correct.
UNCONTROLLED = {"grey_market_retailer", "content_affiliate"}

# ---------------------------------------------------------------- layer 1: structural
# Institutional TLDs. Genuinely category-independent: a .gov or .edu host is an authority
# in any category, and no LLM call should be spent deciding that.
_AUTHORITY_TLD = re.compile(
    r"\.(?:gov|mil|int)$|\.gov\.[a-z]{2}$|\.go\.[a-z]{2}$"
    r"|\.edu$|\.edu\.[a-z]{2}$|\.ac\.[a-z]{2}$|\.ac\.uk$"
    r"|\.nhs\.uk$|\.who\.int$|\.europa\.eu$")

# Hostname shapes that indicate a practising clinic or practitioner. Structural, not
# category vocabulary — it is the ORGANISATIONAL form being detected.
_PRACTITIONER_RX = re.compile(
    r"(?:^|[.-])(?:dr|doctor|drs)[.-]?[a-z]|"
    r"(?:clinic|clinics|practice|surgery|medspa|medcenter|medicalcenter"
    r"|medicalgroup|healthgroup|associates|partners|institute)(?:\.|$)|"
    r"(?:dermatolog|dentist|orthodont|optometr|podiatr|veterinar)[a-z]*"
    r"(?:group|partners|associates|center|centre|clinic|care)?(?:\.|$)")

_CDN_RX = re.compile(
    r"^(?:cdn|static|assets|images|img|media|files|s3|storage)[.-]|"
    r"\.(?:cloudfront|akamaized|akamai|fastly|imgix|cloudinary|amazonaws|"
    r"googleusercontent|wp|squarespace-cdn)\.")

def structural_type(domain: str) -> tuple[str, str] | tuple[None, None]:
    """Free, deterministic pre-classification. Returns (source_type, reason) or (None, None)."""
    d = (domain or "").strip().lower().lstrip(".").rstrip(".")
    if not d or " " in d:
        return None, None
    if _CDN_RX.search(d):
        return "cdn_asset", "CDN or asset-host hostname shape"
    if _AUTHORITY_TLD.search(d):
        return "editorial_authority", "institutional TLD (government, academic or treaty body)"
    if _PRACTITIONER_RX.search(d):
        return "practitioner", "hostname indicates a practising clinician, clinic or practice"
    return None, None

# ---------------------------------------------------------------- layer 2: LLM pack
PACK_PROMPT = """You are classifying the web sources that AI assistants cite when they
answer consumer questions about the "{category}" category.

Each domain below was cited by at least one AI answer. Classify each into exactly one
bucket. The purpose is to establish WHOSE VOICE the engines are listening to, so the
distinction between an independent publisher and a commission-monetised review site is the
most consequential judgement you will make.

BUCKETS
{buckets}

DECISION RULES
- "editorial_authority" requires a genuine editorial or scientific standard: a masthead, a
  named editorial team, a peer-review process, or statutory standing. A major news outlet
  or established magazine qualifies even when it runs commerce links.
- "content_affiliate" is the default for a site that exists primarily to rank in search and
  earn commission. Signals: the hostname is a category keyword plus a generic noun; the
  site publishes only product round-ups and comparisons; there is no identifiable editorial
  team. Be willing to use this — under-using it is the most common error and it materially
  understates how much of the brand's narrative is written by uncontrolled parties.
- "practitioner" is a real clinician, clinic or practice publishing under their own name.
  Professional standing, but no editorial standard and often commercially interested.
- "retailer" means the products are legitimately sold there. "grey_market_retailer" means
  parallel import or unauthorised resale — typically a thin storefront selling imported
  stock, with no brand relationship.
- "other" only when you genuinely cannot tell. Say so in `reason`; do not guess.

For each domain give a confidence between 0 and 1. Below 0.6 a human will review it, so an
honest low confidence is more useful than a confident guess.

Return STRICT JSON — an array, one object per input domain, nothing else:
[{{"domain":"","source_type":"","confidence":0.0,"reason":"one short line"}}]

DOMAINS (with citation counts — higher counts matter more, so weight your care accordingly)
{domains}"""

def _domain_block(rows, limit=60):
    return json.dumps([{"domain": r["domain"], "citations": r.get("citations")}
                       for r in rows[:limit]], indent=1)

def build_source_pack(gateway, domain_rows, category, model="gemini-2.5-flash",
                      module="audit", batch=50, min_confidence=0.6, progress_cb=None):
    """Classify unclassified domains. Structural rules first, then LLM for the remainder.

    domain_rows: [{"domain": str, "citations": int}] — from unclassified_domains().
    Returns a pack dict ready for human review.
    """
    entries, needs_llm, errors = {}, [], []

    # ---- layer 1: free ----
    for r in domain_rows:
        d = (r.get("domain") or "").strip().lower()
        if not d or " " in d or len(d) > 90:
            continue                                  # a page title, not a domain
        st, why = structural_type(d)
        if st:
            entries[d] = {"domain": d, "source_type": st, "confidence": 1.0,
                          "reason": why, "method": "structural",
                          "citations": r.get("citations", 0), "reviewed": False}
        else:
            needs_llm.append(r)

    # ---- layer 2: LLM ----
    buckets = "\n".join(f"- \"{k}\": {v}" for k, v in SOURCE_TYPES.items())
    n_batches = max(1, -(-len(needs_llm) // batch))
    for i in range(0, len(needs_llm), batch):
        chunk = needs_llm[i:i + batch]
        res = gateway.generate_json(
            model, PACK_PROMPT.format(category=category or "this category",
                                      buckets=buckets,
                                      domains=_domain_block(chunk, batch)),
            expect="list", module=module, task="source_pack",
            temperature=0.0, max_tokens=4000,
            coerce_keys=("domains", "classifications", "results", "items"))
        rows = res.get("data") or []
        if not res.get("ok"):
            errors.append(f"batch {i // batch + 1}: {res.get('failure')}")
        by_dom = {(r.get("domain") or "").strip().lower(): r
                  for r in rows if isinstance(r, dict)}
        for r in chunk:
            d = r["domain"].strip().lower()
            got = by_dom.get(d)
            st = str((got or {}).get("source_type", "")).strip()
            if got and st in SOURCE_TYPES:
                try:
                    conf = float(got.get("confidence") or 0)
                except Exception:
                    conf = 0.0
                entries[d] = {"domain": d, "source_type": st, "confidence": round(conf, 2),
                              "reason": (got.get("reason") or "")[:160], "method": "llm",
                              "citations": r.get("citations", 0),
                              "reviewed": False,
                              "needs_review": conf < min_confidence}
            else:
                entries[d] = {"domain": d, "source_type": "other", "confidence": 0.0,
                              "reason": "the classifier returned nothing for this domain",
                              "method": "failed", "citations": r.get("citations", 0),
                              "reviewed": False, "needs_review": True}
        if progress_cb:
            progress_cb(min((i + batch) / max(len(needs_llm), 1), 1.0))

    cited = sum(e.get("citations", 0) for e in entries.values())
    return {
        "pack_version": PACK_VERSION, "category": category, "model": model,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
        "n_domains": len(entries),
        "n_structural": sum(1 for e in entries.values() if e["method"] == "structural"),
        "n_llm": sum(1 for e in entries.values() if e["method"] == "llm"),
        "n_failed": sum(1 for e in entries.values() if e["method"] == "failed"),
        "n_needs_review": sum(1 for e in entries.values() if e.get("needs_review")),
        "citations_covered": cited,
        "by_type": dict(Counter(e["source_type"] for e in entries.values())),
        "citations_by_type": dict(Counter(
            {}, **{})) or _citations_by_type(entries),
        "errors": errors,
    }

def _citations_by_type(entries):
    c = Counter()
    for e in entries.values():
        c[e["source_type"]] += e.get("citations", 0)
    return dict(c)

# ---------------------------------------------------------------- unclassified harvest
def unclassified_domains(raw_path, stype_fn, top=400, min_citations=2):
    """Domains the current classifier returns 'other' for. Page titles are excluded —
    they are an unresolved-redirect problem, not a classification problem."""
    from url_resolver import looks_like_domain
    if not Path(raw_path).exists():
        return [], {}
    counts, total, titles = Counter(), 0, 0
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("success"):
                continue
            for s in (r.get("sources") or []):
                d = (s.get("domain") or "").strip().lower()
                if not d:
                    continue
                total += 1
                if s.get("domain_is_title") or not looks_like_domain(d):
                    titles += 1
                    continue
                if stype_fn(d) == "other":
                    counts[d] += 1
    rows = [{"domain": d, "citations": n, "share": round(n / max(total, 1), 5)}
            for d, n in counts.most_common(top) if n >= min_citations]
    return rows, {"total_citations": total, "excluded_titles": titles,
                  "distinct_unclassified": len(counts),
                  "unclassified_citations": sum(counts.values()),
                  "covered_by_top_n": sum(r["citations"] for r in rows)}

# ---------------------------------------------------------------- apply
def pack_lookup(pack, min_confidence=0.0):
    """Returns {domain: source_type}. Reviewed entries are trusted regardless of the
    model's confidence — a human decision outranks a score."""
    out = {}
    for d, e in ((pack or {}).get("entries") or {}).items():
        st = e.get("source_type")
        if st not in SOURCE_TYPES or st == "other":
            continue
        if e.get("reviewed") or (e.get("confidence") or 0) >= min_confidence:
            out[d] = st
    return out

def save_source_pack(parent, target, pack):
    from store import save_artifact
    return save_artifact(parent, target, ARTIFACT, pack,
                         {"pack_version": PACK_VERSION, "model": pack.get("model"),
                          "n_domains": pack.get("n_domains")})

def load_source_pack(parent, target):
    from store import load_artifact
    return load_artifact(parent, target, ARTIFACT)

def pack_summary(pack):
    if not pack:
        return {}
    e = (pack.get("entries") or {}).values()
    return {"domains": len(list(e)), "by_type": pack.get("by_type"),
            "citations_by_type": pack.get("citations_by_type"),
            "needs_review": sum(1 for x in e if x.get("needs_review")),
            "reviewed": sum(1 for x in e if x.get("reviewed")),
            "citations_covered": pack.get("citations_covered"),
            "errors": pack.get("errors")}