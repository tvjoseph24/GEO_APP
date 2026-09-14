"""Normalise the Claims Register before veracity: drop unjudgeable stubs, merge
near-duplicates, strip meta-framing ('X claims that...' == 'X ...')."""
from __future__ import annotations

import re
from metrics_spec import brand_stop, fold


_META = re.compile(
    r"^\s*(?:the\s+brand\s+|it\s+)?(?:claims?|states?|says?|markets?|advertises?|"
    r"positions?|promotes?)\s+(?:that\s+|its\s+|it\s+as\s+|itself\s+as\s+)?", re.I)
_TRAIL = re.compile(r"\bas\s+(?:supporting|providing|offering|helping)\b", re.I)
STOP = set("the a an and or of for with to in on is are be its their this that from as "
           "by it has have had was were will can may also more most than then them these "
           "such other others some claims claim markets marketed supports support".split())

def canonical_fact(claim: str, brand: str = "") -> str:
    t = str(claim or "").strip()
    if brand:
        t = re.sub(r"(?<![a-z0-9])" + re.escape(brand) + r"(?:'s)?(?![a-z0-9])", "",
                   t, flags=re.I)
    t = _META.sub("", t)
    t = _TRAIL.sub("", t)
    return re.sub(r"\s+", " ", t).strip(" .,-'")

def _tokens(s, stop):
    return {w for w in re.findall(r"[a-z][a-z\-]{2,}", fold(s)) if w not in stop}

def prepare_claims(claims, brand="", category="", competitors=(),
                   min_tokens=4, jaccard=0.7, stop=None):
    """Returns (kept, report). Merged claims carry `_merged_from` for traceability."""
    stop = stop or brand_stop(brand, competitors, category)
    kept, dropped, merged = [], [], []
    for c in (claims or []):
        raw = (c.get("claim") or "").strip()
        fact = canonical_fact(raw, brand)
        toks = _tokens(fact, stop)
        if len(fact.split()) < min_tokens or len(toks) < 2:
            dropped.append({"claim": raw, "why": "too short / unjudgeable"})
            continue
        hit = None
        for k in kept:
            a, b = k["_tokens"], toks
            if a and b and len(a & b) / len(a | b) >= jaccard:
                hit = k
                break
        if hit is not None:
            hit.setdefault("_merged_from", []).append(raw)
            merged.append({"kept": hit.get("claim"), "merged": raw})
            if len(raw) > len(hit.get("claim", "")):        # keep the richer wording
                hit["claim"], hit["_canonical_fact"], hit["_tokens"] = raw, fact, toks
            continue
        d = dict(c)
        d["_canonical_fact"] = fact
        d["_tokens"] = toks
        kept.append(d)
    for k in kept:
        k.pop("_tokens", None)
    return kept, {"input": len(claims or []), "kept": len(kept),
                  "dropped": dropped, "merged": merged,
                  "n_dropped": len(dropped), "n_merged": len(merged)}