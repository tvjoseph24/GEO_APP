"""Collapse the gap register into testable clusters.

Clustering is deterministic (reproducible), keyed on
(family, pillar, target), with token-overlap merging for claim families so the prebiotic
claim cluster becomes ONE experiment instead of nine.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from metrics_spec import brand_stop, fold

FAMILY = {
    "CLAIM_ABSENT": "claim", "CLAIM_FRAGILE": "claim", "CLAIM_DILUTED": "claim",
    "CLAIM_NO_EVIDENCE": "claim",                       # W-15 type was unmapped
    "CLAIM_DISTORTED": "correction", "HALLUCINATION": "correction",
    "COMPETITOR_BLEED": "correction", "STALE_CLAIM": "correction",
    "UNIT_CONFLATION": "correction",
    "UNSUBSTANTIATED_HELP": "substantiation",
    "REASON_DEFICIT": "reason", "FLAT_MENTION": "reason",
    "PROVENANCE_THIN": "provenance", "GREY_MARKET_VOICE": "provenance",
    "SOURCE_ABSENCE": "placement",
    "VISIBILITY_GAP": "visibility",
    "RANK_GAP": "rank",                                 # split from visibility
    "NEGATIVE_FRAMING": "objection", "ADVISORY_DEFERRAL": "objection",
    "REFUSAL": "objection",
    "OBJECTION_EXPOSURE": "objection",                  # W-13 type was unmapped -> E4
    "DEFEND_POSITION": "defend",                        # W-13 type was unmapped
    "ENGINE_DIVERGENCE": "engine", "FRESHNESS_STALE": "freshness",
    "VISUAL_UNREADABLE": "visual",
    "VISUAL_UNASSESSED": "visual",       # was unmapped -> "other" -> led with E0
    # W-55. `social` is its own family: the remedy is participation and correction, not
    # publishing — so it must not pool with `placement`, whose remedy is an editorial pitch.
    "SOCIAL_ABSENCE": "social", "SOCIAL_NEGATIVE": "social",
}
SEV = {"critical": 5, "high": 4, "medium": 3, "low": 2, "none": 1}


def _toks(s, stop):
    return {w for w in re.findall(r"[a-z][a-z\-]{3,}", fold(s or "")) if w not in stop}


def _topic(gaps, stop, k=3):
    c = Counter()
    for g in gaps:
        c.update(_toks(g.get("claim") or g.get("gap"), stop))
    return "/".join(w for w, _ in c.most_common(k)) or "general"

def _merge_by_overlap(gaps, stop, jaccard=0.35, max_size=10):
    buckets = []
    for g in sorted(gaps, key=lambda x: -SEV.get(x.get("severity"), 0)):
        t = _toks(g.get("claim") or g.get("gap"), stop)
        placed = False
        for b in buckets:
            if len(b["members"]) >= max_size:
                continue
            if b["tokens"] and t and \
                    len(b["tokens"] & t) / len(b["tokens"] | t) >= jaccard:
                b["members"].append(g)
                b["tokens"] |= t
                placed = True
                break
        if not placed:
            buckets.append({"tokens": set(t), "members": [g]})
    return [b["members"] for b in buckets]

# Minimum cluster slots reserved per family, so the severity sort cannot delete an
# entire experiment template from the programme. W-36/W-64: a pure severity sort put
# 40 of 53 clusters on E4 and produced zero E1/E2/E5/E9 designs.
FAMILY_QUOTA = {
    "claim": 0.30,          # ceiling, not a floor — claim families dominate by volume
    "correction": 0.15,
    "substantiation": 0.15,
    "reason": 0.10,
    "visibility": 0.10,
    "rank": 0.08,
    "provenance": 0.05,
    "placement": 0.05,
    "objection": 0.08,
    "engine": 0.05,
    "defend": 0.04,
    "visual": 0.05,
    "social": 0.06,
}

def allocate_quota(clusters, max_clusters, quota=None):
    """Select `max_clusters` with guaranteed family spread.

    W-36/W-64: a pure severity sort put 40 of 53 clusters on the claim family, deleting
    E1/E2/E5/E9 from the experiment programme entirely.

    Three passes:
      1. one slot per family present, most severe family first
      2. fill by severity, subject to a per-family ceiling
      3. round-robin the remainder, so an over-represented family cannot reclaim every
         leftover slot simply by being more numerous
    """
    quota = {**FAMILY_QUOTA, **(quota or {})}
    if len(clusters) <= max_clusters:
        return clusters

    by_fam = defaultdict(list)
    for c in clusters:
        by_fam[c["family"]].append(c)
    for f in by_fam:
        by_fam[f].sort(key=lambda c: (-SEV.get(c.get("severity"), 0), -c["n_gaps"]))

    picked_ids, picked, used = set(), [], Counter()

    def _take(c):
        cid = id(c)
        if cid in picked_ids:
            return False
        picked_ids.add(cid)
        picked.append(c)
        used[c["family"]] += 1
        return True

    fam_order = sorted(by_fam, key=lambda f: -SEV.get(by_fam[f][0].get("severity"), 0))

    # ---- pass 1: one slot per family ----
    for f in fam_order:
        if len(picked) >= max_clusters:
            break
        _take(by_fam[f][0])

    # ---- pass 2: fill by severity, respecting ceilings ----
    ceilings = {f: max(1, int(round(max_clusters * quota.get(f, 0.05))))
                for f in by_fam}
    pool = sorted((c for f in by_fam for c in by_fam[f]),
                  key=lambda c: (-SEV.get(c.get("severity"), 0), -c["n_gaps"]))
    for c in pool:
        if len(picked) >= max_clusters:
            break
        if used[c["family"]] < ceilings.get(c["family"], 1):
            _take(c)

    # ---- pass 3: round-robin the remainder ----
    while len(picked) < max_clusters:
        progressed = False
        for f in fam_order:
            if len(picked) >= max_clusters:
                break
            nxt = next((c for c in by_fam[f] if id(c) not in picked_ids), None)
            if nxt is not None and _take(nxt):
                progressed = True
        if not progressed:
            break

    picked.sort(key=lambda c: (-SEV.get(c["severity"], 0), -c["n_gaps"]))
    if picked:
        picked[0]["_quota"] = {"allocated": dict(used), "ceilings": ceilings,
                              "families_present": len(by_fam),
                              "families_kept": len({c["family"] for c in picked})}
    return picked

def cluster_gaps(gaps, *, only_actionable=True, max_clusters=20,
                 target_brand="", competitors=(), category="", stop=None):
    pool = [g for g in gaps
            if not only_actionable or g.get("classification") == "actionable"]
    if not pool:
        return []
    stop = stop or brand_stop(target_brand, competitors, category)

    grouped = defaultdict(list)
    for g in pool:
        fam = FAMILY.get(g.get("type"), "other")
        if fam in ("claim", "correction", "substantiation"):
            key = (fam, g.get("pillar"), "_")                 # merged by overlap below
        elif fam == "reason":
            key = (fam, g.get("pillar"), g.get("intent") or "all")
        elif fam in ("placement", "social"):
            key = (fam, g.get("pillar"),
                   (g.get("evidence") or {}).get("source_type", "mixed"))
        elif fam == "visibility":
            key = (fam, g.get("pillar"), g.get("intent") or "query")
        else:
            key = (fam, g.get("pillar"), "_")
        grouped[key].append(g)

    clusters, n = [], 0
    for (fam, pillar, sub), members in grouped.items():
        chunks = (_merge_by_overlap(members, stop)
                  if fam in ("claim", "correction", "substantiation")
                     and len(members) > 3 else [members])
        for chunk in chunks:
            n += 1
            sev = max(chunk, key=lambda g: SEV.get(g.get("severity"), 0)).get("severity")
            engines = sorted({e for g in chunk for e in (g.get("engines") or [])})
            intents = sorted({g.get("intent") for g in chunk if g.get("intent")})
            queries = [g.get("query") for g in chunk if g.get("query")]
            claims = [g.get("claim") for g in chunk if g.get("claim")]
            dims = sorted({g.get("dimension") for g in chunk if g.get("dimension")})
            doms = [g.get("domain") for g in chunk if g.get("domain")]
            bases = [g.get("metric_baseline") for g in chunk
                     if isinstance(g.get("metric_baseline"), (int, float))]
            clusters.append({
                "cluster_id": f"C{n:02d}",
                "family": fam, "pillar": pillar, "sub": sub,
                "topic": _topic(chunk, stop),
                "title": _title(fam, pillar, sub, chunk, stop),
                "severity": sev,
                "priority": max(SEV.get(sev, 3) * len(chunk), SEV.get(sev, 3)),
                "n_gaps": len(chunk),
                "member_ids": [g["gap_id"] for g in chunk],
                "types": sorted({g["type"] for g in chunk}),
                "engines": engines, "intents": intents,
                "dimensions": dims,
                "claims": claims[:12], "queries": queries[:10], "domains": doms[:10],
                "baseline": (round(sum(bases) / len(bases), 4) if bases else None),
                "evidence_quotes": _quotes(chunk),
                "member_gaps": [{k: g.get(k) for k in
                                 ("gap_id", "type", "severity", "gap", "claim",
                                  "metric_baseline", "engines", "evidence")}
                                for g in chunk],
            })
    clusters.sort(key=lambda c: (-SEV.get(c["severity"], 0), -c["n_gaps"]))
    kept = allocate_quota(clusters, max_clusters)
    covered = {g for c in kept for g in c["member_ids"]}
    orphaned = [g["gap_id"] for g in pool if g["gap_id"] not in covered]
    if kept:
        kept[0]["_coverage"] = {
            "actionable_gaps": len(pool), "clustered": len(covered),
            "orphaned": len(orphaned), "orphaned_ids": orphaned[:40],
            "clusters_truncated": max(0, len(clusters) - max_clusters)}
    return kept

def _title(fam, pillar, sub, chunk, stop):
    topic = _topic(chunk, stop, 2)
    return {
        "claim": f"Publish & reinforce claims: {topic}",
        "correction": f"Correct distorted machine facts: {topic}",
        "substantiation": f"Substantiate what engines already assert: {topic}",
        "reason": f"Close reason-density deficit on {sub} queries: {topic}",
        "provenance": "Take back citation control of the brand narrative",
        "placement": f"Earn presence on {sub} sources",
        "visibility": f"Win unprompted {sub} queries",
        "objection": "Become the brand named inside professional deferral",
        "engine": "Transfer the winning engine's pattern to the laggard",
        "rank": f"Move from mentioned to first-recommended on {sub} queries: {topic}",
        "defend": f"Defend a leading position on {sub} queries",
        "visual": f"Make claims machine-readable from imagery: {topic}",
        "social": f"Earn and correct community presence on {sub} sources: {topic}",

    }.get(fam, f"{fam}: {topic}")

def _quotes(chunk, limit=6):
    out = []
    for g in chunk:
        ev = g.get("evidence") or {}
        q = ev.get("quote")
        if q:
            out.append({"gap_id": g["gap_id"], "quote": str(q)[:280],
                        "note": ev.get("delta_note") or ev.get("why") or ""})
        if len(out) >= limit:
            break
    return out

def cluster_summary(clusters):
    return {"n_clusters": len(clusters),
            "gaps_covered": sum(c["n_gaps"] for c in clusters),
            "by_family": dict(Counter(c["family"] for c in clusters)),
            "by_pillar": dict(Counter(c["pillar"] for c in clusters)),
            "by_severity": dict(Counter(c["severity"] for c in clusters))}