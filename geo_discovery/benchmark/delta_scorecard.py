"""Delta Scorecard assembly + Gap Register with testable hypotheses."""
import json
from datetime import datetime, timezone
from store import project_dir

def build_delta_scorecard(veracity, reason_rich, provenance, winnability,
                          leaderboard, target_brand):
    """Assemble Ground-Truth vs Machine-Reality per claim + strategic signals."""
    deltas = []
    for v in veracity:
        verdict = v.get("verdict")
        dtype = {"correct":"correct", "missing":"missing",
                 "contradicted":"hallucination"}.get(verdict, "missing")
        severity = {"hallucination":"critical","missing":"high","correct":"none"}[dtype]
        deltas.append({"claim": v.get("claim"), "machine_verdict": verdict,
                       "delta_type": dtype, "severity": severity})
    return {"target_brand": target_brand,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "claim_deltas": deltas,
            "provenance": provenance,
            "leaderboard": leaderboard.to_dict("records") if hasattr(leaderboard,"to_dict") else [],
            "winnability_summary": _winnability_summary(winnability)}

def _winnability_summary(wdf):
    if wdf is None or len(wdf) == 0: return {}
    return wdf["winnability"].value_counts().to_dict()

def build_gap_register(delta_scorecard, winnability, reason_rich, target_brand):
    """Each gap gets a testable hypothesis (feeds Module 4 Experiment)."""
    gaps = []
    gid = 0
    # gaps from claim deltas
    for d in delta_scorecard["claim_deltas"]:
        if d["delta_type"] in ("missing", "hallucination"):
            gid += 1
            pillar = "Claims-Substantiation" if d["delta_type"]=="hallucination" else "Content"
            gaps.append({
                "gap_id": f"G{gid:03d}",
                "gap": f"Claim '{d['claim']}' is {d['delta_type']}",
                "delta_type": d["delta_type"], "severity": d["severity"],
                "pillar": pillar,
                "hypothesis": f"Making '{d['claim']}' explicit + corroborated will "
                              f"increase correct attribution for {target_brand}.",
                "priority": {"critical":5,"high":4,"med":3}.get(d["severity"],2)})
    # gaps from winnable-but-buried queries
    if winnability is not None:
        for _, w in winnability[winnability["winnability"]=="winnable"].iterrows():
            gid += 1
            gaps.append({
                "gap_id": f"G{gid:03d}",
                "gap": f"Winnable query under-served: '{w['query']}'",
                "delta_type": "under-surfaced", "severity": "high",
                "pillar": "Content",
                "hypothesis": f"Adding specific claims for '{w['query']}' will move "
                              f"{target_brand} toward first-mention.",
                "priority": 4})
    return sorted(gaps, key=lambda g: -g["priority"])

GAP_CLASSIFY_PROMPT = """You are a GEO strategist. For each GAP below, classify whether
CLOSING it would plausibly improve how AI assistants RECOMMEND the brand "{brand}" in
REAL user queries (like the ones listed).

Be STRICT. Packaging-refresh taglines ('New Look', 'Same Taste'), pack counts,
and market-specific slogans are NOISE unless a real query asks about them.
Only mark 'actionable' if closing the gap plausibly changes a RECOMMENDATION.

Classifications:
- "actionable": relevant to recommendation/visibility queries; closing it could realistically
   improve how/whether the brand is recommended or described.
- "noise": query-irrelevant trivia (packaging counts like '48 sticks', founding dates,
   market-specific taglines) that no recommendation query would surface. NOT worth effort.
- "monitor": borderline / context-dependent / brand-strategic (e.g. CSR claims) — keep but low priority.

Return STRICT JSON list mirroring input gap_ids:
[{{"gap_id":"", "classification":"actionable|noise|monitor", "reason":"one line"}}]

QUERIES BEING AUDITED (what people actually ask):
{queries}

GAPS:
{gaps}"""

def classify_gaps(gateway, gaps, query_master, target_brand,
                  model="gemini-2.5-flash", module="benchmark"):
    """Batched LLM classification of gaps → actionable/noise/monitor. One call."""
    if not gaps:
        return gaps
    gaps_txt = json.dumps([{"gap_id": g["gap_id"], "gap": g["gap"]} for g in gaps])[:6000]
    queries_txt = json.dumps([q["query"] for q in query_master][:30])[:3000]
    try:
        r = gateway.generate(model,
            GAP_CLASSIFY_PROMPT.format(brand=target_brand, queries=queries_txt, gaps=gaps_txt),
            json_mode=True, module=module, task="gap_classify", temperature=0.1)
        classifications = {c["gap_id"]: c for c in (r["parsed"] or [])
                           if isinstance(c, dict) and "gap_id" in c}
    except Exception:
        classifications = {}
    # attach; default to 'monitor' if LLM didn't classify (safe middle)
    for g in gaps:
        c = classifications.get(g["gap_id"], {})
        g["classification"] = c.get("classification", "monitor")
        g["classification_reason"] = c.get("reason", "not classified — review")
    # sort: actionable first, then monitor, then noise; within, by priority
    order = {"actionable": 0, "monitor": 1, "noise": 2}
    return sorted(gaps, key=lambda g: (order.get(g["classification"], 1), -g["priority"]))

def save_benchmark(parent_brand, target_brand, delta, gaps, hitlist, models_used):
    from store import save_artifact
    save_artifact(parent_brand, target_brand, "delta_scorecard", delta, models_used)
    save_artifact(parent_brand, target_brand, "gap_register", gaps, models_used)
    save_artifact(parent_brand, target_brand, "source_hitlist",
                  hitlist.to_dict("records") if hasattr(hitlist,"to_dict") else hitlist,
                  models_used)