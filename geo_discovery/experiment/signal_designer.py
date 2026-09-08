"""Signal-driven experiment designer. Reads benchmark diagnostics → proposes
experiments targeting PROVEN weaknesses (not claim-gaps against ceilings)."""
import json

def _need_queries(query_master, mention_by_intent):
    """Return need-based queries — where the brand must earn mention unprompted."""
    return [q for q in query_master if q.get("intent") == "need"]

def design_from_signals(metrics_bundle, query_master, claims, target_brand,
                        gap_register=None):
    """Generate weakness-driven experiment designs from benchmark diagnostics.
    Returns a list of design dicts (same shape the runner expects)."""
    designs = []
    did = 0
    flags = metrics_bundle.get("weakness_flags", [])
    mention_by_intent = {r["intent"]: r["mention_rate"]
                         for r in metrics_bundle.get("mention_by_intent", [])}
    first_by_intent = {r["intent"]: r.get("first_rate", 0)
                       for r in metrics_bundle.get("first_mention_by_intent", [])}

    # ---------- SIGNAL 1: INTENT GAP (the Garnier need-based weakness) ----------
    intent_gap = next((f for f in flags if f["type"] in ("intent_gap","need_first_low")), None)
    if intent_gap:
        # target the need-based queries where the brand is invisible
        need_qs = _need_queries(query_master, mention_by_intent)
        for q in need_qs[:4]:            # cap to top few (cost control)
            did += 1
            designs.append({
                "design_id": f"S{did:03d}",
                "signal": "intent_gap",
                "priority_signal": "high",
                "experiment_id": "E4",   # gap-closure: win the unbranded need query
                "experiment_name": "Need-Query Visibility",
                "isolates": "does enriched, relevant content win an UNBRANDED need query",
                "hypothesis": f"Enriched content will move {target_brand} toward first-mention "
                              f"for the need query '{q['query']}' (currently invisible: "
                              f"need mention {mention_by_intent.get('need',0):.0%}).",
                "question": q["query"],
                "gap": f"Need-based invisibility: '{q['query']}'",
                "include": True, "needs_visual": False, "mode": "A/B (fast)",
                "current_rate": mention_by_intent.get("need", 0),   # baseline for context
            })

    # ---------- SIGNAL 2: RICHNESS DEFICIT ----------
    rich_flag = next((f for f in flags if f["type"] == "richness_deficit"), None)
    if rich_flag:
        # pick a need query to test specificity on
        need_qs = _need_queries(query_master, mention_by_intent)
        if need_qs:
            did += 1
            designs.append({
                "design_id": f"S{did:03d}",
                "signal": "richness_deficit", "priority_signal": "medium",
                "experiment_id": "E2",   # specificity
                "experiment_name": "Specificity / Richness",
                "isolates": "does quantified, specific content beat vague content",
                "hypothesis": f"Quantified/specific claims will lift {target_brand} "
                              f"(richness below competitors).",
                "question": need_qs[0]["query"],
                "gap": rich_flag["detail"],
                "include": True, "needs_visual": False, "mode": "A/B (fast)",
            })

    # ---------- SIGNAL 3: RANK GAP (present but buried) ----------
    # if brand is mentioned on need but never FIRST → test climbing rank
    if mention_by_intent.get("need", 0) > 0.2 and first_by_intent.get("need", 0) < 0.1:
        need_qs = _need_queries(query_master, mention_by_intent)
        if need_qs:
            did += 1
            designs.append({
                "design_id": f"S{did:03d}",
                "signal": "rank_gap", "priority_signal": "high",
                "experiment_id": "E2",
                "experiment_name": "Rank-Climb (present but buried)",
                "isolates": "does structure move brand from mentioned → first-recommended",
                "hypothesis": f"{target_brand} is mentioned but rarely first on need queries; "
                              f"enriched structure will improve rank.",
                "question": need_qs[0]["query"],
                "gap": "Present-but-buried on need queries",
                "include": True, "needs_visual": False, "mode": "A/B (fast)",
            })

    # ---------- SIGNAL 4: retained CLAIM GAPS (lower priority, tagged) ----------
    if gap_register:
        bf_queries = _brand_free_queries(query_master)
        for i, g in enumerate(gap_register):
            if g.get("classification") != "actionable":
                continue
            did += 1
            # brand-free question relevant to the claim
            claim_hint = g.get("gap", "")
            q = _pick_brand_free(query_master, hint=claim_hint)
            if not q:
                continue  # skip if no brand-free query available
            designs.append({
                "design_id": f"S{did:03d}", "signal": "claim_gap",
                "priority_signal": "low",
                "experiment_id": _map_claim_gap(g),
                "experiment_name": "Claim-Gap (brand-free test)",
                "isolates": "does the claim help win a BRAND-FREE query",
                "hypothesis": g.get("hypothesis", ""),
                "question": q,  # ← BRAND-FREE, not "is Garnier's X true"
                "gap": g.get("gap", ""), "gap_id": g.get("gap_id"),
                "include": False, "needs_visual": False, "mode": "A/B (fast)"})
    return designs

def _map_claim_gap(g):
    dt = g.get("delta_type","")
    if dt == "hallucination": return "E3"
    if "negative" in g.get("gap","").lower(): return "E9"
    return "E4"

def _brand_free_queries(query_master):
    """Return ONLY brand-free queries (need/attribute/persona/longtail) — never branded."""
    return [q["query"] for q in query_master
            if q.get("intent") in ("need", "attribute", "persona", "longtail")]

def _pick_brand_free(query_master, claims=None, hint=None):
    """Pick a brand-free query. Prefer one relevant to the claim/hint."""
    bf = _brand_free_queries(query_master)
    if not bf:
        return None
    if hint:
        hint_words = set(hint.lower().split())
        for q in bf:
            if hint_words & set(q.lower().split()):
                return q
    return bf[0]