"""Bundle all Tier-A/B diagnostics into a persistable metrics object.
This is the enabler for the signal-driven experiment designer (Priority 2)."""
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

def _df_to_records(df):
    """Safely convert a DataFrame (or None) to list-of-dicts."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return []
    return df.reset_index().to_dict("records") if df.index.name else df.to_dict("records")

def build_metrics_bundle(df, target_brand, competitors, veracity, reason_richness,
                         source_mix_dict):
    """Assemble all diagnostics into ONE serializable bundle.
    df = the tidy audit DataFrame (from loader.load_audit)."""
    bundle = {"target_brand": target_brand}

    # ---- MENTION RATE BY INTENT (the key weakness signal) ----
    mbi = (df.groupby("intent")["mentioned"]
             .agg(hits="sum", runs="count").reset_index())
    mbi["mention_rate"] = (mbi["hits"] / mbi["runs"]).round(3)
    lo, hi = proportion_confint(mbi["hits"], mbi["runs"], alpha=0.05, method="wilson")
    mbi["ci_low"], mbi["ci_high"] = lo.round(3), hi.round(3)
    bundle["mention_by_intent"] = mbi.to_dict("records")

    # ---- FIRST-MENTION BY INTENT ----
    fbi = (df.groupby("intent")["is_first"]
             .agg(hits="sum", runs="count").reset_index())
    fbi["first_rate"] = (fbi["hits"] / fbi["runs"]).round(3)
    bundle["first_mention_by_intent"] = fbi.to_dict("records")

    # ---- RANK WHEN PRESENT (where buried) ----
    present = df[df["first_index"] >= 0]
    if not present.empty:
        rank = (present.groupby("intent")["first_index"]
                  .agg(avg_char_pos="mean", n="count").round(1).reset_index())
        bundle["rank_when_present_by_intent"] = rank.to_dict("records")
    else:
        bundle["rank_when_present_by_intent"] = []

    # ---- REFUSAL BY INTENT ----
    rbi = df.groupby("intent")["refusal"].mean().round(3).reset_index()
    rbi.columns = ["intent", "refusal_rate"]
    bundle["refusal_by_intent"] = rbi.to_dict("records")

    # ---- CATEGORY-AWARE METRIC ROUTING (the query-design fix) ----
    BRAND_NAMED = {"branded", "comparative"}
    routed = {}
    for r in bundle["mention_by_intent"]:
        intent = r["intent"]
        if intent == "branded":
            routed[intent] = {"metric": "veracity",
                              "note": "brand named → measures ACCURACY not visibility",
                              "mention_rate": r["mention_rate"]}
        elif intent == "comparative":
            routed[intent] = {"metric": "win_rate",
                              "note": "brand named → measures POSITIONING vs competitors",
                              "mention_rate": r["mention_rate"]}
        else:
            routed[intent] = {"metric": "visibility",
                              "note": "brand-free → mention rate is MEANINGFUL",
                              "mention_rate": r["mention_rate"]}
    bundle["metric_by_category"] = routed

    # ---- MENTION BY ENGINE × INTENT (cross-engine weakness) ----
    mei = (df.groupby(["engine","intent"])["mentioned"]
             .agg(hits="sum", runs="count").reset_index())
    mei["mention_rate"] = (mei["hits"]/mei["runs"]).round(3)
    bundle["mention_by_engine_intent"] = mei.to_dict("records")

    # ---- REASON RICHNESS (brand vs competitors) ----
    bundle["reason_richness"] = _df_to_records(reason_richness)

    # ---- VERACITY SUMMARY ----
    vcount = {"correct":0, "missing":0, "contradicted":0}
    for v in (veracity or []):
        verdict = v.get("verdict","")
        if verdict in vcount: vcount[verdict] += 1
    bundle["veracity_summary"] = vcount

    # ---- SOURCE MIX (per engine, already computed) ----
    bundle["source_mix"] = source_mix_dict or {}

    # ---- WEAKNESS FLAGS (pre-computed for the designer) ----
    bundle["weakness_flags"] = _detect_weaknesses(bundle, target_brand)
    return bundle

def _detect_weaknesses(bundle, target_brand):
    """Pre-flag weaknesses for the Priority-2 designer.
    Category-aware: only flags VISIBILITY categories (brand-free) — because branded/
    comparative queries name the brand, so their mention rate isn't a visibility signal."""
    flags = []

    routed = bundle.get("metric_by_category", {})
    mbi = {r["intent"]: r["mention_rate"] for r in bundle.get("mention_by_intent", [])}
    fr  = {r["intent"]: r.get("first_rate", 0) for r in bundle.get("first_mention_by_intent", [])}

    # visibility categories = brand-free (mention rate is MEANINGFUL here)
    visibility_intents = [k for k, v in routed.items() if v.get("metric") == "visibility"]
    # fallback if routing absent: treat all non-branded/comparative as visibility
    if not visibility_intents:
        visibility_intents = [i for i in mbi if i not in ("branded", "comparative")]

    # ---- 1. VISIBILITY GAP: low unprompted mention on brand-free queries ----
    vis_rates = {i: mbi.get(i, 0) for i in visibility_intents if i in mbi}
    if vis_rates:
        worst = min(vis_rates, key=vis_rates.get)
        worst_rate = vis_rates[worst]
        if worst_rate < 0.3:
            flags.append({"type": "visibility_gap", "severity": "high",
                          "detail": f"Low unprompted visibility on '{worst}' queries "
                                    f"({worst_rate:.0%}) — brand rarely surfaces when not named",
                          "target_intent": worst})

    # ---- 2. NEED-BASED SPECIFIC (the Garnier signal) ----
    # need-based is the flagship visibility category; flag it explicitly if weak
    if "need" in mbi and mbi["need"] < 0.3:
        flags.append({"type": "need_visibility_gap", "severity": "high",
                      "detail": f"Need-based mention only {mbi['need']:.0%} — "
                                f"brand invisible for unbranded consumer needs",
                      "target_intent": "need"})

    # ---- 3. RANK GAP: present but rarely FIRST on brand-free queries ----
    for i in visibility_intents:
        if mbi.get(i, 0) > 0.3 and fr.get(i, 0) < 0.1:
            flags.append({"type": "rank_gap", "severity": "medium",
                          "detail": f"On '{i}' queries brand is mentioned "
                                    f"({mbi[i]:.0%}) but rarely first ({fr.get(i,0):.0%}) "
                                    f"— present but buried",
                          "target_intent": i})
            break   # one rank-gap flag is enough to trigger the experiment

    # ---- 4. RICHNESS DEFICIT vs competitors ----
    rr = bundle.get("reason_richness", [])
    if rr:
        tgt = next((x for x in rr if x.get("brand") == target_brand), None)
        others = [x.get("avg_richness", 0) for x in rr if x.get("brand") != target_brand]
        if tgt and others and tgt.get("avg_richness", 0) < max(others):
            flags.append({"type": "richness_deficit", "severity": "medium",
                          "detail": f"{target_brand} richness ({tgt.get('avg_richness')}) "
                                    f"below top competitor ({max(others)}) — "
                                    f"content less specific/quantified",
                          "target_intent": "need"})

    return flags

def save_metrics_bundle(parent_brand, target_brand, bundle):
    from store import save_artifact
    save_artifact(parent_brand, target_brand, "benchmark_metrics", bundle,
                  {"note": "diagnostics for signal-driven experiments"})