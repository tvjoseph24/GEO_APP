"""Bundle all Tier-A/B diagnostics into a persistable metrics object.
This is the enabler for the signal-driven experiment designer (Priority 2)."""
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
import signal_vocab as SV

def _df_to_records(df):
    """Safely convert a DataFrame (or None) to list-of-dicts."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return []
    return df.reset_index().to_dict("records") if df.index.name else df.to_dict("records")

def build_metrics_bundle(df, target_brand, competitors, veracity, reason_richness,
                         source_mix_dict, *, provenance=None, framing=None,
                         flatness=None, reason_gap=None, consistency=None,
                         parent_brand_for_delta=None, social=None,
                         alias_overrides=None, claims_register=None):
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

    # ---- VERACITY SUMMARY (v2 statuses) ----
    v = veracity or {}
    pc = v.get("per_claim", [])
    from collections import Counter as _C
    st = _C(r.get("status") for r in pc)
    bundle["veracity_summary"] = {
        "method": v.get("method"), "span_count": v.get("span_count"),
        "claims_considered": v.get("claims_considered"),
        "claims_with_evidence": v.get("claims_with_evidence"),
        "locked_in": st.get("locked_in", 0), "fragile": st.get("fragile", 0),
        "partial_only": st.get("partial_only", 0), "distorted": st.get("distorted", 0),
        "absent": st.get("absent", 0),
        "hallucination_rate": v.get("hallucination_rate"),
        "unsubstantiated_rate": v.get("unsubstantiated_rate"),
        "competitor_bleed_rate": v.get("competitor_bleed_rate"),
        "engine_scores": v.get("engine_scores", {}),
        "calibration": v.get("calibration")}
    bundle["provenance"] = provenance or {}
    bundle["framing"] = framing or {}
    bundle["flatness"] = flatness or {}
    bundle["reason_gap"] = _df_to_records(reason_gap)
    bundle["consistency"] = consistency or {}

    # ---- SOCIAL / COMMUNITY (W-55) ----
    # Computed here rather than passed in, so any engagement — including one whose
    # Benchmark predates this module — gets it on the next run.
    if social is None:
        try:
            from benchmark.social import social_report
            social = social_report(df, target_brand, competitors, alias_overrides)
        except Exception as e:
            social = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    bundle["social"] = social or {}

    # ---- VISUAL (W-57) ----
    # What the engines' own imagery shows. Distinct from the creative bench, which tests
    # assets the analyst uploads.
    try:
        from benchmark.visual import visual_report
        bundle["visual"] = visual_report(df, target_brand, competitors,
                                         claims=claims_register,
                                         alias_overrides=alias_overrides)
    except Exception as e:
        bundle["visual"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}

    # ---- SOURCE MIX (per engine, already computed) ----
    bundle["source_mix"] = source_mix_dict or {}

    # ---- ADVISORY REFERRAL BY INTENT (regulated-category signal) ----
    if "advisory_referral" in df.columns:
        abi = df.groupby("intent")["advisory_referral"].mean().round(3).reset_index()
        abi.columns = ["intent", "advisory_rate"]
        bundle["advisory_rate_by_intent"] = abi.to_dict("records")
    else:
        bundle["advisory_rate_by_intent"] = []

    # ---- POLARITY (mention rate alone overstates health) ----
    if "polarity" in df.columns:
        pol = df[df["mentioned"]]["polarity"].value_counts(normalize=True).round(3)
        bundle["polarity_summary"] = {"pos_share": float(pol.get("pos", 0)),
                                      "neu_share": float(pol.get("neu", 0)),
                                      "neg_share": float(pol.get("neg", 0))}
    else:
        bundle["polarity_summary"] = {}

    bundle["n_runs"] = int(len(df))
    bundle["engines"] = sorted(df["engine"].dropna().unique().tolist())
    bundle["metrics_spec_version"] = 1

    # ---- TOP-3 AND RECOMMENDATION RATE BY INTENT ----
    if "rank" in df.columns and df["rank"].notna().any():
        d3 = df[df["rank"].notna()].copy()
        d3["top3"] = d3["rank"] <= 3
        t3 = d3.groupby("intent")["top3"].agg(hits="sum", runs="count").reset_index()
        t3["top3_rate"] = (t3["hits"] / t3["runs"]).round(3)
        bundle["top3_by_intent"] = t3.to_dict("records")
    else:
        bundle["top3_by_intent"] = []

    if "outcome" in df.columns:
        dr = df.copy()
        dr["rec"] = dr["outcome"].isin(["brand_recommended", "brand_first"])
        rb = dr.groupby("intent")["rec"].agg(hits="sum", runs="count").reset_index()
        rb["recommendation_rate"] = (rb["hits"] / rb["runs"]).round(3)
        bundle["recommendation_by_intent"] = rb.to_dict("records")
    else:
        bundle["recommendation_by_intent"] = []

    # ---- RETRIEVAL DELTA: "can it be found" vs "does it perform once found" ----
    # `_f` was assigned inside the try and read outside it — an unbound NameError on any
    # load failure, uncaught, crashing Benchmark.
    _need = next((r["mention_rate"] for r in bundle["mention_by_intent"]
                  if r["intent"] == "need"), None)
    _f = []
    try:
        from store import load_artifact as _la
        _f = _la(parent_brand_for_delta, target_brand, "validated_findings") or []
    except Exception:
        _f = []
    # A creative-bench arm A is ~20 words of minimal text because the ASSET carries the
    # information. Averaging it in as "the brand's own document in context" understates
    # in_context_worst by an order of magnitude — observed: a bench arm at 1% would have
    # the scorecard report "with generic brand copy present, visibility falls to 1%".
    _text_f = [x for x in _f if x.get("mode") != "creative"
               and x.get("source") != "bench"]
    _n_excluded = len(_f) - len(_text_f)
    _rates = [r["rate"] for x in _text_f for r in (x.get("rate_table") or [])
              if r.get("arm") == "A" and r.get("metric") == "mention"]
    if _need is not None and _rates:
        hi, lo = max(_rates), min(_rates)
        _med = sorted(_rates)[len(_rates) // 2]
        bundle["retrieval_delta"] = {
            "unprompted_mention": _need,
            "in_context_best": round(hi, 3),
            "in_context_worst": round(lo, 3),
            "in_context_median": round(_med, 3),
            # The headline gap uses the MEDIAN control arm, not the best. Measured control
            # arms span 4% to 100% depending on how substantive the document is, so
            # quoting the best case overstates the gap by the full spread.
            "gap_pp": round((_med - _need) * 100, 1),
            "gap_pp_best_case": round((hi - _need) * 100, 1),
            "control_arm_spread_pp": round((hi - lo) * 100, 1),
            "interpretation": (
                f"With the brand's own document in context, mention ranges from {lo:.0%} "
                f"to {hi:.0%} across {len(_rates)} measured control arm(s) — median "
                f"{_med:.0%}. Unprompted, in live retrieval, it is {_need:.1%}. Two "
                f"separate constraints: content must clear a substantiveness threshold, "
                f"and must then be retrievable. The headline gap uses the MEDIAN, because "
                f"the best case describes only the most substantive document tested."),
            "basis": f"audit need-intent mention against the median of {len(_rates)} TEXT "
                     f"experiment control arm(s); the best case is reported separately",
            "n_control_arms": len(_rates),
            "creative_findings_excluded": _n_excluded}
    else:
        bundle["retrieval_delta"] = {
            "_note": (f"No text-experiment control arm available "
                      f"({len(_f)} finding(s), {_n_excluded} creative). The retrieval "
                      f"delta needs a text design's arm-A rate as its in-context "
                      f"reference."
                      if _f else
                      "No experiment findings — run Experiment, or `reanalyse.py` if runs "
                      "exist in the raw log.")}


    # ---- WEAKNESS FLAGS (pre-computed for the designer) ----
    bundle["weakness_flags"] = _detect_weaknesses(bundle, target_brand)
    bundle["visual_pages_total"] = (bundle["visual"] or {}).get(
        "visual_pages_returned", 0)
    bundle["images_interrogated"] = (bundle["visual"] or {}).get("images_read", 0)
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
            flags.append({"type": SV.VISIBILITY_GAP, "severity": "high",
                          "detail": f"Low unprompted visibility on '{worst}' queries "
                                    f"({worst_rate:.0%}) — brand rarely surfaces when not named",
                          "target_intent": worst})

    # ---- 2. NEED-BASED SPECIFIC (the Garnier signal) ----
    # need-based is the flagship visibility category; flag it explicitly if weak
    if "need" in mbi and mbi["need"] < 0.3:
        flags.append({"type": SV.NEED_VISIBILITY_GAP, "severity": "high",
                      "detail": f"Need-based mention only {mbi['need']:.0%} — "
                                f"brand invisible for unbranded consumer needs",
                      "target_intent": "need"})

    # ---- 3. RANK GAP: present but rarely FIRST on brand-free queries ----
    for i in visibility_intents:
        if mbi.get(i, 0) > 0.3 and fr.get(i, 0) < 0.1:
            flags.append({"type": SV.RANK_GAP, "severity": "medium",
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
            flags.append({"type": SV.RICHNESS_DEFICIT, "severity": "medium",
                          "detail": f"{target_brand} richness ({tgt.get('avg_richness')}) "
                                    f"below top competitor ({max(others)}) — "
                                    f"content less specific/quantified",
                          "target_intent": "need"})
    # ---- 5. REFUSAL / ADVISORY pressure (regulated categories) ----
    for r in bundle.get("refusal_by_intent", []):
        if r.get("refusal_rate", 0) >= 0.3:
            flags.append({"type": SV.REFUSAL_HEAVY, "severity": "high",
                          "detail": f"'{r['intent']}' queries refused "
                                    f"{r['refusal_rate']:.0%} of the time — "
                                    f"category-level constraint, not brand weakness",
                          "target_intent": r["intent"]})
            break
    adv = bundle.get("advisory_rate_by_intent", [])
    for r in adv:
        if r.get("advisory_rate", 0) >= 0.5:
            flags.append({"type": SV.ADVISORY_HEAVY, "severity": "medium",
                          "detail": f"'{r['intent']}' answers defer to a health "
                                    f"professional {r['advisory_rate']:.0%} of the time — "
                                    f"recommendations are hedged, not absent",
                          "target_intent": r["intent"]})
            break
    # ---- 6. NEGATIVE FRAMING ----
    neg = bundle.get("polarity_summary", {})
    if neg.get("neg_share", 0) >= 0.15:
        flags.append({"type": SV.NEGATIVE_FRAMING, "severity": "high",
                      "detail": f"{neg['neg_share']:.0%} of brand mentions are "
                                f"negatively framed — mention rate overstates health",
                      "target_intent": "negative"})

    # ---- 7. VERACITY / PROVENANCE / FLATNESS ----
    vs = bundle.get("veracity_summary", {})
    if (vs.get("absent", 0) + vs.get("distorted", 0)) >= 3:
        flags.append({"type": SV.VERACITY_GAP, "severity": "high",
                      "detail": f"{vs.get('absent', 0)} claim(s) absent and "
                                f"{vs.get('distorted', 0)} distorted in machine reality",
                      "target_intent": "branded"})
    if (vs.get("hallucination_rate") or 0) >= 0.1:
        flags.append({"type": SV.VERACITY_GAP, "severity": "high",
                      "detail": f"{vs['hallucination_rate']:.0%} of assertions about the "
                                f"brand are unsupported by the claims register",
                      "target_intent": "branded"})
    pv = bundle.get("provenance", {})
    if pv and pv.get("self_referential_pct", 0) < 0.05 and \
            (pv.get("grey_market_pct", 0) + pv.get("affiliate_pct", 0)) > 0.15:
        flags.append({"type": SV.RICHNESS_DEFICIT, "severity": "high",
                      "detail": f"Brand controls only "
                                f"{pv['self_referential_pct']:.0%} of citations vs "
                                f"{pv.get('grey_market_pct', 0):.0%} grey-market + "
                                f"{pv.get('affiliate_pct', 0):.0%} affiliate — the brand "
                                f"does not own its own machine narrative",
                      "target_intent": "need"})
    fl = bundle.get("flatness", {})
    if fl.get("no_substantive_share", 0) >= 0.30:
        flags.append({"type": SV.RICHNESS_DEFICIT, "severity": "high",
                      "detail": f"{fl['no_substantive_share']:.0%} of brand mentions "
                                f"carry no substantive reason — named but not argued for",
                      "target_intent": "need"})

    # ---- 8. SOCIAL / COMMUNITY (W-55) ----
    soc = bundle.get("social") or {}
    if soc and not soc.get("error"):
        for it in (soc.get("by_intent") or []):
            if it["social_trigger_rate"] >= 0.20 and it.get("community_over_editorial"):
                flags.append({
                    "type": SV.SOCIAL_GAP, "severity": "high",
                    "detail": f"'{it['intent']}' queries cite a community source in "
                              f"{it['social_trigger_rate']:.0%} of runs — more often than "
                              f"an editorial one. Consensus outweighs authority here",
                    "target_intent": it["intent"]})
                break
        if soc.get("n_threads_negative"):
            flags.append({
                "type": SV.SOCIAL_NEGATIVE, "severity": "high",
                "detail": f"{soc['n_threads_negative']} cited community thread(s) carry "
                          f"negative framing — the engines are using them as evidence",
                "target_intent": "negative"})
        _d = ((soc.get("framing") or {}).get("delta_with_minus_without") or {})
        if _d.get("hedged", 0) >= 0.10:
            flags.append({
                "type": SV.SOCIAL_NEGATIVE, "severity": "medium",
                "detail": f"Answers citing a community source hedge "
                          f"{_d['hedged'] * 100:.0f}pp more often — scepticism enters "
                          f"through the source mix, not the brand's own content",
                "target_intent": "need"})
    return flags



def save_metrics_bundle(parent_brand, target_brand, bundle):
    from store import save_artifact
    save_artifact(parent_brand, target_brand, "benchmark_metrics", bundle,
                  {"note": "diagnostics for signal-driven experiments"})