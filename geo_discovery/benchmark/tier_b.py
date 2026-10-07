"""Tier B — Diagnostic. Thin orchestration over veracity.py / richness.py, plus
upgraded provenance and consistency. Public names kept stable for benchmark_tab."""
from __future__ import annotations

from collections import Counter

import pandas as pd

from benchmark import richness as R
from benchmark import veracity as V
from benchmark.spans import build_span_corpus
from metrics_spec import fold

# ---------------------------------------------------------------- VERACITY
def run_veracity(df, claims, target_brand, *, use_llm=False, gateway=None,
                 judge_model="gemini-2.5-flash", competitors=(),
                 alias_overrides=None, max_claims=40, spans_per_engine=4,
                 halluc_sample=48, cache=None, progress_cb=None):
    """LLM path = the full engine. Keyword path = span-scoped calibration only."""
    if use_llm and gateway:
        res = V.run_veracity_v2(gateway, df, claims, target_brand,
                                judge_model=judge_model, competitors=competitors,
                                alias_overrides=alias_overrides,
                                max_claims=max_claims,
                                spans_per_engine=spans_per_engine,
                                halluc_sample=halluc_sample, cache=cache,
                                progress_cb=progress_cb)
        kw = V.veracity_keyword(df, claims, target_brand, competitors=competitors)
        res["keyword_rows"] = kw
        res["calibration"] = V.calibration_report(res.get("per_claim"), kw)
        return res
    kw = V.veracity_keyword(df, claims, target_brand, competitors=competitors)
    sd = build_span_corpus(df, target_brand, other_brands=competitors,
                           alias_overrides=alias_overrides)
    return {"method": "keyword_only", "span_count": int(len(sd)),
            "keyword_rows": kw, "per_claim": [], "judgements": [],
            "hallucinations": [], "engine_scores": {},
            "note": "Enable the LLM veracity audit for verdicts, hallucination "
                    "detection and per-engine coverage."}

# ---------------------------------------------------------------- RICHNESS
def reason_richness(df, brands, lexicon=None, claims=None, category="",
                    gateway=None, model=None, alias_overrides=None):
    lex = lexicon or R.build_lexicon(claims or [], category, gateway, model)
    agg, detail = R.reason_richness_v2(df, brands, lex, alias_overrides)
    return agg, detail, lex

# ---------------------------------------------------------------- PROVENANCE
# W-56: canonical definitions now live in source_profile so the classifier, the
# provenance maths and the gap detector cannot drift apart.
CORROBORATING = {"editorial_authority", "trade_press", "reference"}
NON_CONTENT = {"cdn_asset"}

def claim_provenance(df):
    """WHO the engines are listening to.

    W-56: adds `practitioner` (clinics and clinicians — professional standing, no
    editorial standard) and a RELIABILITY flag. With a large unclassified share every
    percentage below is a lower bound, and `voice_control_ratio` in particular can sit on
    the wrong side of 1.0 — the threshold that decides whether the headline reads
    'citation control is adequate' or 'third parties outweigh the brand'.
    """
    from source_profile import CORROBORATING as CORR, UNCONTROLLED
    t = Counter()
    for tally in df.get("source_type_tally", []):
        t.update(tally or {})
    for k in NON_CONTENT:
        t.pop(k, None)
    total = sum(t.values()) or 1
    by_type = {k: round(v / total, 3) for k, v in t.items()}
    corr = sum(t.get(k, 0) for k in CORR)
    unc = sum(t.get(k, 0) for k in UNCONTROLLED)
    other = t.get("other", 0)

    per_engine = {}
    for eng, g in df.groupby("engine"):
        et = Counter()
        for tally in g.get("source_type_tally", []):
            et.update(tally or {})
        for k in NON_CONTENT:
            et.pop(k, None)
        tt = sum(et.values()) or 1
        per_engine[eng] = {
            "self_referential_pct": round(et.get("brand_owned", 0) / tt, 3),
            "corroborated_pct": round(sum(et.get(k, 0) for k in CORR) / tt, 3),
            "practitioner_pct": round(et.get("practitioner", 0) / tt, 3),
            "competitor_owned_pct": round(et.get("competitor_owned", 0) / tt, 3),
            "grey_market_pct": round(et.get("grey_market_retailer", 0) / tt, 3),
            "community_pct": round(et.get("community_reviews", 0) / tt, 3),
            "affiliate_pct": round(et.get("content_affiliate", 0) / tt, 3),
            "unclassified_pct": round(et.get("other", 0) / tt, 3),
            "citations": int(tt)}

    unclassified_share = other / total
    # The ratio is only meaningful once most citations are classified. Reporting it at a
    # high unclassified share is how a 3.98x reading survived alongside ~2,900
    # misclassified affiliate citations that make the true value ~0.41x.
    ratio = round(t.get("brand_owned", 0) / max(unc, 1), 3)
    reliable = unclassified_share <= 0.15

    return {"by_type": by_type,
            "total_citations": int(total),
            "self_referential_pct": round(t.get("brand_owned", 0) / total, 3),
            "corroborated_pct": round(corr / total, 3),
            "practitioner_pct": round(t.get("practitioner", 0) / total, 3),
            "competitor_owned_pct": round(t.get("competitor_owned", 0) / total, 3),
            "grey_market_pct": round(t.get("grey_market_retailer", 0) / total, 3),
            "community_pct": round(t.get("community_reviews", 0) / total, 3),
            "affiliate_pct": round(t.get("content_affiliate", 0) / total, 3),
            "retailer_pct": round(t.get("retailer", 0) / total, 3),
            "unclassified_pct": round(unclassified_share, 3),
            "per_engine": per_engine,
            # W-55: community_pct was computed per engine and surfaced nowhere. Engines
            # weight community voice very differently, and that belongs in an overlay.
            "community_pct_by_engine": {e: v.get("community_pct")
                                        for e, v in per_engine.items()},
            "voice_control_ratio": ratio if reliable else None,
            "voice_control_ratio_provisional": ratio,
            "reliability": {
                "classified_share": round(1 - unclassified_share, 3),
                "reliable": reliable,
                "note": ("Provenance is reliable: "
                         f"{1 - unclassified_share:.0%} of citations are classified."
                         if reliable else
                         f"{unclassified_share:.0%} of citations are UNCLASSIFIED, so "
                         f"every percentage here is a lower bound and "
                         f"voice_control_ratio is withheld. Generate a source pack in "
                         f"the Audit tab and re-run `backfill.py`. Reported "
                         f"provisionally as {ratio}."),
                "practitioner_note": (
                    "`practitioner` — independent professionals who practise in the "
                    "category — is reported separately and is NOT counted in "
                    "voice_control_ratio. They hold real expertise but no editorial "
                    "standard, and usually a commercial interest in what they recommend. "
                    "Whether they count as controllable voice is a category judgement: "
                    "in some categories a practitioner endorsement functions as "
                    "authority, in others the practitioner is effectively a reseller. "
                    "Decide it explicitly rather than by default.")}}

# ---------------------------------------------------------------- CONSISTENCY
def answer_consistency(df, target_brand, locked=0.90, fragile_hi=0.50):
    """Stability of brand presence per query.

    W-16: locked >= 0.90 and fragile < 0.50 left the 0.50-0.90 band in NO bucket — the
    counts did not sum to the query total, and the most actionable queries generated no
    gap. An `emerging` band now makes the partition exhaustive.
    """
    out = {"overall": {}, "by_engine": {}, "by_web_mode": {}, "divergence": [],
           "thresholds": {"locked": locked, "fragile_hi": fragile_hi},
           "metric": "mentioned"}

    def _buckets(series):
        return {"locked_in": series[series >= locked].index.tolist(),
                "emerging": series[(series >= fragile_hi) & (series < locked)].index.tolist(),
                "fragile": series[(series > 0) & (series < fragile_hi)].index.tolist(),
                "absent": series[series == 0].index.tolist()}

    g = df.groupby("query")["mentioned"].mean()
    b = _buckets(g)
    b["n_queries"] = int(len(g))
    b["_partition_ok"] = (len(b["locked_in"]) + len(b["emerging"])
                          + len(b["fragile"]) + len(b["absent"])) == len(g)
    out["overall"] = b

    for eng, ge in df.groupby("engine"):
        q = ge.groupby("query")["mentioned"].mean()
        e = _buckets(q)
        e["mean_rate"] = round(float(q.mean()), 3) if len(q) else 0.0
        e["n_queries"] = int(len(q))
        out["by_engine"][eng] = e

    if "web_on" in df and df["web_on"].nunique() > 1:
        for wo, gw in df.groupby("web_on"):
            k = "retrieval(web-on)" if wo else "memory(web-off)"
            out["by_web_mode"][k] = {
                "mention_rate": round(float(gw["mentioned"].mean()), 3),
                "first_rate": round(float(gw["is_first"].mean()), 3),
                "n": int(len(gw))}
    else:
        out["by_web_mode"] = {"_note": "single retrieval mode audited — memory and "
                                       "retrieval cannot be separated"}

    if df["engine"].nunique() > 1:
        piv = df.pivot_table(index="query", columns="engine", values="mentioned",
                             aggfunc="mean")
        spread = (piv.max(axis=1) - piv.min(axis=1)).sort_values(ascending=False)
        for q, s in spread[spread > 0.4].items():
            out["divergence"].append(
                {"query": q, "metric": "mentioned",
                 "spread_pp": round(float(s) * 100, 1),
                 "best_engine": piv.loc[q].idxmax(), "worst_engine": piv.loc[q].idxmin()})
    return out

# ---------------------------------------------------------------- FRAMING
def sentiment_framing(df, target_brand):
    """Mention rate alone overstates brand health. Hedged and advisory mentions are not
    wins, and a single directed criticism inside a positive answer is still evidence.

    W-06: `has_negative` and `n_negative_mentions` are computed per response by
    metrics_spec but were dropped at this aggregation boundary — so a category returning
    neg_share of exactly 0.000 across thousands of mentions gave no way to distinguish
    'no criticism exists' from 'criticism exists but never dominates a response'.
    """
    m = df[df["mentioned"] == True]                                  # noqa: E712
    if m.empty:
        return {"mentions": 0}
    pol = m["polarity"].value_counts(normalize=True).round(3).to_dict() \
        if "polarity" in m else {}
    out = {"mentions": int(len(m)),
           "pos_share": float(pol.get("pos", 0)),
           "neu_share": float(pol.get("neu", 0)),
           "neg_share": float(pol.get("neg", 0)),
           "hedged_share": round(float(m["hedged"].mean()), 3) if "hedged" in m else None,
           "advisory_share": round(float(m["advisory_referral"].mean()), 3)
           if "advisory_referral" in m else None}

    # ---- negative EVIDENCE, independent of dominant polarity ----
    if "has_negative" in m.columns:
        hn = m["has_negative"].fillna(False).astype(bool)
        out["any_negative_share"] = round(float(hn.mean()), 3)
        out["responses_with_negative"] = int(hn.sum())
        if "n_negative_mentions" in m.columns:
            out["negative_mentions_total"] = int(
                m["n_negative_mentions"].fillna(0).sum())
        out["negative_detection_note"] = (
            f"{out['responses_with_negative']} response(s) contain at least one "
            f"directed criticism ({out['any_negative_share']:.1%}), of which "
            f"{out['neg_share']:.1%} were negative overall. A large gap between these "
            f"two figures means criticism is present but diluted by surrounding praise."
            if out["responses_with_negative"] else
            "No directed criticism detected in any response. Verify the category cue set "
            "before concluding the brand is never criticised.")
    else:
        out["negative_detection_note"] = (
            "has_negative not present in the corpus — run backfill to populate it.")

    out["by_engine"] = {}
    for eng, g in m.groupby("engine"):
        p = g["polarity"].value_counts(normalize=True) if "polarity" in g else {}
        row = {"pos": round(float(p.get("pos", 0)), 3),
               "neu": round(float(p.get("neu", 0)), 3),
               "neg": round(float(p.get("neg", 0)), 3),
               "advisory": round(float(g["advisory_referral"].mean()), 3)
               if "advisory_referral" in g else None}
        if "has_negative" in g.columns:
            row["any_negative"] = round(
                float(g["has_negative"].fillna(False).astype(bool).mean()), 3)
        out["by_engine"][eng] = row
    return out