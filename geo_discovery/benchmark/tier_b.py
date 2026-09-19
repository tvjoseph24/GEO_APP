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
CORROBORATING = {"editorial_authority", "trade_press", "reference"}
NON_CONTENT = {"cdn_asset"}

def claim_provenance(df):
    """WHO the engines are listening to. Now separates authorised retail from
    grey-market resale and SEO-affiliate content — different strategic problems."""
    t = Counter()
    for tally in df.get("source_type_tally", []):
        t.update(tally or {})
    for k in NON_CONTENT:
        t.pop(k, None)
    total = sum(t.values()) or 1
    by_type = {k: round(v / total, 3) for k, v in t.items()}
    corr = sum(t.get(k, 0) for k in CORROBORATING)

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
            "corroborated_pct": round(sum(et.get(k, 0) for k in CORROBORATING) / tt, 3),
            "competitor_owned_pct": round(et.get("competitor_owned", 0) / tt, 3),
            "grey_market_pct": round(et.get("grey_market_retailer", 0) / tt, 3),
            "community_pct": round(et.get("community_reviews", 0) / tt, 3),
            "affiliate_pct": round(et.get("content_affiliate", 0) / tt, 3),
            "unclassified_pct": round(et.get("other", 0) / tt, 3),
            "citations": int(tt)}

    return {"by_type": by_type,
            "total_citations": int(total),
            "self_referential_pct": round(t.get("brand_owned", 0) / total, 3),
            "corroborated_pct": round(corr / total, 3),
            "competitor_owned_pct": round(t.get("competitor_owned", 0) / total, 3),
            "grey_market_pct": round(t.get("grey_market_retailer", 0) / total, 3),
            "community_pct": round(t.get("community_reviews", 0) / total, 3),
            "affiliate_pct": round(t.get("content_affiliate", 0) / total, 3),
            "retailer_pct": round(t.get("retailer", 0) / total, 3),
            "unclassified_pct": round(t.get("other", 0) / total, 3),
            "per_engine": per_engine,
            "voice_control_ratio": round(t.get("brand_owned", 0) /
                                         max(t.get("grey_market_retailer", 0) +
                                             t.get("content_affiliate", 0), 1), 3)}

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
    """Mention rate alone overstates brand health. Hedged/advisory mentions are not wins."""
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
    out["by_engine"] = {}
    for eng, g in m.groupby("engine"):
        p = g["polarity"].value_counts(normalize=True) if "polarity" in g else {}
        out["by_engine"][eng] = {
            "pos": round(float(p.get("pos", 0)), 3),
            "neu": round(float(p.get("neu", 0)), 3),
            "neg": round(float(p.get("neg", 0)), 3),
            "advisory": round(float(g["advisory_referral"].mean()), 3)
            if "advisory_referral" in g else None}
    return out