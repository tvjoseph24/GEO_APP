"""Tier A — Descriptive metrics, per engine × intent."""
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

def mention_rate(df):
    g = df.groupby(["engine", "intent"])["mentioned"].agg(hits="sum", runs="count").reset_index()
    lo, hi = proportion_confint(g["hits"], g["runs"], alpha=0.05, method="wilson")
    g["rate"] = (g["hits"]/g["runs"]).round(3)
    g["ci_low"], g["ci_high"] = lo.round(3), hi.round(3)
    return g

def first_mention_rate(df):
    return (df.groupby(["engine", "intent"])["is_first"]
              .agg(hits="sum", runs="count").assign(
                  rate=lambda x: (x.hits/x.runs).round(3)).reset_index())

def rank_when_present(df):
    d = df[df["first_index"] >= 0]
    return (d.groupby(["engine","intent"])["first_index"]
              .agg(avg_pos="mean", n="count").round(1).reset_index())

def leaderboard(df, target_brand, competitors):
    """First-mention share per brand across all runs.

    W-73: compared _first_brand() output against the slug brand name, so the target's
    first_share silently read 0 whenever brand_mentions keys were title-cased.
    """
    from metrics_spec import same_brand
    brands = [target_brand] + list(competitors)
    n = len(df)
    firsts = df["brand_mentions"].apply(_first_brand)
    rows = []
    for b in brands:
        hits = int(sum(1 for fb in firsts if same_brand(fb, b)))
        rows.append({"brand": b, "first_mentions": hits,
                     "first_share": round(hits / n, 3) if n else 0})
    return pd.DataFrame(rows).sort_values("first_share", ascending=False)

def _first_brand(bm):
    present = [(b, v.get("first_index", -1)) for b, v in (bm or {}).items()
               if isinstance(v, dict) and v.get("first_index", -1) >= 0]
    return min(present, key=lambda x: x[1])[0] if present else None

def source_mix(df):
    from collections import Counter
    out = {}
    for eng, g in df.groupby("engine"):
        t = Counter()
        for tally in g["source_type_tally"]:
            t.update(tally or {})
        total = sum(t.values()) or 1
        out[eng] = {k: round(v/total, 3) for k, v in t.items()}
    return out

def refusal_and_search(df):
    return (df.groupby("engine").agg(
        refusal_rate=("refusal", "mean"),
        search_rate=("web_on", "mean"),
        avg_sources=("num_sources", "mean")).round(3).reset_index())