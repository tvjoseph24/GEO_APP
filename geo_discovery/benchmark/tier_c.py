"""Tier C — Prescriptive: winnability triage, source hit-list, cross-engine map."""
import json
import pandas as pd
from urllib.parse import urlparse
from collections import Counter

# ============ WINNABILITY TRIAGE (Approach C) ============
# Tunable thresholds (config constants — per D-i design)
WINNABILITY_THRESHOLDS = {
    "refusal_unwinnable": 0.5,
    "incumbent_dominant": 0.70,
    "brand_weak_first": 0.10,
    "brand_present": 0.5,
}

def _ownable_claim(query, claims):
    """Does the brand have a claim relevant to this query?"""
    ql = query.lower()
    return any(any(w in ql for w in c.get("claim","").lower().split() if len(w) > 4)
               for c in claims)

def classify_winnability(df, target_brand, competitors, claims,
                         thresholds=None):
    """RULES: classify each query. Returns label + triggering signals (transparent)."""
    T = {**WINNABILITY_THRESHOLDS, **(thresholds or {})}
    rows = []
    for query, g in df.groupby("query"):
        n = len(g)
        mention = g["mentioned"].mean()
        first = g["is_first"].mean()
        refusal = g["refusal"].mean()
        # incumbent dominance among competitors
        comp_first = Counter()
        for bm in g["brand_mentions"]:
            fb = _first_brand(bm)
            if fb and fb != target_brand: comp_first[fb] += 1
        incumbent, inc_share = ("", 0)
        if comp_first:
            incumbent, cnt = comp_first.most_common(1)[0]
            inc_share = cnt / n
        ownable = _ownable_claim(query, claims)

        # RULE TREE
        if refusal >= T["refusal_unwinnable"]:
            label, reason = "unwinnable", f"refusal_rate={refusal:.0%} (engine declines)"
        elif inc_share >= T["incumbent_dominant"] and first < T["brand_weak_first"]:
            label, reason = "unwinnable", f"{incumbent} dominates first-mention ({inc_share:.0%}); brand weak ({first:.0%})"
        elif mention >= T["brand_present"] and not ownable:
            label, reason = "commoditized", f"present ({mention:.0%}) but no ownable claim"
        elif ownable and (0 < first < T["brand_present"] or (mention < T["brand_present"] and inc_share < T["incumbent_dominant"])):
            label, reason = "winnable", f"ownable claim + room (first={first:.0%}, incumbent={inc_share:.0%})"
        else:
            label, reason = "commoditized", "default: no strong signal either way"

        # confidence: distance from nearest threshold boundary
        conf = _confidence(label, refusal, inc_share, first, mention, T)
        rows.append({"query": query, "winnability": label, "confidence": conf,
                     "signals": {"mention_rate": round(mention,3), "first_rate": round(first,3),
                                 "refusal_rate": round(refusal,3), "incumbent": incumbent,
                                 "incumbent_share": round(inc_share,3), "ownable_claim": ownable,
                                 "n": n},
                     "rule_reason": reason})
    return pd.DataFrame(rows)

def _first_brand(bm):
    present = [(b, v.get("first_index",-1)) for b,v in (bm or {}).items()
               if v.get("first_index",-1) >= 0]
    return min(present, key=lambda x:x[1])[0] if present else None

def _confidence(label, refusal, inc_share, first, mention, T):
    """Distance-from-threshold → confidence (per D-iv)."""
    if label == "unwinnable":
        d = max(refusal - T["refusal_unwinnable"], inc_share - T["incumbent_dominant"])
    elif label == "winnable":
        d = abs(mention - T["brand_present"])
    else:
        d = 0.15  # commoditized = medium default
    return round(min(1.0, 0.5 + abs(d)), 2)

# LLM rationale (opt-in) — NARRATES the rule decision, does NOT re-decide (per D-iii)
RATIONALE_PROMPT = """Write ONE concise sentence explaining this winnability verdict for a
brand strategist. Use the signals; do NOT change the verdict.
Query: "{query}" | Verdict: {label} | Signals: {signals}"""

def add_llm_rationale(gateway, wdf, model="gemini-2.5-flash", module="benchmark"):
    out = []
    for _, r in wdf.iterrows():
        try:
            resp = gateway.generate(model, RATIONALE_PROMPT.format(
                query=r["query"], label=r["winnability"], signals=json.dumps(r["signals"])),
                module=module, task="winnability_rationale", temperature=0.3)
            out.append(resp["text"].strip())
        except Exception:
            out.append(r["rule_reason"])
    wdf = wdf.copy(); wdf["rationale"] = out
    return wdf

# ============ SOURCE HIT-LIST ============
def source_hitlist(df, target_brand):
    """Top domains driving answers + whether brand is present on them."""
    dom_count, dom_type = Counter(), {}
    for srcs in df["sources"]:
        for s in (srcs or []):
            d = s.get("domain", "")
            if d:
                dom_count[d] += 1
                dom_type[d] = s.get("source_type", "other")
    rows = [{"domain": d, "times_cited": c, "source_type": dom_type.get(d, "other")}
            for d, c in dom_count.most_common(30)]
    return pd.DataFrame(rows)

# ============ CROSS-ENGINE AGREEMENT ============
def cross_engine_map(df, target_brand):
    """Where engines agree (structural truth) vs diverge (engine-specific)."""
    per = df.groupby(["engine","query"])["is_first"].mean().unstack(0)
    if per.shape[1] < 2:
        return {"note": "single engine — no cross-engine comparison", "detail": None}
    engines = list(per.columns)
    agree_win = per[(per > 0.5).all(axis=1)].index.tolist()      # win everywhere
    agree_lose = per[(per == 0).all(axis=1)].index.tolist()      # lose everywhere
    diverge = per[per.max(axis=1) - per.min(axis=1) > 0.4].index.tolist()
    return {"engines": engines, "win_everywhere": agree_win,
            "lose_everywhere": agree_lose, "diverges_by_engine": diverge}