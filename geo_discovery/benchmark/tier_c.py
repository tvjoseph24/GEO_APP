"""Tier C — Prescriptive: winnability triage, source hit-list, cross-engine map."""
import json
import re
from collections import Counter
from urllib.parse import urlparse

import pandas as pd

from metrics_spec import fold

# ============ WINNABILITY TRIAGE (Approach C) ============
# Tunable thresholds (config constants — per D-i design)
# ============ WINNABILITY TRIAGE ============
WINNABILITY_THRESHOLDS = {
    "refusal_unwinnable": 0.5,
    "incumbent_dominant": 0.70,
    "brand_weak_first": 0.10,
    "brand_present": 0.5,
    "already_winning_mention": 0.90,
    "already_winning_first": 0.50,
}

# Intents where the brand is NAMED in the prompt — mention is 100% by construction, so
# visibility cannot be assessed.
BRAND_NAMED_INTENTS = {"branded", "comparative"}
# Intents where surfacing is not the objective.
OBJECTION_INTENTS = {"negative"}

def _ownable_claim(query, claims, stop=None, min_overlap=2):
    """Does the brand hold a claim substantively relevant to this query?

    W-13: previously matched ANY single word over 4 characters, so with a large register
    every query was "ownable" — making the commoditized rule dead code and pushing half
    the corpus into a fixed-confidence default.
    """
    stop = stop or set()
    qt = {w for w in re.findall(r"[a-z][a-z\-]{3,}", fold(query)) if w not in stop}
    if not qt:
        return False, None, 0
    best, best_n = None, 0
    for c in (claims or []):
        ct = {w for w in re.findall(r"[a-z][a-z\-]{3,}", fold(c.get("claim", "")))
              if w not in stop}
        n = len(qt & ct)
        if n > best_n:
            best, best_n = c.get("claim"), n
    return best_n >= min_overlap, best, best_n

def classify_winnability(df, target_brand, competitors, claims, thresholds=None,
                         category="", stop=None):
    """Rule-based triage. Verdicts:
        already_winning    — the brand leads; defend, do not fund visibility work
        winnable           — ownable claim plus headroom
        commoditized       — present but nothing to own
        unwinnable         — refused, or a dominant incumbent
        objection_defence  — negative-intent query; surfacing is NOT the objective
        not_applicable     — brand named in the prompt; visibility unmeasurable
    """
    from metrics_spec import same_brand
    from metrics_spec import brand_stop as _bs
    T = {**WINNABILITY_THRESHOLDS, **(thresholds or {})}
    stop = stop or _bs(target_brand, competitors, category)
    rows = []

    for query, g in df.groupby("query"):
        n = len(g)
        intent = (g["intent"].dropna().iloc[0] if g["intent"].notna().any() else None)
        mention = float(g["mentioned"].mean())
        first = float(g["is_first"].mean())
        refusal = float(g["refusal"].mean())
        neg_share = (float((g["polarity"] == "neg").mean())
                     if "polarity" in g else 0.0)

        comp_first = Counter()
        for bm in g["brand_mentions"]:
            fb = _first_brand(bm)
            if fb and not same_brand(fb, target_brand):
                comp_first[fb] += 1
        incumbent, inc_share = ("", 0.0)
        if comp_first:
            incumbent, cnt = comp_first.most_common(1)[0]
            inc_share = cnt / n

        ownable, which_claim, overlap = _ownable_claim(query, claims, stop)

        # ---------------- rule tree ----------------
        if intent in BRAND_NAMED_INTENTS:
            label = "not_applicable"
            reason = (f"'{intent}' query names the brand — mention is {mention:.0%} by "
                      f"construction; measures accuracy/positioning, not visibility")
            conf = 1.0
        elif intent in OBJECTION_INTENTS:
            label = "objection_defence"
            reason = (f"negative-intent query (mention {mention:.0%}, "
                      f"{neg_share:.0%} negatively framed) — the objective is to be "
                      f"framed fairly when named, NOT to surface more often")
            conf = round(min(1.0, 0.5 + mention / 2), 2)
        elif refusal >= T["refusal_unwinnable"]:
            label, reason = "unwinnable", f"refusal_rate={refusal:.0%} (engine declines)"
            conf = round(min(1.0, 0.5 + (refusal - T["refusal_unwinnable"])), 2)
        elif mention >= T["already_winning_mention"] and first >= T["already_winning_first"]:
            label = "already_winning"
            reason = (f"brand already leads: mention {mention:.0%}, first {first:.0%} "
                      f"— defend, do not fund visibility work here")
            conf = round(min(1.0, 0.5 + (first - T["already_winning_first"])), 2)
        elif inc_share >= T["incumbent_dominant"] and first < T["brand_weak_first"]:
            label = "unwinnable"
            reason = (f"{incumbent} dominates first-mention ({inc_share:.0%}); "
                      f"brand weak ({first:.0%})")
            conf = round(min(1.0, 0.5 + (inc_share - T["incumbent_dominant"])), 2)
        elif ownable and first < T["brand_present"]:
            label = "winnable"
            reason = (f"ownable claim ({overlap} shared term(s)) + headroom "
                      f"(first={first:.0%}, incumbent={inc_share:.0%})")
            conf = round(min(1.0, 0.4 + 0.1 * overlap + 0.3 * (1 - first)), 2)
        elif mention >= T["brand_present"] and not ownable:
            label = "commoditized"
            reason = (f"present ({mention:.0%}) but no ownable claim "
                      f"(best overlap {overlap} term(s))")
            conf = round(min(1.0, 0.5 + mention / 3), 2)
        else:
            label = "commoditized"
            reason = (f"weak on both sides: mention {mention:.0%}, ownable={ownable}, "
                      f"incumbent {inc_share:.0%}")
            conf = 0.4

        rows.append({
            "query": query, "intent": intent, "winnability": label,
            "confidence": conf,
            # headroom × reach — the correct basis for prioritising visibility work
            "opportunity": round((1.0 - first) * n, 1),
            "signals": {"mention_rate": round(mention, 3), "first_rate": round(first, 3),
                        "refusal_rate": round(refusal, 3), "neg_share": round(neg_share, 3),
                        "incumbent": incumbent, "incumbent_share": round(inc_share, 3),
                        "ownable_claim": ownable, "claim_overlap": overlap,
                        "best_claim": (which_claim or "")[:120], "n": n},
            "rule_reason": reason})
    return pd.DataFrame(rows)

def _first_brand(bm):
    present = [(b, v.get("first_index", -1)) for b, v in (bm or {}).items()
               if isinstance(v, dict) and v.get("first_index", -1) >= 0]
    return min(present, key=lambda x: x[1])[0] if present else None

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

def _bucket(brand_rate, stype, comp_density):
    if stype == "cdn_asset":
        return "Ignore (asset host)"
    if stype == "brand_owned":
        return "Own (self-reference risk)"
    if stype == "competitor_owned":
        return "Ignore (competitor property)"
    if stype == "grey_market_retailer":
        return "Displace (uncontrolled reseller)"
    if brand_rate >= 0.7:
        return "Defend"
    if brand_rate <= 0.1 and comp_density >= 1:
        return "Enter (competitor-owned narrative)"
    if 0.1 < brand_rate < 0.7:
        return "Correct (mentioned — verify facts)"
    return "Enter"

def _route(stype):
    return {"editorial_authority": "editorial pitch / expert quote / correction request",
            "trade_press": "PR / trade announcement",
            "retailer": "retailer content ops / A+ content / review seeding",
            "grey_market_retailer": "legal or commercial escalation; publish authoritative "
                                    "alternative to displace",
            "content_affiliate": "affiliate outreach / media kit",
            "community_reviews": "community participation (Pillar 5)",
            "video_social": "creator brief",
            "reference": "Wikipedia/Wikidata accuracy edit",
            "brand_owned": "own-surface publish (Pillar 1/2)"}.get(stype, "outreach")


# ============ SOURCE HIT-LIST ============
def source_hitlist(df, target_brand, competitors=(), alias_overrides=None, top_n=40):
    """Brand-RELATIVE targeting. Ranked by brand_absence_index = influence × absence,
    i.e. the pages that shape answers while not mentioning you."""
    from collections import Counter, defaultdict
    from metrics_spec import build_registry, fold
    reg = build_registry(target_brand, list(competitors), alias_overrides)
    tpat = reg.target.pattern
    cpats = [(c.canonical, c.pattern) for c in reg.competitors]

    rec = defaultdict(lambda: {"cited": 0, "queries": set(), "engines": set(),
                               "brand_runs": 0, "comp_runs": 0, "first_runs": 0,
                               "pos_runs": 0, "types": Counter(), "urls": Counter(),
                               "intents": Counter()})
    for _, r in df.iterrows():
        low = fold(r.get("response_text") or "")
        has_brand = bool(tpat.search(low))
        n_comp = sum(1 for _, p in cpats if p.search(low))
        for s in (r.get("sources") or []):
            d = s.get("domain") or ""
            if not d:
                continue
            e = rec[d]
            e["cited"] += 1
            e["queries"].add(r.get("query"))
            e["engines"].add(r.get("engine"))
            e["intents"][r.get("intent")] += 1
            e["types"][s.get("source_type", "other")] += 1
            u = s.get("resolved_url") or s.get("url")
            if u:
                e["urls"][u] += 1
            e["brand_runs"] += int(has_brand)
            e["comp_runs"] += n_comp
            e["first_runs"] += int(bool(r.get("is_first")))
            e["pos_runs"] += int(r.get("polarity") == "pos")

    base_first = float(df["is_first"].mean()) if "is_first" in df else 0.0
    rows = []
    for d, e in rec.items():
        n = e["cited"]
        bpr = e["brand_runs"] / n
        st = e["types"].most_common(1)[0][0]
        rows.append({
            "domain": d, "times_cited": n, "n_queries": len(e["queries"]),
            "n_engines": len(e["engines"]), "source_type": st,
            "top_intent": (e["intents"].most_common(1)[0][0] if e["intents"] else None),
            "brand_present_rate": round(bpr, 3),
            "competitor_density": round(e["comp_runs"] / n, 2),
            "brand_absence_index": round(n * (1 - bpr) * len(e["engines"]), 1),
            "first_mention_lift": round(e["first_runs"] / n - base_first, 3),
            "positive_rate": round(e["pos_runs"] / n, 3),
            "top_url": (e["urls"].most_common(1)[0][0] if e["urls"] else None),
            "bucket": _bucket(bpr, st, e["comp_runs"] / n),
            "access_route": _route(st),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (out.sort_values(["brand_absence_index", "times_cited"], ascending=False)
            .head(top_n).reset_index(drop=True))

# ============ CROSS-ENGINE AGREEMENT ============
def cross_engine_map(df, target_brand, win_hi=0.5, lose_lo=0.05, diverge=0.40):
    """Where engines agree (structural truth) vs diverge (engine-specific play).

    W-16: the three buckets were exclusive FILTERS, not a partition — half the queries
    appeared nowhere, and `lose_everywhere` required an exact 0.0 so one first-mention
    in one run disqualified a query. The metric is now labelled, because
    answer_consistency reports divergence on `mentioned` while this uses `is_first`.
    """
    per = df.groupby(["engine", "query"])["is_first"].mean().unstack(0)
    if per.shape[1] < 2:
        return {"note": "single engine — no cross-engine comparison",
                "metric": "is_first", "detail": None}
    spread = per.max(axis=1) - per.min(axis=1)
    win = per[(per > win_hi).all(axis=1)].index.tolist()
    lose = per[(per <= lose_lo).all(axis=1)].index.tolist()
    div = spread[spread > diverge].index.tolist()
    assigned = set(win) | set(lose) | set(div)
    mixed = [q for q in per.index if q not in assigned]
    return {
        "engines": list(per.columns), "metric": "is_first",
        "thresholds": {"win_hi": win_hi, "lose_lo": lose_lo, "diverge": diverge},
        "win_everywhere": win,
        "lose_everywhere": lose,
        "diverges_by_engine": div,
        "mixed_middle": mixed,
        "coverage": {"total_queries": int(len(per)), "assigned": len(assigned),
                     "unassigned": len(mixed)},
        "divergence_detail": [
            {"query": q, "spread_pp": round(float(spread[q]) * 100, 1),
             "best_engine": per.loc[q].idxmax(), "worst_engine": per.loc[q].idxmin()}
            for q in div],
    }