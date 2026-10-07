"""social.py — community and social source analysis. W-55.

Pillar 5 was structurally unreachable: no gap type routed to Social/Community, so
build_pillar_strategies marked it `_skipped` and its prompt was never called. The data was
captured from the first audit and never read.

Nothing new is collected. Every figure here is arithmetic over `source_type_tally` and
`sources`, both already on every audit row.

Design notes
------------
* The unit of action is a THREAD, not a domain. `top_url` is already captured per source
  and was discarded in favour of domain aggregation — so the hit-list said "engage
  reddit.com", which is not an action anyone can take.
* Community work is the easiest place in the framework to recommend something a brand
  should not do. This module reports where community sources carry weight; the prompt
  constrains what may be recommended about it.
* Community is a MEMORY play. It cannot be published into, so it has a slow clock and is
  measured on a proxy — thread count, not mention rate this quarter.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from metrics_spec import build_registry, fold

# Buckets that are community or social voice. Distinct from `content_affiliate`, which
# reads like editorial and is a publishing problem, and from `retailer`, whose reviews are
# commerce-attached.
SOCIAL_TYPES = ("community_reviews", "video_social")

# Platforms where a brand may legitimately participate under its own name, as opposed to
# review aggregators where it cannot. Drives the ROUTE, not the measurement — a brand can
# correct a factual error on Reddit; it cannot post on Trustpilot as itself.
PARTICIPATORY = ("reddit.com", "quora.com", "stackexchange", "mumsnet", "netmums",
                 "babycenter", "babycentre", "whattoexpect", "thebump", "madeformums",
                 "youtube.com", "youtu.be", "tiktok.com", "instagram.com",
                 "facebook.com", "peanut-app.io", "discord", "substack.com")

MIN_CITATIONS_FOR_THREAD = 2      # a single citation is noise, not a thread that matters
MIN_RUNS_FOR_RATE = 15            # below this a share is not worth reporting

def _is_social(stype):
    return stype in SOCIAL_TYPES

def _participatory(domain):
    d = (domain or "").lower()
    return any(p in d for p in PARTICIPATORY)

# ---------------------------------------------------------------- 1 · intent
def by_intent(df, min_runs=MIN_RUNS_FOR_RATE):
    """Which query categories trigger a community citation?

    The framework describes Pillar 5 as "query-triggered", and this is the figure that
    makes that concrete. It has never been computed.
    """
    rows = []
    for intent, g in df.groupby("intent", dropna=False):
        n = len(g)
        if n < min_runs:
            continue
        tally = Counter()
        runs_with_social = 0
        for t in g.get("source_type_tally", []):
            t = t or {}
            tally.update(t)
            if any(t.get(s) for s in SOCIAL_TYPES):
                runs_with_social += 1
        total_cites = sum(tally.values()) or 1
        soc = sum(tally.get(s, 0) for s in SOCIAL_TYPES)
        lo, hi = proportion_confint(runs_with_social, n, alpha=0.05, method="wilson")
        rows.append({
            "intent": intent, "runs": int(n),
            "runs_citing_social": int(runs_with_social),
            "social_trigger_rate": round(runs_with_social / n, 3),
            "ci_low": round(lo, 3), "ci_high": round(hi, 3),
            "social_share_of_citations": round(soc / total_cites, 3),
            "community_reviews": int(tally.get("community_reviews", 0)),
            "video_social": int(tally.get("video_social", 0)),
            "editorial_authority": int(tally.get("editorial_authority", 0)),
            "total_citations": int(total_cites),
            # a category that cites community MORE than editorial is one where consensus
            # outweighs authority — a different strategic problem
            "community_over_editorial": bool(
                soc > tally.get("editorial_authority", 0))})
    return sorted(rows, key=lambda r: -r["social_trigger_rate"])

# ---------------------------------------------------------------- 2 · framing
def by_framing(df, target_brand):
    """Does a community citation correlate with hedged, advisory or negative framing?

    If community-cited answers hedge more, that says where scepticism enters the answer —
    which is a finding about the SOURCE MIX, not about the brand's content.
    """
    m = df[df["mentioned"] == True].copy()                        # noqa: E712
    if m.empty or "source_type_tally" not in m:
        return {"mentions": 0,
                "note": "the brand is never mentioned in this corpus"}

    def _has_social(t):
        return any((t or {}).get(s) for s in SOCIAL_TYPES)

    m["_social"] = m["source_type_tally"].apply(_has_social)
    out = {"mentions": int(len(m)),
           "mentions_with_social_citation": int(m["_social"].sum()),
           "social_citation_share_of_mentions": round(float(m["_social"].mean()), 3)}

    def _block(g, label):
        if len(g) < MIN_RUNS_FOR_RATE:
            return {"_note": f"only {len(g)} mention(s) — too few to report a rate"}
        pol = (g["polarity"].value_counts(normalize=True)
               if "polarity" in g else pd.Series(dtype=float))
        b = {"mentions": int(len(g)),
             "pos": round(float(pol.get("pos", 0)), 3),
             "neu": round(float(pol.get("neu", 0)), 3),
             "neg": round(float(pol.get("neg", 0)), 3)}
        for col, key in (("hedged", "hedged"), ("advisory_referral", "advisory"),
                         ("has_negative", "any_negative")):
            if col in g.columns and g[col].notna().any():
                b[key] = round(float(g[col].fillna(False).astype(bool).mean()), 3)
        return b

    with_s, without_s = m[m["_social"]], m[~m["_social"]]
    out["with_social_citation"] = _block(with_s, "with")
    out["without_social_citation"] = _block(without_s, "without")

    # ---- the comparison, stated plainly ----
    a, b = out["with_social_citation"], out["without_social_citation"]
    deltas, notes = {}, []
    for k in ("pos", "neg", "hedged", "advisory", "any_negative"):
        if isinstance(a.get(k), float) and isinstance(b.get(k), float):
            deltas[k] = round(a[k] - b[k], 3)
    out["delta_with_minus_without"] = deltas
    if deltas.get("hedged", 0) >= 0.10:
        notes.append(f"Answers citing a community source hedge "
                     f"{deltas['hedged'] * 100:.0f}pp more often. Scepticism is entering "
                     f"through the source mix, not through the brand's own content — so "
                     f"the remedy is corroboration and community presence, not messaging.")
    if deltas.get("neg", 0) >= 0.05 or deltas.get("any_negative", 0) >= 0.05:
        notes.append(f"Negative framing is "
                     f"{max(deltas.get('neg', 0), deltas.get('any_negative', 0)) * 100:.0f}pp "
                     f"more likely where a community source is cited. Community consensus "
                     f"is shaping how the brand is described.")
    if deltas.get("pos", 0) <= -0.10:
        notes.append(f"Positive framing is {abs(deltas['pos']) * 100:.0f}pp LESS likely "
                     f"where a community source is cited — community voice is currently "
                     f"working against the brand.")
    elif deltas.get("pos", 0) >= 0.10:
        notes.append(f"Positive framing is {deltas['pos'] * 100:.0f}pp MORE likely where a "
                     f"community source is cited — existing community voice is an asset to "
                     f"protect, not a gap to fill.")
    out["interpretation"] = notes
    if "has_negative" in m.columns and not m["has_negative"].fillna(False).any():
        out["negative_detection_caveat"] = (
            "No directed criticism was detected anywhere in this corpus, so the negative "
            "comparison above is uninformative rather than reassuring. Verify the category "
            "cue set before concluding community voice is benign.")
    return out

# ---------------------------------------------------------------- 3 · threads
def threads(df, target_brand, competitors=(), alias_overrides=None,
            min_citations=MIN_CITATIONS_FOR_THREAD, top=30):
    """Thread-level rows. THE actionable output of this module.

    `top_url` is captured on every source and was discarded in favour of domain
    aggregation, so the hit-list recommended engaging `reddit.com` — which is not
    something anyone can do. A thread is the unit of action.
    """
    reg = build_registry(target_brand, list(competitors), alias_overrides)
    tpat = reg.target.pattern
    cpats = [(c.canonical, c.pattern) for c in reg.competitors]

    rec = defaultdict(lambda: {"cited": 0, "queries": set(), "engines": set(),
                               "intents": Counter(), "brand_runs": 0, "comp": Counter(),
                               "hedged": 0, "neg": 0, "pos": 0, "advisory": 0,
                               "domain": "", "stype": "", "title": ""})
    for _, r in df.iterrows():
        low = fold(r.get("response_text") or "")
        has_brand = bool(tpat.search(low))
        named = [n for n, p in cpats if p.search(low)]
        for s in (r.get("sources") or []):
            if not _is_social(s.get("source_type")):
                continue
            url = s.get("resolved_url") or s.get("url")
            # An unresolved redirect is not a thread anyone can visit, and its `domain` is
            # a page title. Excluded rather than presented as a target.
            if not url or "vertexaisearch" in str(url) or s.get("domain_is_title"):
                continue
            e = rec[url]
            e["cited"] += 1
            e["domain"] = s.get("domain") or e["domain"]
            e["stype"] = s.get("source_type") or e["stype"]
            e["title"] = (s.get("title") or e["title"])[:160]
            e["queries"].add(r.get("query"))
            e["engines"].add(r.get("engine"))
            e["intents"][r.get("intent")] += 1
            e["brand_runs"] += int(has_brand)
            for n in named:
                e["comp"][n] += 1
            e["hedged"] += int(bool(r.get("hedged")))
            e["advisory"] += int(bool(r.get("advisory_referral")))
            if r.get("polarity") == "neg":
                e["neg"] += 1
            elif r.get("polarity") == "pos":
                e["pos"] += 1

    rows = []
    for url, e in rec.items():
        n = e["cited"]
        if n < min_citations:
            continue
        bpr = e["brand_runs"] / n
        top_comp = e["comp"].most_common(1)[0] if e["comp"] else (None, 0)
        rows.append({
            "url": url, "domain": e["domain"], "source_type": e["stype"],
            "title": e["title"] or "(no title captured)",
            "times_cited": n,
            "n_queries": len(e["queries"]), "n_engines": len(e["engines"]),
            "top_intent": (e["intents"].most_common(1)[0][0] if e["intents"] else None),
            "brand_present_rate": round(bpr, 3),
            # influence weighted by absence: the threads that shape answers WITHOUT
            # mentioning the brand
            "brand_absence_index": round(n * (1 - bpr) * len(e["engines"]), 1),
            "top_competitor": top_comp[0],
            "competitor_mentions": top_comp[1],
            "hedged_rate": round(e["hedged"] / n, 3),
            "advisory_rate": round(e["advisory"] / n, 3),
            "negative_rate": round(e["neg"] / n, 3),
            "positive_rate": round(e["pos"] / n, 3),
            "participatory": _participatory(e["domain"]),
            "bucket": _bucket(bpr, e["neg"] / n, _participatory(e["domain"])),
            "route": _route(e["stype"], _participatory(e["domain"]), bpr)})
    return sorted(rows, key=lambda r: -r["brand_absence_index"])[:top]

def _bucket(brand_rate, neg_rate, participatory):
    if neg_rate >= 0.15:
        return ("Correct (negative, participatory)" if participatory
                else "Monitor (negative, not participatory)")
    if brand_rate >= 0.6:
        return "Defend (brand already present)"
    if brand_rate <= 0.1:
        return ("Enter (absent, participatory)" if participatory
                else "Earn (absent, not participatory)")
    return "Strengthen (inconsistently present)"

def _route(stype, participatory, brand_rate):
    """The ACTION, and the constraint on it. Community work is the easiest place in the
    framework to recommend something a brand should not do."""
    if stype == "video_social":
        return ("creator brief or disclosed partnership — the brand cannot post as a "
                "third party")
    if not participatory:
        return ("no direct participation possible: earn mention through the other pillars, "
                "or request a factual correction from the platform")
    if brand_rate >= 0.6:
        return "monitor and protect: the brand is already present and correctly described"
    return ("disclosed participation under an identified brand account, or a factual "
            "correction where the thread is wrong. NEVER undisclosed seeding")

# ---------------------------------------------------------------- 4 · engines
def by_engine(df):
    """Engines weight community voice differently, and that belongs in an overlay.

    `provenance.per_engine.community_pct` exists and is never surfaced anywhere.
    """
    out = {}
    for eng, g in df.groupby("engine"):
        t = Counter()
        runs_with_social = 0
        for tal in g.get("source_type_tally", []):
            tal = tal or {}
            t.update(tal)
            if any(tal.get(s) for s in SOCIAL_TYPES):
                runs_with_social += 1
        tot = sum(t.values()) or 1
        n = max(len(g), 1)
        out[eng] = {
            "runs": int(len(g)),
            "social_share_of_citations": round(
                sum(t.get(s, 0) for s in SOCIAL_TYPES) / tot, 3),
            "community_reviews_pct": round(t.get("community_reviews", 0) / tot, 3),
            "video_social_pct": round(t.get("video_social", 0) / tot, 3),
            "editorial_pct": round(t.get("editorial_authority", 0) / tot, 3),
            "social_trigger_rate": round(runs_with_social / n, 3),
            "citations": int(tot)}
    shares = {e: v["social_share_of_citations"] for e, v in out.items()}
    if not shares:
        return out
    hi = max(shares, key=shares.get)
    # Checked FIRST and without requiring two engines. A single-engine audit is common,
    # and "no engine listens to community voice" is a finding in its own right — it does
    # not depend on having a second engine to compare against.
    if shares[hi] < 0.02:
        out["_divergence"] = (
            f"No engine cites community sources materially (highest {shares[hi]:.1%}). "
            f"Community work is unlikely to move machine visibility in this category, "
            f"whatever its brand value.")
    elif len(shares) > 1:
        lo = min(shares, key=shares.get)
        if shares[lo] > 0 and shares[hi] / max(shares[lo], 0.001) >= 2.0:
            out["_divergence"] = (
                f"{hi} cites community sources {shares[hi] / max(shares[lo], 0.001):.1f}× "
                f"as often as {lo} ({shares[hi]:.1%} against {shares[lo]:.1%}). Community "
                f"work will move {hi} materially and {lo} barely — this belongs in an "
                f"engine overlay, not the overall strategy.")
    return out

# ---------------------------------------------------------------- 5 · report
def social_report(df, target_brand, competitors=(), alias_overrides=None):
    """The bundle. Consumed by metrics_bundle, gap_detect and the Pillar-5 prompt."""
    intents = by_intent(df)
    fr = by_framing(df, target_brand)
    th = threads(df, target_brand, competitors, alias_overrides)
    eng = by_engine(df)

    tally = Counter()
    for t in df.get("source_type_tally", []):
        tally.update(t or {})
    tot = sum(tally.values()) or 1
    social_cites = sum(tally.get(s, 0) for s in SOCIAL_TYPES)
    overall = round(social_cites / tot, 4)

    trig = [r for r in intents if r["social_trigger_rate"] >= 0.15]
    absent_threads = [t for t in th if t["brand_present_rate"] <= 0.1]
    neg_threads = [t for t in th if t["negative_rate"] >= 0.15]

    # A category where community citation is negligible should say so, loudly. Pillar 5 is
    # a real cost, and recommending it where engines do not listen is the most expensive
    # mistake available in this pillar.
    if overall < 0.02:
        verdict, why = "negligible", (
            f"Community and social sources are only {overall:.1%} of all citations. "
            f"Engines are not listening to community voice in this category, so community "
            f"work will not move machine visibility — whatever its value for brand "
            f"affinity. Pillar 5 should be minimal or empty, and the strategy should say "
            f"why rather than leaving it unexplained.")
    elif trig:
        _rates = ", ".join("{:.0%}".format(r["social_trigger_rate"]) for r in trig)
        verdict, why = "query_triggered", (
            f"Community sources are {overall:.1%} of citations overall, but they are "
            f"concentrated: {', '.join(r['intent'] for r in trig)} "
            f"{'queries' if len(trig) > 1 else 'queries'} trigger a community citation "
            f"in {_rates} of runs respectively. That is where community work pays, and "
            f"nowhere else.")
    else:
        verdict, why = "diffuse", (
            f"Community sources are {overall:.1%} of citations, spread evenly across query "
            f"types with no category above 15%. There is no concentrated opportunity — "
            f"treat this as background corroboration rather than a targeted play.")

    return {
        "overall_social_share": overall,
        "community_reviews_share": round(tally.get("community_reviews", 0) / tot, 4),
        "video_social_share": round(tally.get("video_social", 0) / tot, 4),
        "total_citations": int(tot),
        "verdict": verdict, "why": why,
        "by_intent": intents,
        "triggered_intents": [r["intent"] for r in trig],
        "framing": fr,
        "threads": th,
        "n_threads": len(th),
        "n_threads_brand_absent": len(absent_threads),
        "n_threads_negative": len(neg_threads),
        "participatory_threads": sum(1 for t in th if t["participatory"]),
        "by_engine": eng,
        "clock": "memory",
        "clock_note": (
            "Community is a MEMORY play with a slow clock. A brand cannot publish into a "
            "community, so this is participation, correction and earned mention — measured "
            "on a proxy (threads where the brand is present and correctly described), not "
            "on mention rate this quarter."),
        "constraint": (
            "Only two plays are legitimate: disclosed participation under an identified "
            "brand account, and factual correction where a thread is wrong. Undisclosed "
            "seeding is off the table — it is a disclosure breach on most platforms and a "
            "regulatory one in a health-adjacent category."),
    }

def social_summary(rep):
    """One line for the QA gate and the diagnostic dump."""
    if not rep:
        return {}
    return {"verdict": rep.get("verdict"),
            "overall_share": rep.get("overall_social_share"),
            "triggered_intents": rep.get("triggered_intents"),
            "threads": rep.get("n_threads"),
            "threads_brand_absent": rep.get("n_threads_brand_absent"),
            "threads_negative": rep.get("n_threads_negative"),
            "engine_divergence": (rep.get("by_engine") or {}).get("_divergence")}