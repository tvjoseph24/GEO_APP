"""Reason-richness v2 — brand-scoped CONTENT QUALITY, category-agnostic.

Replaces: presence of a skincare keyword list inside an arbitrary ±240-CHARACTER window
around the FIRST brand mention (structurally 0 for any non-skincare brand, truncated
mid-sentence, blind to reasons stated before the brand name, and treats a richly
NEGATIVE description as a win).
"""
from __future__ import annotations

import json
import re

import pandas as pd

from benchmark.spans import build_multi_brand_corpus
from metrics_spec import brand_stop, fold


# ---- generic, category-independent reason dimensions ----
PATTERNS = {
    "quantified": [r"\b\d+(?:[.,]\d+)?\s?%", r"\b\d+(?:[.,]\d+)?\s?(?:mg|mcg|ml|g|kg|iu|oz|kcal)\b",
                   r"\b\d+\s?[- ]?(?:hour|hr|day|week|month|year)s?\b",
                   r"\bup to \d+", r"\b\d+\s?x\b", r"\b\d+\s?(?:fold|times)\b",
                   r"\bstage\s?\d\b", r"\b\d+\+?\s?months?\b"],
    "clinical": [r"\bclinical(?:ly)?\b", r"\bstud(?:y|ies)\b", r"\btrial(?:s)?\b",
                 r"\bn\s?=\s?\d+", r"\bpeer[- ]reviewed\b", r"\brandomi[sz]ed\b",
                 r"\bmeta[- ]analys", r"\bproven\b", r"\bresearch(?:ed)?\b",
                 r"\bpublished\b", r"\bevidence[- ]based\b"],
    "certification": [r"\bcertified\b", r"\bapproved\b", r"\bseal\b", r"\baccredited\b",
                      r"\bcompliant\b", r"\bFDA\b", r"\bEFSA\b", r"\bUSDA\b",
                      r"\borganic\b", r"\bnon[- ]GMO\b", r"\bhalal\b", r"\bkosher\b",
                      r"\bdemeter\b", r"\bEU regulation\b", r"\bcodex\b"],
    "mechanism": [r"\bbecause\b", r"\bwhich (?:helps|allows|means|supports|aids)\b",
                  r"\bworks by\b", r"\bby (?:supporting|restoring|reducing|providing|mimicking)",
                  r"\bresult(?:s|ing) in\b", r"\bso that\b", r"\bthis (?:helps|means)\b",
                  r"\bin order to\b", r"\bcontributes to\b"],
    "comparative": [r"\bmore .{3,25} than\b", r"\bcompared (?:to|with)\b", r"\bunlike\b",
                    r"\boutperform", r"\bbetter than\b", r"\bcloser to\b",
                    r"\bcloser resembl", r"\bversus\b", r"\bwhereas\b"],
    # use_specificity is populated at runtime from the lexicon, not hardcoded
    "use_specificity": [r"\bfor (?:[a-z]{4,}(?:\s[a-z]{4,})?)\b",
                        r"\bages? \d+", r"\bfrom \d+\s?(?:months?|years?|weeks?)\b",
                        r"\bsuitable for\b", r"\bdesigned for\b",
                        r"\b(?:stage|step|type|level)\s?\d\b"],
    "provenance": [r"\baccording to\b", r"\bas (?:reported|noted|stated) by\b",
                   r"\bcites?\b", r"\bsource[sd]?:", r"\brecommended by\b",
                   r"\bper the\b", r"\bstudies (?:show|suggest)\b"],
    "hedge": [r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bsome (?:parents|users|studies)\b",
              r"\bgenerally\b", r"\breportedly\b", r"\bclaims? (?:that|to)\b",
              r"\bmarketed as\b", r"\bit'?s said\b", r"\bdepend(?:s|ing)\b",
              r"\bnot (?:clinically )?proven\b"],
    "negative": [r"\bdrawback", r"\bdownside", r"\bcriticis", r"\bcriticiz",
                 r"\bcomplain", r"\bexpensive\b", r"\bpricey\b", r"\brecall",
                 r"\bcontrovers", r"\bconcern(?:s|ed)? (?:about|over)\b",
                 r"\bnot (?:suitable|ideal|recommended)\b"],
}
_C = {k: [re.compile(p, re.I) for p in v] for k, v in PATTERNS.items()}

# active-with-dose, e.g. "1.2 g of GOS", "0.3% DHA" — strongest single winner signal
DOSED = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:%|mg|mcg|g|ml|iu)\s?(?:of\s+)?"
                   r"([a-z][a-z\-\s]{2,28})", re.I)

WEIGHTS = {"quantified": 2.0, "dosed_actives": 3.0, "clinical": 2.0,
           "certification": 1.5, "mechanism": 1.5, "comparative": 1.0,
           "use_specificity": 1.0, "provenance": 1.5, "named_actives": 1.0,
           "hedge": -0.8, "negative": -1.0}

LEXICON_PROMPT = """For the product category "{category}", list the terminology that
expert content in this category uses.

Return STRICT JSON:
{{"actives": ["ingredient / functional component names, 20-30 items"],
  "certifications": ["certification, standard and regulatory body names, 10-15 items"],
  "attributes": ["differentiating product attributes, 10-15 items"]}}

Be specific and factual. Lowercase. No brand names."""



def build_lexicon(claims, category, gateway=None, model=None, module="benchmark",
                  cap=40, brand="", competitors=(), stop=None):
    """Curated, not dumped. Multi-word phrases preferred; generic tokens excluded."""
    stop = stop or brand_stop(brand, competitors, category)

    terms = {"actives": set(), "certs": set(), "attributes": set()}

    def _ok(term):
        term = term.strip()
        return (4 < len(term) <= 40
                and all(w not in stop for w in term.split())
                and not term.isdigit())

    for c in (claims or []):
        txt = fold(c.get("claim", ""))
        cat = c.get("category", "")
        # only compound / hyphenated / acronym-like terms carry lexical signal
        cands = re.findall(r"\b[a-z][a-z\-]{4,}(?:\s[a-z][a-z\-]{4,})?\b", txt)
        cands += [m.lower() for m in re.findall(r"\b[A-Z]{2,}(?:/[A-Z]{2,})?\b",
                                               c.get("claim", ""))]
        for t in cands:
            if not _ok(t):
                continue
            if cat == "ingredient":
                terms["actives"].add(t)
            elif cat == "credential":
                terms["certs"].add(t)
    if gateway and model and category:
        try:
            res = gateway.generate_json(model, LEXICON_PROMPT.format(category=category),
                                        expect="dict", module=module, task="lexicon",
                                        temperature=0.0, max_tokens=1500)
            d = res["data"] or {}
            terms["actives"].update(fold(x) for x in d.get("actives", []) if _ok(fold(x)))
            terms["certs"].update(fold(x) for x in d.get("certifications", [])
                                  if _ok(fold(x)))
            terms["attributes"].update(fold(x) for x in d.get("attributes", [])
                                       if _ok(fold(x)))
        except Exception:
            pass
    # longest first, capped — prevents one bloated category dominating the score
    return {k: sorted(v, key=lambda x: (-len(x), x))[:cap] for k, v in terms.items()}

def _lex_hits(low, terms):
    return sum(1 for t in terms
               if re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", low))
def richness_vector(span, lexicon):
    low = fold(span)
    tok = max(len(span.split()), 1)
    v = {k: sum(len(rx.findall(low)) for rx in rxs) for k, rxs in _C.items()}
    v["named_actives"] = _lex_hits(low, lexicon.get("actives", []))
    v["certification"] += _lex_hits(low, lexicon.get("certs", []))
    v["dosed_actives"] = len(DOSED.findall(low))
    v["reason_types"] = sum(1 for k in ("quantified", "clinical", "certification",
                                        "mechanism", "comparative", "use_specificity",
                                        "provenance")
                            if v.get(k, 0) > 0)
    v["substantive_types"] = sum(1 for k in SUBSTANTIVE if v.get(k, 0) > 0)
    v["contextual_types"] = sum(1 for k in CONTEXTUAL if v.get(k, 0) > 0)
    v["n_tokens"] = tok
    v["score_raw"] = round(sum(WEIGHTS.get(k, 0) * v.get(k, 0) for k in WEIGHTS), 2)
    v["score_per_100t"] = round(v["score_raw"] / tok * 100, 2)
    return v

DIMS = ["quantified", "dosed_actives", "clinical", "certification", "mechanism",
        "comparative", "use_specificity", "provenance", "named_actives"]
# Substantive = the reasons that differentiate. Contextual = scope-setting that nearly
# every answer contains, so it must not count towards "has a reason".
SUBSTANTIVE = ["dosed_actives", "clinical", "certification", "mechanism",
               "provenance", "quantified"]
CONTEXTUAL = ["use_specificity", "comparative", "named_actives"]
def reason_richness_v2(df, brands, lexicon, alias_overrides=None):
    """Returns (leaderboard, per-span detail). Normalised per 100 tokens so this
    measures DENSITY of reasons, not verbosity."""
    corpus = build_multi_brand_corpus(df, brands, alias_overrides=alias_overrides)
    if corpus.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for _, s in corpus.iterrows():
        v = richness_vector(s["span"], lexicon)
        v.update({"brand": s["brand"], "engine": s["engine"], "intent": s["intent"],
                  "query": s["query"], "span_id": s["span_id"],
                  "n_co_mentioned": s["n_co_mentioned"]})
        rows.append(v)
    detail = pd.DataFrame(rows)
    agg = (detail.groupby("brand").agg(
        spans=("score_raw", "count"),
        avg_richness=("score_raw", "mean"),
        richness_per_100t=("score_per_100t", "mean"),
        avg_tokens=("n_tokens", "mean"),
        reason_types=("reason_types", "mean"),
        quantified=("quantified", "mean"),
        dosed_actives=("dosed_actives", "mean"),
        clinical=("clinical", "mean"),
        certification=("certification", "mean"),
        mechanism=("mechanism", "mean"),
        provenance=("provenance", "mean"),
        hedge=("hedge", "mean"),
        negative=("negative", "mean"))
        .round(2).sort_values("avg_richness", ascending=False).reset_index())
    return agg, detail

def reason_gap(detail, target_brand, min_spans=8, min_deficit=0.25):
    """Per-100-token normalised. Raw per-span counts are confounded by span length —
    the target's spans are 2-3x longer because the brand is named in the prompt."""
    if detail is None or detail.empty:
        return pd.DataFrame()
    d = detail.copy()
    for c in DIMS:
        d[c + "_p100"] = d[c] / d["n_tokens"].clip(lower=1) * 100
    out = []
    for intent, g in d.groupby("intent", dropna=False):
        tgt = g[g.brand == target_brand]
        comp = g[g.brand != target_brand]
        if len(tgt) < min_spans or comp.empty:
            continue
        cb = comp.groupby("brand")["score_per_100t"].agg(["mean", "count"])
        cb = cb[cb["count"] >= min_spans]
        if cb.empty:
            continue
        leader = cb["mean"].idxmax()
        ldr = comp[comp.brand == leader]
        for c in DIMS:
            t, l = float(tgt[c + "_p100"].mean()), float(ldr[c + "_p100"].mean())
            gap_abs = l - t
            rel = gap_abs / max(l, 1e-6)
            # require BOTH an absolute and a relative gap — kills small-sample noise
            if gap_abs < min_deficit or rel < 0.35:
                continue
            n_eff = min(len(tgt), len(ldr))
            out.append({"intent": intent, "dimension": c, "leader": leader,
                        "leader_per_100t": round(l, 2), "brand_per_100t": round(t, 2),
                        "deficit_per_100t": round(gap_abs, 2),
                        "relative_deficit": round(rel, 2),
                        "leader_raw": round(float(ldr[c].mean()), 2),
                        "brand_raw": round(float(tgt[c].mean()), 2),
                        "leader_spans": int(len(ldr)), "brand_spans": int(len(tgt)),
                        "evidence_strength": "ok" if n_eff >= 20 else "low_n"})
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values("deficit_per_100t",
                                        ascending=False).reset_index(drop=True)

def flatness_report(detail, target_brand):
    """Is the brand ARGUED FOR, or merely listed?
    `substantive` excludes scope-setting language every answer contains."""
    if detail is None or detail.empty:
        return {}
    t = detail[detail.brand == target_brand]
    if t.empty:
        return {}
    n = len(t)
    zero_sub = int((t.substantive_types == 0).sum())
    shallow = int((t.substantive_types <= 1).sum())
    comp = detail[detail.brand != target_brand]
    out = {
        "spans": int(n),
        "bare_mentions": int((t.reason_types == 0).sum()),
        "bare_share": round(float((t.reason_types == 0).mean()), 3),
        "no_substantive_reason": zero_sub,
        "no_substantive_share": round(zero_sub / n, 3),
        "shallow_share": round(shallow / n, 3),
        "mean_reason_types": round(float(t.reason_types.mean()), 2),
        "mean_substantive_types": round(float(t.substantive_types.mean()), 2),
        "mean_richness_per_100t": round(float(t.score_per_100t.mean()), 2),
        "hedge_per_span": round(float(t.hedge.mean()), 2),
        "substantive_mix": {k: round(float(t[k].mean()), 2) for k in SUBSTANTIVE},
    }
    if not comp.empty:
        cb = comp.groupby("brand")["score_per_100t"].agg(["mean", "count"])
        cb = cb[cb["count"] >= 3]
        if not cb.empty:
            leader = cb["mean"].idxmax()
            out["leader"] = leader
            out["leader_richness_per_100t"] = round(float(cb.loc[leader, "mean"]), 2)
            out["richness_ratio_vs_leader"] = round(
                out["mean_richness_per_100t"] / max(cb.loc[leader, "mean"], 0.01), 2)
            ls = comp[comp.brand == leader]
            out["leader_mean_substantive_types"] = round(
                float(ls.substantive_types.mean()), 2)
            out["substantive_delta_vs_leader"] = round(
                float(t.substantive_types.mean() - ls.substantive_types.mean()), 2)
            out["ahead_of_leader_on_substance"] = out["substantive_delta_vs_leader"] > 0
    return out