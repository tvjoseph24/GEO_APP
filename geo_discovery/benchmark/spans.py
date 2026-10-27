"""Brand-span extraction — the unit of ALL Tier-B analysis.

Rationale: judging veracity on whole responses is both context-explosive and imprecise.
A brand span is the sentence window that actually talks about a given brand, so we can
(a) fit the whole corpus through an LLM affordably, (b) attribute claims per-engine,
(c) run a reverse hallucination scan, (d) score richness per brand fairly.
"""
from __future__ import annotations

import re

import pandas as pd

from metrics_spec import (Brand, build_registry, fold, make_brand, sentence_spans)

# continuation sentences that keep describing the brand without renaming it
PRONOUN_CONT = re.compile(
    r"^\s*(?:it|its|this|these|they|their|the\s+(?:formula|product|brand|milk|range|"
    r"powder|blend|stage\s?\d))\b", re.I)

def brand_spans(text: str, brand: Brand, *, window_before=1, window_after=2,
                max_cont=3) -> list[dict]:
    """Sentence windows about `brand`. Overlapping windows are merged."""
    if not text:
        return []
    folded = fold(text)
    sents = sentence_spans(text)
    if not sents:
        return []
    hit_sent = []
    for m in brand.pattern.finditer(folded):
        for k, (a, b) in enumerate(sents):
            if a <= m.start() < b:
                hit_sent.append(k)
                break
    if not hit_sent:
        return []

    ranges = []
    for i in sorted(set(hit_sent)):
        lo = max(0, i - window_before)
        hi = min(len(sents), i + window_after + 1)
        j, c = hi, 0
        while j < len(sents) and c < max_cont and PRONOUN_CONT.match(text[sents[j][0]:sents[j][1]]):
            j += 1
            c += 1
        ranges.append((lo, j))

    merged = []
    for lo, hi in sorted(ranges):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append([lo, hi])

    out = []
    for lo, hi in merged:
        a, b = sents[lo][0], sents[hi - 1][1]
        out.append({"span": text[a:b].strip(), "char_start": a, "char_end": b,
                    "n_sentences": hi - lo})
    return [s for s in out if s["span"]]

def span_id(row, span_text, char_start) -> str:
    """Stable identity for a brand span, independent of row position.

    Keyed on the run that produced it plus the span's own content, so appending audit runs
    cannot shift it. The previous positional ID made the veracity cache unsound: a cached
    judgement keyed on `412_0` was returned for whatever span happened to occupy index 412
    after the next audit batch.
    """
    import hashlib
    parts = [str(row.get("query") or ""), str(row.get("engine") or ""),
             str(row.get("model") or ""), str(row.get("run_index") or ""),
             str(row.get("web_on")), str(char_start),
             (span_text or "")[:300]]
    return "s" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:14]

def build_span_corpus(df, brand_name, *, aliases=(), other_brands=(),
                      alias_overrides=None) -> pd.DataFrame:
    """One row per brand span across the whole audit. THE Tier-B substrate."""
    reg = build_registry(brand_name, list(other_brands), alias_overrides)
    target = reg.target
    others = [c for c in reg.competitors]
    rows = []
    for idx, r in df.iterrows():
        text = r.get("response_text") or ""
        if not text:
            continue
        for k, s in enumerate(brand_spans(text, target)):
            low = fold(s["span"])
            co = [c.canonical for c in others if c.pattern.search(low)]
            srcs = r.get("sources") or []
            rows.append({
                # CONTENT hash, not a positional index. The index shifts when audit runs
                # are appended, so the veracity cache — keyed on span IDs — returned a hit
                # for a different span. Also the stable key embeddings require.
                "span_id": span_id(r, s["span"], s["char_start"]),
                "row_idx": idx,
                "span_id_positional": f"{idx}_{k}",  # transitional, for cache migration
                "engine": r.get("engine"), "model": r.get("model"),
                "query": r.get("query"), "intent": r.get("intent"),
                "web_on": r.get("web_on"), "run_index": r.get("run_index"),
                "span": s["span"], "char_start": s["char_start"],
                "n_sentences": s["n_sentences"],
                "n_tokens": len(s["span"].split()),
                "co_mentioned": co, "n_co_mentioned": len(co),
                "polarity": r.get("polarity"), "hedged": r.get("hedged"),
                "advisory_referral": r.get("advisory_referral"),
                "cited_domains": [x.get("domain") for x in srcs if x.get("domain")],
                "cited_types": sorted({x.get("source_type", "other") for x in srcs}),
            })
    return pd.DataFrame(rows)

def build_multi_brand_corpus(df, brands, *, alias_overrides=None) -> pd.DataFrame:
    """Span corpus for the target AND every competitor — for head-to-head richness."""
    frames = []
    for b in brands:
        others = [x for x in brands if fold(x) != fold(b)]
        f = build_span_corpus(df, b, other_brands=others,
                              alias_overrides=alias_overrides)
        if not f.empty:
            f["brand"] = b
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()