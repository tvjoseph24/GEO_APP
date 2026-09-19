"""Retrieval, claim normalisation, winnability and bucket completeness.
Covers W-04, W-08, W-13, W-16."""
import pandas as pd
import pytest

# ---------------------------------------------------------------- W-04 retrieval
from benchmark.veracity import (MIN_SCORE, _num_pattern, _score_span, _word_pattern,
                                claim_terms)

def _score(claim, span):
    nums, words = claim_terms(claim)
    return _score_span(span.lower(), _num_pattern(nums), _word_pattern(words), len(words))

def test_units_become_searchable_terms():
    nums, words = claim_terms("Electric Range Rover adds up to 220 km in 10 minutes")
    assert "220" in nums and "10" in nums
    assert "km" in words, "unit was discarded — 220 km would retrieve on 220 alone"

def test_short_tokens_kept_when_derived():
    _, w = claim_terms("The 4.4-litre V8 produces 390 kW and 750 Nm")
    assert "kw" in w and "nm" in w
    assert any("v8" in x for x in w), "alphanumeric token dropped"
    _, w2 = claim_terms("Offers SPF 50 and UVA protection")
    assert "spf" in w2 and "uva" in w2, "acronyms dropped"

def test_numeric_boundary_prevents_false_match():
    p = _num_pattern(["100"])
    assert p.search("adds 100 km")
    assert not p.search("costs 1,000 pounds"), "100 matched inside 1,000"
    assert not p.search("range of 1000 km")
    p2 = _num_pattern(["6"])
    assert not p2.search("0-100 in 6.0 s"), "bare 6 matched 6.0"

def test_word_boundary_prevents_false_match():
    p = _word_pattern(["audio"])
    assert p.search("in-car audio system")
    assert not p.search("popular with audiophiles"), "audio matched audiophile"

def test_offtopic_span_excluded_ontopic_retained():
    claim = ("A new standard for in-car audio reaches its crescendo with the "
             "introduction of Electrostatic technology.")
    off = "Terrain Response 2 technology is standard across the line, with sand and mud."
    on = ("With the introduction of electrostatic technology for in-car audio, "
          "setting a new standard.")
    assert _score(claim, off) < MIN_SCORE, "off-topic span still admitted"
    assert _score(claim, on) >= MIN_SCORE, "on-topic span excluded"

def test_threshold_not_length_dependent():
    """A short claim must not be satisfied by a single generic word match."""
    short = "Contains colloidal oatmeal"
    generic = "This product contains various beneficial ingredients for the skin."
    assert _score(short, generic) < MIN_SCORE

# ---------------------------------------------------------------- W-08 claim prep
from benchmark.claim_prep import (canonical_fact, classify_claim, prepare_claims,
                                  split_multiclaim)

def test_meta_framing_stripped():
    a = canonical_fact("Aveeno claims that its lotion contains colloidal oatmeal",
                       "Aveeno")
    b = canonical_fact("Aveeno's lotion contains colloidal oatmeal", "Aveeno")
    assert set(a.lower().split()) & set(b.lower().split()) >= {"contains", "colloidal"}

def test_analyst_commentary_excluded():
    kind, why = classify_claim(
        "Design pedigree — publicly invoked via the Louvre exhibition and decades of "
        "design-award recognition, used as evidence for a design-led claim to luxury "
        "status.")
    assert kind == "analyst_note", f"got {kind}: {why}"

def test_advertising_copy_is_brand_voice():
    kind, _ = classify_claim("Protects against everything but admiring glances")
    assert kind == "brand_voice"

def test_factual_claims_survive():
    for t in ("Contains 1% colloidal oatmeal",
              "Dermatologist recommended for sensitive skin",
              "Certified by the National Eczema Association"):
        assert classify_claim(t)[0] == "factual", t

def test_asymmetric_containment_merges_factual_duplicates():
    """Containment, not symmetric Jaccard. The pair below scores 0.44 on Jaccard —
    below the 0.70 threshold — because one entry carries extra supporting prose."""
    claims = [
        {"claim": "The lotion contains 1% colloidal oatmeal.", "category": "ingredient"},
        {"claim": "Aveeno claims that the lotion contains 1% colloidal oatmeal, which "
                  "is clinically shown to soothe dry itchy skin.",
         "category": "ingredient"},
    ]
    kept, rep = prepare_claims(claims, brand="Aveeno", category="skincare")
    assert len(kept) == 1, f"containment merge failed: {rep}"
    assert rep["merged"][0]["basis"] == "containment"

def test_brand_voice_pair_is_excluded_not_merged():
    """Classification precedes merging by design — a claim with no verifiable anchor is
    excluded regardless of duplication, but the report flags the duplication."""
    claims = [
        {"claim": "Range Rover claims refinement and comfort coexist with genuine "
                  "all-terrain performance.", "category": "efficacy"},
        {"claim": "Range Rover's public claims cluster around recurring themes: "
                  "refinement and comfort coexisting with all-terrain performance.",
         "category": "efficacy"},
    ]
    kept, rep = prepare_claims(claims, brand="Range Rover", category="SUV")
    assert kept == []
    assert set(rep["dropped_by_kind"]) == {"brand_voice", "analyst_note"}

def test_differing_numbers_never_merge():
    """Numerics are part of the token set AND separately vetoed. Without this,
    '1% oatmeal' and '5% oatmeal' have identical tokens and collapse into one claim."""
    claims = [{"claim": "Contains 1% colloidal oatmeal.", "category": "ingredient"},
              {"claim": "Contains 5% colloidal oatmeal.", "category": "ingredient"}]
    kept, rep = prepare_claims(claims, brand="Aveeno", category="skincare")
    assert len(kept) == 2, f"different concentrations were merged: {rep['merged']}"
def test_long_paragraph_is_split_not_dropped():
    long = ("The lineup includes the Daily Moisturising Lotion which is a best seller. "
            "It contains colloidal oatmeal at 1% concentration. The Baby range is "
            "tear-free and pediatrician tested. All variants are dermatologist tested "
            "and fragrance free for sensitive skin types.")
    assert len(long.split()) > 25
    assert len(split_multiclaim(long)) > 1
    kept, rep = prepare_claims([{"claim": long, "category": "feature"}], brand="Aveeno")
    assert rep["n_split"] == 1 and len(kept) >= 2

def test_report_explains_every_exclusion():
    _, rep = prepare_claims([{"claim": "Pure joy"},
                             {"claim": "Contains 1% colloidal oatmeal"}], brand="Aveeno")
    assert rep["n_dropped"] >= 1
    assert all(d.get("why") for d in rep["dropped"])

# ---------------------------------------------------------------- W-13 winnability
from benchmark.tier_c import _ownable_claim, classify_winnability

CLAIMS = [{"claim": "Contains 1% colloidal oatmeal for eczema-prone sensitive skin"},
          {"claim": "Fragrance free and dermatologist recommended"}]

def test_ownable_requires_substantive_overlap():
    ok, _, n = _ownable_claim("best luxury watch for diving", CLAIMS, set())
    assert not ok and n < 2, "single generic word made an unrelated query ownable"
    ok2, _, n2 = _ownable_claim("moisturiser for eczema prone sensitive skin",
                                CLAIMS, set())
    assert ok2 and n2 >= 2

def _mk(query, intent, mention, first, n=100, neg=0.0):
    bm = [{"Aveeno": {"mentioned": True, "first_index": 0 if i < first * n else 50},
           "CeraVe": {"mentioned": True, "first_index": 10}}
          if i < mention * n else {"CeraVe": {"mentioned": True, "first_index": 0}}
          for i in range(n)]
    return pd.DataFrame({
        "query": [query] * n, "intent": [intent] * n,
        "mentioned": [i < mention * n for i in range(n)],
        "is_first": [i < first * n for i in range(n)],
        "refusal": [False] * n,
        "polarity": ["neg" if i < neg * n else "pos" for i in range(n)],
        "brand_mentions": bm})

def test_already_winning_verdict_exists():
    r = classify_winnability(_mk("best gentle cleanser", "need", 0.97, 0.95),
                             "Aveeno", ["CeraVe"], CLAIMS)
    assert r.iloc[0]["winnability"] == "already_winning", r.iloc[0]["rule_reason"]

def test_negative_intent_routes_to_objection_defence():
    r = classify_winnability(
        _mk("which moisturisers cause irritation?", "negative", 1.0, 0.14, neg=0.3),
        "Aveeno", ["CeraVe"], CLAIMS)
    assert r.iloc[0]["winnability"] == "objection_defence", \
        "negative query treated as a visibility target"

def test_branded_query_is_not_applicable():
    r = classify_winnability(_mk("Is Aveeno good for eczema?", "branded", 1.0, 0.9),
                             "Aveeno", ["CeraVe"], CLAIMS)
    assert r.iloc[0]["winnability"] == "not_applicable"

def test_opportunity_prefers_headroom_over_extremeness():
    low = classify_winnability(_mk("q1", "need", 0.24, 0.01), "Aveeno", ["CeraVe"],
                               CLAIMS).iloc[0]
    high = classify_winnability(_mk("q2", "need", 1.00, 0.95), "Aveeno", ["CeraVe"],
                                CLAIMS).iloc[0]
    assert low["opportunity"] > high["opportunity"], \
        "a saturated query outranked one with real headroom"

# ---------------------------------------------------------------- W-16 buckets
from benchmark.tier_b import answer_consistency
from benchmark.tier_c import cross_engine_map

def _corpus():
    rows = []
    for q, rate in (("locked", 0.95), ("emerging", 0.70), ("fragile", 0.30),
                    ("absent", 0.0)):
        for i in range(20):
            rows.append({"query": q, "engine": "e1", "mentioned": i < rate * 20,
                         "is_first": i < rate * 10, "web_on": True})
    return pd.DataFrame(rows)

def test_consistency_partition_is_exhaustive():
    out = answer_consistency(_corpus(), "Aveeno")
    ov = out["overall"]
    assert "emerging" in ov and len(ov["emerging"]) == 1
    assert ov["_partition_ok"], "buckets do not sum to the query total"

def test_cross_engine_reports_unassigned():
    df = _corpus()
    df2 = df.copy(); df2["engine"] = "e2"
    out = cross_engine_map(pd.concat([df, df2]), "Aveeno")
    assert out["metric"] == "is_first"
    cov = out["coverage"]
    assert cov["assigned"] + cov["unassigned"] == cov["total_queries"]