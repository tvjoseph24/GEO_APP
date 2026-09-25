"""Creative-bench integration into Optimise. No LLM, no network."""
import pytest

from optimise.roadmap import OVERLAP_GROUPS, VISUAL_SURFACE_FACTOR, _group, project
from optimise.strategy_llm import (DISPROVEN_PILLARS, TEMPLATE_PILLAR, _bench_row,
                                   _significant_rows, deterministic_validate,
                                   evidence_summary, pillar_budget,
                                   pillar_evidence_verdict)

VIS = "Visual/Multimodal"

def _bench_win(did="V01", **kw):
    return {"design_id": did, "template": "VBENCH", "mode": "creative",
            "pillar": VIS, "is_creative_bench": True, "bakeoff_verdict": "winner",
            "winning_creative": "B", "lift": 62.5, "lift_unit": "pp",
            "best_metric": "mention_rate", "underpowered": False,
            "evidence_class": "PROVEN_ENGINE_SPECIFIC", **kw}

def _bench_equiv(did="V02"):
    return {"design_id": did, "template": "VBENCH", "mode": "creative", "pillar": VIS,
            "is_creative_bench": True, "bakeoff_verdict": "equivalent",
            "verdict": "no_lift", "underpowered": False, "lift": 0.0,
            "evidence_class": "DISPROVEN"}

def _si(wins=(), nulls=()):
    return {"wins": list(wins), "nulls": list(nulls), "substantiation_backlog": [],
            "claims": [], "gaps": [], "delta": {}, "metrics": {}}

# ---------------------------------------------------------------- routing
def test_vbench_is_in_both_evidence_maps():
    assert VIS in DISPROVEN_PILLARS["VBENCH"]
    assert VIS in TEMPLATE_PILLAR["VBENCH"]

def test_a_proven_bakeoff_makes_the_visual_pillar_proven():
    """Without VBENCH in DISPROVEN_PILLARS the pillar reads UNTESTED — and UNTESTED
    instructs the prompt that every action must be DIAGNOSED, so a 62.5pp proven finding
    would produce unevidenced actions."""
    pv = pillar_evidence_verdict(_si(wins=[_bench_win()]), VIS)
    assert pv["stance"] == "PROVEN_MECHANISM" and "V01" in pv["proven_ids"]

def test_text_and_creative_evidence_coexist():
    si = _si(wins=[{"design_id": "C19", "template": "E1",
                    "pillar": "Technical/Structured-Data", "underpowered": False},
                   _bench_win()])
    assert pillar_evidence_verdict(si, "Technical/Structured-Data")["stance"] \
        == "PROVEN_MECHANISM"
    assert pillar_evidence_verdict(si, VIS)["stance"] == "PROVEN_MECHANISM"

# ---------------------------------------------------------------- equivalence
def test_equivalence_is_not_a_disproven_mechanism():
    """'These two creatives are indistinguishable' does not disprove that creative
    matters. Counted as DISPROVEN_MECHANISM the prompt is told the opposite."""
    pv = pillar_evidence_verdict(_si(nulls=[_bench_equiv()]), VIS)
    assert pv["stance"] == "TESTED_EQUIVALENT"
    assert pv["equivalent_ids"] == ["V02"] and pv["disproven_ids"] == []
    assert "does NOT disprove" in pv["guidance"]
    assert "do_not_do" in pv["guidance"], "an equivalence is still an exclusion"

def test_a_proven_bakeoff_outranks_an_equivalent_one():
    pv = pillar_evidence_verdict(_si(wins=[_bench_win()], nulls=[_bench_equiv()]), VIS)
    assert pv["stance"] == "PROVEN_MECHANISM"
    assert pv["equivalent_ids"] == ["V02"], "the exclusion is still reported"

def test_equivalence_budget_sits_between_disproven_and_untested():
    assert pillar_budget(_si(), VIS, "DISPROVEN_MECHANISM") < \
           pillar_budget(_si(), VIS, "TESTED_EQUIVALENT") <= \
           pillar_budget(_si(), VIS, "UNTESTED")

def test_a_text_null_still_disproves_its_pillar():
    """The equivalence carve-out must not weaken a genuine well-powered null."""
    pv = pillar_evidence_verdict(
        _si(nulls=[{"design_id": "C20", "template": "E1", "underpowered": False,
                    "runs_per_cell": 100}]), "Technical/Structured-Data")
    assert pv["stance"] == "DISPROVEN_MECHANISM" and pv["disproven_ids"] == ["C20"]

# ---------------------------------------------------------------- prompt payload
def test_significant_rows_use_bench_keys():
    """A/B keys read off a bench row put all-null rows into the pillar prompt."""
    f = {"bakeoff": [{"engine": "gemini", "metric": "mention_rate", "verdict": "winner",
                      "ranking": [{"arm": "B", "rank": 1, "rate": 0.52,
                                   "ci_lo": 0.42, "ci_hi": 0.61, "n": 100},
                                  {"arm": "A", "rank": 2, "rate": 0.01,
                                   "ci_lo": 0.00, "ci_hi": 0.05, "n": 100}]}]}
    rows = _significant_rows(f, is_bench=True)
    assert rows and rows[0]["rate_pct"] == 52.0 and rows[0]["arm"] == "B"
    assert all(r.get("rate_pct") is not None for r in rows)

def test_bench_row_carries_the_authority_reading():
    """The most valuable bench reading reached no prompt."""
    f = {"mode": "creative", "winning_arm": "B", "attribution": "isolation",
         "bakeoff_summary": {"verdict": "winner",
                             "mention_without_recommendation": ["gemini"],
                             "mention_without_recommendation_note": "Route to "
                                                                   "Corroboration.",
                             "gap_spread_pp": 23.0, "magnitude_diverges": True}}
    r = _bench_row(f)
    assert r["mention_without_recommendation"] == ["gemini"]
    assert "Corroboration" in r["mention_without_recommendation_note"]
    assert "multimodal" in r["surface_applicability"].lower()
    assert r["attribution_caveat"] is None

def test_bundle_attribution_carries_its_caveat():
    r = _bench_row({"mode": "creative", "attribution": "bundle",
                    "bakeoff_summary": {}})
    assert r["attribution_caveat"] and "shipped package" in r["attribution_caveat"]

def test_evidence_summary_reports_bench_separately():
    ev = evidence_summary({**_si(wins=[_bench_win()], nulls=[_bench_equiv()]),
                           "gaps": [], "substantiation_backlog": [],
                           "metrics": {}, "delta": {}})
    assert ev["n_creative_bench"] == 2
    assert ev["creative_winners"] == {"V01": "B"}
    assert ev["creative_equivalent"] == ["V02"]
    assert ev["n_well_powered_nulls"] == 0, \
        "an equivalence must not be counted as a well-powered text null"

# ---------------------------------------------------------------- roadmap
def test_vbench_shares_the_visual_overlap_group():
    """Different groups mean no decay between them, so the same mechanism double-counts."""
    assert "VBENCH" in OVERLAP_GROUPS["visual"]
    d2t = {"V01": "VBENCH", "X7": "E7"}
    a = {"evidence_refs": ["V01"], "pillar": VIS}
    b = {"evidence_refs": ["X7"], "pillar": VIS}
    assert _group(a, d2t) == _group(b, d2t) == "visual"

def test_bench_effect_is_discounted_on_the_text_axis():
    """62.5pp measured in context on a multimodal read would otherwise pin the projection
    at the cap in wave 0."""
    waves = {w: {"actions": [], "load": {}} for w in
             ["Wave 0 (0-2 weeks)", "Wave 1 (2-8 weeks)",
              "Wave 2 (2-4 months)", "Wave 3 (2-4 quarters)"]}
    waves["Wave 0 (0-2 weeks)"]["actions"] = ["P3-1"]
    idx = {"P3-1": {"action_id": "P3-1", "pillar": VIS,
                    "evidence_class": "PROVEN_ENGINE_SPECIFIC",
                    "evidence_refs": ["V01"],
                    "expected_effect": {"value": 62.5, "unit": "pp",
                                        "basis": "measured"}}}
    t0 = project(waves, idx, 35.0, design_to_template={"V01": "VBENCH"})[0]
    assert t0["added_pp"] < 20.0, "capped and modality-discounted"
    assert t0["projected_pct"] < 60.0, "must not reach the cap on one visual finding"
    mm = t0["multimodal_effects"]
    assert mm and mm[0]["designs"] == ["V01"]
    assert mm[0]["multimodal_effect_pp"] > mm[0]["counted_toward_text_projection_pp"], \
        "the full multimodal effect must be reported, not lost"

def test_a_text_effect_is_not_discounted():
    waves = {w: {"actions": [], "load": {}} for w in
             ["Wave 0 (0-2 weeks)", "Wave 1 (2-8 weeks)",
              "Wave 2 (2-4 months)", "Wave 3 (2-4 quarters)"]}
    waves["Wave 0 (0-2 weeks)"]["actions"] = ["P1-1"]
    idx = {"P1-1": {"action_id": "P1-1", "pillar": "Content",
                    "evidence_class": "PROVEN", "evidence_refs": ["C19"],
                    "expected_effect": {"value": 10.0, "unit": "pp",
                                        "basis": "measured"}}}
    t0 = project(waves, idx, 35.0, design_to_template={"C19": "E1"})[0]
    assert t0["added_pp"] == pytest.approx(10.0, abs=0.1)
    assert not t0["multimodal_effects"]

# ---------------------------------------------------------------- validation
def test_bench_action_must_name_its_surface():
    si = {**_si(wins=[_bench_win()]), "claims": [], "gaps": []}
    bad = {"action_id": "P3-1", "pillar": VIS, "title": "Improve brand visibility",
           "what": "roll out the better version everywhere",
           "evidence_class": "PROVEN_ENGINE_SPECIFIC", "evidence_refs": ["V01"],
           "kill_criterion": "stop if flat",
           "expected_effect": {"value": 62.5, "unit": "pp", "basis": "measured",
                               "source": "V01"}}
    errs = deterministic_validate(si, {VIS: {"actions": [bad]}},
                                 {"not_doing": [{"item": "x"}], "big_bets": []})
    assert any(e["error"] == "bench_effect_without_surface" for e in errs)

def test_bench_action_naming_the_surface_passes():
    si = {**_si(wins=[_bench_win()]), "claims": [], "gaps": []}
    good = {"action_id": "P3-1", "pillar": VIS,
            "title": "Ship creative B across multimodal surfaces",
            "what": "replace the pack image on image search and shopping listings, and "
                    "transcribe the on-pack claims into the PDP copy",
            "evidence_class": "PROVEN_ENGINE_SPECIFIC", "evidence_refs": ["V01"],
            "kill_criterion": "stop if flat",
            "expected_effect": {"value": 62.5, "unit": "pp", "basis": "measured",
                                "source": "V01 (multimodal, asset in context)"}}
    errs = deterministic_validate(si, {VIS: {"actions": [good]}},
                                 {"not_doing": [{"item": "x"}], "big_bets": []})
    assert not any(e["error"] == "bench_effect_without_surface" for e in errs)