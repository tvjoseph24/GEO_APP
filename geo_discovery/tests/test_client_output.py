"""P3: the defects that were visible in a client-facing document."""
import pytest

WAVES = ["Wave 0 (0-2 weeks)", "Wave 1 (2-8 weeks)",
         "Wave 2 (2-4 months)", "Wave 3 (2-4 quarters)"]

def _waves(**placement):
    w = {k: {"actions": [], "load": {}} for k in WAVES}
    for wave, ids in placement.items():
        w[wave]["actions"] = ids
    return w

def _act(aid, pillar="Content", cls="PROVEN", refs=(), value=10.0, basis="measured"):
    return {"action_id": aid, "pillar": pillar, "evidence_class": cls,
            "evidence_refs": list(refs),
            "expected_effect": {"value": value, "unit": "pp", "basis": basis}}

# ---------------------------------------------------------------- N1
def test_a_finding_cannot_pay_out_in_two_waves():
    """Observed live: VM-1 (wave 1) and VM-5 (wave 3) both cited V01, contributing 10.9pp
    each — the same measured effect banked twice."""
    from optimise.roadmap import project
    waves = _waves(**{WAVES[1]: ["VM-1"], WAVES[3]: ["VM-5"]})
    idx = {"VM-1": _act("VM-1", "Visual/Multimodal", "PROVEN_ENGINE_SPECIFIC",
                        ["V01"], 62.5),
           "VM-5": _act("VM-5", "Visual/Multimodal", "PROVEN_ENGINE_SPECIFIC",
                        ["V01"], 62.5)}
    tr = project(waves, idx, 35.0, design_to_template={"V01": "VBENCH"})
    assert tr[1]["added_pp"] > 0
    assert tr[3]["added_pp"] == 0.0
    assert "VM-5" in tr[3]["double_counted_excluded"]

def test_same_mechanism_decays_across_waves():
    """P2-1, P2-2 and P2-3 all sat in the `structure` group across three waves. Decay was
    per wave, so each paid near-full strength."""
    from optimise.roadmap import project
    waves = _waves(**{WAVES[0]: ["P2-1"], WAVES[1]: ["P2-2"], WAVES[2]: ["P2-3"]})
    idx = {k: _act(k, "Technical/Structured-Data", "PROVEN",
                   [f"C{i}"], 19.0)
           for i, k in enumerate(["P2-1", "P2-2", "P2-3"], 1)}
    d2t = {"C1": "E1", "C2": "E1", "C3": "E1"}
    tr = project(waves, idx, 35.0, design_to_template=d2t)
    a0, a1, a2 = tr[0]["added_pp"], tr[1]["added_pp"], tr[2]["added_pp"]
    assert a0 > a1 > a2, "each successive contribution in a group must decay"
    assert a2 < a0 * 0.5
    assert tr[-1]["projected_pct"] < 85.0, \
        "three structure actions must not carry a 35% baseline to the cap"

def test_different_mechanisms_do_not_decay_against_each_other():
    from optimise.roadmap import project
    waves = _waves(**{WAVES[0]: ["A1", "A2"]})
    idx = {"A1": _act("A1", "Content", "PROVEN", ["C2"], 10.0),
           "A2": _act("A2", "Corroboration", "PROVEN", ["C3"], 10.0)}
    tr = project(waves, idx, 35.0,
                 design_to_template={"C2": "E2", "C3": "E3"})
    assert tr[0]["added_pp"] == pytest.approx(20.0, abs=0.5)

def test_decay_factors_are_recorded_per_contributor():
    from optimise.roadmap import project
    waves = _waves(**{WAVES[0]: ["A1"], WAVES[1]: ["A2"]})
    idx = {"A1": _act("A1", "Content", "PROVEN", ["C1"], 10.0),
           "A2": _act("A2", "Content", "PROVEN", ["C2"], 10.0)}
    tr = project(waves, idx, 35.0,
                 design_to_template={"C1": "E2", "C2": "E2"})
    assert tr[0]["contributors"][0]["decay_factor"] == 1.0
    assert tr[1]["contributors"][0]["decay_factor"] < 1.0
    assert tr[1]["contributors"][0]["nth_in_group"] == 2

# ---------------------------------------------------------------- N2
def test_duplicate_action_ids_are_renamed_not_dropped():
    """`VM-3 → VM-4 → VM-4` was a self-loop: two actions shared VM-4, and idx kept only
    one, so the other was invisible to the roadmap, projection and briefs."""
    from optimise.roadmap import dedupe_action_ids
    acts = [{"action_id": "VM-4", "pillar": "Visual/Multimodal", "title": "First"},
            {"action_id": "VM-4", "pillar": "Visual/Multimodal", "title": "Second"},
            {"action_id": "VM-5", "pillar": "Visual/Multimodal", "title": "Third"}]
    renamed = dedupe_action_ids(acts)
    assert [a["action_id"] for a in acts] == ["VM-4", "VM-4.2", "VM-5"]
    assert renamed[0]["was"] == "VM-4" and renamed[0]["now"] == "VM-4.2"
    assert acts[1]["id_collision"]
    assert len({a["action_id"] for a in acts}) == 3

def test_triple_collision_resolves():
    from optimise.roadmap import dedupe_action_ids
    acts = [{"action_id": "X"} for _ in range(3)]
    dedupe_action_ids(acts)
    assert [a["action_id"] for a in acts] == ["X", "X.2", "X.3"]

def test_no_self_loop_after_dedupe():
    from optimise.roadmap import build_dag, dedupe_action_ids
    acts = [{"action_id": "A", "pillar": "Content",
             "dependencies": ["B"]},
            {"action_id": "B", "pillar": "Content"},
            {"action_id": "B", "pillar": "Content"}]
    dedupe_action_ids(acts)
    dag = build_dag(acts)
    assert not any(k in v for k, v in dag.items()), "no node may depend on itself"

# ---------------------------------------------------------------- N8
def _si_retrieval(mention, first, nulls=None):
    """A minimal strategy input. Every key route_gaps_to_pillars touches."""
    return {"gaps": [], "substantiation_backlog": [], "wins": [],
            "nulls": nulls if nulls is not None else [
                {"design_id": "C20", "template": "E1", "underpowered": False}],
            "metrics": {
                "mention_by_intent": [{"intent": "need", "mention_rate": mention}],
                "first_mention_by_intent": [{"intent": "need", "first_rate": first}]}}

def test_mentioned_but_never_chosen_counts_as_a_retrieval_constraint():
    """Observed live: need mention 35.4%, first-rec 0.7%, reported as
    retrieval_is_primary_constraint=False because only mention was tested."""
    from optimise.strategy_llm import route_gaps_to_pillars
    si = _si_retrieval(0.354, 0.007)
    _, is_constraint = route_gaps_to_pillars(si)
    assert is_constraint is True
    assert si["_retrieval_detail"]["pattern"] == "mentioned_not_chosen"
    assert "not chosen" in si["_retrieval_detail"]["note"]

def test_absent_also_counts():
    from optimise.strategy_llm import route_gaps_to_pillars
    si = _si_retrieval(0.04, 0.0)
    _, c = route_gaps_to_pillars(si)
    assert c is True and si["_retrieval_detail"]["pattern"] == "absent"

def test_healthy_visibility_is_not_a_retrieval_constraint():
    from optimise.strategy_llm import route_gaps_to_pillars
    si = _si_retrieval(0.60, 0.35)
    _, c = route_gaps_to_pillars(si)
    assert c is False and si["_retrieval_detail"]["pattern"] == "neither"

def test_a_bench_null_does_not_imply_a_retrieval_constraint():
    """A creative equivalence says nothing about whether text content form works."""
    from optimise.strategy_llm import route_gaps_to_pillars
    si = _si_retrieval(0.04, 0.0, nulls=[
        {"design_id": "V02", "template": "E1", "underpowered": False,
         "is_creative_bench": True}])
    _, c = route_gaps_to_pillars(si)
    assert c is False

def test_routing_tolerates_a_sparse_strategy_input():
    """Routing is the FIRST thing Optimise does. It read si['gaps'] and
    si['substantiation_backlog'] by subscript while every other consumer used .get(),
    so one missing key took down the whole phase rather than degrading."""
    from optimise.strategy_llm import route_gaps_to_pillars
    routed, c = route_gaps_to_pillars({})
    assert routed == {} and c is False
    routed2, _ = route_gaps_to_pillars({"gaps": None, "nulls": None,
                                        "substantiation_backlog": None, "metrics": {}})
    assert routed2 == {}

def test_pillar_verdict_tolerates_a_sparse_input():
    from optimise.strategy_llm import pillar_evidence_verdict
    assert pillar_evidence_verdict({}, "Content")["stance"] == "UNTESTED"

# ---------------------------------------------------------------- W-43
def test_placeholder_exclusion_basis_is_a_blocker():
    """All three not_doing entries read 'disproven by design_id' — the format spec copied
    verbatim into the section where a client is told not to spend money."""
    from optimise.strategy_llm import deterministic_validate
    si = {"wins": [], "nulls": [], "claims": [], "gaps": [],
          "substantiation_backlog": [], "metrics": {}}
    strat = {"not_doing": [{"item": "reformatting",
                            "basis": "disproven by design_id | unwinnable"}],
             "big_bets": [], "structural_truths": []}
    errs = deterministic_validate(si, {}, strat)
    e = next(x for x in errs if x["error"] == "placeholder_in_exclusion_basis")
    assert e["severity"] == "blocker"

def test_a_real_basis_passes():
    from optimise.strategy_llm import deterministic_validate
    si = {"wins": [], "nulls": [], "claims": [], "gaps": [],
          "substantiation_backlog": [], "metrics": {}}
    strat = {"not_doing": [{"item": "reformatting",
                            "basis": "disproven by C20 (100 runs/arm, both engines)"}],
             "big_bets": [], "structural_truths": []}
    errs = deterministic_validate(si, {}, strat)
    assert not [e for e in errs if "placeholder" in e["error"]]

def test_an_exclusion_with_no_basis_is_flagged():
    from optimise.strategy_llm import deterministic_validate
    si = {"wins": [], "nulls": [], "claims": [], "gaps": [],
          "substantiation_backlog": [], "metrics": {}}
    errs = deterministic_validate(si, {}, {"not_doing": [{"item": "x", "basis": ""}],
                                           "big_bets": [], "structural_truths": []})
    assert any(e["error"] == "exclusion_without_basis" for e in errs)

# ---------------------------------------------------------------- W-44
def test_a_structural_truth_needs_cross_engine_evidence():
    from optimise.strategy_llm import deterministic_validate
    si = {"wins": [{"design_id": "C25", "significant_engines": ["azure/gpt-4o"],
                    "per_engine_lift_pp": {"azure/gpt-4o": 34.0,
                                           "gemini-2.5-flash": 4.0}}],
          "nulls": [], "claims": [], "gaps": [], "substantiation_backlog": [],
          "metrics": {"engines": ["azure/gpt-4o", "gemini-2.5-flash"]}}
    strat = {"not_doing": [{"item": "x", "basis": "disproven by C20"}], "big_bets": [],
             "structural_truths": [{"truth": "Specificity wins everywhere",
                                    "evidence": "C25 showed +34pp",
                                    "applies_to_engines": ["all"]}]}
    errs = deterministic_validate(si, {}, strat)
    assert any(e["error"] == "universality_unspecified" for e in errs)

def test_single_engine_evidence_is_not_structural():
    from optimise.strategy_llm import deterministic_validate
    si = {"wins": [{"design_id": "C25", "significant_engines": ["azure/gpt-4o"],
                    "per_engine_lift_pp": {"azure/gpt-4o": 34.0}}],
          "nulls": [], "claims": [], "gaps": [], "substantiation_backlog": [],
          "metrics": {"engines": ["azure/gpt-4o", "gemini-2.5-flash"]}}
    strat = {"not_doing": [{"item": "x", "basis": "disproven by C20"}], "big_bets": [],
             "structural_truths": [{"truth": "Specificity wins",
                                    "evidence": "C25",
                                    "applies_to_engines": ["azure/gpt-4o",
                                                           "gemini-2.5-flash"]}]}
    errs = deterministic_validate(si, {}, strat)
    e = next(x for x in errs
             if x["error"] == "structural_truth_on_single_engine_evidence")
    assert "engine_specific_signals" in e["fix"]

# ---------------------------------------------------------------- N7
def test_funnel_reports_unaddressed_gaps():
    """85 gaps became 26 actions with no account of the other 59."""
    from optimise.strategy_llm import funnel_attrition
    si = {"gaps": [{"gap_id": f"G{i:03d}", "type": "CLAIM_ABSENT",
                    "severity": "high" if i < 8 else "medium",
                    "pillar": "Content", "gap": f"gap {i}"} for i in range(40)]}
    pillars = {"Content": {"actions": [
        {"action_id": "P1-1", "pillar": "Content", "evidence_refs": ["G000", "G001"]}]}}
    fn = funnel_attrition(si, pillars, {"Content": si["gaps"]})
    assert fn["gaps_actionable"] == 40
    assert fn["gaps_addressed"] == 2
    assert fn["gaps_unaddressed"] == 38
    assert len(fn["unaddressed_critical"]) == 6
    assert "not referenced by any action" in fn["note"]

def test_funnel_reports_empty_pillars():
    from optimise.strategy_llm import funnel_attrition
    fn = funnel_attrition({"gaps": []},
                          {"Content": {"actions": [{"action_id": "A"}]},
                           "Social/Community": {"actions": []}}, None)
    assert fn["pillars_empty"] == ["Social/Community"]

def test_many_unaddressed_criticals_is_a_major_error():
    from optimise.strategy_llm import deterministic_validate
    si = {"wins": [], "nulls": [], "claims": [], "metrics": {},
          "substantiation_backlog": [],
          "gaps": [{"gap_id": f"G{i}", "type": "CLAIM_ABSENT", "severity": "critical",
                    "pillar": "Content", "gap": "x"} for i in range(8)]}
    errs = deterministic_validate(si, {}, {"not_doing": [{"item": "x", "basis": "C20"}],
                                           "big_bets": [], "structural_truths": []})
    e = next(x for x in errs if x["error"] == "critical_gaps_unaddressed")
    assert "oversight" in e["fix"]

# ---------------------------------------------------------------- W-50
def test_retrieval_gap_uses_the_median_control_arm(tmp_path, monkeypatch):
    """Control arms span 4% to 100% depending on how substantive the document is, so the
    best case overstates the gap by the full spread."""
    import pandas as pd
    import store as S
    from benchmark.metrics_bundle import build_metrics_bundle
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    S.save_artifact("p", "t", "validated_findings", [
        {"design_id": "C01", "mode": "ab",
         "rate_table": [{"arm": "A", "metric": "mention", "rate": r}]}
        for r in (0.04, 0.50, 1.00)], {})
    df = pd.DataFrame([{"engine": "e1", "intent": "need", "mentioned": True,
                        "is_first": False, "first_index": 3, "refusal": False,
                        "num_sources": 2, "source_type_tally": {}, "polarity": "neu",
                        "rank": 1, "outcome": "brand_recommended",
                        "advisory_referral": False, "brand_mentions": {},
                        "classification": {}}])
    rd = build_metrics_bundle(df, "t", [], {}, None, {},
                              parent_brand_for_delta="p")["retrieval_delta"]
    assert rd["in_context_median"] == 0.5
    assert rd["gap_pp"] < rd["gap_pp_best_case"]
    assert rd["control_arm_spread_pp"] == pytest.approx(96.0, abs=1)
    assert "median" in rd["interpretation"].lower()

# ---------------------------------------------------------------- W-45
def _prov(ratio, reliable=True, unclassified=0.05, own=0.03):
    return {"self_referential_pct": own, "grey_market_pct": 0.10,
            "affiliate_pct": 0.05, "corroborated_pct": 0.04,
            "practitioner_pct": 0.02, "competitor_owned_pct": 0.03,
            "voice_control_ratio": ratio, "unclassified_pct": unclassified,
            "reliability": {"reliable": reliable,
                            "note": "too many citations unclassified"}}

def _hl(provenance=None, framing=None, veracity=None, flatness=None, metrics=None):
    """Named-argument wrapper. _headline is positional
    (veracity, provenance, framing, flatness, metrics, brand) and getting the order wrong
    silently tests the wrong parameter."""
    from benchmark.delta_scorecard import _headline
    return _headline(veracity, provenance, framing, flatness, metrics or {}, "Brand X")

def test_provenance_headline_cannot_contradict_its_own_ratio():
    """It read 'citation control is weak' beside a ratio of 3.98x — a hardcoded verdict
    next to a computed number."""
    weak = _hl(provenance=_prov(0.2))
    assert any("is weak" in x for x in weak)
    ok = _hl(provenance=_prov(3.0, own=0.30))
    assert not any("is weak" in x for x in ok), \
        "the verdict must be DERIVED from the ratio, not asserted"
    assert any("is adequate" in x for x in ok)
    marginal = _hl(provenance=_prov(1.4))
    assert any("is marginal" in x for x in marginal)

def test_unreliable_provenance_states_so_rather_than_judging():
    out = _hl(provenance=_prov(None, reliable=False, unclassified=0.79))
    assert any("cannot be assessed" in x for x in out)
    assert not any("is weak" in x or "is adequate" in x or "is marginal" in x
                   for x in out)

def test_professional_referral_is_category_neutral_and_conditional():
    """The wording named a health professional in every category, and stated the figure
    even at zero."""
    none = _hl(framing={"mentions": 100, "pos_share": 0.3, "neu_share": 0.7,
                        "advisory_share": 0.0})
    assert not any("professional" in x for x in none)
    some = _hl(framing={"mentions": 100, "pos_share": 0.3, "neu_share": 0.7,
                        "advisory_share": 0.25})
    _line = next(x for x in some if "professional" in x)
    assert "health" not in _line.lower(), "category-specific wording"
    assert "qualified professional" in _line

def test_headline_names_the_rank_case():
    """Mentioned and not chosen is a different problem with a different remedy, and the
    mention figure alone conceals it."""
    out = _hl(metrics={
        "mention_by_intent": [{"intent": "need", "mention_rate": 0.354, "runs": 1610}],
        "first_mention_by_intent": [{"intent": "need", "first_rate": 0.007}]})
    assert any("retrieved and then not chosen" in x for x in out)
    healthy = _hl(metrics={
        "mention_by_intent": [{"intent": "need", "mention_rate": 0.60, "runs": 1610}],
        "first_mention_by_intent": [{"intent": "need", "first_rate": 0.35}]})
    assert not any("not chosen" in x for x in healthy)