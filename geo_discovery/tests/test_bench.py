"""Creative bench: design construction, text modes, legibility verdicts, N-way stats."""
import pytest

from experiment import bench as B
from experiment import visual as VIS
from experiment.bench_stats import analyze_bakeoff, bakeoff_summary, mde_nway

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 600

# ---------------------------------------------------------------- design
def test_design_ids_do_not_collide_with_hypothesis_or_signal_designs():
    """C01 and S001 are the other namespaces. completed_runs and aggregate_from_raw key
    on design_id, so a collision would merge runs across incompatible modes."""
    existing = [{"design_id": "C01"}, {"design_id": "S001"}, {"design_id": "V01"}]
    d = B.build_bench_design(question="best moisturiser", existing_designs=existing)
    assert d["design_id"] == "V02"

def test_bench_runs_in_its_own_mode():
    d = B.build_bench_design(question="q")
    assert d["run_mode"] == "creative"
    assert d["visual_locked"] is True, "progressive would replace the manipulation"

def test_arm_count_is_clamped():
    assert B.build_bench_design(question="q", n_arms=99)["n_visual_arms"] == 5
    assert B.build_bench_design(question="q", n_arms=1)["n_visual_arms"] == 2

@pytest.mark.parametrize("tp,metric", [("none", "mention_rate"),
                                       ("minimal", "mention_rate"),
                                       ("full", "top1_unconditional")])
def test_full_text_targets_rank_not_mention(tp, metric):
    """With claims in the text, mention saturates — the finding must come from rank."""
    assert B.build_bench_design(question="q",
                                text_presence=tp)["prediction"]["metric"] == metric

def test_isolation_and_bundle_declare_different_variables():
    assert B.build_bench_design(question="q", attribution="isolation")["isolates"] \
           == "creative execution"
    assert "package" in B.build_bench_design(question="q",
                                             attribution="bundle")["isolates"]

# ---------------------------------------------------------------- validation
def test_bench_design_of_four_arms_requires_four_assets():
    store = {}
    d = B.build_bench_design(question="q", n_arms=4)
    for arm in "ABC":
        VIS.attach_asset(store, d, arm, PNG + arm.encode(), "image/png")
    txt = dict.fromkeys("ABCD", "Brand X cream from Brand X.")
    v = VIS.validate(store, d, B.BENCH_TEMPLATE, txt)
    assert not v["ok"], "a missing fourth asset must be caught"
    VIS.attach_asset(store, d, "D", PNG + b"D", "image/png")
    assert VIS.validate(store, d, B.BENCH_TEMPLATE, txt)["ok"]

def test_isolation_rejects_differing_text():
    d = B.build_bench_design(question="q", attribution="isolation")
    v = {"arm_keys": ["A", "B"], "text_presence": "minimal", "attribution": "isolation",
         "variant_A": {"text": "Brand X cream from Brand X."},
         "variant_B": {"text": "Brand X lotion from Brand X."}}
    c = B.bench_check(v, "Brand X", [], {}, d)
    assert not c["ok"] and "Isolation mode requires identical text" in c["invalid"][0]

def test_bundle_is_runnable_but_not_attributable():
    d = B.build_bench_design(question="q", attribution="bundle")
    v = {"arm_keys": ["A", "B"], "text_presence": "minimal", "attribution": "bundle",
         "variant_A": {"text": "Brand X cream from Brand X."},
         "variant_B": {"text": "Brand X lotion from Brand X."}}
    c = B.bench_check(v, "Brand X", ["bundle warning"], {}, d)
    assert c["ok"] and c["severity"] == "caveated"
    assert "bundle_not_attributable" in c["notes"]

def test_minimal_mode_rejects_a_claim_in_the_text():
    """If the claim is in the text the model answers without looking at the image."""
    claim = "Contains 1% colloidal oatmeal"
    d = B.build_bench_design(question="q", text_presence="minimal",
                             claims_allowed=[claim])
    v = {"arm_keys": ["A", "B"], "text_presence": "minimal", "attribution": "isolation",
         "variant_A": {"text": f"Brand X cream. {claim}. Brand X."},
         "variant_B": {"text": f"Brand X cream. {claim}. Brand X."}}
    c = B.bench_check(v, "Brand X", [], {}, d)
    assert not c["ok"] and "claim_leak" in c["notes"]

def test_none_mode_flags_the_floor_and_full_mode_the_ceiling():
    for tp, note in (("none", "no_brand_in_arm"), ("full", "binary_at_ceiling")):
        d = B.build_bench_design(question="q", text_presence=tp)
        v = {"arm_keys": ["A", "B"], "text_presence": tp, "attribution": "isolation",
             "variant_A": {"text": "x"}, "variant_B": {"text": "x"}}
        c = B.bench_check(v, "Brand X", [], {}, d)
        assert note in c["notes"] and c["severity"] == "caveated"

# ---------------------------------------------------------------- legibility
def _read(arm, brand="Brand X", text=None, claims=None, tokens=258, legible="clear"):
    return {"arm": arm, "brand_name_read": brand,
            "legible_text": text if text is not None else ["Brand X", "Moisturiser"],
            "visible_claims": claims if claims is not None else ["24-hour hydration"],
            "numbers_read": [], "certifications_read": [],
            "text_legibility": legible, "dominant_subject": "pack",
            "image_tokens": tokens, "payload_delivered": bool(tokens)}

def test_identical_machine_reads_block_the_run():
    """The highest-value check: if the model reads both assets the same way, the null is
    known in advance for two vision calls instead of 800 runs."""
    v = B._legibility_verdict({"A": _read("A"), "B": _read("B")}, "Brand X",
                              {"text_presence": "minimal"})
    assert not v["ok"] and "read IDENTICALLY" in v["blockers"][0]
    assert v["distinct_machine_reads"] == 1

def test_different_machine_reads_pass():
    v = B._legibility_verdict(
        {"A": _read("A", claims=["24-hour hydration"]),
         "B": _read("B", claims=["24-hour hydration", "dermatologist tested"],
                    text=["Brand X", "Moisturiser", "Dermatologist tested"])},
        "Brand X", {"text_presence": "minimal"})
    assert v["ok"] and v["distinct_machine_reads"] == 2

def test_no_payload_on_any_arm_blocks():
    v = B._legibility_verdict({"A": _read("A", tokens=0), "B": _read("B", tokens=0)},
                              "Brand X", {"text_presence": "minimal"})
    assert not v["ok"] and "not reaching the model" in v["blockers"][0]

def test_unreadable_brand_with_no_text_is_a_dead_experiment():
    v = B._legibility_verdict(
        {"A": _read("A", brand="", text=["hydration"]),
         "B": _read("B", brand="", text=["moisture"])},
        "Brand X", {"text_presence": "none"})
    assert any("floor at zero" in c for c in v["cautions"])
    assert any("minimal" in c for c in v["cautions"]), "must name the remedy"

def test_video_arms_are_skipped_not_failed():
    v = B._legibility_verdict(
        {"A": _read("A"), "B": {"arm": "B", "skipped": "video"}},
        "Brand X", {"text_presence": "minimal"})
    assert v["arms_skipped"] == ["B"]
    assert not v["blockers"], "a skipped video arm must not block the run"

def test_measured_tokens_are_reported():
    v = B._legibility_verdict({"A": _read("A", tokens=1290)}, "Brand X",
                              {"text_presence": "minimal"})
    assert v["measured_tokens"]["A"] == 1290

# ---------------------------------------------------------------- N-way stats
def _er(mode="creative", **arms):
    return {"design_id": "V01", "mode": mode,
            "engines": {"e1": {k: {"n": v[1], "mentioned": v[0], "is_first": 0}
                               for k, v in arms.items()}}}

def test_multiplicity_costs_sensitivity():
    """Four arms means six comparisons. The two-arm MDE materially overstates what the
    design can detect."""
    two = mde_nway(0.5, 100, 2)
    four = mde_nway(0.5, 100, 4)
    assert four > two
    assert (four - two) * 100 > 2, "the penalty must be material enough to report"

def test_clear_winner_is_a_decision():
    rows = analyze_bakeoff(_er(A=(20, 100), B=(60, 100)))
    r = next(x for x in rows if x["metric"] == "mention_rate")
    assert r["omnibus_significant"] and r["winner"] == "B"
    assert r["verdict"] == "winner" and r["winner_distinguishable"]
    assert r["gap_ci_pp"] and r["gap_ci_pp"][0] > 0

def test_equivalent_arms_are_reported_as_equivalent_not_as_a_winner():
    rows = analyze_bakeoff(_er(A=(50, 100), B=(52, 100)))
    r = next(x for x in rows if x["metric"] == "mention_rate")
    assert r["verdict"] == "equivalent"
    assert "human grounds" in r["decision"]

def test_a_leader_not_separated_from_second_is_not_a_decision():
    """The failure this exists to prevent: promoting a noise winner from a four-way test."""
    rows = analyze_bakeoff(_er(A=(10, 100), B=(45, 100), C=(50, 100), D=(12, 100)))
    r = next(x for x in rows if x["metric"] == "mention_rate")
    assert r["omnibus_significant"], "B/C clearly beat A/D"
    assert r["winner"] == "C" and r["runner_up"] == "B"
    assert r["verdict"] == "leader_not_separated"
    assert not r["winner_distinguishable"]
    assert "confirmation run" in r["decision"]

def test_ceiling_and_floor_are_named_not_reported_as_a_null():
    ceil = analyze_bakeoff(_er(A=(100, 100), B=(100, 100)))
    r = next(x for x in ceil if x["metric"] == "mention_rate")
    assert r["verdict"] == "no_variance" and "ceiling" in r["omnibus_note"]
    assert "top-1" in r["decision"]

    floor = analyze_bakeoff(_er(A=(0, 100), B=(0, 100)))
    r2 = next(x for x in floor if x["metric"] == "mention_rate")
    assert r2["verdict"] == "no_variance" and "floor" in r2["omnibus_note"]

def test_pairwise_only_runs_when_the_omnibus_passes():
    """Six comparisons on an equivalent set is the multiplicity trap the gate exists for."""
    rows = analyze_bakeoff(_er(A=(50, 100), B=(51, 100), C=(49, 100), D=(50, 100)))
    r = next(x for x in rows if x["metric"] == "mention_rate")
    assert r["pairwise"] == []

def test_pairwise_is_fdr_corrected():
    rows = analyze_bakeoff(_er(A=(10, 100), B=(30, 100), C=(50, 100), D=(70, 100)))
    r = next(x for x in rows if x["metric"] == "mention_rate")
    assert r["pairwise"] and all("p_fdr" in x for x in r["pairwise"])
    assert all(x["p_fdr"] >= x["p_value"] - 1e-9 for x in r["pairwise"]), \
        "an FDR-adjusted p can never be smaller than the raw p"

def test_ranking_is_ordered_and_has_intervals():
    rows = analyze_bakeoff(_er(A=(20, 100), B=(60, 100), C=(40, 100)))
    r = next(x for x in rows if x["metric"] == "mention_rate")
    assert [x["arm"] for x in r["ranking"]] == ["B", "C", "A"]
    assert all(x["ci_lo"] <= x["rate"] <= x["ci_hi"] for x in r["ranking"])
    assert all(x["rank"] == i for i, x in enumerate(r["ranking"], 1))

# ---------------------------------------------------------------- cross-engine
def test_engine_disagreement_is_reported_not_averaged():
    """Averaging two engines that prefer different creatives yields a figure describing
    neither — and 'creative B wins' would be actively wrong."""
    er = {"design_id": "V01", "mode": "creative", "engines": {
        "e1": {"A": {"n": 100, "mentioned": 20, "is_first": 0},
               "B": {"n": 100, "mentioned": 70, "is_first": 0}},
        "e2": {"A": {"n": 100, "mentioned": 70, "is_first": 0},
               "B": {"n": 100, "mentioned": 20, "is_first": 0}}}}
    summ = bakeoff_summary(analyze_bakeoff(er), prefer="mention_rate")
    assert summ["verdict"] == "engine_specific"
    assert not summ["engines_agree"]
    assert summ["winner"] is None, "there is no single winning creative"
    assert "engine overlay" in summ["decision"]

def test_agreeing_engines_produce_one_decision():
    er = {"design_id": "V01", "mode": "creative", "engines": {
        "e1": {"A": {"n": 100, "mentioned": 20, "is_first": 0},
               "B": {"n": 100, "mentioned": 70, "is_first": 0}},
        "e2": {"A": {"n": 100, "mentioned": 25, "is_first": 0},
               "B": {"n": 100, "mentioned": 65, "is_first": 0}}}}
    summ = bakeoff_summary(analyze_bakeoff(er), prefer="mention_rate")
    assert summ["verdict"] == "winner" and summ["winner"] == "B"
    assert summ["engines_agree"]
