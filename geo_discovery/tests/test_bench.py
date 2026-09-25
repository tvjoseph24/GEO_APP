"""Creative bench: design construction, text modes, legibility verdicts, N-way stats."""
import pytest

from experiment import bench as B
from experiment import visual as VIS
from experiment.bench_stats import analyze_bakeoff, bakeoff_summary, mde_nway

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 600
# A plausible MP4 header. probe() reads the MIME, not the bytes, so the content only needs
# to be long enough to clear the minimum-size check.
MP4 = b"\x00\x00\x00 ftypisom" + b"\x00" * 600

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
    d = B.build_bench_design(question="q", text_presence=tp)
    assert d["prediction"]["metric"] == metric
    assert d["question_answered"] == B.TEXT_PRESENCE[tp]["question"]
    assert d["metric_note"] and d["prediction"]["note"], \
        "the metric caveat must travel with the design, not live only in a UI caption"

def test_unknown_text_mode_is_rejected_at_construction():
    """A silent fallback to 'minimal' would produce a valid-looking finding answering a
    question nobody asked — and would fail much later, inside text generation."""
    with pytest.raises(ValueError, match="text_presence"):
        B.build_bench_design(question="q", text_presence="med")
    with pytest.raises(ValueError, match="attribution"):
        B.build_bench_design(question="q", attribution="bundled")

def test_claims_are_rejected_outside_full_text_mode():
    """In minimal/none the claim is what the ASSET must carry. Putting it in the text is
    the one failure that voids a visual experiment."""
    with pytest.raises(ValueError, match="permits no claims"):
        B.build_bench_design(question="q", text_presence="minimal",
                             claims_allowed=["Contains 1% colloidal oatmeal"])
    d = B.build_bench_design(question="q", text_presence="full",
                             claims_allowed=["Contains 1% colloidal oatmeal"])
    assert d["claims_allowed"] == ["Contains 1% colloidal oatmeal"]

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
    assert c["ok"] is False and c["severity"] == "invalid"
    assert "isolation_text_differs" in c["notes"]
    assert c["checks"]["text_identical"] is False

def test_bundle_is_runnable_but_not_attributable():
    d = B.build_bench_design(question="q", attribution="bundle")
    v = {"arm_keys": ["A", "B"], "text_presence": "minimal", "attribution": "bundle",
         "variant_A": {"text": "Brand X cream from Brand X."},
         "variant_B": {"text": "Brand X lotion from Brand X."}}
    c = B.bench_check(v, "Brand X", ["bundle warning"], {}, d)
    assert c["ok"] and c["severity"] == "caveated"
    assert "bundle_not_attributable" in c["notes"]

def test_minimal_mode_rejects_a_claim_in_the_text():
    """Defence in depth. build_bench_design blocks the claim at construction; bench_check
    blocks it again if a claim reaches the generated text by any other route — an LLM
    restating a fact it inferred from the asset read, for instance."""
    claim = "Contains 1% colloidal oatmeal"
    d = B.build_bench_design(question="q", text_presence="minimal")
    d["claims_allowed"] = [claim]        # bypass construction, as a leak would
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
    known in advance for two vision calls instead of 800 runs.

    Asserted on the `notes` token rather than the message text — prose is for the
    operator and will be reworded; the token is the contract.
    """
    v = B._legibility_verdict({"A": _read("A"), "B": _read("B")}, "Brand X",
                              {"text_presence": "minimal"})
    assert v["ok"] is False
    assert "assets_read_identically" in v["notes"]
    assert v["distinct_machine_reads"] == 1
    assert v["blockers"], "a blocker must carry an explanation for the operator"

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
    assert v["ok"] is False
    assert "payload_not_delivered" in v["notes"]

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

def test_all_arms_errored_is_a_blocker_not_a_pass():
    """Empty `live` short-circuited every check and returned ok=True — a guard reporting
    'fine' because it was handed no data."""
    v = B._legibility_verdict({"A": {"arm": "A", "error": "fetch failed"},
                               "B": {"arm": "B", "error": "timeout"}},
                              "Brand X", {"text_presence": "minimal"})
    assert not v["ok"] and "no_readable_arm" in v["notes"]
    assert v["arms_errored"] == ["A", "B"]

def test_string_instead_of_list_does_not_corrupt_the_read():
    """A model returning a bare string would report 18 claims instead of 1."""
    v = B._legibility_verdict(
        {"A": {"arm": "A", "brand_name_read": "Brand X",
               "visible_claims": "24-hour hydration", "legible_text": "Brand X",
               "text_legibility": "clear", "image_tokens": 258,
               "payload_delivered": True}},
        "Brand X", {"text_presence": "minimal"})
    assert v["comparison"][0]["n_claims"] == 1

def test_an_arm_the_model_reads_nothing_from_is_flagged():
    v = B._legibility_verdict(
        {"A": _read("A"),
         "B": {"arm": "B", "brand_name_read": "", "legible_text": [],
               "visible_claims": [], "numbers_read": [], "certifications_read": [],
               "text_legibility": "illegible", "image_tokens": 258,
               "payload_delivered": True}},
        "Brand X", {"text_presence": "minimal"})
    assert "empty_machine_read" in v["notes"]
    assert any("invisible to" in c for c in v["cautions"])

# tests/test_bench.py — append

# ---------------------------------------------------------------- live-run findings
def test_provider_without_token_reporting_is_not_a_delivery_failure():
    """OpenAI folds image tokens into prompt_tokens, so a LiteLLM engine reports zero on
    EVERY vision call. A token-only test marked a clean finding as caveated."""
    from experiment.results import _visual_validity
    er = {"attribution": "isolation", "engines": {
        "azure/gpt-4o": {
            "A": {"n": 100, "mentioned": 26, "visual": {
                "provider": "litellm", "reports_media_tokens": False,
                "runs_with_image": 100, "runs_with_image_tokens": 0,
                "image_tokens_total": 0}},
            "B": {"n": 100, "mentioned": 100, "visual": {
                "provider": "litellm", "reports_media_tokens": False,
                "runs_with_image": 100, "runs_with_image_tokens": 0,
                "image_tokens_total": 0}}}}}
    v = _visual_validity(er)
    assert v["verdict"] == "OK", "a silent provider must not read as a delivery failure"
    assert v["behavioural_evidence"], \
        "identical text with a 74pp arm gap IS proof the image arrived"
    assert "litellm" in v["providers_not_reporting_tokens"]
    assert v["runs_on_silent_providers"] == 200

def test_silent_provider_with_no_arm_difference_stays_partial():
    """Without token evidence AND without a behavioural gap, delivery is unverified."""
    from experiment.results import _visual_validity
    er = {"attribution": "isolation", "engines": {"azure/gpt-4o": {
        "A": {"n": 100, "mentioned": 50, "visual": {
            "provider": "litellm", "reports_media_tokens": False,
            "runs_with_image": 100, "runs_with_image_tokens": 0,
            "image_tokens_total": 0}},
        "B": {"n": 100, "mentioned": 52, "visual": {
            "provider": "litellm", "reports_media_tokens": False,
            "runs_with_image": 100, "runs_with_image_tokens": 0,
            "image_tokens_total": 0}}}}}
    assert _visual_validity(er)["verdict"] == "PARTIAL"

def test_reporting_provider_with_zero_tokens_is_still_invalid():
    """Vertex DOES publish the breakdown, so zero there is a genuine failure."""
    from experiment.results import _visual_validity
    er = {"attribution": "isolation", "engines": {"gemini-2.5-flash": {
        "A": {"n": 100, "mentioned": 1, "visual": {
            "provider": "vertex", "reports_media_tokens": True,
            "runs_with_image": 100, "runs_with_image_tokens": 0,
            "image_tokens_total": 0}}}}}
    assert _visual_validity(er)["verdict"] == "INVALID"

def test_mixed_providers_report_the_measurable_rate():
    from experiment.results import _visual_validity
    er = {"attribution": "isolation", "engines": {
        "gemini-2.5-flash": {"A": {"n": 100, "mentioned": 1, "visual": {
            "provider": "vertex", "reports_media_tokens": True,
            "runs_with_image": 200, "runs_with_image_tokens": 200,
            "image_tokens_total": 176400}},
                             "B": {"n": 100, "mentioned": 52}},
        "azure/gpt-4o": {"A": {"n": 100, "mentioned": 26, "visual": {
            "provider": "litellm", "reports_media_tokens": False,
            "runs_with_image": 200, "runs_with_image_tokens": 0,
            "image_tokens_total": 0}},
                         "B": {"n": 100, "mentioned": 100}}}}
    v = _visual_validity(er)
    assert v["verdict"] == "OK"
    assert v["measurable_delivery_rate"] == 1.0, \
        "the verifiable half was 100%, not 50% — the old test averaged across a provider " \
        "that cannot report"
    assert v["runs_on_silent_providers"] == 200

def test_mde_is_undefined_at_floor_and_ceiling():
    """W-21 reproduced in bench_stats: a clamp made 0% and 100% both read ~3.9pp."""
    floor = analyze_bakeoff(_er(A=(0, 100), B=(0, 100)))
    r = next(x for x in floor if x["metric"] == "mention_rate")
    assert r["mde_pp"] is None and "floored" in r["mde_note"]

    ceil = analyze_bakeoff(_er(A=(100, 100), B=(100, 100)))
    r2 = next(x for x in ceil if x["metric"] == "mention_rate")
    assert r2["mde_pp"] is None and "saturated" in r2["mde_note"]

def test_mde_uses_the_pooled_rate_not_the_winner():
    """Computed on the winner, a 100% arm clamped to 0.99 and produced an absurdly small
    MDE for a design with no headroom."""
    rows = analyze_bakeoff(_er(A=(26, 100), B=(100, 100)))
    r = next(x for x in rows if x["metric"] == "mention_rate")
    assert r["mde_pp"] is not None
    assert r["mde_pp"] > 5, f"pooled 63% should give a substantial MDE, got {r['mde_pp']}"

def test_bench_power_verdict_is_exploratory_not_underpowered():
    """A bake-off predicts no specific lift, so the confirmatory power question does not
    apply. Gating on it would block every bench run."""
    from experiment.power import power_verdict
    pilot = {"p0_mention": 0.26, "p0_first": 0.08, "n_total": 40,
             "ceiling_risk": False, "floor_risk": False, "mde_table": {},
             "piloted_with_asset": True}
    pv = power_verdict(pilot, {"_bench": True, "n_visual_arms": 2,
                               "prediction": {"metric": "mention_rate"}}, 100)
    assert pv["verdict"] == "exploratory" and pv["exploratory"] is True
    assert pv["mde_pp"] and "nothing to power FOR" in pv["note"]

def test_bench_floor_names_the_competitor_field_as_the_suspect():
    from experiment.power import power_verdict
    pilot = {"p0_mention": 0.0, "p0_first": 0.0, "n_total": 40,
             "ceiling_risk": False, "floor_risk": True, "mde_table": {}}
    pv = power_verdict(pilot, {"_bench": True, "n_visual_arms": 2,
                               "prediction": {"metric": "mention_rate"}}, 100)
    assert pv["verdict"] == "floor"
    assert "competitor field" in pv["note"], \
        "at 9 words against a full field the floor is a configuration problem"

def test_no_variance_reports_no_winner():
    """A stable sort on equal rates left arm A first, so a floor-on-every-arm test
    reported winner='A' at a 0.0% rate."""
    rows = analyze_bakeoff(_er(A=(0, 100), B=(0, 100)))
    r = next(x for x in rows if x["metric"] == "mention_rate")
    assert r["verdict"] == "no_variance"
    assert r["winner"] is None and r["winner_rate"] is None
    assert r["leader"] == "A", "the top-ranked arm is still reported, separately"

def test_equivalent_reports_no_winner():
    rows = analyze_bakeoff(_er(A=(50, 100), B=(52, 100)))
    r = next(x for x in rows if x["metric"] == "mention_rate")
    assert r["verdict"] == "equivalent" and r["winner"] is None
    assert r["leader"] == "B"

def test_saturated_winner_gap_is_a_lower_bound():
    """Arm B at 100/100: +74pp understates, and no further creative could beat it."""
    rows = analyze_bakeoff(_er(A=(26, 100), B=(100, 100)))
    r = next(x for x in rows if x["metric"] == "mention_rate")
    assert r["winner_at_ceiling"] and r["gap_is_lower_bound"]
    assert "LOWER BOUND" in r["decision"] and "headroom" in r["decision"]

def test_magnitude_divergence_flagged_even_when_engines_agree():
    """Observed live: both engines picked B, with gaps of 51pp and 74pp. The pooled
    62.5pp describes neither, and engines_agree=True hid it."""
    er = {"design_id": "V01", "mode": "creative", "engines": {
        "gemini": {"A": {"n": 100, "mentioned": 1, "is_first": 0},
                   "B": {"n": 100, "mentioned": 52, "is_first": 0}},
        "gpt4o": {"A": {"n": 100, "mentioned": 26, "is_first": 8},
                  "B": {"n": 100, "mentioned": 100, "is_first": 47}}}}
    summ = bakeoff_summary(analyze_bakeoff(er), prefer="mention_rate")
    assert summ["engines_agree"] and summ["winner"] == "B"
    assert summ["magnitude_diverges"] and summ["gap_spread_pp"] == 23.0
    assert "describes neither" in summ["decision"]

def test_secondary_metric_result_is_surfaced():
    """top-1 moved 8% to 47% on one engine, FDR p<0.001 — a second validated result the
    summary silently dropped."""
    er = {"design_id": "V01", "mode": "creative", "engines": {
        "gpt4o": {"A": {"n": 100, "mentioned": 26, "is_first": 8},
                  "B": {"n": 100, "mentioned": 100, "is_first": 47}}}}
    summ = bakeoff_summary(analyze_bakeoff(er), prefer="mention_rate")
    sec = summ["secondary_results"]
    assert sec and sec[0]["metric"] == "top1_unconditional"
    assert sec[0]["gap_pp"] == 39.0
    assert "8% to 47%" in summ["decision"] or "8% to 47" in summ["decision"]

def test_mention_without_recommendation_routes_to_corroboration():
    """Mentioned 52% of the time, recommended first 0% of the time, under either creative.
    That is an authority constraint and no creative change will move it."""
    er = {"design_id": "V01", "mode": "creative", "engines": {
        "gemini": {"A": {"n": 100, "mentioned": 1, "is_first": 0},
                   "B": {"n": 100, "mentioned": 52, "is_first": 0}}}}
    summ = bakeoff_summary(analyze_bakeoff(er), prefer="mention_rate")
    assert summ["mention_without_recommendation"] == ["gemini"]
    assert "Corroboration" in summ["mention_without_recommendation_note"]
    assert "no creative change" in summ["mention_without_recommendation_note"]

def test_payload_estimate_is_not_wildly_low():
    """441 estimated against 1,806 measured on a sub-megapixel asset."""
    pytest.importorskip("PIL")
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(buf, format="PNG")
    store, d = {}, _design_v("V90")
    VIS.attach_asset(store, d, "A", buf.getvalue(), "image/png")
    est = VIS.estimate_asset_tokens(VIS.assets(store, d)["A"])
    assert est >= 800, f"estimate {est} is still far below the ~1806 observed"

def test_measured_tokens_win_over_the_estimate():
    store, d = {}, _design_v("V91")
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    store["V91"]["A"]["measured_tokens"] = 1806
    pc = VIS.estimate_payload_cost(store, d, 10, 1, 1e-6)
    assert pc["total_media_tokens"] == 18060 and pc["measured"] is True
    assert "real figure" in pc["note"]

def _design_v(did):
    from experiment import visual as _V   # noqa: F401
    return {"design_id": did, "needs_visual": True, "n_visual_arms": 2}

def test_every_engine_is_analysed():
    """The guard for a truncated loop. `return out` inside the engine loop discards every
    engine after the first — and that reads as unanimous agreement rather than as missing
    data, so it is invisible in the output. Observed once."""
    er = {"design_id": "V01", "mode": "creative", "engines": {
        "e1": {"A": {"n": 100, "mentioned": 10, "is_first": 2},
               "B": {"n": 100, "mentioned": 60, "is_first": 20}},
        "e2": {"A": {"n": 100, "mentioned": 20, "is_first": 4},
               "B": {"n": 100, "mentioned": 70, "is_first": 30}},
        "e3": {"A": {"n": 100, "mentioned": 30, "is_first": 6},
               "B": {"n": 100, "mentioned": 80, "is_first": 40}}}}
    rows = analyze_bakeoff(er)
    assert sorted({r["engine"] for r in rows}) == ["e1", "e2", "e3"]
    assert len(rows) == 6, "three engines x two metrics"

def test_silent_provider_note_names_the_engine_not_the_gateway():
    """'litellm publishes no breakdown' names a gateway. An operator needs the engine."""
    from experiment.results import _visual_validity
    er = {"attribution": "isolation", "engines": {"azure/gpt-4o": {
        "A": {"n": 100, "mentioned": 26, "visual": {
            "provider": "litellm", "model": "azure/gpt-4o",
            "reports_media_tokens": False, "runs_with_image": 100,
            "runs_with_image_tokens": 0, "image_tokens_total": 0}},
        "B": {"n": 100, "mentioned": 100, "visual": {
            "provider": "litellm", "model": "azure/gpt-4o",
            "reports_media_tokens": False, "runs_with_image": 100,
            "runs_with_image_tokens": 0, "image_tokens_total": 0}}}}}
    v = _visual_validity(er)
    assert v["engines_not_reporting_tokens"] == ["azure/gpt-4o"]
    assert "azure/gpt-4o" in v["note"]

def test_a_single_cell_run_is_incomplete_not_a_null():
    """76 rows in ONE cell reported as a well-powered null → DISPROVEN → a client-facing
    exclusion built on a single interrupted arm."""
    from experiment.results import assemble_findings
    er = {"design_id": "C12", "mode": "progressive", "experiment_id": "E1",
          "engines": {"gemini": {"A": {"n": 76, "mentioned": 10, "is_first": 2}}}}
    f = assemble_findings([er], "Brand X")[0]
    assert f["verdict"] == "insufficient_data"
    assert f["evidence_class"] == "INCONCLUSIVE", \
        "an incomplete run must never be DISPROVEN — that becomes a client exclusion"
    assert f["underpowered"] is True
    assert any("interrupted" in w for w in f["variance_warnings"])

def test_two_arms_on_one_engine_is_comparable():
    from experiment.results import assemble_findings
    er = {"design_id": "X1", "mode": "ab", "experiment_id": "E0",
          "engines": {"gemini": {"A": {"n": 100, "mentioned": 20, "is_first": 5},
                                 "B": {"n": 100, "mentioned": 60, "is_first": 25}}}}
    f = assemble_findings([er], "Brand X")[0]
    assert f["verdict"] != "insufficient_data"

def test_rate_table_is_produced_on_the_live_path():
    """Only reanalyse.py produced it, so metrics_bundle.retrieval_delta — which reads it —
    was empty unless someone happened to run a re-analysis."""
    from experiment.results import assemble_findings
    er = {"design_id": "X1", "mode": "ab", "experiment_id": "E0",
          "engines": {"gemini": {"A": {"n": 100, "mentioned": 30, "is_first": 5},
                                 "B": {"n": 100, "mentioned": 70, "is_first": 25}}}}
    f = assemble_findings([er], "Brand X")[0]
    rt = f.get("rate_table") or []
    assert rt, "rate_table must be produced by the live aggregator"
    a = next(r for r in rt if r["arm"] == "A" and r["metric"] == "mention")
    assert a["rate"] == 0.3 and a["ci_lo"] < 0.3 < a["ci_hi"]

def test_creative_findings_excluded_from_retrieval_delta(tmp_path, monkeypatch):
    """A bench arm A is 20 words of minimal text. Counted as 'the brand's own document in
    context' it drags in_context_worst to ~1%."""
    import pandas as pd
    import store as S
    from benchmark.metrics_bundle import build_metrics_bundle
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    S.save_artifact("p", "t", "validated_findings", [
        {"design_id": "C01", "mode": "ab",
         "rate_table": [{"arm": "A", "metric": "mention", "rate": 0.80}]},
        {"design_id": "V01", "mode": "creative", "source": "bench",
         "rate_table": [{"arm": "A", "metric": "mention", "rate": 0.01}]}], {})
    df = pd.DataFrame([{"engine": "e1", "intent": "need", "mentioned": True,
                        "is_first": False, "first_index": 3, "refusal": False,
                        "num_sources": 2, "source_type_tally": {}, "polarity": "neu",
                        "rank": 1, "outcome": "brand_recommended",
                        "advisory_referral": False,
                        "brand_mentions": {}, "classification": {}}])
    b = build_metrics_bundle(df, "t", [], {}, None, {}, parent_brand_for_delta="p")
    rd = b["retrieval_delta"]
    assert rd["n_control_arms"] == 1 and rd["creative_findings_excluded"] == 1
    assert rd["in_context_worst"] == 0.8, \
        "the bench arm must not become the worst-case in-context reference"

def test_bench_id_reserves_against_the_raw_log(tmp_path, monkeypatch):
    """The bug this prevents: session state is wiped on restart, so _next_id re-issued V01
    and its runs merged with the V01 already on disk. Both carry mode='creative', so the
    contamination check would not fire."""
    import json
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    d = S.project_dir("p", "t")
    with open(d / "experiment_raw.jsonl", "w") as f:
        for arm in "AB":
            f.write(json.dumps({"design_id": "V01", "arm": arm,
                                "mode": "creative", "model": "m", "run": 0}) + "\n")
        f.write(json.dumps({"design_id": "_pilot_V01", "arm": "A",
                            "mode": "pilot", "model": "m", "run": 0}) + "\n")
    assert "V01" in B.reserved_design_ids("p", "t", [])
    assert B.build_bench_design(question="q", parent="p",
                                target="t")["design_id"] == "V02"

def test_bench_id_reserves_against_saved_designs(tmp_path, monkeypatch):
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    S.save_artifact("p", "t", "bench_designs",
                    [{"design_id": "V01"}, {"design_id": "V02"}], {})
    assert B.build_bench_design(question="q", parent="p",
                                target="t")["design_id"] == "V03"

def test_bench_id_unions_all_three_sources(tmp_path, monkeypatch):
    import json
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    d = S.project_dir("p", "t")
    with open(d / "experiment_raw.jsonl", "w") as f:
        f.write(json.dumps({"design_id": "V03", "mode": "creative"}) + "\n")
    S.save_artifact("p", "t", "bench_designs", [{"design_id": "V01"}], {})
    nxt = B.build_bench_design(question="q", parent="p", target="t",
                               existing_designs=[{"design_id": "V02"}])["design_id"]
    assert nxt == "V04", "session, saved designs and the raw log must ALL be reserved"

def test_video_is_read_when_the_model_supports_it():
    """Video was skipped by design. But Gemini accepts video natively, and for video the
    pre-check is the ONLY way to get a real token count — the estimate guesses duration."""
    class _G:
        def __init__(self):
            self.tasks = []

        def generate(self, model, prompt, **kw):
            self.tasks.append(kw.get("task"))
            return {"parsed": {"brand_name_read": "Brand X",
                               "legible_text": ["Brand X"],
                               "visible_claims": ["24-hour hydration"],
                               "numbers_read": [], "certifications_read": [],
                               "text_legibility": "clear", "dominant_subject": "pack",
                               "on_screen_duration": "most"},
                    "image_tokens": 13500}

    store, d = {}, B.build_bench_design(question="q")
    VIS.attach_asset(store, d, "A", MP4, "video/mp4")
    VIS.attach_asset(store, d, "B", PNG, "image/png")
    g = _G()
    out = B.precheck_legibility(g, store, d, "gemini-2.5-flash",
                               target_brand="Brand X")
    assert "legibility_precheck_video" in g.tasks
    assert out["video_capable_model"] is True
    assert out["arms_skipped"] == []
    assert out["measured_video_tokens"] == 13500
    assert VIS.assets(store, d)["A"]["measured_tokens"] == 13500
    assert any("re-sent on every run" in c for c in out["cautions"])

def test_video_skipped_with_a_reason_on_a_vision_only_model():
    class _G:
        def generate(self, *a, **k):
            raise AssertionError("must not call a vision-only model with video")

    store, d = {}, B.build_bench_design(question="q")
    VIS.attach_asset(store, d, "A", MP4, "video/mp4")
    out = B.precheck_legibility(_G(), store, d, "azure/gpt-4o", target_brand="Brand X")
    assert out["arms_skipped"] == ["A"]
    assert "video-capable" in out["reads"]["A"]["note"] or \
           "no video support" in out["reads"]["A"]["note"]

def test_fleeting_brand_exposure_is_cautioned():
    class _G:
        def generate(self, *a, **k):
            return {"parsed": {"brand_name_read": "Brand X", "legible_text": ["Brand X"],
                               "visible_claims": [], "numbers_read": [],
                               "certifications_read": [], "text_legibility": "partial",
                               "dominant_subject": "lifestyle",
                               "on_screen_duration": "fleeting"},
                    "image_tokens": 9000}

    store, d = {}, B.build_bench_design(question="q")
    VIS.attach_asset(store, d, "A", MP4, "video/mp4")
    out = B.precheck_legibility(_G(), store, d, "gemini-2.5-flash",
                               target_brand="Brand X")
    assert any("fleetingly legible" in c for c in out["cautions"])