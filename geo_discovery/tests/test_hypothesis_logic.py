"""Prediction repair, metric selection, family wiring and quota allocation.
Covers W-37, W-64, W-76 and the two Phase-1b wiring defects."""
import pytest

from benchmark.hypothesis import (FAMILY_TEMPLATES, choose_metric, mde,
                                  normalise_prediction)

# ---------------------------------------------------------------- prediction repair
def test_expected_below_baseline_is_read_as_a_lift():
    """Observed on 13 of 53 hypotheses: the LLM wrote the MDE into expected_pp."""
    p, notes = normalise_prediction(
        {"baseline_pp": 35.4, "expected_pp": 19.0}, mde_pp=18.9)
    assert p["expected_pp"] == 54.4
    assert p["lift_pp"] == 19.0
    assert notes and "lift" in notes[0]

def test_fractions_are_converted_to_percentage_points():
    p, notes = normalise_prediction(
        {"baseline_pp": 0.354, "expected_pp": 0.55}, mde_pp=18.9)
    assert p["baseline_pp"] == 35.4
    assert p["expected_pp"] == 55.0
    assert any("fraction" in n for n in notes)

def test_explicit_lift_is_preferred():
    p, _ = normalise_prediction({"baseline_pp": 0.7, "lift_pp": 12.0}, mde_pp=8.0)
    assert p["expected_pp"] == 12.7 and p["lift_pp"] == 12.0

def test_no_prediction_survives_as_zero():
    """A degenerate prediction must never pass validation silently."""
    p, _ = normalise_prediction({"baseline_pp": 35.4, "expected_pp": 35.4}, mde_pp=18.9)
    assert p["lift_pp"] > 0, "a no-change prediction was not repaired"

def test_absurd_required_n_no_longer_possible():
    p, _ = normalise_prediction(
        {"baseline_pp": 0.007, "expected_pp": 0.196}, mde_pp=18.9)
    assert p["baseline_pp"] == 0.7 and p["lift_pp"] > 18.0

# ---------------------------------------------------------------- metric selection
MBI = {"need": 0.354, "persona": 0.735, "branded": 1.0, "freshness": 0.062}
FBI = {"need": 0.007, "persona": 0.212, "branded": 1.0, "freshness": 0.062}

def test_rank_problem_targets_first_rec_rate():
    """Aveeno: 35.4% mention on need queries, 0.7% first. That is rank, not visibility —
    yet 40 of 53 hypotheses targeted mention_rate."""
    metric, p0, note = choose_metric("need", MBI, FBI)
    assert metric == "first_rec_rate"
    assert p0 == pytest.approx(0.007)
    assert "RANK" in note

def test_saturated_mention_targets_first_rec_rate():
    metric, p0, _ = choose_metric("persona", MBI, FBI)
    assert metric == "first_rec_rate" and p0 == pytest.approx(0.212)

def test_genuine_absence_targets_mention_rate():
    metric, p0, _ = choose_metric("freshness", MBI, FBI)
    assert metric == "mention_rate" and p0 == pytest.approx(0.062)

def test_objection_family_always_uses_mention():
    metric, _, note = choose_metric("negative", {"negative": 0.136},
                                    {"negative": 0.0}, family="objection")
    assert metric == "mention_rate" and "FRAMING" in note

# ---------------------------------------------------------------- family wiring
def test_every_gap_family_has_templates():
    from benchmark.gap_cluster import FAMILY
    for fam in set(FAMILY.values()):
        assert fam in FAMILY_TEMPLATES, f"family '{fam}' has no template shortlist"

def test_objection_family_leads_with_e9():
    assert FAMILY_TEMPLATES["objection"][0] == "E9"

def test_new_gap_types_are_mapped():
    from benchmark.gap_cluster import FAMILY
    for t, fam in (("OBJECTION_EXPOSURE", "objection"),
                   ("DEFEND_POSITION", "defend"),
                   ("CLAIM_NO_EVIDENCE", "claim"),
                   ("RANK_GAP", "rank")):
        assert FAMILY.get(t) == fam, f"{t} maps to {FAMILY.get(t)}, expected {fam}"

# ---------------------------------------------------------------- quota
from benchmark.gap_cluster import allocate_quota

def _cl(i, fam, sev="high", n=2):
    return {"cluster_id": f"C{i:02d}", "family": fam, "severity": sev, "n_gaps": n}

def test_quota_guarantees_family_spread():
    """A pure severity sort put 40 of 53 clusters on the claim family, deleting
    E1/E2/E5/E9 from the programme entirely."""
    clusters = ([_cl(i, "claim", "critical") for i in range(30)]
                + [_cl(100 + i, "objection", "medium") for i in range(2)]
                + [_cl(200 + i, "reason", "medium") for i in range(2)]
                + [_cl(300 + i, "provenance", "medium") for i in range(2)])
    kept = allocate_quota(clusters, 12)
    fams = {c["family"] for c in kept}
    assert {"objection", "reason", "provenance"} <= fams, \
        f"severity sort deleted families: {fams}"
    n_claim = sum(1 for c in kept if c["family"] == "claim")
    assert n_claim <= 6, f"claim family took {n_claim} of 12 slots"

def test_quota_is_a_no_op_below_the_cap():
    clusters = [_cl(i, "claim") for i in range(5)]
    assert allocate_quota(clusters, 20) == clusters

# ---------------------------------------------------------------- polarity
def test_negative_evidence_is_not_suppressed():
    """neg_share of exactly 0.000 across 2,911 skincare mentions is not credible —
    a single directed criticism in a long multi-mention answer was being discarded."""
    from metrics_spec import build_registry, measure
    reg = build_registry("Aveeno", ["CeraVe"])
    txt = ("Aveeno is a popular choice for sensitive skin. Some users report that "
           "Aveeno can cause irritation on very reactive skin. Aveeno remains widely "
           "recommended by dermatologists, and Aveeno is affordable.")
    m = measure(txt, reg)
    assert m["has_negative"] is True, "directed criticism was discarded"
    assert m["n_negative_mentions"] >= 1