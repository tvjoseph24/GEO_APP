"""Per-design pilots, unconditional rank, metric selection. Covers W-20, W-24, W-29."""
import json

import pytest

# ---------------------------------------------------------------- W-24 rank
from experiment.exp_stats import analyze_rank

def _log(tmp_path, rows):
    f = tmp_path / "experiment_raw.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows))
    return str(f)

def _runs(design, arm, model, n_total, ranks):
    """ranks: list of rank values; None means the brand was absent."""
    return [{"design_id": design, "mode": "ab", "arm": arm, "model": model,
             "run": i, "rank": (ranks[i] if i < len(ranks) else None),
             "mentioned": ranks[i] is not None if i < len(ranks) else False,
             "error": None}
            for i in range(n_total)]

def test_unconditional_top1_exposes_the_real_movement(tmp_path):
    """The observed failure: n_A=57, n_B=100, improvement reported as 0 while
    A_top1_pct read 100 — a CONDITIONAL share. True movement was 57% -> 100%."""
    a = _runs("C1", "A", "g", 100, [1] * 57)          # 57 present, all rank 1
    b = _runs("C1", "B", "g", 100, [1] * 100)         # 100 present, all rank 1
    rk = analyze_rank(_log(tmp_path, a + b), "C1", mode="ab")
    r = rk[0]
    assert r["A_mean_rank"] == r["B_mean_rank"] == 1.0
    assert r["improvement"] == 0.0
    assert r["A_top1_cond_pct"] == 100.0        # conditional — the misleading figure
    assert r["A_top1_uncond_pct"] == 57.0      # unconditional — the real one
    assert r["B_top1_uncond_pct"] == 100.0
    assert r["top1_uncond_lift_pp"] == 43.0

def test_presence_divergence_is_flagged(tmp_path):
    a = _runs("C2", "A", "g", 100, [1] * 50)
    b = _runs("C2", "B", "g", 100, [1] * 95)
    r = analyze_rank(_log(tmp_path, a + b), "C2", mode="ab")[0]
    assert "conditioning_warning" in r
    assert "different populations" in r["conditioning_warning"]

def test_degenerate_test_is_not_reported_as_a_p_value(tmp_path):
    """Fully-tied arrays previously produced p=1.0 via a bare except, indistinguishable
    from a computed result."""
    a = _runs("C3", "A", "g", 60, [1] * 60)
    b = _runs("C3", "B", "g", 60, [1] * 60)
    r = analyze_rank(_log(tmp_path, a + b), "C3", mode="ab")[0]
    assert r["test_ok"] is False
    assert "zero variance" in (r["test_error"] or "")
    assert r["significant"] is False

def test_fdr_is_applied_across_comparisons(tmp_path):
    rows = []
    for arm, rk in (("A", [2] * 60), ("B", [1] * 60), ("C", [2] * 60), ("D", [2] * 60)):
        rows += _runs("C4", arm, "g", 60, rk)
    out = analyze_rank(_log(tmp_path, rows), "C4", mode="ab")
    assert out[0]["_meta"]["fdr_applied"] is True
    assert all(r.get("p_fdr") is not None for r in out if r.get("test_ok"))

def test_excluded_engines_are_reported(tmp_path):
    """Silent exclusion removed the rescue metric exactly when arm A was weakest."""
    a = _runs("C5", "A", "g", 100, [1] * 2)          # only 2 present
    b = _runs("C5", "B", "g", 100, [1] * 100)
    out = analyze_rank(_log(tmp_path, a + b), "C5", mode="ab")
    meta = out[0]["_meta"]
    assert meta["n_comparisons"] == 0
    assert meta["engines_excluded"][0]["engine"] == "g"
    assert "present" in meta["engines_excluded"][0]["why"]

# ---------------------------------------------------------------- W-20 power
from experiment.power import mde, power_verdict

def _pilot(m, f, n=40):
    return {"design_id": "C1", "p0_mention": m, "p0_first": f, "n_total": n,
            "ceiling_risk": m >= 0.90, "floor_risk": m <= 0.05,
            "at_floor": m <= 0.02, "at_ceiling": m >= 0.98,
            "mde_table": {} if (m <= 0.02 or m >= 0.98) else {"100": {}},
            "per_engine": {}}

def test_missing_pilot_does_not_claim_power():
    v = power_verdict(None, {"prediction": {"lift_pp": 10}}, 100)
    assert v["verdict"] == "no_pilot" and v["mde_pp"] is None

def test_pilot_baseline_overrides_the_audit_baseline():
    """The audit said 0.7%; the pilot measures 45% because the brand's own document is
    in context. The hypothesis was sized against the wrong number."""
    v = power_verdict(_pilot(0.60, 0.45),
                      {"prediction": {"metric": "first_rec_rate",
                                      "baseline_pp": 0.7, "lift_pp": 15.0}}, 100)
    assert v["pilot_p0"] == 0.45
    assert "baseline_shift" in v
    assert "0.7%" in v["baseline_shift"] and "45" in v["baseline_shift"]

def test_metric_override_changes_the_verdict():
    """At a 97% mention baseline there is no headroom, but first-rec at 30% has plenty.
    The selected metric must drive the verdict — W-29's advice to 'switch the primary
    metric' was previously unactionable because no control existed."""
    p = _pilot(0.97, 0.30)
    assert power_verdict(p, {"prediction": {"lift_pp": 25}}, 100,
                         metric_override="mention_rate")["verdict"] == "ceiling"
    assert power_verdict(p, {"prediction": {"lift_pp": 25}}, 100,
                         metric_override="first_rec_rate")["verdict"] == "powered"

def test_mde_table_suppressed_at_both_extremes():
    """mde() clamps p0 to [0.01, 0.99], so floor and ceiling yield the SAME ~3.9pp at
    N=100 — one number for two opposite situations."""
    assert _pilot(0.00, 0.00)["mde_table"] == {}
    assert _pilot(1.00, 1.00)["mde_table"] == {}
    assert round(mde(0.01, 100) * 100, 1) == round(mde(0.99, 100) * 100, 1)

# ---------------------------------------------------------------- W-29 metric
from experiment.exp_stats import verdict

def _rows(metric, a, b, sig, engines=("g", "x")):
    return [{"engine": e, "arm": "B", "metric": metric, "A_rate": a, "B_rate": b,
             "lift_pp": (b - a) * 100, "significant": sig} for e in engines]

def test_selected_metric_is_honoured():
    """The metric was previously picked post-hoc — whichever reached significance first —
    so mention_rate's win would have been reported even when first_rec_rate was chosen."""
    rows = _rows("mention_rate", 0.30, 0.90, True) + \
           _rows("first_rec_rate", 0.25, 0.27, False)
    v = verdict(rows, 5.0, primary_metric="first_rec_rate")
    assert v["best_metric"] == "first_rec_rate"
    assert v["verdict"] == "no_lift", v
    assert v["metric_was_selected"] is True

def test_floor_and_ceiling_are_distinguished_on_the_selected_metric():
    """A null at 8% control is a FLOOR problem; at 95% it is a CEILING problem. Same
    verdict word would hide entirely different remedies."""
    low = verdict(_rows("first_rec_rate", 0.08, 0.09, False), 5.0,
                  primary_metric="first_rec_rate")
    high = verdict(_rows("first_rec_rate", 0.95, 0.96, False), 5.0,
                   primary_metric="first_rec_rate")
    assert low["verdict"] == "lost" and high["verdict"] == "already_winning"

def test_post_hoc_path_unchanged_when_no_metric_given():
    rows = _rows("mention_rate", 0.30, 0.90, True) + \
           _rows("first_rec_rate", 0.10, 0.12, False)
    v = verdict(rows, 5.0)
    assert v["best_metric"] == "mention_rate"

# ---------------------------------------------------------------- recovered clusters
from benchmark.hypothesis import validate

def test_substantiation_deficit_with_a_registered_claim_is_testable():
    """8 of 10 not_testable clusters had usable claims — including the ONLY E9
    objection design, deleted by a mechanism label rather than by evidence."""
    claims = [{"claim": "The brand's positioning is built on dermatologist endorsement."}]
    h = [{"mechanism": "substantiation_deficit", "generated_ok": True,
          "claims_used": ["The brand's positioning is built on dermatologist endorsement."],
          "prediction": {"metric": "mention_rate", "baseline_pp": 13.6, "lift_pp": 15.0},
          "experiment": {"template": "E9", "isolated_variable": "rebuttal",
                         "arm_A": "no rebuttal present",
                         "arm_B": "substantiated rebuttal present",
                         "target_query": "cream for very dry chapped hands"},
          "falsification": "no lift means the claim is insufficient",
          "pillar": "Content", "_packet_baseline": 0.136,
          "_brand_free": ["cream for very dry chapped hands"],
          "_target_queries": ["cream for very dry chapped hands"]}]
    out = validate(h, claims=claims, brand="Aveeno", n_planned=100)
    assert out[0]["status"] == "ready", out[0].get("validation_errors")
    assert out[0].get("also_route") == "optimise_pillar6"

def test_substantiation_deficit_without_claims_stays_not_testable():
    h = [{"mechanism": "substantiation_deficit", "generated_ok": True,
          "claims_used": [],
          "prediction": {"metric": "mention_rate", "baseline_pp": 40.0, "lift_pp": 20.0},
          "experiment": {"template": "E4", "isolated_variable": "claim",
                         "arm_A": "without", "arm_B": "with",
                         "target_query": "best cream"},
          "falsification": "x", "pillar": "Content", "_packet_baseline": 0.4,
          "_brand_free": ["best cream"], "_target_queries": ["best cream"]}]
    assert validate(h, claims=[], brand="Aveeno")[0]["status"] == "not_testable"

# ---------------------------------------------------------------- family wiring
def test_visual_gap_types_reach_the_visual_family():
    from benchmark.gap_cluster import FAMILY
    from benchmark.hypothesis import FAMILY_TEMPLATES
    assert FAMILY["VISUAL_UNASSESSED"] == "visual"
    assert FAMILY["VISUAL_UNREADABLE"] == "visual"
    assert FAMILY_TEMPLATES["visual"][0] == "E7"