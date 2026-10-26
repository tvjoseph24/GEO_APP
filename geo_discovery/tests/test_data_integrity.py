"""P0/P1: cost persistence, finding merge, reconciliation, evidence completeness."""
import json

import pytest

from experiment.results import merge_findings
from meter import Meter

def _f(did, mode="ab", cls="PROVEN", runs=100, verdict="validated", **kw):
    return {"design_id": did, "mode": mode, "evidence_class": cls,
            "runs_per_cell": runs, "verdict": verdict, **kw}

# ---------------------------------------------------------------- cost
def test_restore_sets_the_cursor_so_events_are_not_rewritten():
    """Without this the first autosave re-appends every restored event and the ledger
    doubles on each restart."""
    m = Meter()
    m.restore([{"cost_usd": 1.0, "tok_in": 10, "tok_out": 5, "seconds": 1.0}] * 30)
    assert m._save_cursor == 30
    assert m.total_cost() == pytest.approx(30.0)

def test_autosave_appends_only_new_events():
    seen = []

    def cb(meter, cursor):
        seen.append([e["task"] for e in meter.events[cursor:]])
        return len(meter.events)

    m = Meter(autosave_every=3)
    m.autosave_cb = cb
    for i in range(7):
        m.record("gemini-2.5-flash", "vertex", 10, 5, 0.1, "audit", f"t{i}")
    m.flush()
    flat = [t for batch in seen for t in batch]
    assert flat == [f"t{i}" for i in range(7)], "every event written exactly once"
    assert len(seen) == 3, "two threshold writes plus the flush"

def test_autosave_failure_does_not_abort_a_run():
    """A failed cost write must never kill a paid run."""
    def cb(meter, cursor):
        raise IOError("disk full")

    m = Meter(autosave_every=1)
    m.autosave_cb = cb
    m.record("gemini-2.5-flash", "vertex", 10, 5, 0.1, "audit", "t")
    assert m.calls == 1

def test_cost_ledger_round_trips(tmp_path, monkeypatch):
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    m = Meter(budget_usd=250.0)
    for i in range(5):
        m.record("gemini-2.5-flash", "vertex", 100, 50, 0.2, "audit", f"t{i}")
    cur = S.save_cost_log("p", "t", m, 0)
    assert cur == 5

    back = S.load_cost_log("p", "t")
    assert len(back) == 5
    m2 = Meter()
    m2.restore(back)
    assert m2.total_cost() == pytest.approx(m.total_cost())
    assert S.load_cost_meta("p", "t")["budget_usd"] == 250.0, \
        "the cap must survive a restart, or a raised cap reverts and aborts the next run"

def test_ledger_survives_a_torn_final_line(tmp_path, monkeypatch):
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    d = S.project_dir("p", "t")
    with open(d / S.COST_LEDGER, "w") as f:
        f.write(json.dumps({"cost_usd": 1.0, "module": "audit"}) + "\n")
        f.write(json.dumps({"cost_usd": 2.0, "module": "audit"}) + "\n")
        f.write('{"cost_usd": 3.0, "modu')          # interrupted write
    ev = S.load_cost_log("p", "t")
    assert len(ev) == 2, "a torn line must not lose the events before it"

def test_cost_summary_groups_by_phase(tmp_path, monkeypatch):
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    m = Meter()
    m.record("gemini-2.5-flash", "vertex", 1000, 500, 1.0, "audit", "run")
    m.record("gemini-2.5-flash", "vertex", 2000, 1000, 1.0, "benchmark", "judge")
    S.save_cost_log("p", "t", m, 0)
    cs = S.cost_summary("p", "t")
    assert set(cs["by_module"]) == {"audit", "benchmark"}
    assert cs["by_module"]["benchmark"] > cs["by_module"]["audit"]

# ---------------------------------------------------------------- merge
def test_second_batch_does_not_delete_the_first():
    """THE bug: emit wrote the session's findings wholesale, so running a text batch and
    then a bench batch left Optimise reading only the bench findings."""
    text = [_f("C01"), _f("C19", mode="progressive"), _f("C20", cls="DISPROVEN",
                                                         verdict="no_lift")]
    bench = [_f("V01", mode="creative", cls="PROVEN_ENGINE_SPECIFIC")]
    merged, rep = merge_findings(text, bench)
    assert {f["design_id"] for f in merged} == {"C01", "C19", "C20", "V01"}
    assert rep["added"] == 1 and rep["carried_forward"] == 3

def test_a_rerun_at_higher_n_supersedes():
    prior = [_f("V01", mode="creative", cls="PROVEN_WITH_CAVEAT", runs=20)]
    fresh = [_f("V01", mode="creative", cls="PROVEN_ENGINE_SPECIFIC", runs=100)]
    merged, rep = merge_findings(prior, fresh)
    assert len(merged) == 1
    assert merged[0]["runs_per_cell"] == 100
    assert merged[0]["supersedes"] == ["V01@20runs/arm"]
    assert rep["superseded"][0]["dropped_runs"] == 20

def test_a_weaker_rerun_does_not_overwrite_stronger_evidence():
    """An interrupted re-run must not replace a complete one."""
    prior = [_f("C01", cls="PROVEN", runs=100)]
    fresh = [_f("C01", cls="INCONCLUSIVE", runs=12, verdict="no_lift")]
    merged, _ = merge_findings(prior, fresh)
    assert merged[0]["evidence_class"] == "PROVEN"
    assert merged[0]["runs_per_cell"] == 100
    assert merged[0]["supersedes"] == ["C01@12runs/arm"]

def test_mode_is_part_of_the_identity():
    """An A/B and a progressive run of the same design answer different questions."""
    merged, _ = merge_findings([_f("C19", mode="ab")],
                               [_f("C19", mode="progressive")])
    assert len(merged) == 2

def test_merge_is_order_independent_for_distinct_designs():
    a, b = [_f("C01")], [_f("V01", mode="creative")]
    assert ({f["design_id"] for f in merge_findings(a, b)[0]}
            == {f["design_id"] for f in merge_findings(b, a)[0]})

def test_save_merges_with_what_is_on_disk(tmp_path, monkeypatch):
    import store as S
    from experiment.results import save_experiment_results
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    save_experiment_results("p", "t", [{"design_id": "C01", "mode": "ab"}],
                            [_f("C01")], [{"design_id": "C01"}], {})
    rep = save_experiment_results(
        "p", "t", [{"design_id": "V01", "mode": "creative"}],
        [_f("V01", mode="creative")], [{"design_id": "V01", "source": "bench"}], {})
    saved = S.load_artifact("p", "t", "validated_findings")
    assert {f["design_id"] for f in saved} == {"C01", "V01"}
    assert rep["carried_forward"] == 1
    assert len(S.load_artifact("p", "t", "experiment_results")) == 2
    assert {e["design_id"] for e in
            S.load_artifact("p", "t", "experiment_variants")["experiments"]} \
        == {"C01", "V01"}

def test_merge_false_replaces_deliberately(tmp_path, monkeypatch):
    import store as S
    from experiment.results import save_experiment_results
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    save_experiment_results("p", "t", [], [_f("C01")], [], {})
    save_experiment_results("p", "t", [], [_f("V01", mode="creative")], [], {},
                            merge=False)
    assert {f["design_id"] for f in
            S.load_artifact("p", "t", "validated_findings")} == {"V01"}

# ---------------------------------------------------------------- reconcile
def _raw(tmp_path, monkeypatch, rows):
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    d = S.project_dir("p", "t")
    with open(d / "experiment_raw.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return S

def test_coverage_detects_findings_missing_from_the_artifact(tmp_path, monkeypatch):
    from experiment.reconcile import coverage
    rows = [{"design_id": d, "mode": m, "arm": a, "model": "gemini", "run": i,
             "mentioned": True, "is_first": False, "error": None}
            for d, m in (("C01", "ab"), ("V01", "creative"))
            for a in "AB" for i in range(5)]
    S = _raw(tmp_path, monkeypatch, rows)
    S.save_artifact("p", "t", "validated_findings", [_f("V01", mode="creative")], {})
    cov = coverage("p", "t")
    assert ("C01", "ab") in cov["missing"]
    assert ("V01", "creative") not in cov["missing"]

def test_pilots_are_excluded_from_reconciliation(tmp_path, monkeypatch):
    from experiment.reconcile import load_rows
    rows = ([{"design_id": "C01", "mode": "ab", "arm": "A", "model": "m", "run": 0}]
            + [{"design_id": "_pilot_C01", "mode": "pilot", "arm": "A",
                "model": "m", "run": i} for i in range(20)])
    _raw(tmp_path, monkeypatch, rows)
    df, pilots = load_rows("p", "t")
    assert pilots == 20 and len(df) == 1
    assert "_pilot_C01" not in set(df["design_id"])

def test_reconcile_rebuilds_both_batches(tmp_path, monkeypatch):
    from experiment.reconcile import reconcile
    rows = [{"design_id": d, "mode": m, "arm": a, "model": "gemini-2.5-flash", "run": i,
             "mentioned": (a == "B"), "is_first": False, "error": None,
             "question": "best lotion"}
            for d, m in (("C01", "ab"), ("V01", "creative"))
            for a in "AB" for i in range(40)]
    S = _raw(tmp_path, monkeypatch, rows)
    S.save_artifact("p", "t", "validated_findings", [_f("V01", mode="creative")], {})
    findings, rep = reconcile("p", "t", save=True)
    assert rep["saved"] and rep["n_designs"] == 2
    assert {f["design_id"] for f in
            S.load_artifact("p", "t", "validated_findings")} == {"C01", "V01"}

def test_reconcile_refuses_to_destroy_bakeoff_data(tmp_path, monkeypatch):
    """Losing a bake-off is silent and unrecoverable without a re-run."""
    from experiment.reconcile import reconcile
    rows = [{"design_id": "C01", "mode": "ab", "arm": a, "model": "m", "run": i,
             "mentioned": True, "is_first": False, "error": None}
            for a in "AB" for i in range(40)]
    S = _raw(tmp_path, monkeypatch, rows)
    S.save_artifact("p", "t", "validated_findings",
                    [_f("V01", mode="creative",
                        bakeoff=[{"engine": "e", "metric": "mention_rate"}])], {})
    _, rep = reconcile("p", "t", save=True)
    assert not rep["saved"] and "V01" in rep["bakeoff_at_risk"]
    assert "Refusing to save" in rep["error"]

# ---------------------------------------------------------------- evidence policy
def test_all_evidence_reaches_the_prompts_never_only_the_new():
    """Filtering by 'not yet used' would flip a pillar from DISPROVEN_MECHANISM to
    UNTESTED — changing the instruction from 'do not recommend more of the same' to
    'propose anything with a validating experiment'."""
    from optimise.strategy_llm import pillar_evidence_verdict
    old_null = {"design_id": "C20", "template": "E1", "underpowered": False,
                "runs_per_cell": 100, "first_seen_in_strategy": "S001"}
    si_all = {"wins": [], "nulls": [old_null], "substantiation_backlog": []}
    si_new_only = {"wins": [], "nulls": [], "substantiation_backlog": []}
    assert pillar_evidence_verdict(si_all,
                                   "Technical/Structured-Data")["stance"] \
        == "DISPROVEN_MECHANISM"
    assert pillar_evidence_verdict(si_new_only,
                                   "Technical/Structured-Data")["stance"] \
        == "UNTESTED", "this is the wrong answer a filter would produce"

def test_incomplete_evidence_is_a_blocker():
    from optimise.strategy_llm import deterministic_validate
    si = {"wins": [], "nulls": [], "claims": [], "gaps": [],
          "substantiation_backlog": [],
          "evidence_coverage": {"missing": [("C01", "ab"), ("C19", "progressive")]}}
    errs = deterministic_validate(si, {}, {"not_doing": [{"item": "x"}],
                                           "big_bets": []})
    e = next(x for x in errs if x["error"] == "findings_missing_from_artifact")
    assert e["severity"] == "blocker" and "Reconcile" in e["fix"]

def test_complete_evidence_raises_no_coverage_error():
    from optimise.strategy_llm import deterministic_validate
    si = {"wins": [], "nulls": [], "claims": [], "gaps": [],
          "substantiation_backlog": [], "evidence_coverage": {"missing": []}}
    errs = deterministic_validate(si, {}, {"not_doing": [{"item": "x"}],
                                           "big_bets": []})
    assert not [e for e in errs if e["error"] == "findings_missing_from_artifact"]

def test_first_seen_stamp_marks_only_unstamped(tmp_path, monkeypatch):
    import store as S
    from optimise.strategy_llm import mark_evidence_seen
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    S.save_artifact("p", "t", "validated_findings",
                    [_f("C01", first_seen_in_strategy="S001"), _f("V01",
                                                                  mode="creative")], {})
    new = mark_evidence_seen("p", "t", None, "S002")
    assert new == ["V01"]
    saved = {f["design_id"]: f["first_seen_in_strategy"]
             for f in S.load_artifact("p", "t", "validated_findings")}
    assert saved == {"C01": "S001", "V01": "S002"}

# geo_discovery/tests/test_data_integrity.py — append

def test_reconcile_survives_rows_missing_optional_fields(tmp_path, monkeypatch):
    """The raw log is append-only and has gained fields over time. An older row simply
    lacks `question`, and a bare g['question'] raised KeyError on the whole
    reconciliation — which is the recovery path, so a crash there means no recovery."""
    from experiment.reconcile import reconcile
    rows = [{"design_id": "C01", "mode": "ab", "arm": a, "model": "gemini-2.5-flash",
             "run": i, "mentioned": (a == "B"), "is_first": False, "error": None}
            for a in "AB" for i in range(40)]          # no question, comp_mode, position
    _raw(tmp_path, monkeypatch, rows)
    findings, rep = reconcile("p", "t", save=True)
    assert rep.get("saved") and rep["n_designs"] == 1
    # The point is that it did not crash. CARRY skips None, so an absent question stays
    # absent rather than becoming an explicit null.
    assert findings[0].get("question") is None
    assert findings[0]["design_id"] == "C01"

def test_question_reaches_the_finding_when_present(tmp_path, monkeypatch):
    """It was set on the experiment RESULT and never copied onto the finding, so the
    report rendered a blank on the live path while reanalyse showed the query."""
    from experiment.reconcile import reconcile
    rows = [{"design_id": "C01", "mode": "ab", "arm": a, "model": "m", "run": i,
             "mentioned": (a == "B"), "is_first": False, "error": None,
             "question": "best body lotion for dry skin", "comp_mode": "full"}
            for a in "AB" for i in range(40)]
    _raw(tmp_path, monkeypatch, rows)
    findings, _ = reconcile("p", "t", save=True)
    assert findings[0]["question"] == "best body lotion for dry skin"
    assert findings[0]["comp_mode"] == "full"

def test_reconcile_survives_rows_missing_metric_fields(tmp_path, monkeypatch):
    """`mentioned` and `is_first` were also bare — a row written before the flat-field
    migration would crash on those instead."""
    from experiment.reconcile import reconcile
    rows = [{"design_id": "C01", "mode": "ab", "arm": a, "model": "m", "run": i,
             "error": None} for a in "AB" for i in range(40)]
    _raw(tmp_path, monkeypatch, rows)
    _, rep = reconcile("p", "t", save=True)
    assert rep.get("saved"), rep.get("error")
