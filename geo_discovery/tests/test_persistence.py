"""W-30/W-31/W-32 and W-46: persistence, span identity, the claims denominator."""
import pytest

from benchmark.diagnostics import (build_diagnostics, claims_denominator,
                                   content_hash, diagnostics_summary, drift_check,
                                   load_diagnostics, save_diagnostics,
                                   save_manifest, save_prepared_claims)
from benchmark.spans import span_id

def _row(query="best lotion", engine="gemini", model="gemini-2.5-flash",
         run_index=0, web_on=True):
    return {"query": query, "engine": engine, "model": model,
            "run_index": run_index, "web_on": web_on}

# ---------------------------------------------------------------- W-31
def test_span_id_is_independent_of_row_position():
    """The positional ID made the veracity cache unsound: a judgement keyed on `412_0`
    was returned for whatever span occupied index 412 after the next audit batch."""
    a = span_id(_row(), "Brand X is gentle.", 120)
    b = span_id(_row(), "Brand X is gentle.", 120)
    assert a == b and a.startswith("s")

def test_span_id_distinguishes_different_content():
    assert span_id(_row(), "Brand X is gentle.", 0) != \
           span_id(_row(), "Brand X is harsh.", 0)

def test_span_id_distinguishes_the_run_that_produced_it():
    """Two runs of the same query can yield identical text, and they are different
    observations."""
    assert span_id(_row(run_index=0), "same text", 0) != \
           span_id(_row(run_index=1), "same text", 0)
    assert span_id(_row(engine="gemini"), "same text", 0) != \
           span_id(_row(engine="gpt4o"), "same text", 0)
    assert span_id(_row(web_on=True), "same text", 0) != \
           span_id(_row(web_on=False), "same text", 0)

def test_span_id_distinguishes_position_within_a_response():
    assert span_id(_row(), "Brand X appears twice.", 10) != \
           span_id(_row(), "Brand X appears twice.", 400)

def test_span_corpus_carries_a_content_hash_id():
    import pandas as pd
    from benchmark.spans import build_span_corpus
    df = pd.DataFrame([{**_row(), "intent": "need", "sources": [],
                        "response_text": "Brand X is a gentle moisturiser. It helps."}])
    c = build_span_corpus(df, "Brand X")
    assert not c.empty
    assert c.iloc[0]["span_id"].startswith("s")
    assert len(c.iloc[0]["span_id"]) == 15
    assert c.iloc[0]["span_id_positional"] == "0_0", "transitional key retained"

# ---------------------------------------------------------------- cache version
def test_a_prompt_change_invalidates_the_cache():
    """Without a prompt version, a judge-prompt fix silently did not apply to any claim
    already cached — the cache returned the verdict the old prompt produced."""
    import hashlib
    from benchmark.veracity import PROMPT_VERSION

    def _key(version):
        return hashlib.sha256(
            ("a claim|gemini|" + version + "|s123").encode()).hexdigest()[:24]

    assert _key(PROMPT_VERSION) != _key("v2")
    assert PROMPT_VERSION != "v2", "must be bumped past the unversioned scheme"

# ---------------------------------------------------------------- W-30
def _ver():
    return {"method": "veracity_v2", "span_count": 300,
            "judgements": [{"claim": "c", "span_id": f"s{i}", "verdict": "supported",
                            "quote": "verbatim text"} for i in range(50)],
            "hallucinations": [{"span_id": "s1", "assertion": "a",
                                "classification": "unsubstantiated"}],
            "per_claim": [{"claim": "c", "engine": "e", "status": "locked_in"}],
            "claims_judged": 40, "judge_yield": 0.95,
            "quote_verification_failures": 2,
            "calibration": {"agreement": 0.85, "n_compared": 30}}

def test_diagnostics_capture_the_evidence_trail():
    d = build_diagnostics(veracity=_ver(), judge_model="gemini-2.5-pro")
    assert d["n_span_judgements"] == 50
    assert len(d["span_judgements"]) == 50
    assert d["span_judgements"][0]["quote"] == "verbatim text", \
        "the verbatim quote IS the evidence trail"
    assert d["judge"]["judge_yield"] == 0.95
    assert d["judge"]["calibration"]["agreement"] == 0.85

def test_diagnostics_are_capped_and_say_so():
    """Span judgements can run to tens of thousands of rows, and an artifact nobody can
    open is not persistence."""
    v = {**_ver(), "judgements": [{"span_id": f"s{i}"} for i in range(5000)]}
    d = build_diagnostics(veracity=v, span_limit=100)
    assert len(d["span_judgements"]) == 100
    assert d["n_span_judgements"] == 5000
    assert d["span_judgements_truncated"] is True

def test_diagnostics_accept_a_dataframe():
    """Benchmark mixes DataFrames, lists and None."""
    import pandas as pd
    d = build_diagnostics(
        winnability=pd.DataFrame([{"query": "q", "winnability": "winnable"}]),
        richness_agg=pd.DataFrame([{"brand": "B", "richness_per_100t": 3.2}]),
        reason_gap=None)
    assert d["winnability"][0]["winnability"] == "winnable"
    assert d["richness_aggregate"][0]["brand"] == "B"
    assert d["reason_gap"] == []

def test_diagnostics_round_trip(tmp_path, monkeypatch):
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    save_diagnostics("p", "t", build_diagnostics(veracity=_ver(),
                                                 judge_model="gemini-2.5-pro"))
    back = load_diagnostics("p", "t")
    assert back["n_span_judgements"] == 50
    s = diagnostics_summary(back)
    assert s["persisted"] and s["judge_model"] == "gemini-2.5-pro"

def test_absent_diagnostics_are_declared_not_assumed():
    s = diagnostics_summary(None)
    assert s["persisted"] is False
    assert "only in the session that produced them" in s["note"]

# ---------------------------------------------------------------- denominator
def test_prepared_register_is_the_denominator(tmp_path, monkeypatch):
    """Preparation SPLITS multi-fact entries, so dividing by the raw count could report
    more than 100% assessed."""
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    raw = [{"claim": f"raw {i}"} for i in range(10)]
    kept = [{"claim": f"prepared {i}"} for i in range(14)]      # splitting expanded it
    save_prepared_claims("p", "t", kept, {"input": 10, "kept": 14, "n_split": 6,
                                          "n_merged": 1, "n_dropped": 3}, raw)
    den = claims_denominator("p", "t", raw)
    assert den["denominator"] == 14 and den["basis"] == "prepared"
    assert den["n_raw"] == 10

def test_denominator_falls_back_and_declares_it(tmp_path, monkeypatch):
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    den = claims_denominator("p", "t", [{"claim": "x"}] * 7)
    assert den["denominator"] == 7 and den["basis"] == "raw_register_fallback"
    assert "may be wrong in either direction" in den["note"]

def test_coverage_cannot_exceed_one_hundred_percent(tmp_path, monkeypatch):
    import store as S
    from report.contract import coverage_declaration
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    S.save_artifact("p", "t", "claims_register", [{"claim": f"c{i}"} for i in range(10)],
                    {})
    S.save_artifact("p", "t", "delta_scorecard",
                    {"evidence_base": {"claims_considered": 41}}, {})
    save_prepared_claims("p", "t", [{"claim": f"p{i}"} for i in range(41)],
                         {"input": 10, "kept": 41, "n_split": 31}, [{"claim": "c"}] * 10)
    cov = coverage_declaration("p", "t")
    assert cov["claims_in_register"] == 41
    assert cov["claims_in_raw_register"] == 10
    assert cov["claims_assessed_pct"] == 100.0, "never above 100"

def test_heavy_exclusion_is_reported(tmp_path, monkeypatch):
    """53% of one register was discarded before veracity ran and nothing downstream knew."""
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    p = save_prepared_claims(
        "p", "t", [{"claim": f"k{i}"} for i in range(41)],
        {"input": 112, "kept": 41, "n_dropped": 59, "n_merged": 12, "n_split": 0,
         "dropped_by_kind": {"brand_voice": 50, "too_short": 9},
         "dropped": [{"claim": "free of fragrances", "kind": "brand_voice",
                      "why": "no verifiable anchor"}]}, [{"claim": "r"}] * 112)
    assert p["preparation"]["n_dropped"] == 59
    assert p["preparation"]["dropped_by_kind"]["brand_voice"] == 50
    assert load_diagnostics("p", "t") is None     # separate artifact
    assert "neither an upper nor a lower bound" in p["denominator_note"]

# ---------------------------------------------------------------- drift
def test_drift_detects_a_grown_audit_log(tmp_path, monkeypatch):
    """An audit backfilled after Benchmark ran leaves the scorecard describing a corpus
    that no longer exists, and nothing noticed."""
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    save_manifest("p", "t", df_rows=3220, engines=["gemini"], intents=["need"],
                  judge_model="m", prepared_claims=[{"claim": "c"}],
                  source_pack={"pack_version": 1, "n_domains": 200})
    d = drift_check("p", "t", current_rows=5515,
                    current_pack={"pack_version": 1})
    assert d["has_drift"] and any("3,220" in x and "5,515" in x for x in d["drift"])

def test_drift_detects_a_changed_source_pack(tmp_path, monkeypatch):
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    save_manifest("p", "t", df_rows=100, engines=[], intents=[], judge_model="m",
                  prepared_claims=[], source_pack={"pack_version": 1})
    d = drift_check("p", "t", current_rows=100, current_pack={"pack_version": 2})
    assert d["has_drift"] and any("source pack has changed" in x for x in d["drift"])

def test_no_drift_when_nothing_moved(tmp_path, monkeypatch):
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    save_manifest("p", "t", df_rows=100, engines=[], intents=[], judge_model="m",
                  prepared_claims=[], source_pack={"pack_version": 1})
    d = drift_check("p", "t", current_rows=100, current_pack={"pack_version": 1})
    assert not d["has_drift"]

def test_absent_manifest_is_declared(tmp_path, monkeypatch):
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    d = drift_check("p", "t", current_rows=100)
    assert d["has_manifest"] is False
    assert "may describe a corpus that has since been backfilled" in d["note"]

def test_prepared_claims_hash_is_stable_and_discriminating():
    a = content_hash(["claim one", "claim two"])
    assert a == content_hash(["claim one", "claim two"])
    assert a != content_hash(["claim one", "claim three"])

# ---------------------------------------------------------------- W-32
def test_new_artifacts_are_loadable_and_reportable():
    from report.contract import CONTRACT
    from store import ARTIFACT_NAMES
    for name in ("benchmark_diagnostics", "prepared_claims", "benchmark_manifest",
                 "gap_register", "source_pack"):
        assert name in ARTIFACT_NAMES, f"{name} would be dropped on a session refresh"
    _c = {r.artifact for r in CONTRACT}
    for name in ("benchmark_diagnostics", "prepared_claims", "benchmark_manifest",
                 "gap_register"):
        assert name in _c, f"{name} is loadable but nothing in the report can use it"

# ---------------------------------------------------------------- W-46
def test_basis_none_with_a_null_value_is_accepted():
    """Three of ten blockers in a live run came from the critic flagging the very form the
    pillar prompt mandates — which trains an analyst to override the gate."""
    from optimise.strategy_llm import deterministic_validate
    si = {"wins": [], "nulls": [], "claims": [], "gaps": [], "metrics": {},
          "substantiation_backlog": []}
    a = {"action_id": "P1-1", "pillar": "Content", "title": "Publish the detail",
         "what": "publish the specific HMO content on the PDP",
         "evidence_class": "DIAGNOSED", "validating_experiment": "E4 at N>=100",
         "kill_criterion": "stop if flat at re-audit",
         "expected_effect": {"metric": "mention_rate", "value": None, "basis": "none",
                             "qualitative": "moderate"}}
    errs = deterministic_validate(si, {"Content": {"actions": [a]}},
                                 {"not_doing": [{"item": "x", "basis": "C20"}],
                                  "big_bets": [], "structural_truths": []})
    assert not [e for e in errs if e["target"] == "P1-1"
                and e["severity"] == "blocker"]
    assert not [e for e in errs if e["error"] == "value_without_basis"]

def test_a_number_with_basis_none_is_flagged():
    """A figure from nowhere is the actual fault the critic should be catching."""
    from optimise.strategy_llm import deterministic_validate
    si = {"wins": [], "nulls": [], "claims": [], "gaps": [], "metrics": {},
          "substantiation_backlog": []}
    a = {"action_id": "P1-2", "pillar": "Content", "title": "x", "what": "y",
         "evidence_class": "DIAGNOSED", "validating_experiment": "E4",
         "kill_criterion": "stop",
         "expected_effect": {"value": 18.0, "basis": "none"}}
    errs = deterministic_validate(si, {"Content": {"actions": [a]}},
                                 {"not_doing": [{"item": "x", "basis": "C20"}],
                                  "big_bets": [], "structural_truths": []})
    assert any(e["error"] == "value_without_basis" for e in errs)

def test_an_inference_must_name_its_source():
    from optimise.strategy_llm import deterministic_validate
    si = {"wins": [], "nulls": [], "claims": [], "gaps": [], "metrics": {},
          "substantiation_backlog": []}
    a = {"action_id": "P1-3", "pillar": "Content", "title": "x", "what": "y",
         "evidence_class": "DIAGNOSED", "validating_experiment": "E4",
         "kill_criterion": "stop",
         "expected_effect": {"value": 6.0, "basis": "inferred"}}
    errs = deterministic_validate(si, {"Content": {"actions": [a]}},
                                 {"not_doing": [{"item": "x", "basis": "C20"}],
                                  "big_bets": [], "structural_truths": []})
    assert any(e["error"] == "inference_without_source" for e in errs)

def test_a_missing_qualitative_band_is_minor_not_blocking():
    from optimise.strategy_llm import deterministic_validate
    si = {"wins": [], "nulls": [], "claims": [], "gaps": [], "metrics": {},
          "substantiation_backlog": []}
    a = {"action_id": "P1-4", "pillar": "Content", "title": "x", "what": "y",
         "evidence_class": "DIAGNOSED", "validating_experiment": "E4",
         "kill_criterion": "stop",
         "expected_effect": {"value": None, "basis": "none"}}
    errs = deterministic_validate(si, {"Content": {"actions": [a]}},
                                 {"not_doing": [{"item": "x", "basis": "C20"}],
                                  "big_bets": [], "structural_truths": []})
    e = next(x for x in errs if x["error"] == "no_qualitative_band")
    assert e["severity"] == "minor"