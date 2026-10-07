"""W-55 community and social analysis. Pure functions over a synthetic corpus."""
import pandas as pd
import pytest

from benchmark.social import (PARTICIPATORY, SOCIAL_TYPES, _bucket, _participatory,
                              _route, by_engine, by_framing, by_intent, social_report,
                              social_summary, threads)

def _row(intent="need", engine="gemini", text="Try Brand X for dry skin.",
         tally=None, sources=None, mentioned=True, polarity="neu",
         hedged=False, advisory=False, negative=False, query="q1"):
    return {"intent": intent, "engine": engine, "query": query,
            "response_text": text, "mentioned": mentioned, "polarity": polarity,
            "hedged": hedged, "advisory_referral": advisory, "has_negative": negative,
            "source_type_tally": tally or {}, "sources": sources or []}

def _src(domain, stype, url=None, title=""):
    return {"domain": domain, "source_type": stype,
            "resolved_url": url or f"https://{domain}/thread/1",
            "title": title, "domain_is_title": False}

# ---------------------------------------------------------------- 1 · intent
def test_community_citation_is_reported_per_intent():
    """The 'query-triggered' half of Pillar 5's definition, never previously computed."""
    rows = ([_row(intent="negative", tally={"community_reviews": 3}) for _ in range(20)]
            + [_row(intent="branded", tally={"editorial_authority": 2})
               for _ in range(20)])
    out = {r["intent"]: r for r in by_intent(pd.DataFrame(rows))}
    assert out["negative"]["social_trigger_rate"] == 1.0
    assert out["branded"]["social_trigger_rate"] == 0.0
    assert out["negative"]["community_over_editorial"] is True
    assert out["branded"]["community_over_editorial"] is False

def test_thin_intents_are_excluded_not_reported_as_zero():
    rows = [_row(intent="freshness", tally={"community_reviews": 1}) for _ in range(3)]
    assert by_intent(pd.DataFrame(rows)) == []

def test_intent_rates_carry_confidence_intervals():
    rows = [_row(tally={"community_reviews": 1} if i % 2 else {}) for i in range(40)]
    r = by_intent(pd.DataFrame(rows))[0]
    assert r["ci_low"] < r["social_trigger_rate"] < r["ci_high"]

# ---------------------------------------------------------------- 2 · framing
def test_hedging_delta_is_detected_and_explained():
    """If community-cited answers hedge more, scepticism enters through the SOURCE MIX,
    not the brand's own content — a different remedy."""
    rows = ([_row(tally={"community_reviews": 2}, hedged=True) for _ in range(20)]
            + [_row(tally={"editorial_authority": 2}, hedged=False) for _ in range(20)])
    fr = by_framing(pd.DataFrame(rows), "Brand X")
    assert fr["delta_with_minus_without"]["hedged"] == pytest.approx(1.0, abs=0.01)
    assert any("source mix" in i for i in fr["interpretation"])

def test_positive_community_voice_is_reported_as_an_asset():
    rows = ([_row(tally={"community_reviews": 2}, polarity="pos") for _ in range(20)]
            + [_row(tally={"editorial_authority": 2}, polarity="neu")
               for _ in range(20)])
    fr = by_framing(pd.DataFrame(rows), "Brand X")
    assert any("asset to protect" in i for i in fr["interpretation"])

def test_no_negatives_anywhere_is_caveated_not_treated_as_reassurance():
    """Observed live: neg_share 0.0 across 2,911 mentions while the backfill found
    sixteen lawsuit cues. A zero there is uninformative, not benign."""
    rows = [_row(tally={"community_reviews": 1}, negative=False) for _ in range(30)]
    fr = by_framing(pd.DataFrame(rows), "Brand X")
    assert "negative_detection_caveat" in fr
    assert "uninformative rather than reassuring" in fr["negative_detection_caveat"]

# ---------------------------------------------------------------- 3 · threads
def test_threads_are_the_unit_not_domains():
    """top_url was captured and discarded in favour of domain aggregation, so the hit-list
    recommended engaging reddit.com — not an action anyone can take."""
    a = "https://reddit.com/r/skin/comments/aaa"
    b = "https://reddit.com/r/skin/comments/bbb"
    rows = ([_row(sources=[_src("reddit.com", "community_reviews", a, "Best for eczema?")])
             for _ in range(5)]
            + [_row(sources=[_src("reddit.com", "community_reviews", b, "Worst buys")])
               for _ in range(3)])
    th = threads(pd.DataFrame(rows), "Brand X")
    assert {t["url"] for t in th} == {a, b}
    assert {t["title"] for t in th} == {"Best for eczema?", "Worst buys"}

def test_single_citation_threads_are_noise():
    rows = [_row(sources=[_src("reddit.com", "community_reviews",
                               f"https://reddit.com/{i}")]) for i in range(5)]
    assert threads(pd.DataFrame(rows), "Brand X") == []

def test_unresolved_redirects_are_excluded():
    """A page title is not a thread anyone can visit."""
    rows = [{**_row(), "sources": [
        {"domain": "Best Baby Products 2024", "source_type": "community_reviews",
         "resolved_url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/x",
         "domain_is_title": True}]} for _ in range(5)]
    assert threads(pd.DataFrame(rows), "Brand X") == []

def test_absence_index_ranks_influence_without_the_brand():
    present = "https://reddit.com/p"
    absent = "https://reddit.com/a"
    rows = ([_row(text="Brand X is great.",
                  sources=[_src("reddit.com", "community_reviews", present)])
             for _ in range(6)]
            + [_row(text="Rival Y is best.", engine="gpt4o",
                    sources=[_src("reddit.com", "community_reviews", absent)])
               for _ in range(6)]
            + [_row(text="Rival Y is best.",
                    sources=[_src("reddit.com", "community_reviews", absent)])
               for _ in range(6)])
    th = threads(pd.DataFrame(rows), "Brand X", competitors=["Rival Y"])
    assert th[0]["url"] == absent, "the thread shaping answers WITHOUT the brand ranks first"
    assert th[0]["brand_present_rate"] == 0.0
    assert th[0]["top_competitor"] == "Rival Y"
    assert th[0]["n_engines"] == 2

def test_video_social_threads_are_included():
    rows = [_row(sources=[_src("youtube.com", "video_social",
                               "https://youtube.com/watch?v=abc", "Review")])
            for _ in range(4)]
    th = threads(pd.DataFrame(rows), "Brand X")
    assert th and th[0]["source_type"] == "video_social"

# ---------------------------------------------------------------- routing
def test_non_participatory_sources_get_no_direct_action():
    """A brand cannot post on a review aggregator as itself. Recommending it would be a
    recommendation to do something impossible."""
    assert not _participatory("trustpilot.com")
    r = _route("community_reviews", False, 0.0)
    assert "no direct participation" in r and "other pillars" in r

def test_participatory_sources_require_disclosure():
    assert _participatory("reddit.com")
    r = _route("community_reviews", True, 0.0)
    assert "disclosed" in r and "NEVER undisclosed seeding" in r

def test_video_route_is_a_creator_brief():
    assert "creator brief" in _route("video_social", True, 0.0)

def test_negative_threads_bucket_to_correction():
    assert "Correct" in _bucket(0.0, 0.3, True)
    assert "Monitor" in _bucket(0.0, 0.3, False)
    assert "Defend" in _bucket(0.8, 0.0, True)
    assert "Enter" in _bucket(0.0, 0.0, True)
    assert "Earn" in _bucket(0.0, 0.0, False)

# ---------------------------------------------------------------- 4 · engines
def test_engine_divergence_is_reported():
    rows = ([_row(engine="gemini", tally={"community_reviews": 8,
                                          "editorial_authority": 2}) for _ in range(20)]
            + [_row(engine="gpt4o", tally={"community_reviews": 1,
                                           "editorial_authority": 9}) for _ in range(20)])
    out = by_engine(pd.DataFrame(rows))
    assert "_divergence" in out and "overlay" in out["_divergence"]

def test_negligible_community_weight_is_stated():
    rows = ([_row(engine="gemini", tally={"editorial_authority": 10})
             for _ in range(20)]
            + [_row(engine="gpt4o", tally={"editorial_authority": 10})
               for _ in range(20)])
    out = by_engine(pd.DataFrame(rows))
    assert "No engine cites community sources materially" in out["_divergence"]

def test_negligible_weight_is_stated_on_a_single_engine_audit():
    """A single-engine audit is common, and 'no engine listens to community voice' is a
    finding that should not require a second engine to compare against."""
    rows = [_row(tally={"editorial_authority": 10}) for _ in range(20)]
    out = by_engine(pd.DataFrame(rows))
    assert "_divergence" in out
    assert "No engine cites community sources materially" in out["_divergence"]

def test_negligible_check_precedes_the_ratio_check():
    """Two engines at 1% and 0.4% are a 2.5x ratio, but both are negligible. Reporting
    that as an engine-overlay opportunity would be a recommendation to spend on a channel
    neither engine listens to."""
    rows = ([_row(engine="a", tally={"community_reviews": 1,
                                     "editorial_authority": 99}) for _ in range(20)]
            + [_row(engine="b", tally={"editorial_authority": 100})
               for _ in range(20)])
    out = by_engine(pd.DataFrame(rows))
    assert "No engine cites community sources materially" in out["_divergence"]
    assert "overlay" not in out["_divergence"]

# ---------------------------------------------------------------- 5 · verdict
def test_negligible_verdict_says_pillar_5_should_be_empty():
    """Recommending community work where the engines do not listen is the most expensive
    mistake available in this pillar."""
    rows = [_row(tally={"editorial_authority": 50, "community_reviews": 1})
            for _ in range(30)]
    rep = social_report(pd.DataFrame(rows), "Brand X")
    assert rep["verdict"] == "negligible"
    assert "will not move machine visibility" in rep["why"]
    assert "should be minimal or empty" in rep["why"]

def test_query_triggered_verdict_names_the_intents():
    rows = ([_row(intent="negative", tally={"community_reviews": 5}) for _ in range(20)]
            + [_row(intent="branded", tally={"editorial_authority": 5})
               for _ in range(20)])
    rep = social_report(pd.DataFrame(rows), "Brand X")
    assert rep["verdict"] == "query_triggered"
    assert rep["triggered_intents"] == ["negative"]

def test_report_always_carries_the_clock_and_the_constraint():
    rep = social_report(pd.DataFrame([_row(tally={"community_reviews": 2})
                                      for _ in range(20)]), "Brand X")
    assert rep["clock"] == "memory"
    assert "cannot publish into a community" in rep["clock_note"]
    assert "Undisclosed seeding is off the table" in rep["constraint"]

def test_summary_is_flat_and_json_safe():
    import json
    rep = social_report(pd.DataFrame([_row(tally={"community_reviews": 2})
                                      for _ in range(20)]), "Brand X")
    json.dumps(social_summary(rep))

# ---------------------------------------------------------------- gaps and routing
def test_social_gaps_are_detected_and_route_to_pillar_5():
    """Pillar 5 was structurally unreachable: no gap type mapped to it, so
    build_pillar_strategies marked it _skipped and its prompt was never called."""
    from benchmark.gap_detect import detect_gaps
    from optimise.strategy_llm import GAP_TYPE_PILLAR
    assert GAP_TYPE_PILLAR["SOCIAL_ABSENCE"] == "Social/Community"
    assert GAP_TYPE_PILLAR["SOCIAL_NEGATIVE"] == "Social/Community"

    metrics = {"social": {
        "verdict": "query_triggered", "overall_social_share": 0.18,
        "total_citations": 1000,
        "by_intent": [{"intent": "negative", "runs": 100, "runs_citing_social": 40,
                       "social_trigger_rate": 0.40, "community_over_editorial": True,
                       "community_reviews": 90, "editorial_authority": 20,
                       "ci_low": 0.3, "ci_high": 0.5}],
        "threads": [{"url": "https://reddit.com/x", "domain": "reddit.com",
                     "title": "Worst lotions", "source_type": "community_reviews",
                     "times_cited": 9, "n_engines": 2, "n_queries": 3,
                     "top_intent": "negative", "brand_present_rate": 0.0,
                     "brand_absence_index": 18.0, "top_competitor": "Rival Y",
                     "competitor_mentions": 9, "negative_rate": 0.4,
                     "hedged_rate": 0.2, "participatory": True,
                     "bucket": "Correct (negative, participatory)",
                     "route": "disclosed participation"}],
        "framing": {"delta_with_minus_without": {"hedged": 0.2},
                    "interpretation": ["Answers citing a community source hedge more."]}}}
    gaps = detect_gaps({}, metrics=metrics)
    types = {g["type"] for g in gaps}
    assert "SOCIAL_ABSENCE" in types and "SOCIAL_NEGATIVE" in types
    assert all(g["pillar"] == "Social/Community"
               for g in gaps if g["type"].startswith("SOCIAL_"))
    thread_gap = next(g for g in gaps if g["type"] == "SOCIAL_ABSENCE"
                      and (g.get("evidence") or {}).get("url"))
    assert thread_gap["evidence"]["route"], "the gap must carry an executable route"

def test_negligible_social_still_produces_a_gap_so_the_pillar_can_explain_itself():
    from benchmark.gap_detect import detect_gaps
    gaps = detect_gaps({}, metrics={"social": {
        "verdict": "negligible", "overall_social_share": 0.004,
        "total_citations": 5000, "why": "engines do not cite community sources"}})
    g = next(x for x in gaps if x["type"] == "SOCIAL_ABSENCE")
    assert g["severity"] == "low", "an honest empty pillar is not a high-severity gap"

def test_social_family_is_mapped_in_clustering_and_hypothesis():
    """Three gap types have already shipped unmapped and silently defaulted to E0."""
    from benchmark.gap_cluster import FAMILY, FAMILY_QUOTA
    from benchmark.hypothesis import FAMILY_TEMPLATES
    assert FAMILY["SOCIAL_ABSENCE"] == FAMILY["SOCIAL_NEGATIVE"] == "social"
    assert "social" in FAMILY_QUOTA
    assert FAMILY_TEMPLATES["social"][0] == "E3", \
        "earned presence cannot be injected; the proxy is a cited consensus"

def test_prohibited_community_plays_are_blocked():
    from optimise.strategy_llm import deterministic_validate
    si = {"wins": [], "nulls": [], "claims": [], "gaps": [],
          "substantiation_backlog": []}
    bad = {"action_id": "P5-1", "pillar": "Social/Community",
           "title": "Seed positive reviews on forums",
           "what": "post anonymously in relevant threads",
           "evidence_class": "DIAGNOSED", "validating_experiment": "E3",
           "kill_criterion": "stop if flat", "clock": "memory",
           "expected_effect": {"basis": "none", "value": None}}
    errs = deterministic_validate(si, {"Social/Community": {"actions": [bad]}},
                                 {"not_doing": [{"item": "x"}], "big_bets": []})
    assert any(e["error"] == "prohibited_community_play" for e in errs)
    assert any(e["severity"] == "blocker" for e in errs
               if e["error"] == "prohibited_community_play")

def test_community_action_must_use_the_memory_clock():
    from optimise.strategy_llm import deterministic_validate
    si = {"wins": [], "nulls": [], "claims": [], "gaps": [],
          "substantiation_backlog": []}
    a = {"action_id": "P5-2", "pillar": "Social/Community",
         "title": "Correct the factual error in the cited thread",
         "what": "respond under the brand account with the substantiated figure",
         "evidence_class": "DIAGNOSED", "validating_experiment": "E3",
         "kill_criterion": "stop if flat", "clock": "retrieval",
         "expected_effect": {"basis": "none", "value": None}}
    errs = deterministic_validate(si, {"Social/Community": {"actions": [a]}},
                                 {"not_doing": [{"item": "x"}], "big_bets": []})
    assert any(e["error"] == "community_action_on_retrieval_clock" for e in errs)

def test_a_legitimate_community_action_passes():
    from optimise.strategy_llm import deterministic_validate
    si = {"wins": [], "nulls": [], "claims": [], "gaps": [],
          "substantiation_backlog": []}
    a = {"action_id": "P5-3", "pillar": "Social/Community",
         "title": "Correct the ingredient error in the cited eczema thread",
         "what": "respond under the identified brand account citing the registered "
                 "concentration, and request a moderator correction",
         "evidence_class": "DIAGNOSED", "validating_experiment": "E3",
         "kill_criterion": "stop if the thread is locked", "clock": "memory",
         "leading_indicator": "engine-cited threads where the brand is correctly described",
         "expected_effect": {"basis": "none", "value": None,
                             "qualitative": "moderate"}}
    errs = deterministic_validate(si, {"Social/Community": {"actions": [a]}},
                                 {"not_doing": [{"item": "x"}], "big_bets": []})
    assert not [e for e in errs if e["target"] == "P5-3"
                and e["severity"] == "blocker"]