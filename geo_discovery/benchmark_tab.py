"""Benchmark tab. Refactored from benchmark_app.py → render(ctx)."""
import json
import streamlit as st
import pandas as pd
from store import save_artifact
from benchmark.loader import load_audit, load_context
from benchmark import tier_a, tier_b, tier_c
from benchmark.delta_scorecard import (build_delta_scorecard,
                                       save_benchmark, classify_gaps)
from human_gate import HumanGate
from store import project_dir, show_audit_samples
from benchmark.loader import load_audit, load_context
from benchmark import tier_a, tier_b, tier_c, richness as RICH
from benchmark.qa import run_qa, qa_verdict
from benchmark.veracity import estimate_cost as ver_cost
from benchmark.delta_scorecard import (build_delta_scorecard,
                                       save_benchmark, classify_gaps)
from benchmark.diagnostics import (build_diagnostics, drift_check, load_diagnostics,
                                   diagnostics_summary, save_diagnostics,
                                   save_manifest, save_prepared_claims)
from benchmark.metrics_bundle import build_metrics_bundle, save_metrics_bundle
from benchmark.gap_detect import build_gap_register, detect_gaps, gap_summary
from store import load_artifact
from benchmark.gap_cluster import cluster_gaps, cluster_summary
from benchmark.hypothesis import (generate_hypotheses, validate, hypothesis_summary,
                                  mde, n_for_effect)
from store import save_cache, load_cache
import theme

def render(ctx):
    theme.section("Benchmark",
                  "The Delta — Ground Truth versus Machine Reality")
    p, t = ctx.parent_brand, ctx.target_brand
    gateway = ctx.gateway
    judge_model = ctx.models.get("judge", "gemini-2.5-flash")
    ss = st.session_state
    loaded = ss.get("loaded_artifacts", {})
    if "delta_scorecard" in loaded and "bm_delta" not in ss:
        st.success("✅ Existing Benchmark results loaded (read-only). Re-run to regenerate & edit.")
        with st.expander("⭐ Delta Scorecard", expanded=True):
            dc = loaded["delta_scorecard"]
            st.dataframe(pd.DataFrame(dc.get("claim_deltas", [])), use_container_width=True)
            if dc.get("leaderboard"):
                st.write("Leaderboard");
                st.dataframe(pd.DataFrame(dc["leaderboard"]), use_container_width=True)
        with st.expander("📄 Actionable Gap Register"):
            st.dataframe(pd.DataFrame(loaded.get("gap_register_actionable", [])), use_container_width=True)
        st.divider()

    # ---- has the corpus moved since the saved Benchmark ran? ----
    if "delta_scorecard" in loaded:
        from source_profile import load_source_pack
        _n_rows = 0
        try:
            import json as _j
            _rp = project_dir(p, t) / "audit_raw.jsonl"
            if _rp.exists():
                with open(_rp, encoding="utf-8") as _f:
                    _n_rows = sum(1 for _l in _f
                                  if _l.strip() and _j.loads(_l).get("success"))
        except Exception:
            _n_rows = None
        _dr = drift_check(p, t, current_rows=_n_rows,
                          current_pack=load_source_pack(p, t))
        if _dr.get("has_drift"):
            st.error("🚫 **The saved Benchmark describes a different corpus.**\n\n"
                     + "\n\n".join(f"• {x}" for x in _dr["drift"]))
        elif not _dr.get("has_manifest"):
            st.caption(f"📌 {_dr.get('note')}")
        else:
            st.caption(f"✅ Benchmark ran on {_dr['benchmarked_rows']:,} audit run(s) at "
                       f"{str(_dr.get('benchmarked_utc', ''))[:19]}, source pack "
                       f"v{_dr.get('source_pack_version')} — the corpus has not moved "
                       f"since.")
        _ds = diagnostics_summary(load_diagnostics(p, t))
        if _ds.get("persisted"):
            st.caption(f"📁 Diagnostics persisted: {_ds['span_judgements']:,} span "
                       f"judgement(s), {_ds['reverse_scan_assertions']:,} reverse-scan "
                       f"assertion(s), {_ds['winnability_rows']} winnability row(s). "
                       f"Judge `{_ds.get('judge_model')}`, yield "
                       f"{_ds.get('judge_yield')}.")
        else:
            st.caption(f"📌 {_ds.get('note')}")

    use_llm_veracity = st.checkbox("LLM veracity audit (+cost)", key="bm_ver_cb")
    use_llm_winnability = st.checkbox("LLM winnability rationale (+cost)", key="bm_win_cb")

    try:
        df = load_audit(p, t);
        c = load_context(p, t)
    except Exception as e:
        st.error(f"Load failed: {e}. Complete Audit first.");
        return

    # ---- competitors + aliases FIRST: claim prep and the lexicon both need them ----
    _cset = c["competitors"] or []
    competitors = [x["brand"] for x in _cset
                   if isinstance(x, dict) and x.get("brand")
                   and x.get("role") != "self"
                   and x["brand"].strip().lower() != t.strip().lower()]
    alias_ov = {x["brand"]: x.get("aliases", []) for x in _cset
                if isinstance(x, dict) and x.get("brand") and x.get("aliases")}
    st.caption(f"Tracking {len(competitors)} competitor(s): {', '.join(competitors)}")

    # ---- normalise the claims register (drop stubs, merge near-duplicates) ----
    from benchmark.claim_prep import prepare_claims
    _claims_raw = c["claims"]
    c["claims"], ss["bm_claim_prep"] = prepare_claims(
        _claims_raw, brand=t, category=ctx.category, competitors=competitors)
    _cp = ss["bm_claim_prep"]
    # Persisted immediately. This is the register veracity ACTUALLY runs against, and the
    # correct denominator for every coverage figure — the report computed coverage against
    # the raw register, so with splitting it could print over 100% assessed.
    save_prepared_claims(p, t, c["claims"], _cp, _claims_raw)
    if _cp["n_dropped"] or _cp["n_merged"] or _cp["n_split"]:
        st.caption(f"🧹 Claims register: {_cp['input']} → {_cp['kept']} "
                   f"({_cp['n_split']} split, {_cp['n_merged']} merged, "
                   f"{_cp['n_dropped']} dropped as unjudgeable). Coverage is reported "
                   f"against the {_cp['kept']} prepared claim(s), not the "
                   f"{_cp['input']} raw entries.")
        if _cp["n_dropped"] / max(_cp["input"], 1) >= 0.40:
            st.warning(f"⚠️ {_cp['n_dropped']} of {_cp['input']} register entries "
                       f"({_cp['n_dropped'] / max(_cp['input'], 1):.0%}) were excluded as "
                       f"unjudgeable. Check the reasons below — a whole claim FAMILY may "
                       f"be falling outside the classifier, in which case veracity is "
                       f"silently narrow.")
        with st.expander("Claim normalisation detail — what was split, merged and dropped"):
            st.caption(f"Dropped by kind: {_cp.get('dropped_by_kind')}")
            if _cp.get("dropped"):
                st.dataframe(pd.DataFrame(_cp["dropped"]),
                             use_container_width=True, hide_index=True)
            st.json({k: v for k, v in _cp.items() if k != "dropped"}, expanded=False)

    st.success(f"Loaded {len(df)} audit runs.")
    # show samples (in the recovery/load section)
    raw_path = str(project_dir(p, t) / "audit_raw.jsonl")
    ######### Audit samples ##########
    st.subheader("🔍 Sample audit responses")
    _b1, _b2 = st.tabs(["Random sample", "❌ Brand absent"])
    with _b1:
        show_audit_samples(raw_path, n=2, key="bm_samp",
                           stratify=("engine", "intent"), target_brand=t)
    with _b2:
        show_audit_samples(raw_path, n=2, key="bm_samp_abs",
                           stratify=("engine", "intent"), only_absent=True,
                           target_brand=t)

    st.subheader("Veracity depth (cost control)")
    _n_claims = len(c["claims"])
    c1, c2, c3 = st.columns(3)
    claim_pct = c1.slider("Claims to judge (% of register)", 5, 100, 40,
                          key="bm_claimpct")
    max_claims = max(5, int(round(_n_claims * claim_pct / 100)))
    c1.caption(f"{max_claims} of {_n_claims} claim(s)")
    spans_per_engine = c2.slider("Spans per claim / engine", 2, 8, 4, key="bm_spans")
    halluc_pct = c3.slider("Reverse-scan depth (% of spans)", 1, 25, 5,
                           key="bm_hpct")
    if claim_pct < 100:
        st.caption(f"⚠️ {100 - claim_pct}% of the register will NOT be assessed. "
                   f"Claims are prioritised efficacy → ingredient → credential → safety "
                   f"→ feature, then by proof length, so the sample is biased toward the "
                   f"BEST-substantiated claims. Veracity scores read high as a result. "
                   f"This is declared in the report's coverage section.")
    _cc1, _cc2 = st.columns(2)
    n_planned = _cc1.number_input("Planned experiment N per arm (for the power check)",
                                  20, 400, 100, step=10, key="bm_nplanned")
    _cc2.slider("Cluster budget (% of actionable gaps)", 10, 100, 40,
                key="bm_clusterpct",
                help="Clusters become hypotheses become experiments. Below 100% some "
                     "actionable gaps will receive no hypothesis and cannot be tested.")
    halluc_sample = max(12, int(round(len(df) * 2.0 * halluc_pct / 100)))
    n_calls, n_cost = ver_cost(max_claims, 200, spans_per_engine,
                               df["engine"].nunique(), halluc_sample)
    st.caption(f"≈{n_calls} judge calls ≈ ${n_cost} (cached across re-runs)")

    if st.button("🔬 Run Benchmark", key="bm_run"):
        try:
            if "bm_cache" not in ss:
                ss["bm_cache"] = load_cache(p, t, f"veracity_{judge_model}")
                if ss["bm_cache"]:
                    st.caption(f"♻️ Reusing {len(ss['bm_cache'])} cached judge result(s)")
            prog = st.progress(0.0, text="Starting…")
            with st.spinner("Analyzing…"):
                prog.progress(0.05, text="Tier A…")
                ss["bm_mention"] = tier_a.mention_rate(df)
                ss["bm_lb"] = tier_a.leaderboard(df, t, competitors)
                ss["bm_srcmix"] = tier_a.source_mix(df)

                prog.progress(0.12, text="Lexicon…")
                ss["bm_lex"] = RICH.build_lexicon(
                    c["claims"], ctx.category, gateway, judge_model,
                    brand=t, competitors=competitors)

                prog.progress(0.18, text="Reason richness (all brands)…")
                agg, detail, _ = tier_b.reason_richness(
                    df, [t] + competitors, lexicon=ss["bm_lex"],
                    alias_overrides=alias_ov)
                ss["bm_reason"], ss["bm_reason_detail"] = agg, detail
                ss["bm_reason_gap"] = RICH.reason_gap(detail, t)
                ss["bm_flatness"] = RICH.flatness_report(detail, t)

                prog.progress(0.28, text="Veracity engine…")
                ss["bm_veracity"] = tier_b.run_veracity(
                    df, c["claims"], t, use_llm=use_llm_veracity, gateway=gateway,
                    judge_model=judge_model, competitors=competitors,
                    alias_overrides=alias_ov, max_claims=max_claims,
                    spans_per_engine=spans_per_engine, halluc_sample=halluc_sample,
                    cache=ss["bm_cache"],
                    progress_cb=lambda x: prog.progress(0.28 + 0.42 * x,
                                                        text="Veracity engine…"))

                prog.progress(0.74, text="Provenance / framing / consistency…")
                ss["bm_prov"] = tier_b.claim_provenance(df)
                ss["bm_framing"] = tier_b.sentiment_framing(df, t)
                ss["bm_consistency"] = tier_b.answer_consistency(df, t)

                prog.progress(0.80, text="Community / social sources…")
                from benchmark.social import social_report
                ss["bm_social"] = social_report(df, t, competitors, alias_ov)

                prog.progress(0.84, text="Tier C…")
                ss["bm_win"] = tier_c.classify_winnability(df, t, competitors, c["claims"])
                if use_llm_winnability:
                    ss["bm_win"] = tier_c.add_llm_rationale(gateway, ss["bm_win"], judge_model)
                ss["bm_hitlist"] = tier_c.source_hitlist(df, t, competitors=competitors,
                                                         alias_overrides=alias_ov)
                ss["bm_cross"] = tier_c.cross_engine_map(df, t)

                prog.progress(0.92, text="Metrics bundle + QA…")
                ss["bm_metrics"] = build_metrics_bundle(
                    df, t, competitors, ss["bm_veracity"], ss["bm_reason"],
                    ss["bm_srcmix"], provenance=ss["bm_prov"], framing=ss["bm_framing"],
                    flatness=ss["bm_flatness"], reason_gap=ss["bm_reason_gap"],
                    consistency=ss["bm_consistency"], parent_brand_for_delta=p,
                    social=ss["bm_social"], alias_overrides=alias_ov)
                ss["bm_qa"] = run_qa(df, c["claims"], ss["bm_veracity"], ss["bm_reason"],
                                     t, competitors,
                                     manifest=load_artifact(p, t, "audit_manifest"),
                                     gateway=gateway,
                                     ingested_sources=load_artifact(p, t,
                                                                    "ingested_sources"),
                                     claim_prep=ss.get("bm_claim_prep"),
                                     hitlist=ss.get("bm_hitlist"))

                prog.progress(0.96, text="Delta Scorecard…")
                ss["bm_delta"] = build_delta_scorecard(
                    ss["bm_veracity"], ss["bm_reason"], ss["bm_prov"], ss["bm_win"],
                    ss["bm_lb"], t, framing=ss["bm_framing"],
                    consistency=ss["bm_consistency"], flatness=ss["bm_flatness"],
                    reason_gap=ss["bm_reason_gap"], metrics=ss["bm_metrics"],
                    qa=ss["bm_qa"], social=ss["bm_social"])
                ss["bm_gaps"] = build_gap_register(
                    ss["bm_delta"], winnability=ss["bm_win"], reason_rich=ss["bm_reason"],
                    target_brand=t, metrics=ss["bm_metrics"], hitlist=ss["bm_hitlist"])
                ss["bm_gaps"] = classify_gaps(gateway, ss["bm_gaps"], c["query_master"],
                                              t, model=judge_model, category=ctx.category)
                prog.progress(0.97, text="Clustering gaps…")
                _n_actionable = sum(1 for g in ss["bm_gaps"]
                                    if g.get("classification") == "actionable")
                _max_clusters = max(6, int(round(_n_actionable
                                                 * ss.get("bm_clusterpct", 40) / 100)))
                ss["bm_clusters"] = cluster_gaps(
                    ss["bm_gaps"], target_brand=t, competitors=competitors,
                    category=ctx.category, max_clusters=_max_clusters)
                st.caption(f"Cluster budget: {_max_clusters} from {_n_actionable} "
                           f"actionable gap(s)")

                _cov = (ss["bm_clusters"][0] or {}).get("_coverage", {}) if ss["bm_clusters"] else {}
                if _cov.get("orphaned"):
                    st.warning(f"⚠️ {_cov['orphaned']} of {_cov['actionable_gaps']} actionable "
                               f"gaps are not in any cluster ({_cov.get('clusters_truncated', 0)} "
                               f"clusters truncated). Raise max_clusters or triage these.")

                prog.progress(0.98, text="Generating hypotheses…")
                ss["bm_hyps"] = generate_hypotheses(
                    gateway, ss["bm_clusters"], model=judge_model, brand=t,
                    category=ctx.category, claims=c["claims"], metrics=ss["bm_metrics"],
                    delta=ss["bm_delta"], hitlist=ss["bm_hitlist"],
                    winnability=ss["bm_win"], query_master=c["query_master"],
                    n_planned=int(n_planned),
                    progress_cb=lambda x: prog.progress(0.98 + 0.02 * x,
                                                        text="Generating hypotheses…"))
                ss["bm_hyps"] = validate(ss["bm_hyps"], claims=c["claims"], brand=t,
                                         n_planned=int(n_planned))
                for k in ("bm_delta_gate", "bm_gap_gate"):
                    ss.pop(k, None)
            prog.progress(1.0, text="Complete")
            st.success("Benchmark complete.")
        finally:
            save_cache(p, t, f"veracity_{judge_model}", ss.get("bm_cache", {}))
            ctx.meter.flush()  # judge calls are the largest single spend
            # Persist the analytical trail BEFORE anything can fail downstream. These are
            # the most expensive artifacts the platform produces and they existed in
            # session state only.
            try:
                from source_profile import load_source_pack
                ss["bm_diagnostics"] = save_diagnostics(p, t, build_diagnostics(
                    veracity=ss.get("bm_veracity"),
                    winnability=ss.get("bm_win"),
                    richness_detail=ss.get("bm_reason_detail"),
                    richness_agg=ss.get("bm_reason"),
                    reason_gap=ss.get("bm_reason_gap"),
                    lexicon=ss.get("bm_lex"),
                    cross_engine=ss.get("bm_cross"),
                    consistency=ss.get("bm_consistency"),
                    social=ss.get("bm_social"),
                    hitlist=ss.get("bm_hitlist"),
                    qa=ss.get("bm_qa"),
                    claim_prep=ss.get("bm_claim_prep"),
                    judge_model=judge_model))
                save_manifest(p, t, df_rows=len(df),
                              engines=df["engine"].dropna().unique().tolist(),
                              intents=df["intent"].dropna().unique().tolist(),
                              judge_model=judge_model,
                              prepared_claims=c["claims"],
                              source_pack=load_source_pack(p, t),
                              config={"veracity_llm": use_llm_veracity,
                                      "max_claims": max_claims,
                                      "spans_per_engine": spans_per_engine,
                                      "halluc_sample": halluc_sample,
                                      "n_planned": int(n_planned)})
                st.caption("📁 Diagnostics, prepared register and manifest persisted — "
                           "the evidence trail survives a refresh.")
            except Exception as _e:
                st.warning(f"Diagnostics were not persisted: "
                           f"{type(_e).__name__}: {str(_e)[:120]}. The analysis above is "
                           f"unaffected, but the span-level trail exists only in this "
                           f"session.")

    if "bm_qa" in ss:
        v = qa_verdict(ss["bm_qa"])
        icon = {"ok": "✅", "caution": "🟡", "blocked": "🔴"}[v]
        with st.expander(f"{icon} Data-quality gate — {v.upper()}",
                         expanded=(v != "ok")):
            for ch in ss["bm_qa"]:
                fn = {"blocker": st.error, "warn": st.warning, "info": st.info}[ch["level"]]
                fn(f"**{ch['code']}** — {ch['message']}"
                   + (f"\n\n➡️ {ch['fix']}" if ch["fix"] else ""))
    ##### New Ordering of results #######
    # ================= RESULTS, ORDERED BY TIER =================
    if "bm_delta" in ss:
        st.divider()
        st.subheader("⭐ Headline")
        st.caption("Deterministic — generated from the measured figures, not written by a "
                   "model.")
        for line in ss["bm_delta"].get("headline", []):
            st.markdown(f"- {line}")

        if "bm_metrics" in ss:
            _flags = ss["bm_metrics"].get("weakness_flags", [])
            if _flags:
                with st.expander(f"⚠️ {len(_flags)} weakness signal(s) — these drive the "
                                 f"experiment programme", expanded=True):
                    for f in _flags:
                        (st.error if f["severity"] == "high" else st.warning)(
                            f"**{f['type']}** · {f['detail']}")
            else:
                st.info("No major weakness flags detected.")

        _tA, _tB, _tC = st.tabs([
            "Tier A · Descriptive — what is happening",
            "Tier B · Diagnostic — why",
            "Tier C · Prescriptive — what to do"])

        # ---------------------------------------------------------------- TIER A
        with _tA:
            st.caption("Observable behaviour. Mention, position, framing, who is cited, "
                       "and how stable any of it is across repeated runs. No "
                       "interpretation — the question here is only *what happened*.")

            st.markdown("#### Visibility by query type")
            st.caption("Branded and comparative queries NAME the brand, so their mention "
                       "rate measures description accuracy rather than visibility. The "
                       "brand-free categories are the ones that measure whether it "
                       "surfaces unprompted.")
            if "bm_metrics" in ss:
                st.dataframe(pd.DataFrame(ss["bm_metrics"]["mention_by_intent"]),
                             use_container_width=True, hide_index=True)
                _c1, _c2 = st.columns(2)
                _c1.markdown("**First mention**")
                _c1.dataframe(pd.DataFrame(
                    ss["bm_metrics"].get("first_mention_by_intent") or []),
                    use_container_width=True, hide_index=True)
                _c2.markdown("**Recommended, and top-3**")
                _rec = ss["bm_metrics"].get("recommendation_by_intent") or []
                _t3 = ss["bm_metrics"].get("top3_by_intent") or []
                if _rec or _t3:
                    _r = {x["intent"]: x for x in _rec}
                    _t = {x["intent"]: x for x in _t3}
                    _c2.dataframe(pd.DataFrame([
                        {"intent": i,
                         "recommended": (_r.get(i) or {}).get("recommendation_rate"),
                         "top3": (_t.get(i) or {}).get("top3_rate")}
                        for i in sorted(set(_r) | set(_t))]),
                        use_container_width=True, hide_index=True)
                else:
                    _c2.caption("Not computed for this run.")

            st.markdown("#### Per engine")
            st.caption("Engines behave differently, and those differences drive the "
                       "engine overlays in the strategy rather than the overall plan.")
            st.dataframe(ss["bm_mention"], use_container_width=True)

            st.markdown("#### Competitive leaderboard")
            st.caption("First-mention share across all runs.")
            st.dataframe(ss["bm_lb"], use_container_width=True, hide_index=True)

            st.markdown("#### Framing of brand mentions")
            st.caption("Mention rate alone overstates health. A hedged mention is not a "
                       "recommendation, and an answer that defers to a professional while "
                       "naming brands is a different outcome from a refusal.")
            _fr = ss["bm_framing"]
            if _fr.get("mentions"):
                _fm = st.columns(5)
                _fm[0].metric("Mentions", f"{_fr['mentions']:,}")
                _fm[1].metric("Positive", f"{_fr.get('pos_share', 0):.0%}")
                _fm[2].metric("Neutral", f"{_fr.get('neu_share', 0):.0%}")
                _fm[3].metric("Hedged", f"{(_fr.get('hedged_share') or 0):.0%}")
                _fm[4].metric("Defers to a professional",
                              f"{(_fr.get('advisory_share') or 0):.0%}")
                if _fr.get("negative_detection_note"):
                    st.caption(f"📌 {_fr['negative_detection_note']}")
                if _fr.get("by_engine"):
                    st.dataframe(pd.DataFrame(_fr["by_engine"]).T,
                                 use_container_width=True)
            else:
                st.caption("The brand is never mentioned in this corpus.")

            st.markdown("#### Who the engines cite")
            _pv = ss["bm_prov"]
            _rel = (_pv.get("reliability") or {})
            if not _rel.get("reliable", True):
                st.error(f"🚫 {_rel.get('note')}")
            _pm = st.columns(6)
            _pm[0].metric("Brand-owned", f"{_pv.get('self_referential_pct', 0):.1%}")
            _pm[1].metric("Editorial", f"{_pv.get('corroborated_pct', 0):.1%}")
            _pm[2].metric("Practitioner", f"{_pv.get('practitioner_pct', 0):.1%}")
            _pm[3].metric("Competitor", f"{_pv.get('competitor_owned_pct', 0):.1%}")
            _pm[4].metric("Affiliate / grey",
                          f"{_pv.get('affiliate_pct', 0) + _pv.get('grey_market_pct', 0):.1%}")
            _pm[5].metric("Unclassified", f"{_pv.get('unclassified_pct', 0):.1%}")
            if _pv.get("voice_control_ratio") is not None:
                st.caption(f"Voice-control ratio **{_pv['voice_control_ratio']}** — "
                           f"brand-owned divided by uncontrolled. Below 1.0 means third "
                           f"parties outweigh the brand in machine citations.")
            else:
                st.caption("Voice-control ratio withheld — see the reliability note above. "
                           f"Reported provisionally as "
                           f"{_pv.get('voice_control_ratio_provisional')}.")
            with st.expander("Per engine, and the full source-type mix"):
                if _pv.get("per_engine"):
                    st.dataframe(pd.DataFrame(_pv["per_engine"]).T,
                                 use_container_width=True)
                st.json(ss["bm_srcmix"])

            st.markdown("#### Community and social voice")
            _soc = ss.get("bm_social") or {}
            if _soc and not _soc.get("error"):
                _fn = {"negligible": st.info, "query_triggered": st.warning,
                       "diffuse": st.info}.get(_soc["verdict"], st.info)
                _fn(f"**{_soc['verdict'].replace('_', ' ').title()}** — {_soc['why']}")
                _sm = st.columns(5)
                _sm[0].metric("Share of citations",
                              f"{_soc['overall_social_share']:.1%}")
                _sm[1].metric("Cited threads", _soc.get("n_threads", 0))
                _sm[2].metric("Brand absent", _soc.get("n_threads_brand_absent", 0))
                _sm[3].metric("Negative", _soc.get("n_threads_negative", 0))
                _sm[4].metric("Participatory", _soc.get("participatory_threads", 0))
                if _soc.get("by_intent"):
                    st.caption("Which query types trigger a community citation — the "
                               "query-triggered half of the framework's Pillar-5 "
                               "definition.")
                    st.dataframe(pd.DataFrame(_soc["by_intent"]),
                                 use_container_width=True, hide_index=True)
                for _i in ((_soc.get("framing") or {}).get("interpretation") or []):
                    st.warning(f"⚠️ {_i}")
                if _soc.get("threads"):
                    with st.expander("Cited threads — the unit of action"):
                        st.dataframe(pd.DataFrame(_soc["threads"])[
                                         ["domain", "title", "times_cited", "n_engines", "top_intent",
                                          "brand_present_rate", "brand_absence_index",
                                          "top_competitor", "negative_rate", "participatory",
                                          "bucket", "route"]],
                                     use_container_width=True, hide_index=True)
                        for _t in _soc["threads"]:
                            st.caption(f"[{_t['times_cited']}×] {_t['url']}")
                _be = {k: v for k, v in (_soc.get("by_engine") or {}).items()
                       if not k.startswith("_")}
                if _be:
                    st.dataframe(pd.DataFrame(_be).T, use_container_width=True)
                if (_soc.get("by_engine") or {}).get("_divergence"):
                    st.warning(f"🔀 {_soc['by_engine']['_divergence']}")
                st.caption(f"📌 {_soc.get('clock_note')}")
                st.caption(f"⛔ {_soc.get('constraint')}")
            elif _soc.get("error"):
                st.caption(f"Community analysis failed: {_soc['error']}")

            st.markdown("#### Answer consistency")
            st.caption("The same question, asked repeatedly, does not always return the "
                       "same answer. This is which facts are locked in and which are "
                       "fragile.")
            _cons = ss.get("bm_consistency", {})
            _ov = _cons.get("overall", {})
            _cc = st.columns(4)
            _cc[0].metric("Locked in", len(_ov.get("locked_in", [])),
                          help="≥90% of runs mention the brand")
            _cc[1].metric("Emerging", len(_ov.get("emerging", [])),
                          help="50–90% — present but not locked in, and the most "
                               "actionable band")
            _cc[2].metric("Fragile", len(_ov.get("fragile", [])), help="under 50%")
            _cc[3].metric("Absent", len(_ov.get("absent", [])), help="never mentioned")
            if _ov.get("_partition_ok") is False:
                st.warning(f"Bucket counts do not sum to {_ov.get('n_queries')} queries.")
            _wm = {k: v for k, v in (_cons.get("by_web_mode") or {}).items()
                   if isinstance(v, dict)}
            if _wm:
                st.markdown("**Memory against retrieval**")
                st.caption("Absent with the web ON is a retrieval problem, fixable in "
                           "weeks. Absent with it OFF is a brand-fame problem, fixable in "
                           "quarters.")
                st.dataframe(pd.DataFrame(_wm).T, use_container_width=True)
            elif (_cons.get("by_web_mode") or {}).get("_note"):
                st.caption(f"📌 {_cons['by_web_mode']['_note']}")
            if _cons.get("divergence"):
                st.markdown("**Engine divergence** (>40pp spread)")
                st.dataframe(pd.DataFrame(_cons["divergence"]),
                             use_container_width=True, hide_index=True)
            with st.expander("Per-engine consistency detail"):
                st.json(_cons.get("by_engine", {}))

        # ---------------------------------------------------------------- TIER B
        with _tB:
            st.caption("Why the brand stands where it does. Claim-by-claim accuracy "
                       "against the substantiated register, argument density against "
                       "competitors, and what the engines assert that the brand has never "
                       "claimed.")

            st.markdown("#### Claim accuracy by engine")
            _es = ss["bm_delta"].get("engine_veracity", {})
            if _es:
                st.dataframe(pd.DataFrame(_es).T, use_container_width=True)
            _cal = ss["bm_delta"].get("calibration") or {}
            if _cal.get("agreement") is not None:
                st.caption(f"Keyword↔LLM agreement {_cal['agreement']:.0%} "
                           f"({_cal['disagreements']} disagreement(s) of "
                           f"{_cal['n_compared']}). Low agreement means one method is "
                           f"systematically wrong.")

            st.markdown("#### Claim-level delta")
            st.caption("Every verdict is quote-grounded: the judge must cite text that is "
                       "a literal substring of a real response, and any verdict whose "
                       "quote cannot be verified is discarded.")
            _cd = pd.DataFrame(ss["bm_delta"]["claim_deltas"])
            if not _cd.empty:
                _sev = st.multiselect("Severity",
                                      ["critical", "high", "medium", "none"],
                                      default=["critical", "high", "medium"],
                                      key="bm_sev")
                st.dataframe(_cd[_cd.severity.isin(_sev)], use_container_width=True)

            _v = ss["bm_veracity"]
            if _v.get("error"):
                st.error(_v["error"])
            elif _v.get("method") == "keyword_only":
                st.info(_v.get("note", "Keyword mode — enable the LLM veracity audit for "
                                       "verdicts."))
                st.dataframe(pd.DataFrame(_v.get("keyword_rows", [])),
                             use_container_width=True)
            else:
                _vm = st.columns(4)
                _vm[0].metric("Brand spans", _v.get("span_count", 0))
                _vm[1].metric("Claims judged", _v.get("claims_with_evidence", 0))
                _vm[2].metric("No evidence found", _v.get("claims_absent", 0))
                _vm[3].metric("Quote failures", _v.get("quote_verification_failures", 0))
                _pc = pd.DataFrame(_v.get("per_claim", []))
                if not _pc.empty:
                    _st = sorted(_pc["status"].dropna().unique().tolist())
                    _pick = st.multiselect("Status", _st, default=_st, key="bm_vstatus")
                    _show = [c2 for c2 in ["claim", "claim_category", "engine", "status",
                                           "coverage", "ci_low", "ci_high",
                                           "n_relevant_spans", "supported", "partial",
                                           "distorted", "omitted", "hedged",
                                           "example_quote"] if c2 in _pc.columns]
                    st.dataframe(_pc[_pc.status.isin(_pick)][_show],
                                 use_container_width=True, height=420)
                with st.expander("Span-level judgements, with verbatim quotes"):
                    st.dataframe(pd.DataFrame(_v.get("judgements", [])),
                                 use_container_width=True)

            st.markdown("#### What the engines assert that the brand has not claimed")
            st.caption("The reverse scan. Not *did the engines state our claims* but "
                       "*what are they stating about us?* Most of these are plausible and "
                       "specific, and entirely unsubstantiated.")
            _rr = ss["bm_delta"].get("risk_register", [])
            if _rr:
                st.dataframe(pd.DataFrame(_rr)[["delta_type", "severity", "engine",
                                                "assertion", "evidence_quote", "why"]],
                             use_container_width=True)
            else:
                st.caption("No unsupported assertions detected, or the LLM veracity audit "
                           "was disabled.")
            with st.expander("Every assertion extracted about the brand"):
                st.dataframe(pd.DataFrame(_v.get("hallucinations", [])),
                             use_container_width=True)

            st.markdown("#### Argument density, head to head")
            st.caption("Normalised per 100 tokens, so verbosity does not inflate it. "
                       "Mention rate says the brand appears; this says whether it is "
                       "argued for.")
            st.dataframe(ss["bm_reason"], use_container_width=True)
            if ss.get("bm_flatness"):
                _f = ss["bm_flatness"]
                st.caption(f"**Flatness** — {_f.get('shallow_share', 0):.0%} of "
                           f"{_f.get('spans', 0)} brand mentions carry at most ONE "
                           f"substantive reason; mean "
                           f"{_f.get('mean_substantive_types')} per mention."
                           + (f" Reason density is "
                              f"{_f['richness_ratio_vs_leader']}× {_f.get('leader')}'s."
                              if _f.get("richness_ratio_vs_leader") is not None else ""))
            if ss.get("bm_reason_gap") is not None and len(ss["bm_reason_gap"]):
                st.markdown("**Where competitors argue harder** — direct input to the "
                            "specificity experiments")
                st.dataframe(ss["bm_reason_gap"], use_container_width=True)

        # ---------------------------------------------------------------- TIER C
        with _tC:
            st.caption("What to do about it. Which queries are winnable, which sources to "
                       "target, and where the engines agree enough for a finding to be "
                       "structural rather than engine-specific.")

            st.markdown("#### Winnability triage")
            st.caption("Per query. `not_applicable` means the brand is named in the "
                       "prompt, so visibility cannot be assessed. `objection_defence` "
                       "means the objective is fair framing when named, not surfacing "
                       "more often.")
            _wcols = ["query", "intent", "winnability", "opportunity", "confidence",
                      "rule_reason"] + (["rationale"] if "rationale" in ss["bm_win"]
                                        else [])
            _wcols = [c for c in _wcols if c in ss["bm_win"].columns]
            st.dataframe(ss["bm_win"][_wcols], use_container_width=True,
                         hide_index=True)
            _ws = ss["bm_delta"].get("winnability_summary") or {}
            if _ws:
                st.caption("Mix: " + " · ".join(f"{k} {v}" for k, v in _ws.items()))

            st.markdown("#### Source hit-list")
            st.caption("Ranked by influence weighted by brand absence — the pages that "
                       "shape answers *while not mentioning the brand*. Each carries a "
                       "route, because a publisher is pitched and a practitioner is "
                       "engaged.")
            st.dataframe(ss["bm_hitlist"], use_container_width=True, hide_index=True)

            st.markdown("#### Cross-engine agreement")
            st.caption("Where engines agree, a finding is a structural truth and belongs "
                       "in the overall strategy. Where they diverge, it belongs in an "
                       "engine overlay.")
            _ce = ss["bm_cross"]
            if _ce.get("coverage"):
                _cm = st.columns(4)
                _cm[0].metric("Win everywhere", len(_ce.get("win_everywhere") or []))
                _cm[1].metric("Lose everywhere", len(_ce.get("lose_everywhere") or []))
                _cm[2].metric("Diverges", len(_ce.get("diverges_by_engine") or []))
                _cm[3].metric("Mixed middle", len(_ce.get("mixed_middle") or []))
            if _ce.get("divergence_detail"):
                st.dataframe(pd.DataFrame(_ce["divergence_detail"]),
                             use_container_width=True, hide_index=True)
            with st.expander("Full cross-engine map"):
                st.json(_ce)

    if "bm_hyps" in ss:
        st.divider()
        st.subheader("🧠 Testable hypotheses (clustered, evidence-backed)")
        cs = cluster_summary(ss["bm_clusters"])
        hs = hypothesis_summary(ss["bm_hyps"])
        k = st.columns(4)
        k[0].metric("Clusters", cs["n_clusters"], f"from {cs['gaps_covered']} gaps")
        k[1].metric("Ready", hs["ready"])
        k[2].metric("Needs review", hs["needs_review"])
        k[3].metric("Retrieval / Memory",
                    f"{hs['by_horizon'].get('retrieval', 0)}/"
                    f"{hs['by_horizon'].get('memory', 0)}")
        st.caption(f"Templates: {hs['by_template']} · Mechanisms: {hs['by_mechanism']} "
                   f"· Pillars: {hs['by_pillar']}")

        for h in ss["bm_hyps"]:
            icon = "✅" if h["status"] == "ready" else "⚠️"
            exp = h.get("experiment") or {}
            pred = h.get("prediction") or {}
            with st.expander(f"{icon} {h['cluster_id']} · {h.get('cluster_title')} "
                             f"· {exp.get('template', '?')} · {h.get('pillar', '?')}"):
                st.markdown(f"**Hypothesis** — {h.get('hypothesis')}")
                st.markdown(f"**Root cause** — {h.get('root_cause')}")
                st.caption(f"Mechanism `{h.get('mechanism')}` · horizon "
                           f"`{h.get('horizon')}` · effort {h.get('effort')} · "
                           f"confidence {h.get('confidence')} · covers "
                           f"{h.get('n_gaps')} gap(s): {', '.join(h.get('gap_ids', [])[:8])}")
                c1, c2 = st.columns(2)
                c1.markdown(f"**Arm A** — {exp.get('arm_A')}")
                c2.markdown(f"**Arm B** — {exp.get('arm_B')}")
                st.caption(f"Isolates **{exp.get('isolated_variable')}** · holds constant "
                           f"{exp.get('held_constant')} · query: “{exp.get('target_query')}”")
                st.caption(f"Prediction: {pred.get('baseline_pp')}% → "
                           f"{pred.get('expected_pp')}% on {pred.get('metric')} "
                           f"(detectable ≥{h.get('mde_pp')}pp at N={h.get('_n_planned')})")
                st.markdown(f"**Falsification** — {h.get('falsification')}")
                if h.get("claims_used"):
                    st.markdown("**Claims used**")
                    for cu in h["claims_used"]:
                        st.caption(f"• {cu}")
                if h.get("validation_errors"):
                    st.error("Blockers: " + " · ".join(h["validation_errors"]))
                if h.get("validation_warnings"):
                    st.warning("Warnings: " + " · ".join(h["validation_warnings"]))

    if "bm_gaps" in ss:
        st.subheader("Gap Register (classified)")
        from benchmark.gap_detect import gap_summary
        gs = gap_summary(ss["bm_gaps"])
        k1, k2, k3 = st.columns(3)
        k1.metric("Gaps detected", gs["total"])
        k2.metric("Critical", gs["by_severity"].get("critical", 0))
        k3.metric("High", gs["by_severity"].get("high", 0))
        st.caption(f"By pillar: {gs['by_pillar']}")
        with st.expander("By gap type"):
            st.json(gs["by_type"])

        gdf = pd.DataFrame(ss["bm_gaps"])
        cols = [c2 for c2 in ["gap_id", "type", "classification", "severity", "pillar",
                              "gap", "engines", "metric_baseline",
                              "classification_reason"] if c2 in gdf.columns]
        st.dataframe(gdf[cols], use_container_width=True)

        counts = gdf["classification"].value_counts().to_dict() \
            if "classification" in gdf else {}
        src = gdf["classification_source"].value_counts().to_dict() \
            if "classification_source" in gdf else {}
        st.caption(f"Classification: {counts} · source: {src}")

        errs = (ss["bm_gaps"][0] or {}).get("_classify_errors")
        if errs:
            st.error("Gap classification partially failed — some gaps fell back to "
                     "rules:\n" + "\n".join(f"• {e}" for e in errs))

        n_act = counts.get("actionable", 0)
        if n_act == 0:
            st.error("🔴 Zero actionable gaps. Nothing will flow to Experiment. "
                     "Promote gaps below, or edit the JSON in Gate 2.")
        elif n_act < 3:
            st.warning(f"Only {n_act} actionable gap(s) — thin experiment pipeline.")

        with st.expander("🔁 Bulk re-classify"):
            b1, b2 = st.columns(2)
            promote = b1.multiselect("Promote to actionable",
                                     sorted(gdf["type"].unique()) if "type" in gdf else [],
                                     key="bm_promote")
            if b1.button("Apply promotion", key="bm_promote_go") and promote:
                for g in ss["bm_gaps"]:
                    if g.get("type") in promote:
                        g["classification"] = "actionable"
                        g["classification_source"] = "analyst"
                st.rerun()
            if b2.button("Promote all critical + high", key="bm_promote_sev"):
                for g in ss["bm_gaps"]:
                    if g.get("severity") in ("critical", "high"):
                        g["classification"] = "actionable"
                        g["classification_source"] = "analyst"
                st.rerun()

        st.subheader("🚦 Gate 1: Delta")
        if "bm_delta_gate" not in ss:
            ss["bm_delta_gate"] = HumanGate("delta", lambda fb=None: ss["bm_delta"]); ss["bm_delta_gate"].generate()
        if st.button("✅ Approve Delta", key="bm_ok_delta"): ss["bm_delta_gate"].approve("analyst"); st.success("OK")
        st.caption(f"Delta status: **{ss['bm_delta_gate'].state.status}**")

        st.subheader("🚦 Gate 2: Gap Register")
        edited = st.text_area("Gaps (edit JSON — flip classification/delete noise)",
                              json.dumps(ss["bm_gaps"], indent=2, ensure_ascii=False), height=280, key="bm_gap_edit")
        if "bm_gap_gate" not in ss:
            ss["bm_gap_gate"] = HumanGate("gaps", lambda fb=None: ss["bm_gaps"]); ss["bm_gap_gate"].generate()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Approve Gaps", key="bm_ok_gap"): ss["bm_gap_gate"].approve("analyst"); st.success("OK")
        with c2:
            if st.button("✏️ Save gap edits", key="bm_save_gap"):
                try: ss["bm_gaps"] = json.loads(edited); ss["bm_gap_gate"].edit(ss["bm_gaps"]); st.success("Saved")
                except Exception as e: st.error(f"Invalid JSON: {e}")
        st.caption(f"Gaps status: **{ss['bm_gap_gate'].state.status}**")
        ####### Emit Section ###################
        if (ss.get("bm_delta_gate") and ss["bm_delta_gate"].is_final()
                and ss.get("bm_gap_gate") and ss["bm_gap_gate"].is_final()):
            if st.button("📦 Emit", key="bm_emit"):
                actionable = [g for g in ss["bm_gaps"]
                              if g.get("classification") == "actionable"]
                if not actionable:
                    st.error("Refusing to emit an empty actionable set — Experiment "
                             "would have no input. Promote at least one gap.")
                else:
                    save_benchmark(p, t, ss["bm_delta"], ss["bm_gaps"], ss["bm_hitlist"],
                                   {"veracity_llm": use_llm_veracity,
                                    "judge_model": judge_model})
                    '''
                    save_artifact(p, t, "gap_register_actionable", actionable,
                                  {"classified": True, "n_total": len(ss["bm_gaps"])})
                    '''
                    ready = [h for h in ss["bm_hyps"] if h["status"] == "ready"]
                    save_artifact(p, t, "gap_hypotheses", ss["bm_hyps"],
                                  {"clusters": len(ss["bm_clusters"]),
                                   "n_planned": int(n_planned),
                                   "judge_model": judge_model})
                    save_artifact(p, t, "gap_clusters", ss["bm_clusters"], {})
                    # back-compat: attach the reasoned hypothesis onto member gaps
                    hmap = {gid: h for h in ss["bm_hyps"] for gid in h.get("gap_ids", [])}
                    for g in actionable:
                        h = hmap.get(g["gap_id"])
                        if h:
                            g["hypothesis"] = h.get("hypothesis")
                            g["hypothesis_source"] = "llm_cluster"
                            g["cluster_id"] = h["cluster_id"]
                            g["experiment_id"] = (h.get("experiment") or {}).get("template")
                            g["target_query"] = (h.get("experiment") or {}).get("target_query")
                            g["claims_used"] = h.get("claims_used")
                    save_artifact(p, t, "gap_register_actionable", actionable,
                                  {"classified": True, "n_total": len(ss["bm_gaps"]),
                                   "hypotheses_ready": len(ready)})
                    save_metrics_bundle(p, t, ss["bm_metrics"])
                    # gap_register was write-only: saved by save_benchmark but absent from
                    # the load path and the report contract, so the full classified set
                    # could not be recovered or reported.
                    save_artifact(p, t, "gap_register", ss["bm_gaps"],
                                  {"classified": True,
                                   "n_actionable": len(actionable),
                                   "classify_report": (ss["bm_gaps"][0] or {})
                                  .get("_classify_report")})
                    st.success(f"Emitted. {len(actionable)} actionable gap(s) "
                               f"of {len(ss['bm_gaps'])} → Experiment. Diagnostics, "
                               f"prepared register and the full gap register are "
                               f"persisted.")
        else:
            st.info("Approve BOTH gates to emit.")

        # ---- DIAGNOSTIC DUMP (paste this when reporting) ----
        if "bm_delta" in ss:
            with st.expander("🧾 Diagnostic dump — copy for review"):
                import json as _j
                from collections import Counter as _C
                _v = ss.get("bm_veracity", {})
                _g = ss.get("bm_gaps", []) or []
                _rr = ss.get("bm_reason")
                _rg = ss.get("bm_reason_gap")

                dump = {
                    "classify_errors": (_g[0] or {}).get("_classify_errors") if _g else None,
                    "brand": t, "runs": len(df),
                    "engines": sorted(df["engine"].unique().tolist()),
                    "gap_summary": __import__("benchmark.gap_detect", fromlist=["x"]).gap_summary(_g) if _g else {},
                    "gap_classification_counts": dict(_C(x.get("classification")
                                                         for x in _g)),
                    "gap_classification_source": dict(_C(x.get("classification_source",
                                                               "unset") for x in _g)),
                    "gaps_slim": [{k: x.get(k) for k in
                                   ("gap_id", "type", "severity", "pillar",
                                    "classification", "classification_reason",
                                    "metric_baseline", "gap")} for x in _g[:20]],
                    # ---- NEW: Batch-3 clustering + hypotheses ----
                    "cluster_summary": cluster_summary(ss.get("bm_clusters", [])),
                    "hypothesis_summary": hypothesis_summary(ss.get("bm_hyps", [])),
                    "hypotheses": [{k: h.get(k) for k in
                                    ("cluster_id", "cluster_title", "status", "mechanism",
                                     "pillar", "horizon", "hypothesis", "prediction",
                                     "experiment", "claims_used", "falsification",
                                     "validation_errors", "validation_warnings", "mde_pp")}
                                   for h in ss.get("bm_hyps", [])],

                    "veracity": {
                        "method": _v.get("method"), "span_count": _v.get("span_count"),
                        "spans_by_engine": _v.get("spans_by_engine"),
                        "spans_by_intent": _v.get("spans_by_intent"),
                        "claims_considered": _v.get("claims_considered"),
                        "claims_with_evidence": _v.get("claims_with_evidence"),
                        "claims_absent": _v.get("claims_absent"),
                        "status_counts": dict(_C(r.get("status")
                                                 for r in _v.get("per_claim", []))),
                        "engine_scores": _v.get("engine_scores"),
                        "halluc_counts": _v.get("halluc_counts"),
                        "quote_match_breakdown": _v.get("quote_match_breakdown"),
                        "calibration": _v.get("calibration"),
                        "judge_yield": _v.get("judge_yield"),
                        "judge_failure_count": _v.get("judge_failure_count"),
                        "judge_failures": (_v.get("judge_failures") or [])[:6],
                        "claims_judged": _v.get("claims_judged"),
                        "rescued_retrievals": _v.get("rescued_retrievals"),

                    },
                    "reverse_scan_sample": [
                        {k: h.get(k) for k in ("classification", "risk", "engine",
                                               "assertion", "matched_claim", "why")}
                        for h in (_v.get("hallucinations") or [])[:8]],
                    "claim_delta_sample": [
                        {k: c2.get(k) for k in ("claim", "engine", "machine_status",
                                                "coverage", "n_spans", "delta_note")}
                        for c2 in ss["bm_delta"].get("claim_deltas", [])[:10]],
                    "reason_richness": (_rr.to_dict("records")
                                        if _rr is not None and len(_rr) else []),
                    "reason_gap": (_rg.to_dict("records")
                                   if _rg is not None and len(_rg) else []),
                    "flatness": ss.get("bm_flatness"),
                    "provenance": {k: v2 for k, v2 in (ss.get("bm_prov") or {}).items()
                                   if k != "per_engine"},
                    "framing": ss.get("bm_framing"),
                    "social": __import__("benchmark.social", fromlist=["x"])
                              .social_summary(ss.get("bm_social")),
                    "mention_by_intent": ss["bm_metrics"].get("mention_by_intent"),
                    "first_by_intent": ss["bm_metrics"].get("first_mention_by_intent"),
                    "weakness_flags": [f["type"] for f in
                                       ss["bm_metrics"].get("weakness_flags", [])],
                    "qa": [{"level": c2["level"], "code": c2["code"]}
                           for c2 in ss.get("bm_qa", [])],
                    "headline": ss["bm_delta"].get("headline"),
                }
                st.code(_j.dumps(dump, indent=2, default=str), language="json")