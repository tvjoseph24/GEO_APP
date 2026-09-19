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
    if _cp["n_dropped"] or _cp["n_merged"]:
        st.caption(f"🧹 Claims register: {_cp['input']} → {_cp['kept']} "
                   f"({_cp['n_merged']} merged, {_cp['n_dropped']} dropped as "
                   f"unjudgeable).")
        with st.expander("Claim normalisation detail"):
            st.json(_cp)

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
                    consistency=ss["bm_consistency"],parent_brand_for_delta=p)
                ss["bm_qa"] = run_qa(df, c["claims"], ss["bm_veracity"], ss["bm_reason"],
                                     t, competitors,
                                     manifest=load_artifact(p, t, "audit_manifest"),gateway=gateway)

                prog.progress(0.96, text="Delta Scorecard…")
                ss["bm_delta"] = build_delta_scorecard(
                    ss["bm_veracity"], ss["bm_reason"], ss["bm_prov"], ss["bm_win"],
                    ss["bm_lb"], t, framing=ss["bm_framing"],
                    consistency=ss["bm_consistency"], flatness=ss["bm_flatness"],
                    reason_gap=ss["bm_reason_gap"], metrics=ss["bm_metrics"],
                    qa=ss["bm_qa"])
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

    if "bm_qa" in ss:
        v = qa_verdict(ss["bm_qa"])
        icon = {"ok": "✅", "caution": "🟡", "blocked": "🔴"}[v]
        with st.expander(f"{icon} Data-quality gate — {v.upper()}",
                         expanded=(v != "ok")):
            for ch in ss["bm_qa"]:
                fn = {"blocker": st.error, "warn": st.warning, "info": st.info}[ch["level"]]
                fn(f"**{ch['code']}** — {ch['message']}"
                   + (f"\n\n➡️ {ch['fix']}" if ch["fix"] else ""))

    if "bm_delta" in ss:
        st.subheader("⭐ Headline")
        for line in ss["bm_delta"].get("headline", []):
            st.markdown(f"- {line}")

        st.subheader("🔍 Veracity by engine")
        es = ss["bm_delta"].get("engine_veracity", {})
        if es:
            st.dataframe(pd.DataFrame(es).T, use_container_width=True)
        cal = ss["bm_delta"].get("calibration") or {}
        if cal.get("agreement") is not None:
            st.caption(f"Keyword↔LLM agreement: {cal['agreement']:.0%} "
                       f"({cal['disagreements']} disagreements of {cal['n_compared']})")

        st.subheader("🚨 Risk register (reverse hallucination scan)")
        rr = ss["bm_delta"].get("risk_register", [])
        if rr:
            st.dataframe(pd.DataFrame(rr)[["delta_type", "severity", "engine",
                                           "assertion", "evidence_quote", "why"]],
                         use_container_width=True)
        else:
            st.caption("No unsupported assertions detected "
                       "(or LLM veracity was disabled).")

        st.subheader("📚 Citation provenance — who owns the narrative")
        pv = ss["bm_prov"]
        m = st.columns(5)
        m[0].metric("Brand-owned", f"{pv.get('self_referential_pct', 0):.1%}")
        m[1].metric("Corroborated", f"{pv.get('corroborated_pct', 0):.1%}")
        m[2].metric("Grey market", f"{pv.get('grey_market_pct', 0):.1%}")
        m[3].metric("Competitor-owned", f"{pv.get('competitor_owned_pct', 0):.1%}")
        m[4].metric("Affiliate/SEO", f"{pv.get('affiliate_pct', 0):.1%}")
        st.caption(f"Voice-control ratio (brand-owned ÷ grey+affiliate): "
                   f"{pv.get('voice_control_ratio')} — below 1.0 means third parties "
                   f"outweigh the brand in machine citations.")
        if pv.get("per_engine"):
            st.dataframe(pd.DataFrame(pv["per_engine"]).T, use_container_width=True)

        st.subheader("💬 Framing")
        st.json(ss["bm_framing"])
        if ss.get("bm_flatness"):
            f = ss["bm_flatness"]
            st.caption(f"**Flatness:** {f.get('bare_share', 0):.0%} of "
                       f"{f.get('spans', 0)} brand spans carry no reason at all "
                       f"(mean {f.get('mean_reason_types')} reason types/mention).")

        st.subheader("🏋️ Reason richness — head to head")
        st.dataframe(ss["bm_reason"], use_container_width=True)
        if ss.get("bm_reason_gap") is not None and len(ss["bm_reason_gap"]):
            st.write("**Reason gaps vs the leader (direct E2/E4 input)**")
            st.dataframe(ss["bm_reason_gap"], use_container_width=True)

        st.subheader("📋 Claim-level delta")
        cd = pd.DataFrame(ss["bm_delta"]["claim_deltas"])
        if not cd.empty:
            sev = st.multiselect("Severity", ["critical", "high", "medium", "none"],
                                 default=["critical", "high", "medium"], key="bm_sev")
            st.dataframe(cd[cd.severity.isin(sev)], use_container_width=True)

    if "bm_metrics" in ss:
        st.header("⚠️ Weakness Signals (drive experiments)")
        flags = ss["bm_metrics"].get("weakness_flags", [])
        if flags:
            for f in flags:
                icon = "🔴" if f["severity"] == "high" else "🟡"
                st.warning(f"{icon} **{f['type']}**: {f['detail']}")
        else:
            st.info("No major weakness flags detected.")

        with st.expander("📊 Mention rate by intent (the key signal)"):
            st.dataframe(pd.DataFrame(ss["bm_metrics"]["mention_by_intent"]),
                         use_container_width=True)

    if "bm_delta" in ss:
        st.divider()
        st.subheader("Tier A — Descriptive")
        st.dataframe(ss["bm_mention"], use_container_width=True)
        st.write("**Leaderboard** (first-mention share)")
        st.dataframe(ss["bm_lb"], use_container_width=True)
        with st.expander("Source-type mix per engine"):
            st.json(ss["bm_srcmix"])

        st.subheader("Tier B — Claim-by-claim veracity")
        v = ss["bm_veracity"]
        if v.get("error"):
            st.error(v["error"])
        elif v.get("method") == "keyword_only":
            st.info(v.get("note", "Keyword mode — enable LLM veracity for verdicts."))
            st.dataframe(pd.DataFrame(v.get("keyword_rows", [])),
                         use_container_width=True)
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Brand spans", v.get("span_count", 0))
            m2.metric("Claims judged", v.get("claims_with_evidence", 0))
            m3.metric("Claims with no evidence", v.get("claims_absent", 0))
            m4.metric("Judge quote failures", v.get("quote_verification_failures", 0))

            pc = pd.DataFrame(v.get("per_claim", []))
            if not pc.empty:
                statuses = sorted(pc["status"].dropna().unique().tolist())
                pick = st.multiselect("Status", statuses, default=statuses,
                                      key="bm_vstatus")
                show = [c2 for c2 in ["claim", "claim_category", "engine", "status",
                                      "coverage", "ci_low", "ci_high",
                                      "n_relevant_spans", "supported", "partial",
                                      "distorted", "omitted", "hedged",
                                      "example_quote"] if c2 in pc.columns]
                st.dataframe(pc[pc.status.isin(pick)][show],
                             use_container_width=True, height=420)
            with st.expander("Raw judgements (span-level, with verbatim quotes)"):
                st.dataframe(pd.DataFrame(v.get("judgements", [])),
                             use_container_width=True)
            with st.expander("Reverse scan — every assertion extracted about the brand"):
                st.dataframe(pd.DataFrame(v.get("hallucinations", [])),
                             use_container_width=True)

        st.subheader("Answer consistency")
        cons = ss.get("bm_consistency", {})
        ov = cons.get("overall", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Locked in", len(ov.get("locked_in", [])),
                  help="≥90% of runs mention the brand")
        c2.metric("Emerging", len(ov.get("emerging", [])),
                  help="50–90% — present but not locked in. The most actionable band, "
                       "and previously reported in no bucket at all.")
        c3.metric("Fragile", len(ov.get("fragile", [])), help="under 50%")
        c4.metric("Absent", len(ov.get("absent", [])), help="never mentioned")
        if ov.get("_partition_ok") is False:
            st.warning(f"Bucket counts do not sum to {ov.get('n_queries')} queries — "
                       f"some queries are unaccounted for.")
        _wm = cons.get("by_web_mode") or {}
        _wm_rows = {k: v for k, v in _wm.items() if isinstance(v, dict)}
        if _wm_rows:
            st.write("**Memory vs retrieval**")
            st.dataframe(pd.DataFrame(_wm_rows).T, use_container_width=True)
        elif _wm.get("_note"):
            st.caption(f"Memory vs retrieval: {_wm['_note']}")
        if cons.get("divergence"):
            st.write("**Engine divergence** (>40pp spread)")
            st.dataframe(pd.DataFrame(cons["divergence"]),
                         use_container_width=True)
        with st.expander("Per-engine consistency detail"):
            st.json(cons.get("by_engine", {}))

        st.subheader("Tier C — Prescriptive")
        wcols = ["query", "winnability", "confidence", "rule_reason"] + \
                (["rationale"] if "rationale" in ss["bm_win"] else [])
        st.dataframe(ss["bm_win"][wcols], use_container_width=True)
        st.write("**Source hit-list**")
        st.dataframe(ss["bm_hitlist"], use_container_width=True)
        with st.expander("Cross-engine agreement map"):
            st.json(ss["bm_cross"])

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
                    st.success(f"Emitted. {len(actionable)} actionable gap(s) "
                               f"of {len(ss['bm_gaps'])} → Experiment.")
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
                    "mention_by_intent": ss["bm_metrics"].get("mention_by_intent"),
                    "first_by_intent": ss["bm_metrics"].get("first_mention_by_intent"),
                    "weakness_flags": [f["type"] for f in
                                       ss["bm_metrics"].get("weakness_flags", [])],
                    "qa": [{"level": c2["level"], "code": c2["code"]}
                           for c2 in ss.get("bm_qa", [])],
                    "headline": ss["bm_delta"].get("headline"),
                }
                st.code(_j.dumps(dump, indent=2, default=str), language="json")