"""Benchmark tab. Refactored from benchmark_app.py → render(ctx)."""
import json
import streamlit as st
import pandas as pd
from store import save_artifact
from benchmark.loader import load_audit, load_context
from benchmark import tier_a, tier_b, tier_c
from benchmark.delta_scorecard import (build_delta_scorecard, build_gap_register,
                                       save_benchmark, classify_gaps)
from human_gate import HumanGate
from store import project_dir, show_audit_samples
from benchmark.metrics_bundle import build_metrics_bundle
from benchmark.metrics_bundle import save_metrics_bundle

def render(ctx):
    st.header("📊 Module 3: Benchmark (The Delta)")
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
        df = load_audit(p, t); c = load_context(p, t)
    except Exception as e:
        st.error(f"Load failed: {e}. Complete Audit first."); return
    competitors = [x["brand"] for x in c["competitors"]]
    st.success(f"Loaded {len(df)} audit runs.")
    # show samples (in the recovery/load section)
    raw_path = str(project_dir(p, t) / "audit_raw.jsonl")
    show_audit_samples(raw_path, n=3)


    if st.button("🔬 Run Benchmark", key="bm_run"):
        prog = st.progress(0.0, text="Starting…")
        steps = ["Tier A metrics", "Veracity", "Reason-richness", "Provenance",
                 "Winnability", "Hit-list", "Delta", "Gap register", "Classification"]

        def step(i, label):
            prog.progress(i / len(steps), text=f"{label}…")
        with st.spinner("Analyzing…"):
            step(1,"Tier A");ss["bm_mention"] = tier_a.mention_rate(df)
            ss["bm_lb"] = tier_a.leaderboard(df, t, competitors)
            ss["bm_srcmix"] = tier_a.source_mix(df)
            step(2,"Veracity");ss["bm_veracity"] = tier_b.run_veracity(df, c["claims"], t,
                                    use_llm=use_llm_veracity, gateway=gateway, judge_model=judge_model)
            ss["bm_reason"] = tier_b.reason_richness(df, [t]+competitors)
            step(3,"Claim Provenance");ss["bm_prov"] = tier_b.claim_provenance(df)
            step(4,"Consistency");ss["bm_consistency"] = tier_b.answer_consistency(df, t)
            step(5,"Winnability");ss["bm_win"] = tier_c.classify_winnability(df, t, competitors, c["claims"])
            if use_llm_winnability:
                ss["bm_win"] = tier_c.add_llm_rationale(gateway, ss["bm_win"], judge_model)
            ss["bm_hitlist"] = tier_c.source_hitlist(df, t)
            step(6,"Cross engine map");ss["bm_cross"] = tier_c.cross_engine_map(df, t)
            step(7,"Delta Scorecard");ss["bm_delta"] = build_delta_scorecard(ss["bm_veracity"], ss["bm_reason"],
                                ss["bm_prov"], ss["bm_win"], ss["bm_lb"], t)
            step(8,"Gap Register");ss["bm_gaps"] = build_gap_register(ss["bm_delta"], ss["bm_win"], ss["bm_reason"], t)
            step(9,"Classification");ss["bm_gaps"] = classify_gaps(gateway, ss["bm_gaps"], c["query_master"], t, model=judge_model)
            # NEW: build + stash the metrics bundle (Priority 1)
            ss["bm_metrics"] = build_metrics_bundle(
                df, t, competitors, ss["bm_veracity"], ss["bm_reason"], ss["bm_srcmix"])
            for k in ("bm_delta_gate","bm_gap_gate"): ss.pop(k, None)
        st.success("Benchmark complete.")
        prog.progress(1.0, text="Complete")

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
        st.subheader("Tier A — Descriptive")
        st.dataframe(ss["bm_mention"], use_container_width=True)
        st.write("Leaderboard"); st.dataframe(ss["bm_lb"], use_container_width=True)
        st.json(ss["bm_srcmix"])
        st.subheader("Tier B — Diagnostic")
        st.dataframe(pd.DataFrame(ss["bm_veracity"]), use_container_width=True)
        st.dataframe(ss["bm_reason"], use_container_width=True); st.json(ss["bm_prov"])
        # NEW: answer consistency (locked-in vs fragile claims)
        if "bm_consistency" in ss:
            st.write("**Answer consistency** (locked-in vs fragile across runs)")
            st.json(ss["bm_consistency"])
        st.subheader("Tier C — Prescriptive")
        st.dataframe(ss["bm_win"][["query","winnability","confidence","rule_reason"]
                     + (["rationale"] if "rationale" in ss["bm_win"] else [])], use_container_width=True)
        st.dataframe(ss["bm_hitlist"], use_container_width=True); st.json(ss["bm_cross"])
        st.subheader("⭐ Delta Scorecard")
        st.dataframe(pd.DataFrame(ss["bm_delta"]["claim_deltas"]), use_container_width=True)

    if "bm_gaps" in ss:
        st.subheader("Gap Register (classified)")
        gdf = pd.DataFrame(ss["bm_gaps"])
        cols = [c2 for c2 in ["gap_id","classification","classification_reason","gap",
                              "severity","pillar","priority"] if c2 in gdf.columns]
        st.dataframe(gdf[cols], use_container_width=True)
        counts = gdf["classification"].value_counts().to_dict() if "classification" in gdf else {}
        st.caption(f"Classification: {counts} → only 'actionable' flow to Experiment.")

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

        if (ss.get("bm_delta_gate") and ss["bm_delta_gate"].is_final()
                and ss.get("bm_gap_gate") and ss["bm_gap_gate"].is_final()):
            if st.button("📦 Emit", key="bm_emit"):
                actionable = [g for g in ss["bm_gaps"] if g.get("classification")=="actionable"]
                save_benchmark(p, t, ss["bm_delta"], ss["bm_gaps"], ss["bm_hitlist"],
                               {"veracity_llm": use_llm_veracity})
                save_artifact(p, t, "gap_register_actionable", actionable, {"classified": True})
                save_metrics_bundle(p, t, ss["bm_metrics"])
                st.success(f"Emitted. {len(actionable)} actionable gaps → Experiment.")
        else:
            st.info("Approve BOTH gates to emit.")