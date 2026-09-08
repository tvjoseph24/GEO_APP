"""Module 3 — Benchmark. Audit → Delta Scorecard + classified Gap Register.
Two human gates (Delta + Gap Register). Only 'actionable' gaps flow to Experiment."""
import json
import streamlit as st
import pandas as pd
from config import MODEL_REGISTRY
from meter import Meter
from llm_gateway import LLMGateway
from human_gate import HumanGate
from store import list_brands, save_artifact
from benchmark.loader import load_audit, load_context
from benchmark import tier_a, tier_b, tier_c
from benchmark.delta_scorecard import (build_delta_scorecard, build_gap_register,
                                       save_benchmark, classify_gaps)

st.set_page_config(page_title="GEO · Benchmark", layout="wide")
st.title("📊 GEO Framework — Module 3: Benchmark (The Delta)")

ss = st.session_state
ss.setdefault("meter", Meter(budget_usd=30.0))
gateway = LLMGateway(ss.meter)

# ======================================================================
#  SIDEBAR
# ======================================================================
with st.sidebar:
    st.header("⚙️ Benchmark Config")
    parent_brand = st.text_input("Parent", "Nestlé")
    brands = list_brands(parent_brand)
    target_brand = st.selectbox("Target brand", brands) if brands else st.text_input("Target", "Nescafé")
    use_llm_veracity = st.checkbox("LLM veracity audit (+cost)", value=False)
    use_llm_winnability = st.checkbox("LLM winnability rationale (+cost)", value=False)
    judge_model = st.selectbox("Judge/rationale model",
        [m for m, c in MODEL_REGISTRY.items() if c["provider"] == "vertex"])
    st.divider(); st.metric("Spent", f"${ss.meter.total_cost():.4f}")

# ======================================================================
#  LOAD
# ======================================================================
try:
    df = load_audit(parent_brand, target_brand)
    ctx = load_context(parent_brand, target_brand)
except Exception as e:
    st.error(f"Load failed: {e}. Run Audit first."); st.stop()

competitors = [c["brand"] for c in ctx["competitors"]]
st.success(f"Loaded {len(df)} audit runs for {target_brand}.")

# ======================================================================
#  RUN BENCHMARK
# ======================================================================
if st.button("🔬 Run Benchmark"):
    with st.spinner("Analyzing…"):
        # Tier A
        ss.mention = tier_a.mention_rate(df)
        ss.lb = tier_a.leaderboard(df, target_brand, competitors)
        ss.srcmix = tier_a.source_mix(df)
        # Tier B
        ss.veracity = tier_b.run_veracity(df, ctx["claims"], target_brand,
                        use_llm=use_llm_veracity, gateway=gateway, judge_model=judge_model)
        ss.reason = tier_b.reason_richness(df, [target_brand] + competitors)
        ss.provenance = tier_b.claim_provenance(df)
        ss.consistency = tier_b.answer_consistency(df, target_brand)
        # Tier C
        ss.winnability = tier_c.classify_winnability(df, target_brand, competitors, ctx["claims"])
        if use_llm_winnability:
            ss.winnability = tier_c.add_llm_rationale(gateway, ss.winnability, judge_model)
        ss.hitlist = tier_c.source_hitlist(df, target_brand)
        ss.crosseng = tier_c.cross_engine_map(df, target_brand)
        # Assemble
        ss.delta = build_delta_scorecard(ss.veracity, ss.reason, ss.provenance,
                                         ss.winnability, ss.lb, target_brand)
        ss.gaps = build_gap_register(ss.delta, ss.winnability, ss.reason, target_brand)
        # Classify gaps (actionable / noise / monitor)
        ss.gaps = classify_gaps(gateway, ss.gaps, ctx["query_master"], target_brand,
                                model=judge_model)
        # reset gates so a fresh run re-generates them
        for k in ("delta_gate", "gap_gate"):
            ss.pop(k, None)
    st.success("Benchmark complete.")

# ======================================================================
#  DISPLAY — analysis tiers
# ======================================================================
if "delta" in ss:
    st.header("Tier A — Descriptive")
    st.dataframe(ss.mention, use_container_width=True)
    st.subheader("Leaderboard"); st.dataframe(ss.lb, use_container_width=True)
    st.subheader("Source mix"); st.json(ss.srcmix)

    st.header("Tier B — Diagnostic")
    st.subheader("Veracity"); st.dataframe(pd.DataFrame(ss.veracity), use_container_width=True)
    st.subheader("Reason-richness"); st.dataframe(ss.reason, use_container_width=True)
    st.subheader("Provenance"); st.json(ss.provenance)

    st.header("Tier C — Prescriptive")
    st.subheader("Winnability triage")
    st.dataframe(ss.winnability[["query", "winnability", "confidence", "rule_reason"]
                 + (["rationale"] if "rationale" in ss.winnability else [])],
                 use_container_width=True)
    st.subheader("Source hit-list"); st.dataframe(ss.hitlist, use_container_width=True)
    st.subheader("Cross-engine map"); st.json(ss.crosseng)

    st.header("⭐ Delta Scorecard")
    st.dataframe(pd.DataFrame(ss.delta["claim_deltas"]), use_container_width=True)

# ======================================================================
#  GAP REGISTER (classified)
# ======================================================================
if "gaps" in ss:
    st.header("Gap Register (classified)")
    gdf = pd.DataFrame(ss.gaps)
    show_cols = [c for c in ["gap_id", "classification", "classification_reason",
                             "gap", "severity", "pillar", "priority"] if c in gdf.columns]
    st.dataframe(gdf[show_cols], use_container_width=True)
    counts = gdf["classification"].value_counts().to_dict() if "classification" in gdf else {}
    st.caption(f"Classification: {counts}  →  only 'actionable' flow to Experiment.")

# ======================================================================
#  TWO HUMAN GATES — Delta + Gap Register
# ======================================================================
if "delta" in ss and "gaps" in ss:
    st.header("Validate (human sign-off)")

    # ---- Gate 1: Delta Scorecard ----
    st.subheader("🚦 Gate 1: Delta Scorecard")
    if "delta_gate" not in ss:
        ss.delta_gate = HumanGate("delta", lambda fb=None: ss.delta)
        ss.delta_gate.generate()
    st.caption("Preview of claim deltas (full table above):")
    st.json(ss.delta["claim_deltas"][:5])
    if st.button("✅ Approve Delta"):
        ss.delta_gate.approve("analyst"); st.success("Delta approved.")
    st.caption(f"Delta status: **{ss.delta_gate.state.status}**")

    # ---- Gate 2: Gap Register (editable — override classifications) ----
    st.subheader("🚦 Gate 2: Gap Register")
    edited = st.text_area(
        "Gaps (edit inline — flip classification, delete noise, tune hypotheses)",
        json.dumps(ss.gaps, indent=2, ensure_ascii=False), height=320, key="gap_edit")
    if "gap_gate" not in ss:
        ss.gap_gate = HumanGate("gaps", lambda fb=None: ss.gaps)
        ss.gap_gate.generate()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Approve Gaps"):
            ss.gap_gate.approve("analyst"); st.success("Gaps approved.")
    with c2:
        if st.button("✏️ Save gap edits"):
            try:
                ss.gaps = json.loads(edited)
                ss.gap_gate.edit(ss.gaps); st.success("Gap edits saved & accepted.")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
    st.caption(f"Gaps status: **{ss.gap_gate.state.status}**")

    # ---- EMIT — only when BOTH gates final ----
    st.header("📦 Emit")
    both_final = (ss.get("delta_gate") and ss.delta_gate.is_final()
                  and ss.get("gap_gate") and ss.gap_gate.is_final())
    if both_final:
        if st.button("Emit Delta + Gap Register"):
            actionable = [g for g in ss.gaps if g.get("classification") == "actionable"]
            # full benchmark artifacts
            save_benchmark(parent_brand, target_brand, ss.delta, ss.gaps, ss.hitlist,
                           {"veracity_llm": use_llm_veracity})
            # actionable-only subset → feeds Experiment (Module 4)
            save_artifact(parent_brand, target_brand, "gap_register_actionable",
                          actionable, {"classified": True})
            st.success(f"✅ Emitted. {len(actionable)} actionable gap(s) → Experiment "
                       f"(of {len(ss.gaps)} total).")
    else:
        st.info("Approve BOTH the Delta and the Gap Register to enable emit.")