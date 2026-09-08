"""Module 5 — Optimise. Validated findings → 6-pillar strategy + content briefs.
PROVISIONAL until Phase-2 competitor enrichment."""
import json
import streamlit as st
import pandas as pd
from config import MODEL_REGISTRY
from meter import Meter
from llm_gateway import LLMGateway
from human_gate import HumanGate
from store import list_brands, load_artifact
from optimise.pillar_router import route_all, pillar_summary, PILLARS
from optimise.prioritizer import score_actions
from optimise.brief_gen import generate_brief
from optimise.strategy import build_strategy, build_roadmap, save_strategy

st.set_page_config(page_title="GEO · Optimise", layout="wide")
st.title("🎯 GEO Framework — Module 5: Optimise (Strategy)")
st.warning("⚠️ PROVISIONAL: strategy based on current findings. Absolute lift numbers "
           "pending Phase-2 competitor-content enrichment.")

ss = st.session_state
ss.setdefault("meter", Meter(budget_usd=20.0))
gateway = LLMGateway(ss.meter)




with st.sidebar:
    st.header("⚙️ Optimise Config")
    parent_brand = st.text_input("Parent", "Nestlé")
    brands = list_brands(parent_brand)
    target_brand = st.selectbox("Target brand", brands) if brands else st.text_input("Target", "Nescafé")
    brief_model = st.selectbox("Brief/strategy model",
        [m for m,c in MODEL_REGISTRY.items() if c["provider"]=="vertex"])
    st.divider(); st.metric("Spent", f"${ss.meter.total_cost():.4f}")

# ---- load inputs ----
findings = load_artifact(parent_brand, target_brand, "validated_findings_only") or []
variants = load_artifact(parent_brand, target_brand, "experiment_variants") or []
gaps = load_artifact(parent_brand, target_brand, "gap_register_actionable") or []
hitlist = load_artifact(parent_brand, target_brand, "source_hitlist") or []

if not findings:
    st.error("No validated findings. Run Experiment first & emit."); st.stop()

st.success(f"Loaded {len(findings)} validated finding(s) for {target_brand}.")
gap_lookup = {g["gap_id"]: g for g in gaps}
variant_lookup = {v["gap_id"]: v for v in variants}
hitlist_domains = [h.get("domain") for h in hitlist if h.get("domain")]

# ---- Run optimise ----
if st.button("🎯 Build Strategy + Briefs"):
    with st.spinner("Routing pillars, prioritizing, drafting briefs…"):
        # Step 2: route to pillars
        findings = route_all(findings, gap_lookup)
        # Step 3: prioritize
        findings = score_actions(findings)
        # Step 4: content briefs (from winning variant B)
        briefs = []
        for f in findings:
            v = variant_lookup.get(f["gap_id"], {})
            winning = v.get("variant_B", "") or (v.get("arms", {}).get("E", "") if v.get("arms") else "")
            briefs.append(generate_brief(gateway, f, winning, hitlist_domains,
                                         target_brand, model=brief_model))
        # Step 5: assemble
        ss.findings = findings
        ss.briefs = briefs
        ss.strategy = build_strategy(findings, briefs, target_brand, parent_brand)
        ss.roadmap = build_roadmap(findings)
        ss.pop("strat_gate", None)   # reset gate on re-run
    st.success("Strategy + briefs built.")

# ---- Display ----
if "strategy" in ss:
    st.header("Pillar Summary")
    st.json(ss.strategy["pillar_summary"])

    st.header("📋 Prioritized Roadmap")
    for horizon in ["Now", "Next", "Later"]:
        items = ss.roadmap.get(horizon, [])
        if items:
            st.subheader(f"{horizon} ({len(items)})")
            st.dataframe(pd.DataFrame(items), use_container_width=True)

    st.header("Engine Overlays (engine-specific plays)")
    st.json(ss.strategy["engine_overlays"])

    st.header("📝 Content Briefs")
    for b in ss.briefs:
        with st.expander(f"[{b.get('horizon')}] {b.get('title','(brief)')} — "
                         f"pillar: {b.get('pillar')} | gap {b.get('gap_id')}"):
            with st.expander(f"[{b.get('horizon')}] {b.get('title', '(brief)')} — ..."):
                st.write("**Objective:**", b.get("objective"))
                st.write("**Why it works:**", b.get("why_it_works"))
                st.write("**⭐ PROVEN content to publish (this exact treatment won):**")
                st.text(b.get("proven_content", "")[:2000])  # ← the actual winning variant
                st.write("**Placement:**", b.get("placement"))
                st.write("**Corroboration needed:**", b.get("corroboration"))
                st.write("**Engines targeted:**", b.get("engines_targeted"))  # real engine now
                st.write("**Success metric:**", b.get("success_metric"))

    # ---- Human gate ----
    st.header("🚦 Gate: Strategy Sign-off")
    edited = st.text_area("Strategy (edit inline — adjust actions, horizons, pillars)",
                          json.dumps(ss.strategy, indent=2, ensure_ascii=False),
                          height=300, key="strat_edit")
    if "strat_gate" not in ss:
        ss.strat_gate = HumanGate("strategy", lambda fb=None: ss.strategy)
        ss.strat_gate.generate()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Approve Strategy"):
            ss.strat_gate.approve("analyst"); st.success("Strategy approved.")
    with c2:
        if st.button("✏️ Save edits"):
            try:
                ss.strategy = json.loads(edited); ss.strat_gate.edit(ss.strategy)
                st.success("Edits saved.")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

    # ---- Emit ----
    if ss.get("strat_gate") and ss.strat_gate.is_final():
        if st.button("📦 Emit Strategy + Roadmap + Briefs"):
            save_strategy(parent_brand, target_brand, ss.strategy, ss.roadmap,
                          ss.briefs, {"brief_model": brief_model, "provisional": True})
            st.success("✅ Emitted geo_strategy + roadmap + content_briefs "
                       "(marked provisional).")
    else:
        st.info("Approve the strategy to enable emit.")