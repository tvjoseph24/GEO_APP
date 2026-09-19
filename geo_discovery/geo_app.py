"""GEO Framework — Unified Guided Pipeline (Option B).
Single app, tabbed modules, stage-gating, shared meter + pipeline status.

# Test instance
GEO_STORE=geo_projects_test streamlit run geo_app.py


"""
import streamlit as st
from config import MODEL_REGISTRY, DEFAULT_TASK_MODELS
from meter import Meter
from llm_gateway import LLMGateway
from store import list_brands
from app_context import AppContext, pipeline_status, prereq_met, STAGE_PREREQ

# tab modules (each exposes render(ctx))
import discover_tab, audit_tab, benchmark_tab, experiment_tab, optimise_tab,report_tab
from store import load_all_artifacts
from store import load_cost_log
from store import save_cost_log
import theme

st.set_page_config(page_title="GEO Framework", layout="wide",
                   initial_sidebar_state="expanded")
theme.inject()
theme.title("GEO Framework",
            "Generative Engine Optimisation — measurement, experiment, strategy")

ss = st.session_state
ss.setdefault("meter", Meter(budget_usd=100.0))   # ONE meter for the whole engagement
gateway = LLMGateway(ss.meter)

# ======================================================================
#  SHARED SIDEBAR — brand, models, meter, pipeline status
# ======================================================================
with st.sidebar:
    st.header("Engagement")

    # ---- PARENT: dropdown + add-new (symmetric with target) ----
    from store import list_parents, _slug

    parents = list_parents()
    if parents:
        parent_choice = st.selectbox("Parent / Company", parents + ["➕ New…"])
        if parent_choice == "➕ New…":
            parent_brand = st.text_input("New parent name", "")
        else:
            parent_brand = parent_choice
    else:
        parent_brand = st.text_input("Parent / Company", "L'Oréal")

    # ---- TARGET: dropdown + add-new (existing pattern) ----
    brands = list_brands(parent_brand) if parent_brand else []
    if brands:
        target_choice = st.selectbox("Target brand", brands + ["➕ New…"])
        if target_choice == "➕ New…":
            target_brand = st.text_input("New target brand", "")
        else:
            target_brand = target_choice
    else:
        target_brand = st.text_input("Target brand", "")

    category = st.text_input("Category", "")

    # ---- show resolved storage path (confirms which folder) ----
    if parent_brand and target_brand:
        st.caption(f"Store · {_slug(parent_brand)}/{_slug(target_brand)}")

    st.subheader("Models")
    vertex_models = [m for m,c in MODEL_REGISTRY.items() if c["provider"]=="vertex"]
    all_models = list(MODEL_REGISTRY.keys())
    models = {
        "extraction":  st.selectbox("Extraction", vertex_models, key="m_ext"),
        "query_mining":st.selectbox("Mining", vertex_models, key="m_qm"),
        "vision":      st.selectbox("Vision", [m for m,c in MODEL_REGISTRY.items()
                                               if "vision" in c["supports"]], key="m_vis"),
        "judge":       st.selectbox("Judge", vertex_models, key="m_judge"),
        "brief_model": st.selectbox("Brief/Strategy", vertex_models, key="m_brief"),
        "variant_model": st.selectbox("Variant-draft", vertex_models, key="m_var"),
    }
    # audit/experiment engine multiselect (shared)
    models["audit_engines"] = st.multiselect("Audit/Experiment engines", all_models,
                        default=[m for m in ["gemini-2.5-flash","azure/gpt-4o"] if m in all_models],
                        key="m_engines")

    st.divider()
    st.subheader("Cost")
    st.metric("Total spent", f"${ss.meter.total_cost():.4f}")
    st.metric("LLM calls", ss.meter.calls)

    _cap = st.number_input("Engagement cap ($)", 10.0, 5000.0,
                           float(getattr(ss.meter, "budget_usd", 100.0)),
                           step=25.0, key="eng_cap",
                           help="Hard ceiling enforced on EVERY LLM call, independent of "
                                "any per-phase cap. Raise it before a large run.")
    ss.meter.budget_usd = float(_cap)
    _left = _cap - ss.meter.total_cost()
    (st.error if _left < 5 else st.caption)(f"${_left:.2f} headroom")

    _unpriced = ss.meter.unpriced_models()
    if _unpriced:
        st.caption(f"No PRICING entry (counted as $0): {', '.join(_unpriced)}")
    _tev = getattr(gateway, "transient_events", [])
    if _tev:
        st.caption(f"{len(_tev)} transient API retry/retries this session")

    st.divider()
    st.subheader("Pipeline")
    status = pipeline_status(parent_brand, target_brand)
    ICON = theme.STATUS
    for stage, s in status.items():
        st.write(f"{ICON[s]} {stage} — *{s}*")

# build shared context
ctx = AppContext(parent_brand=parent_brand, target_brand=target_brand,
                 category=category, gateway=gateway, meter=ss.meter, models=models)


# ---- LEVEL-A RECOVERY (must run BEFORE gateway/ctx are bound) ----
brand_key = f"{parent_brand}::{target_brand}"
if ss.get("_loaded_brand") != brand_key and parent_brand and target_brand:
    ss["loaded_artifacts"] = load_all_artifacts(parent_brand, target_brand)
    ss["_loaded_brand"] = brand_key
    prior = load_cost_log(parent_brand, target_brand)
    if prior and ss.get("_cost_restored") != brand_key:
        ss["meter"] = Meter(budget_usd=ss["meter"].budget_usd)
        ss["meter"].restore(prior)
        ss["_cost_restored"] = brand_key
    for prefix in ["disc_", "aud_", "bm_", "exp_", "opt_"]:
        for k in list(ss.keys()):
            if k.startswith(prefix):
                ss.pop(k, None)
    if ss["loaded_artifacts"]:
        st.toast(f"📂 Loaded {len(ss['loaded_artifacts'])} saved artifact(s) "
                 f"for {target_brand}")

# bind gateway to the LIVE meter (previously ctx captured a pre-restore meter)
gateway = LLMGateway(ss["meter"])
ctx = AppContext(parent_brand=parent_brand, target_brand=target_brand,
                 category=category, gateway=gateway, meter=ss["meter"], models=models)

if st.button("Save cost log"):
    save_cost_log(parent_brand, target_brand, ss["meter"])
    st.success("Cost saved.")

# ======================================================================
#  TABS — with stage-gating
# ======================================================================
tabs = st.tabs(theme.TABS)


def gated(stage, render_fn):
    """Render a tab only if prerequisites met; else show a lock message."""
    if not prereq_met(stage, parent_brand, target_brand):
        prev = STAGE_PREREQ[stage]
        st.warning(f"**{stage}** is locked. Complete **{prev}** first "
                   f"(its output artifacts are missing for {target_brand}).")
        return
    render_fn(ctx)

with tabs[0]:
    discover_tab.render(ctx)                         # Discover — no prereq
with tabs[1]:
    gated("Audit", audit_tab.render)
with tabs[2]:
    gated("Benchmark", benchmark_tab.render)
with tabs[3]:
    gated("Experiment", experiment_tab.render)
with tabs[4]:
    gated("Optimise", optimise_tab.render)
with tabs[5]:
    gated("Report", report_tab.render)