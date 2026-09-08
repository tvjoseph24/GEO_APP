"""Audit tab. Refactored from audit_app.py → render(ctx)."""
import json
import streamlit as st
import pandas as pd
from pathlib import Path
from config import MODEL_REGISTRY
from store import load_artifact, project_dir
from audit_runner import run_audit, estimate_cost, visual_interrogation, BudgetExceeded
from baseline import freeze_baseline, compute_snapshot
from human_gate import HumanGate
from store import show_audit_samples


def render(ctx):
    st.header("🔬 Module 2: Audit")
    p, t = ctx.parent_brand, ctx.target_brand
    gateway = ctx.gateway
    ss = st.session_state
    loaded = ss.get("loaded_artifacts", {})
    # Audit's output is the frozen baseline + raw jsonl; show a note if baseline exists

    baselines_dir = project_dir(ctx.parent_brand, ctx.target_brand) / "baselines"
    existing_baselines = list(baselines_dir.glob("*_LOCKED.json")) if baselines_dir.exists() else []
    if existing_baselines:
        st.success(f"✅ {len(existing_baselines)} frozen baseline(s) exist. "
                   f"Latest: {existing_baselines[-1].name}")
        import json as _json
        with st.expander("📄 Latest frozen baseline snapshot"):
            st.json(_json.loads(existing_baselines[-1].read_text()).get("snapshot", {}))
        st.divider()

    qm = load_artifact(p, t, "query_master")
    cs = load_artifact(p, t, "competitive_set")
    if not qm:
        st.error("No Query Master. Complete Discovery first."); return
    competitors = [c["brand"] for c in (cs or [])]
    st.success(f"Loaded {len(qm)} queries, {len(competitors)} competitors.")

    intents = sorted({q.get("intent","other") for q in qm})
    sel_intents = st.multiselect("Intents", intents,
                    default=[i for i in ["branded","need","comparative"] if i in intents], key="aud_int")
    sel_engines = st.multiselect("Engines", list(MODEL_REGISTRY.keys()),
                    default=ctx.models.get("audit_engines", ["gemini-2.5-flash","azure/gpt-4o"]), key="aud_eng")
    n_runs = st.slider("Runs per query (N)", 3, 100, 10, key="aud_n")
    web_choice = st.radio("Mode", ["retrieval (web-on)","memory (web-off)","both"], key="aud_web")
    web_modes = {"retrieval (web-on)":("on",),"memory (web-off)":("off",),"both":("on","off")}[web_choice]
    use_llm_classify = st.checkbox("LLM classification (+cost)", key="aud_llm")
    extract_images = st.checkbox("Interrogate returned images (+cost)", key="aud_img")
    vision_model = ctx.models.get("vision","gemini-2.5-flash")
    judge_model = ctx.models.get("judge","gemini-2.5-flash")
    budget = st.number_input("Budget cap ($)", 1.0, 500.0, 20.0, key="aud_budget")
    hard_stop = st.checkbox("Hard-stop if breached", key="aud_hard")

    queries = [q for q in qm if q.get("intent") in sel_intents]
    engines = [{"name":m,"model":m,"provider":MODEL_REGISTRY[m]["provider"],
                "supports":MODEL_REGISTRY[m]["supports"]} for m in sel_engines]
    raw_path = str(project_dir(p, t) / "audit_raw.jsonl")

    calls, est = estimate_cost(len(queries), len(engines), n_runs, web_modes)
    st.info(f"~{calls:,} calls ≈ ${est}")

    if st.button("🚀 Run Audit", key="aud_run", disabled=not(queries and engines)):
        prog = st.progress(0.0); status = st.empty()
        def cb(done, total, warn=False):
            prog.progress(done/max(total,1))
            status.write(f"{done}/{total} · ${ctx.meter.total_cost():.3f}" + ("  ⚠️ budget breached" if warn else ""))
        try:
            run_audit(gateway, queries, engines, t, competitors, n_runs=n_runs, web_modes=web_modes,
                      out_path=raw_path, use_llm_classify=use_llm_classify, judge_model=judge_model,
                      extract_images=extract_images, vision_model=vision_model,
                      budget=budget, hard_stop=hard_stop, progress_cb=cb)
            st.success("Audit complete.")
        except BudgetExceeded as e:
            st.error(f"🛑 {e}")

    if Path(raw_path).exists():
        rows = [json.loads(l) for l in open(raw_path)]
        imgs = [i for r in rows for i in r.get("image_insights",[]) if i.get("fetched")]
        if imgs:
            st.subheader("🖼️ Returned-image insights")
            st.dataframe(pd.DataFrame(imgs).astype(str), use_container_width=True)

    st.subheader("Visual interrogation (optional)")
    vt = st.file_uploader("Ad(s)/comparison image(s)", type=["png","jpg","jpeg","webp"],
                          accept_multiple_files=True, key="aud_vt")
    vq = st.text_input("Question", "Which product is better and why?", key="aud_vq")
    if st.button("Interrogate visuals", key="aud_vt_btn") and vt:
        imgs = [(f.read(), "image/png" if f.name.lower().endswith("png") else "image/jpeg") for f in vt]
        st.write(visual_interrogation(gateway, imgs, vq, engines, vision_model))

    st.subheader("🚦 Freeze baseline")
    if Path(raw_path).exists():
        snap = compute_snapshot(raw_path, t)
        st.json(snap)
        if "aud_gate" not in ss:
            ss["aud_gate"] = HumanGate("baseline", lambda fb=None: snap); ss["aud_gate"].generate()
        if st.button("✅ Approve & Freeze", key="aud_freeze"):
            ss["aud_gate"].approve("analyst")
            cfg = {"intents":sel_intents,"engines":sel_engines,"n_runs":n_runs,
                   "web_modes":list(web_modes),"llm_classify":use_llm_classify}
            path,_ = freeze_baseline(p, t, raw_path, t, engines, cfg, ctx.meter)
            st.success(f"🔒 Frozen: {path}")
    else:
        st.info("Run the audit first.")
    if Path(raw_path).exists():
        show_audit_samples(raw_path, n=3)