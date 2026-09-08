"""Module 2 — Audit front-end. Loads Discovery artifacts → config → run → baseline.
Portfolio-aware: parent_brand / target_brand."""
import json
from pathlib import Path
import streamlit as st
import pandas as pd

from config import MODEL_REGISTRY, STORE_ROOT
from meter import Meter
from llm_gateway import LLMGateway
from audit_runner import run_audit, estimate_cost, visual_interrogation, BudgetExceeded
from baseline import freeze_baseline, compute_snapshot
from human_gate import HumanGate
from store import load_artifact, list_brands, _slug, project_dir   # ← import _slug + project_dir

st.set_page_config(page_title="GEO · Audit", layout="wide")
st.title("🔬 GEO Framework — Module 2: Audit")

ss = st.session_state
ss.setdefault("meter", Meter(budget_usd=50.0))
gateway = LLMGateway(ss.meter)

# ======================================================================
#  SIDEBAR — portfolio-aware brand selection + audit config
# ======================================================================
with st.sidebar:
    st.header("⚙️ Audit Configuration")

    parent_brand = st.text_input("Parent / Company", "Nestlé")
    existing = list_brands(parent_brand)
    if existing:
        target_brand = st.selectbox("Target brand", existing,
                                    help="Product brands with completed Discovery")
    else:
        target_brand = st.text_input("Target brand", "Nescafé")
        st.warning(f"No completed Discovery found under '{parent_brand}'. "
                   f"Run Discovery for this brand first.")

    # portfolio-aware load (parent, target, name)
    qm  = load_artifact(parent_brand, target_brand, "query_master")
    cs  = load_artifact(parent_brand, target_brand, "competitive_set")
    kpi = load_artifact(parent_brand, target_brand, "kpi_targets")

    if not qm:
        st.error(f"No Query Master for {parent_brand} → {target_brand}. Run Discovery first.")
        st.stop()

    competitors = [c["brand"] for c in (cs or [])]
    st.success(f"Loaded {len(qm)} queries, {len(competitors)} competitors for {target_brand}.")

    # D1: intent selection
    intents = sorted({q.get("intent", "other") for q in qm})
    sel_intents = st.multiselect("Query intents", intents,
                                 default=[i for i in ["branded", "need", "comparative"]
                                          if i in intents])
    # D2: engine selection
    sel_engines = st.multiselect("Engines", list(MODEL_REGISTRY.keys()),
                                 default=[m for m in ["gemini-2.5-flash", "azure/gpt-4o"]
                                          if m in MODEL_REGISTRY])
    # D3: N
    n_runs = st.slider("Runs per query (N)", 3, 100, 10)
    # D4: web mode
    web_choice = st.radio("Mode", ["retrieval (web-on)", "memory (web-off)", "both"])
    web_modes = {"retrieval (web-on)": ("on",), "memory (web-off)": ("off",),
                 "both": ("on", "off")}[web_choice]
    # classifier + image toggles
    use_llm_classify = st.checkbox("LLM classification (richer, +cost)", value=False)
    extract_images   = st.checkbox("Interrogate returned images (+cost)", value=False)
    vision_model = st.selectbox("Vision model",
                    [m for m, c in MODEL_REGISTRY.items() if "vision" in c["supports"]])
    judge_model  = st.selectbox("Classifier model",
                    [m for m, c in MODEL_REGISTRY.items() if c["provider"] == "vertex"])
    # D6: cost governance
    budget = st.number_input("Budget cap ($)", 1.0, 500.0, 20.0)
    hard_stop = st.checkbox("Hard-stop if budget breached", value=False)

    st.divider()
    st.metric("Spent", f"${ss.meter.total_cost():.4f}")
    st.metric("Calls", ss.meter.calls)

# ======================================================================
#  BUILD query + engine lists  ·  portfolio raw path
# ======================================================================
queries = [q for q in qm if q.get("intent") in sel_intents]
engines = [{"name": m, "model": m, "provider": MODEL_REGISTRY[m]["provider"],
            "supports": MODEL_REGISTRY[m]["supports"]} for m in sel_engines]

# portfolio-structured raw path
raw_path = str(project_dir(parent_brand, target_brand) / "audit_raw.jsonl")

# ======================================================================
#  1 · PRE-RUN ESTIMATE
# ======================================================================
st.header("1 · Pre-run estimate")
calls, est = estimate_cost(len(queries), len(engines), n_runs, web_modes)
st.info(f"**{calls:,} calls** ≈ **${est}** "
        f"({len(queries)} queries × {len(engines)} engines × {n_runs} runs × {len(web_modes)} mode(s))"
        + ("  +LLM classify" if use_llm_classify else "")
        + ("  +image interrogation" if extract_images else ""))
if not queries:
    st.warning("No queries match the selected intents.")
if not engines:
    st.warning("Select at least one engine.")

# ======================================================================
#  2 · RUN AUDIT
# ======================================================================
st.header("2 · Run audit")
st.caption(f"Output → `{raw_path}`  (resumable — completed runs are skipped)")

if st.button("🚀 Run Audit", disabled=not (queries and engines)):
    prog = st.progress(0.0); status = st.empty()
    def cb(done, total, warn=False):
        prog.progress(done / max(total, 1))
        status.write(f"{done}/{total} runs · ${ss.meter.total_cost():.3f}"
                     + ("  ⚠️ budget breached (continuing)" if warn else ""))
    try:
        run_audit(gateway, queries, engines, target_brand, competitors,
                  n_runs=n_runs, web_modes=web_modes, out_path=raw_path,
                  use_llm_classify=use_llm_classify, judge_model=judge_model,
                  extract_images=extract_images, vision_model=vision_model,
                  budget=budget, hard_stop=hard_stop, progress_cb=cb)
        st.success("Audit complete.")
    except BudgetExceeded as e:
        st.error(f"🛑 {e}")

# ---- returned-image insights ----
if Path(raw_path).exists():
    rows = [json.loads(l) for l in open(raw_path, encoding="utf-8")]
    img_rows = [ins for r in rows for ins in r.get("image_insights", []) if ins.get("fetched")]
    if img_rows:
        st.subheader("🖼️ Returned-image insights")
        st.dataframe(pd.DataFrame(img_rows).astype(str), use_container_width=True)

# ======================================================================
#  3 · VISUAL INTERROGATION (optional)
# ======================================================================
st.header("3 · Visual interrogation (optional)")
vt_imgs = st.file_uploader("Upload ad(s) / brand-vs-competitor image(s)",
                           type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
vt_query = st.text_input("Question to ask about the image(s)",
                         "Which product is better and why?")
if st.button("Interrogate visuals") and vt_imgs:
    imgs = [(f.read(), "image/png" if f.name.lower().endswith("png") else "image/jpeg")
            for f in vt_imgs]
    res = visual_interrogation(gateway, imgs, vt_query, engines, vision_model)
    st.write(res)

# ======================================================================
#  4 · FREEZE BASELINE (human sign-off)
# ======================================================================
st.header("4 · Freeze baseline (human sign-off)")
if Path(raw_path).exists():
    snap = compute_snapshot(raw_path, target_brand)          # ← target_brand (was 'brand')
    st.subheader("Tier-A snapshot (review before freezing):")
    st.json(snap)

    def regen_baseline(feedback=None):
        return snap    # baseline is data; if unhappy → re-run audit
    if "baseline_gate" not in ss:
        ss.baseline_gate = HumanGate("baseline", regen_baseline)
        ss.baseline_gate.generate()

    if st.button("✅ Approve & Freeze Baseline"):
        ss.baseline_gate.approve("analyst")
        cfg = {"intents": sel_intents, "engines": sel_engines, "n_runs": n_runs,
               "web_modes": list(web_modes), "llm_classify": use_llm_classify,
               "extract_images": extract_images}
        path, _ = freeze_baseline(parent_brand, target_brand, raw_path,
                                  target_brand, engines, cfg, ss.meter)   # ← new signature
        st.success(f"🔒 Baseline frozen (immutable): {path}")
else:
    st.info("Run the audit first to generate a baseline.")