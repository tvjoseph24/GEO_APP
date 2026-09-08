"""Module 4 — Experiment. Actionable gaps → controlled RAG-proxy tests → validated findings."""
import json
import streamlit as st
import pandas as pd
from config import MODEL_REGISTRY, PRICING
from meter import Meter
from llm_gateway import LLMGateway
from human_gate import HumanGate
from store import list_brands, load_artifact
from experiment.gap_mapper import propose_experiments, EXPERIMENTS
from experiment.variant_gen import (generate_text_variants, attach_visual_variant,
                                    build_competitor_context)
from experiment.rag_runner import run_experiment
from experiment.results import assemble_findings, save_experiment_results

def _pick_query(gateway, design, qm, brand, model="gemini-2.5-flash"):
    """LLM picks the audited query most relevant to testing this gap."""
    queries = [q["query"] for q in qm]
    prompt = (f"Gap being tested: {design['gap']}\n"
              f"Hypothesis: {design['hypothesis']}\n\n"
              f"From this list, return the SINGLE most relevant query to test whether "
              f"closing this gap improves {brand}'s recommendation. Return ONLY the query text.\n"
              f"{json.dumps(queries)}")
    try:
        r = gateway.generate(model, prompt, module="experiment", task="query_pick",
                             temperature=0.1)
        pick = r["text"].strip().strip('"')
        # validate it's actually in the list
        for q in queries:
            if q.lower() in pick.lower() or pick.lower() in q.lower():
                return q
    except Exception:
        pass
    return queries[0] if queries else f"best product for {brand}'s category"

st.set_page_config(page_title="GEO · Experiment", layout="wide")
st.title("🧪 GEO Framework — Module 4: Experiment (Prove Causation)")

ss = st.session_state
ss.setdefault("meter", Meter(budget_usd=50.0))
gateway = LLMGateway(ss.meter)

# ---- sidebar ----
with st.sidebar:
    st.header("⚙️ Experiment Config")
    parent_brand = st.text_input("Parent", "Nestlé")
    brands = list_brands(parent_brand)
    target_brand = st.selectbox("Target brand", brands) if brands else st.text_input("Target", "Nescafé")
    sel_engines = st.multiselect("Engines", list(MODEL_REGISTRY.keys()),
                    default=[m for m in ["gemini-2.5-flash","azure/gpt-4o"] if m in MODEL_REGISTRY])
    n_runs = st.slider("Runs per variant (N)", 10, 100, 30)
    temperature = st.slider("Temperature (needs >0 for distribution)", 0.5, 1.0, 0.8)
    variant_model = st.selectbox("Variant-draft model",
                    [m for m,c in MODEL_REGISTRY.items() if c["provider"]=="vertex"])
    target_pos = st.radio("Target doc position", ["top","bottom","shuffle"])
    budget = st.number_input("Budget cap ($)", 1.0, 500.0, 30.0)
    hard_stop = st.checkbox("Hard-stop if breached", value=False)
    st.divider(); st.metric("Spent", f"${ss.meter.total_cost():.4f}")

engines = [{"name": m, "model": m, "provider": MODEL_REGISTRY[m]["provider"],
            "supports": MODEL_REGISTRY[m]["supports"]} for m in sel_engines]

# ---- load actionable gaps ----
gaps = load_artifact(parent_brand, target_brand, "gap_register_actionable")
claims = load_artifact(parent_brand, target_brand, "claims_register") or []
comps = load_artifact(parent_brand, target_brand, "competitive_set") or []
qm = load_artifact(parent_brand, target_brand, "query_master") or []
if not gaps:
    st.error("No actionable gaps found. Run Benchmark first & emit."); st.stop()
st.success(f"Loaded {len(gaps)} actionable gaps for {target_brand}.")

# ---- Step 1+2: propose designs + generate variants ----
st.header("1 · Experiment Designs (auto-mapped from gaps)")
if st.button("🎯 Propose experiments + draft variants"):
    with st.spinner("Mapping gaps + drafting variants…"):
        designs = propose_experiments(gaps)
        for d in designs:
            d["question"] = _pick_query(gateway, d, qm, target_brand)   # ← gateway added
            v = generate_text_variants(gateway, d, claims, target_brand, model=variant_model)
            d["variants"] = v
        ss.designs = designs
        ss.comp_context = build_competitor_context([c["brand"] for c in comps])
    st.success(f"Proposed {len(designs)} experiments with drafted variants.")

# ---- Step 3: DESIGN APPROVAL GATE (before spend) ----
if "designs" in ss:
    st.header("2 · 🚦 Gate 1: Approve Designs (before spend)")
    for i, d in enumerate(ss.designs):
        with st.expander(f"{d['experiment_id']} — {d['experiment_name']} | gap {d['gap_id']}"):
            d["include"] = st.checkbox("Include this experiment", value=d["include"], key=f"inc{i}")
            st.caption(f"Hypothesis: {d['hypothesis']}")
            st.write("**Variant A:**", d["variants"]["variant_A"].get("text","")[:300])
            st.write("**Variant B:**", d["variants"]["variant_B"].get("text","")[:300])
            st.write(f"Question: _{d.get('question')}_")
            d["mode"] = st.radio(f"Mode", ["A/B (fast)", "Progressive (E1 — element attribution)"],
                                 key=f"mode{i}",
                                 help="A/B = one variable. Progressive = 5 cumulative arms "
                                      "(capsule→table→citations→full), ~2.5× cost, shows WHICH element wins.")
            # opt-in visual variant
            d["add_visual_variant"] = st.checkbox("➕ Add visual variant (upload)", key=f"vis{i}")
            if d["add_visual_variant"]:
                up = st.file_uploader("Upload image/video for B variant",
                        type=["png","jpg","jpeg","webp","mp4","mov"], key=f"upl{i}")
                if up:
                    mime = "image/png" if up.name.lower().endswith("png") else \
                           "video/mp4" if up.name.lower().endswith("mp4") else "image/jpeg"
                    if mime.startswith("video"):
                        st.warning("⚠️ Video = higher cost. Confirm in run step.")
                    attach_visual_variant(d, up.read(), mime)
                    st.success("Visual attached to B variant.")

    active = [d for d in ss.designs if d["include"]]
    # pre-run cost estimate
    total_calls = len(active) * 2 * n_runs * len(engines)
    est = round(total_calls * 0.002, 2)
    st.info(f"**{total_calls:,} calls** ≈ **${est}** "
            f"({len(active)} experiments × 2 variants × {n_runs} runs × {len(engines)} engines)")

    if "design_gate" not in ss:
        ss.design_gate = HumanGate("designs", lambda fb=None: active)
        ss.design_gate.generate()
    if st.button("✅ Approve designs & Run experiments"):
        ss.design_gate.approve("analyst")
        # ---- Step 4: RUN ----
        prog = st.progress(0.0); status = st.empty()
        total_runs = sum((5 if d.get("mode", "").startswith("Progressive") else 2)
                         for d in active) * n_runs * len(engines)
        done = [0]
        def cb():
            done[0] += 1; prog.progress(min(done[0]/max(total_runs,1),1.0))
            status.write(f"{done[0]}/{total_runs} runs · ${ss.meter.total_cost():.3f}")
        results = []
        comp_names = [c["brand"] if isinstance(c, dict) else c for c in comps]
        try:
            for d in active:
                # skip experiments with empty variants (Fix #2)
                qw = d.get("variants", {}).get("quality_warning", "")
                if qw:
                    st.warning(qw)
                if "EMPTY" in qw:
                    st.warning(f"⏭️ Skipping {d['gap_id']} — {qw}")
                    continue
                if budget and ss.meter.total_cost() >= budget and hard_stop:
                    st.error("🛑 Budget reached — stopped."); break
                if d.get("mode", "").startswith("Progressive"):
                    from experiment.variant_gen import generate_progressive_variants
                    from experiment.rag_runner import run_progressive_experiment

                    pv = generate_progressive_variants(gateway, d, claims, target_brand,
                                                       model=variant_model)
                    d["progressive_arms"] = pv["arms"]
                    if pv.get("quality_warning"):
                        st.warning(f"{d['gap_id']} variant quality: {pv['quality_warning']}")
                    er = run_progressive_experiment(
                        gateway, d, pv["arms"], ss.comp_context, d["question"],
                        target_brand, comp_names, n_runs=n_runs,  # ← comp_names
                        temperature=temperature, engines=engines,
                        target_pos=target_pos, progress_cb=cb)
                    er["mode"] = "progressive"
                else:
                    er = run_experiment(
                        gateway, d, d["variants"], ss.comp_context, d["question"],
                        target_brand, comp_names,  # ← comp_names (was inline)
                        n_runs=n_runs, temperature=temperature,
                        engines=engines, target_pos=target_pos, progress_cb=cb)
                    er["mode"] = "ab"
                results.append(er)
            ss.exp_results = results
            ss.findings = assemble_findings([e for e in results if e.get("mode")=="ab"])
            st.success("Experiments complete.")
        except Exception as e:
            st.error(f"Run error: {e}")

# ---- Step 6+7: RESULTS + review gate ----
if "findings" in ss:
    st.header("3 · Results")
    for f in ss.findings:
        icon = {"validated":"✅","engine-specific":"🟡","rejected":"❌"}.get(f["verdict"],"?")
        st.subheader(f"{icon} {f['experiment_id']} (gap {f['gap_id']}) — "
                     f"{f['verdict'].upper()} · pooled lift {f['pooled_lift_pp']}pp")
        st.dataframe(pd.DataFrame(f["per_engine"]), use_container_width=True)
        if f["variance_warnings"]:
            st.warning("Variance: " + "; ".join(f["variance_warnings"]))

    st.header("4 · 🚦 Gate 2: Validate Results")
    if "results_gate" not in ss:
        ss.results_gate = HumanGate("results", lambda fb=None: ss.findings)
        ss.results_gate.generate()
    if st.button("✅ Approve & Emit findings"):
        ss.results_gate.approve("analyst")
        save_experiment_results(parent_brand, target_brand, ss.exp_results,
                                ss.findings, ss.designs,  # ← add ss.designs
                                {"variant_model": variant_model})
        validated = [f for f in ss.findings if f["verdict"] != "rejected"]
        st.success(f"Emitted. {len(validated)} validated finding(s) → Optimise. "
                   f"Full variants saved to experiment_variants.json")

# ---- Step 6+7: RESULTS ----
if "exp_results" in ss:
    st.header("3 · Results")

    # ---- A/B results (existing display) ----
    ab_results = [e for e in ss.exp_results if e.get("mode") == "ab"]
    if ab_results and "findings" in ss:
        st.subheader("A/B Experiments")
        for f in ss.findings:
            icon = {"validated":"✅","engine-specific":"🟡","rejected":"❌"}.get(f["verdict"],"?")
            st.markdown(f"{icon} **{f['experiment_id']}** (gap {f['gap_id']}) — "
                        f"**{f['verdict'].upper()}** · pooled lift {f['pooled_lift_pp']}pp")
            st.dataframe(pd.DataFrame(f["per_engine"]), use_container_width=True)
            if f.get("variance_warnings"):
                st.warning("Variance: " + "; ".join(f["variance_warnings"]))

    # ---- PROGRESSIVE (E1) results — element attribution ----
    prog_results = [e for e in ss.exp_results if e.get("mode") == "progressive"]
    if prog_results:
        from experiment.exp_stats import analyze_progressive, element_attribution
        st.subheader("🔬 Progressive (E1) — Element Attribution")
        ARM_LABELS = {"A":"plain", "B":"+capsule", "C":"+table",
                      "D":"+citations", "E":"+full structure"}

        for er in prog_results:
            st.markdown(f"### Gap {er['gap_id']} — which structural element drives lift?")
            rows = analyze_progressive(er)
            attribution = element_attribution(rows)

            # 1. cumulative lift curve (arm vs A) — averaged across engines
            adf = pd.DataFrame(rows)
            cum = (adf.groupby("arm")["vs_A_pp"].mean().reindex(["A","B","C","D","E"]))
            st.caption("**Cumulative lift vs baseline (A)** — averaged across engines")
            st.bar_chart(cum)

            # 2. MARGINAL contribution (guard all-zero / ceiling)
            marg = pd.Series(attribution).reindex(["B", "C", "D", "E"]).dropna()
            marg.index = [f"{k} ({ARM_LABELS.get(k, k)})" for k in ["B", "C", "D", "E"] if k in attribution]
            st.caption("**⭐ Marginal contribution of EACH element**")
            if marg.abs().sum() == 0:
                st.info("⚠️ No differentiation — brand at CEILING on all arms "
                        "(already winning this query). Run progressive on a query "
                        "where the brand does NOT already dominate.")
            else:
                st.bar_chart(marg)
                best = marg.idxmax();
                best_val = marg.max()
                if best_val > 0:
                    st.success(f"🏆 Biggest driver: **{best}** contributed **{best_val:+.1f}pp**.")
                else:
                    st.info("No positive marginal lift — enrichment didn't help here.")

            # 3. the winning element callout
            if len(marg):
                best = marg.idxmax(); best_val = marg.max()
                st.success(f"🏆 Biggest driver: **{best}** contributed **{best_val:+.1f}pp**. "
                           f"→ Prioritise this element in content.")

            # 4. full detail table
            with st.expander("Full per-arm, per-engine detail"):
                st.dataframe(adf[["engine","arm","label","rate","vs_A_pp",
                                  "marginal_pp","p_fdr","significant"]],
                             use_container_width=True)

            # 5. show the arm texts (what was actually tested)
            # 5. show the tested arm content
            with st.expander("View the tested arm content (A→E)"):
                design = next((d for d in ss.designs if d["gap_id"] == er["gap_id"]), None)
                if design and "progressive_arms" in design:
                    for k in ["A", "B", "C", "D", "E"]:
                        st.markdown(f"**{k} ({ARM_LABELS.get(k, k)}):**")
                        st.text(design["progressive_arms"][k].get("text", "")[:500])