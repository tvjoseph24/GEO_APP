"""Experiment tab. Signal-driven design (Priority 2) + rich competitor context (P2-1)."""
import json
import streamlit as st
import pandas as pd
from config import MODEL_REGISTRY
from store import load_artifact, project_dir
from experiment.signal_designer import design_from_signals
from experiment.variant_gen import (generate_text_variants, generate_progressive_variants,
                                    attach_visual_variant)
from experiment.competitor_enrich import (build_rich_competitor_context,
                                          render_competitor_context, save_competitor_context)
from experiment.rag_runner import run_experiment, run_progressive_experiment
from experiment.results import assemble_findings, save_experiment_results
from human_gate import HumanGate
from experiment.signal_designer import _brand_free_queries

def _wkey(prefix, d, i):
    """Stable widget key: identity-based, not position-based."""
    return f"{prefix}_{d.get('design_id') or d.get('gap_id') or i}"


def render(ctx):
    st.header("🧪 Module 4: Experiment (Prove Causation)")
    p, t = ctx.parent_brand, ctx.target_brand
    gateway = ctx.gateway
    variant_model = ctx.models.get("variant_model", "gemini-2.5-flash")
    sel_engines = ctx.models.get("audit_engines", ["gemini-2.5-flash", "azure/gpt-4o"])
    ss = st.session_state

    # ================= RECOVERY (Level-A) =================
    loaded = ss.get("loaded_artifacts", {})
    if "validated_findings_only" in loaded and "exp_findings" not in ss:
        st.success("✅ Existing Experiment findings loaded (read-only). Re-run to regenerate.")
        with st.expander("🧪 Validated Findings", expanded=True):
            for f in loaded["validated_findings_only"]:
                st.markdown(f"**{f.get('experiment_id')}** ({f.get('gap_id', f.get('design_id',''))}) — "
                            f"{f.get('verdict', '').upper()} · {f.get('pooled_lift_pp')}pp")
                if f.get("per_engine"):
                    st.dataframe(pd.DataFrame(f["per_engine"]), use_container_width=True)
        with st.expander("📄 Tested Variants"):
            st.json(loaded.get("experiment_variants", []))


        st.divider()

    # ================= LOAD INPUTS =================
    metrics = load_artifact(p, t, "benchmark_metrics")          # Priority-1 diagnostics
    gaps = load_artifact(p, t, "gap_register_actionable") or []
    claims = load_artifact(p, t, "claims_register") or []
    comps = load_artifact(p, t, "competitive_set") or []
    qm = load_artifact(p, t, "query_master") or []

    if not metrics:
        st.error("No benchmark metrics found. Complete Benchmark first "
                 "(it now saves diagnostics that drive experiment design).")
        return

    comp_names = [c["brand"] if isinstance(c, dict) else c for c in comps]
    raw_path = str(project_dir(p, t) / "audit_raw.jsonl")

    # ================= WEAKNESS SIGNALS (drive design) =================
    st.subheader("⚠️ Weakness Signals (driving experiment design)")
    flags = metrics.get("weakness_flags", [])
    if flags:
        for f in flags:
            icon = "🔴" if f.get("severity") == "high" else "🟡"
            st.warning(f"{icon} **{f['type']}**: {f['detail']}")
    else:
        st.info("No major weakness flags — experiments will lean on claim gaps.")
    with st.expander("📊 Mention rate by intent (the key signal)"):
        st.dataframe(pd.DataFrame(metrics.get("mention_by_intent", [])), use_container_width=True)

    # ================= CONFIG =================
    n_runs = st.slider("Runs per variant (N)", 10, 100, 30, key="exp_n")
    temperature = st.slider("Temperature", 0.5, 1.0, 0.8, key="exp_temp")
    target_pos = st.radio("Target position", ["top", "bottom", "shuffle"], key="exp_pos")
    budget = st.number_input("Budget cap ($)", 1.0, 500.0, 30.0, key="exp_budget")
    hard_stop = st.checkbox("Hard-stop if breached", key="exp_hard")

    engines = [{"name": m, "model": m, "provider": MODEL_REGISTRY[m]["provider"],
                "supports": MODEL_REGISTRY[m]["supports"]} for m in sel_engines]

    # ================= PROPOSE (signal-driven + rich competitors) =================
    if st.button("🎯 Propose weakness-driven experiments", key="exp_propose"):
        with st.spinner("Designing from signals + building competitor context…"):
            prog = st.progress(0.0, text="Building competitor context…")
            # P2-1: rich, frozen competitor context (audit-first, grounded-fallback)
            comp_docs = build_rich_competitor_context(
                gateway, comps, ctx.category, t, raw_path, model=variant_model,
                progress_cb=lambda x: prog.progress(x * 0.3, text="Enriching competitors…"))
            ss["exp_comp_docs"] = comp_docs
            ss["exp_comp_ctx"] = render_competitor_context(comp_docs)
            save_competitor_context(p, t, comp_docs)

            # Priority 2: signal-driven designs (weakness-first, claim-gaps default OFF)
            designs = design_from_signals(metrics, qm, claims, t, gap_register=gaps)
            # draft variants for INCLUDED designs only (cost control)
            active = [d for d in designs if d.get("include")]
            bf = _brand_free_queries(qm)  # compute once
            for i, d in enumerate(active):
                prog.progress(0.3 + 0.7 * (i + 1) / max(len(active), 1),
                              text=f"Drafting variant {i + 1}/{len(active)}")
                if not d.get("question"):
                    d["question"] = bf[0] if bf else f"best {ctx.category}"
                if not d.get("question"):
                    st.warning(f"Skip {d.get('design_id')} — no brand-free query")
                    continue
                if "variants" not in d:
                    d["variants"] = generate_text_variants(
                        gateway, d, claims, t, model=variant_model)

            ss["exp_designs"] = designs
            prog.progress(1.0, text="Done")
        n_active = sum(1 for d in designs if d.get("include"))
        st.success(f"Proposed {len(designs)} experiments ({n_active} weakness-driven active).")

    # ================= GATE 1: DESIGNS + EDITABLE COMPETITOR CONTEXT =================
    if "exp_designs" in ss:
        # editable competitor context (P2-1 — full content shown)
        st.subheader("🏢 Competitor Context (frozen field — edit to enrich)")
        st.caption("What the target competes against. Same for all experiments.")
        edited_ctx = st.text_area("Competitor content (markdown — add claims/tables)",
                                  ss.get("exp_comp_ctx", ""), height=250, key="exp_comp_edit")
        if st.button("💾 Update competitor context", key="exp_comp_save"):
            ss["exp_comp_ctx"] = edited_ctx
            st.success("Competitor context updated — applies to next run.")

        st.subheader("🚦 Gate 1: Approve Designs")
        for i, d in enumerate(ss["exp_designs"]):
            sig = d.get("signal", "claim_gap")
            sig_icon = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(d.get("priority_signal"), "")
            label = f"{sig_icon} [{sig}] {d.get('experiment_name','')} — {d.get('design_id', d.get('gap_id',''))}"
            with st.expander(label, expanded=d.get("include", False)):
                #d["include"] = st.checkbox("Include", value=d.get("include", False), key=f"exp_inc{i}")
                d["include"] = st.checkbox("Include", value=d.get("include", False),
                                           key=_wkey("exp_inc", d, i))
                st.caption(f"**Signal:** {sig} | **Hypothesis:** {d.get('hypothesis','')}")
                st.caption(f"**Question:** {d.get('question')}")
                if "current_rate" in d:
                    st.caption(f"📉 Baseline: {d['current_rate']:.0%} (room to improve)")
                # FULL variant content (tables rendered via markdown)
                if "variants" in d:
                    st.markdown("**Variant A (plain):**")
                    st.text(d["variants"]["variant_A"].get("text", ""))
                    st.markdown("**Variant B (enriched — tables rendered):**")
                    st.markdown(d["variants"]["variant_B"].get("text", ""))
                    if d["variants"].get("quality_warning"):
                        st.warning(d["variants"]["quality_warning"])
                else:
                    st.info("Not drafted (excluded at propose time). Enable + re-propose to draft.")
                #d["mode"] = st.radio("Mode", ["A/B (fast)", "Progressive (E1)"], key=f"exp_mode{i}")
                d["mode"] = st.radio("Mode", ["A/B (fast)", "Progressive (E1)"],
                                     key=_wkey("exp_mode", d, i))

        # ================= COST ESTIMATE + RUN =================
        active = [d for d in ss["exp_designs"] if d.get("include")]
        calls = sum((5 if d.get("mode", "").startswith("Progressive") else 2)
                    for d in active) * n_runs * len(engines)
        st.info(f"~{calls:,} calls ≈ ${calls * 0.002:.2f} "
                f"({len(active)} experiments × arms × {n_runs} runs × {len(engines)} engines)")

        if st.button("✅ Approve & Run", key="exp_run", disabled=not active):
            raw_path = project_dir(p, t) / "experiment_raw.jsonl"
            raw_path.unlink(missing_ok=True)  # fresh file per run
            raw_f = open(raw_path, "a", encoding="utf-8")

            def raw_writer(rec):
                raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n");
                raw_f.flush()
            prog = st.progress(0.0); status = st.empty(); done = [0]
            total = sum((5 if d.get("mode", "").startswith("Progressive") else 2)
                        for d in active) * n_runs * len(engines)
            def cb():
                done[0] += 1
                prog.progress(min(done[0] / max(total, 1), 1.0))
                status.write(f"{done[0]}/{total} · ${ctx.meter.total_cost():.3f}")
            results = []
            bf_run = _brand_free_queries(qm)  # ← hoist, computed once
            for d in active:
                # ── FIX A: draft variant on-demand if not yet drafted ──

                if "variants" not in d:
                    if not d.get("question"):
                        d["question"] = bf_run[0] if bf_run else (
                            f"best {ctx.category}".strip() if ctx.category else "")
                    status.write(f"Drafting variant for {d.get('design_id')}…")
                    d["variants"] = generate_text_variants(gateway, d, claims, t,
                                                           model=variant_model)
                q = (d.get("question") or "").lower()
                if not q.strip():
                    st.warning(f"Skip {d.get('design_id')} — empty question");
                    continue
                if t.lower() in q:
                    st.warning(f"Skip {d.get('design_id')} — question contains target "
                               f"brand '{t}'; mention-rate would be meaningless");
                    continue

                if "EMPTY" in d.get("variants", {}).get("quality_warning", ""):
                    st.warning(f"Skip {d.get('design_id')} — empty variant"); continue
                if budget and ctx.meter.total_cost() >= budget and hard_stop:
                    st.error("🛑 Budget reached"); break
                if d.get("mode", "").startswith("Progressive"):
                    pv = generate_progressive_variants(gateway, d, claims, t, model=variant_model)
                    d["progressive_arms"] = pv["arms"]
                    er = run_progressive_experiment(gateway, d, pv["arms"], ss["exp_comp_ctx"],
                            d["question"], t, comp_names, n_runs=n_runs, temperature=temperature,
                            engines=engines, target_pos=target_pos, progress_cb=cb,raw_writer=raw_writer)
                    er["mode"] = "progressive"
                else:
                    er = run_experiment(gateway, d, d["variants"], ss["exp_comp_ctx"],
                            d["question"], t, comp_names, n_runs=n_runs, temperature=temperature,
                            engines=engines, target_pos=target_pos, progress_cb=cb,raw_writer=raw_writer)
                    er["mode"] = "ab"
                er["design_id"] = d.get("design_id", d.get("gap_id"))
                er["signal"] = d.get("signal")
                results.append(er)
            raw_f.close()
            ss["exp_results"] = results
            ss["exp_findings"] = assemble_findings([e for e in results if e.get("mode")=="ab"], t)   # ← pass target_brand
            st.success("Complete.")

    # ================= RESULTS =================
    if "exp_findings" in ss:
        st.subheader("Results (A/B)")
        for f in ss["exp_findings"]:
            v = f["verdict"]
            icon = {"validated": "✅", "validated_mention": "📈", "engine-specific": "🟡",
                    "engine-specific_mention": "📈", "already_winning": "⬆️",
                    "lost": "⬇️", "rejected": "❌", "no_lift": "➖"}.get(v, "?")
            st.markdown(f"{icon} **{f['experiment_id']}** ({f['design_id']}) — "
                        f"{v.upper()} · best: {f.get('best_metric')} · {f['pooled_lift_pp']}pp")
            df = pd.DataFrame(f["per_engine"])
            # show mention_rate and first_rec_rate separately
            for metric in ["mention_rate", "first_rec_rate"]:
                sub = df[df.metric == metric]
                if not sub.empty:
                    st.caption(f"**{metric}**")
                    st.dataframe(sub[["engine", "A_rate", "B_rate", "lift_pp", "significant"]],
                                 use_container_width=True)


            # ── PER-EXPERIMENT: target vs competitors (INSIDE the loop)
            with st.expander(f"View {f.get('design_id')} — target content vs competitors"):
                d = next((x for x in ss["exp_designs"]
                          if x.get("design_id") == f.get("design_id")), None)
                if d and "variants" in d:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Target (Variant B):**")
                        st.markdown(d["variants"]["variant_B"].get("text", "")[:1500])
                    with col2:
                        st.markdown("**Competitors:**")
                        st.markdown(ss.get("exp_comp_ctx", "")[:1500])

        # ── progressive display (inside if exp_findings, after loop) ──
        prog_r = [e for e in ss.get("exp_results", []) if e.get("mode") == "progressive"]
        if prog_r:
            from experiment.exp_stats import analyze_progressive, element_attribution
            st.subheader("🔬 Progressive — Element Attribution")
            for er in prog_r:
                rows = analyze_progressive(er);
                attr = element_attribution(rows)
                marg = pd.Series(attr).reindex(["B", "C", "D", "E"]).dropna()
                st.markdown(f"**{er.get('design_id')}**")
                # floor-vs-ceiling fix (from earlier)
                a_first = sum(v["A"]["is_first"] for v in er["engines"].values())
                a_n = sum(v["A"]["n"] for v in er["engines"].values())
                a_rate = a_first / max(a_n, 1)
                if marg.abs().sum() == 0:
                    if a_rate >= 0.9:
                        st.info("⬆️ At CEILING — brand already winning (no room to improve)")
                    elif a_rate <= 0.1:
                        st.error("⬇️ At FLOOR — brand absent in all arms (competitor dominates)")
                    else:
                        st.info("No differentiation between arms")
                else:
                    st.bar_chart(marg)

        # ── ONCE: overall competitor content (inside if exp_findings, after loop) ──
        if "exp_comp_ctx" in ss:
            with st.expander("🏢 What the target competed AGAINST (competitor content)"):
                st.markdown(ss["exp_comp_ctx"])

        # ================= GATE 2: VALIDATE + EMIT ================= ← Bug 2 fix (un-nested)
        st.subheader("🚦 Gate 2: Validate")
        if "exp_gate" not in ss:
            ss["exp_gate"] = HumanGate("results", lambda fb=None: ss["exp_findings"])
            ss["exp_gate"].generate()
        if st.button("✅ Approve & Emit", key="exp_emit"):
            ss["exp_gate"].approve("analyst")
            save_experiment_results(
                p, t, ss["exp_results"], ss["exp_findings"], ss["exp_designs"],
                {"variant_model": variant_model},
                competitor_context=ss.get("exp_comp_ctx", ""),
                competitor_docs=ss.get("exp_comp_docs", {}))
            st.success("Emitted (with competitor content).")
