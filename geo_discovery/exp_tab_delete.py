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
from experiment.results import assemble_findings, save_experiment_results
from human_gate import HumanGate
from experiment.signal_designer import _brand_free_queries
from experiment.rag_runner import (run_experiment, run_progressive_experiment,
                                   completed_runs, aggregate_from_raw)
from experiment.design_from_hypothesis import designs_from_hypotheses
from experiment.signal_designer import design_from_signals, _brand_free_queries
from experiment.variant_gen import (generate_text_variants,
                                    generate_progressive_variants,
                                    attach_visual_variant)
from experiment.competitor_enrich import (build_rich_competitor_context,
                                          render_competitor_blocks,
                                          render_competitor_context,
                                          save_competitor_context, parity_report,
                                          freeze_manifest)
from experiment.rag_runner import (run_experiment, run_progressive_experiment,
                                   completed_runs, aggregate_from_raw)
from experiment.power import run_pilot, power_verdict, mde, n_for_effect
from experiment.results import assemble_findings, save_experiment_results

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
    hyps = load_artifact(p, t, "gap_hypotheses") or []
    alias_ov = {x["brand"]: x.get("aliases", []) for x in comps
                if isinstance(x, dict) and x.get("aliases")}
    comp_names = [c["brand"] if isinstance(c, dict) else c for c in comps
                  if (c.get("role") if isinstance(c, dict) else None) != "self"
                  and (c["brand"] if isinstance(c, dict) else c)]
    if hyps:
        ready = sum(1 for h in hyps if h.get("status") == "ready")
        st.success(f"🧠 {len(hyps)} hypothesis(es) from Benchmark — {ready} ready to test.")
    else:
        st.warning("No hypotheses found. Re-run Benchmark to generate them "
                   "(falling back to heuristic signal designs).")

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

    src = st.radio("Design source",
                   ["Hypotheses (Benchmark-derived — recommended)",
                    "Weakness signals (heuristic)"],
                   key="exp_src", horizontal=True)
    target_words = st.slider("Document length per arm (words)", 100, 400, 180,
                             key="exp_words",
                             help="All arms and all competitor docs are matched to this "
                                  "for parity — length is otherwise a confound.")

    if st.button("🎯 Build designs + frozen competitor field", key="exp_propose"):
        if "exp_designs" not in ss:
            st.info("👆 Click **Build designs + frozen competitor field** to begin.")
            return
        st.subheader("🏢 Frozen competitor field")
        edited = st.text_area("Competitor content (markdown)", ss.get("exp_comp_ctx", ""),
                              height=220, key="exp_comp_edit")
        if st.button("💾 Update competitor field", key="exp_comp_save"):
            ss["exp_comp_ctx"] = edited
            ss["exp_comp_manifest"] = freeze_manifest(ss.get("exp_comp_docs", {}), edited)
            st.success("Updated — applies to the next run.")

        active = [d for d in ss["exp_designs"] if d.get("include")]

        if active:
            st.subheader("⚖️ Parity check")
            pr_df = parity_report(
                (active[0]["variants"]["variant_B"].get("text") or ""),
                ss.get("exp_comp_blocks") or [ss.get("exp_comp_ctx", "")])
            st.dataframe(pr_df, use_container_width=True)
            for w in (pr_df.attrs.get("warnings") or []):
                st.warning(f"⚠️ {w}")
            parity_ok = not pr_df.attrs.get("warnings")
            if not parity_ok:
                st.caption("Length/format asymmetry means the experiment measures "
                           "document quality, not your variable.")

        # ---------- PILOT: establish the REAL experimental baseline ----------
        st.subheader("🔬 Step 1 — Pilot (arm A only)")
        st.caption("The audit baseline is the wrong p0: experiments inject the brand "
                   "document with web OFF, so arm-A rates are far higher. The pilot "
                   "measures the true baseline and tells you the N you actually need.")
        pc1, pc2 = st.columns([2, 1])
        pilot_design = pc1.selectbox("Pilot this design",
                                     [d["design_id"] for d in active] or ["—"],
                                     key="exp_pilot_d")
        pilot_n = pc2.number_input("Pilot runs/engine", 10, 60, 20, key="exp_pilot_n")
        if st.button("▶️ Run pilot", key="exp_pilot_go", disabled=not active):
            d0 = next(d for d in active if d["design_id"] == pilot_design)
            with st.spinner(f"Pilot: {pilot_n} × {len(engines)} runs…"):
                ss["exp_pilot"] = run_pilot(
                    gateway, d0["variants"]["variant_A"].get("text", ""),
                    ss["exp_comp_ctx"], d0["question"], t, comp_names,
                    engines=engines, n=int(pilot_n), temperature=temperature,
                    target_pos=target_pos, alias_overrides=alias_ov)
                ss["exp_pilot"]["design_id"] = pilot_design

        if ss.get("exp_pilot"):
            pl = ss["exp_pilot"]
            m = st.columns(4)
            m[0].metric("Arm-A mention", f"{pl['p0_mention']:.0%}")
            m[1].metric("Arm-A first-rec", f"{pl['p0_first']:.0%}")
            m[2].metric("Runs", pl["n_total"])
            m[3].metric("MDE @ N=100", f"{pl['mde_table']['100']['mention_pp']}pp")
            if pl["ceiling_risk"]:
                st.error("⬆️ CEILING — the brand is mentioned in ≥90% of arm-A runs. "
                         "mention_rate cannot move. Switch the primary metric to "
                         "first_rec_rate.")
            if pl["floor_risk"]:
                st.error("⬇️ FLOOR — the brand is absent even with its own document "
                         "injected. Check the target query and competitor field.")
            st.dataframe(pd.DataFrame(pl["per_engine"]).T, use_container_width=True)
            st.write("**N required at the pilot baseline**")
            st.dataframe(pd.DataFrame(pl["mde_table"]).T, use_container_width=True)
            st.write("**Power verdict per selected design**")
            rows = []
            for d in active:
                pv = power_verdict(pl, d.get("_hypothesis") or
                                   {"prediction": d.get("prediction")}, int(n_runs))
                d["_power"] = pv
                rows.append({"design": d["design_id"], **pv})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            bad = [r["design"] for r in rows
                   if r["verdict"] in ("underpowered", "ceiling", "floor",
                                       "no_prediction")]
            if bad:
                st.warning(f"⚠️ {bad} are not adequately powered at N={n_runs}. "
                           f"Raise N, drop to one engine, or switch metric.")

        # ---------- FULL RUN ----------
        st.subheader("🚀 Step 2 — Full run")
        calls = sum((5 if d.get("mode", "").startswith("Progressive") else 2)
                    for d in active) * n_runs * len(engines)
        st.info(f"~{calls:,} calls ≈ ${calls * 0.002:.2f}")
        force = st.checkbox("Override power/parity warnings", key="exp_force")
        gated = [d for d in active
                 if force or (d.get("_power") or {}).get("verdict") == "powered"
                 or not ss.get("exp_pilot")]
        if st.button("✅ Approve & Run", key="exp_run", disabled=not gated):
            raw_path_e = project_dir(p, t) / "experiment_raw.jsonl"
            resume = True
            done = completed_runs(raw_path_e) if resume else set()
            if done:
                st.info(f"↻ Resuming — {len(done)} run(s) already complete")
            raw_f = open(raw_path_e, "a", encoding="utf-8")

            def raw_writer(rec):
                raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n");
                raw_f.flush()

            prog = st.progress(0.0);
            status = st.empty();
            doneN = [0]
            total = max(sum((5 if d.get("mode", "").startswith("Progressive") else 2)
                            for d in gated) * n_runs * len(engines) - len(done), 1)

            def cb():
                doneN[0] += 1
                prog.progress(min(doneN[0] / total, 1.0))
                status.write(f"{doneN[0]}/{total} · ${ctx.meter.total_cost():.3f}")

            results = []
            for d in gated:
                if budget and ctx.meter.total_cost() >= budget and hard_stop:
                    st.error("🛑 Budget reached");
                    break
                kw = dict(comp_context=ss["exp_comp_ctx"], question=d["question"],
                          target_brand=t, competitors=comp_names, n_runs=n_runs,
                          temperature=temperature, engines=engines,
                          target_pos=target_pos, progress_cb=cb,
                          raw_writer=raw_writer, alias_overrides=alias_ov, done=done)
                if d.get("mode", "").startswith("Progressive"):
                    pv = generate_progressive_variants(
                        gateway, d, claims, t, model=variant_model,
                        hypothesis=d.get("_hypothesis"), target_words=target_words)
                    d["progressive_arms"] = pv["arms"]
                    er = run_progressive_experiment(gateway, d, pv["arms"], **kw)
                    er["mode"] = "progressive"
                else:
                    er = run_experiment(gateway, d, d["variants"], **kw)
                    er["mode"] = "ab"
                er["design_id"] = d["design_id"]
                er["signal"] = d.get("signal")
                er["hypothesis"] = d.get("hypothesis")
                er["mechanism"] = d.get("mechanism")
                er["pillar"] = d.get("pillar")
                er["prediction"] = d.get("prediction")
                er["pilot"] = ss.get("exp_pilot")
                agg = aggregate_from_raw(raw_path_e, er["design_id"])
                if agg:
                    for eng in engines:
                        for arm_key, stats in (agg.get(eng["model"]) or {}).items():
                            er["engines"].setdefault(eng["name"], {}) \
                                .setdefault(arm_key, {}).update(stats)
                    er["aggregated_from_raw"] = True
                results.append(er)
            raw_f.close()
            ss["exp_results"] = results
            ss["exp_findings"] = assemble_findings(results, t)
            st.success(f"Complete — {len(results)} experiment(s).")
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
            chk = (d.get("variants") or {}).get("check") or {}
            ok = chk.get("ok", True)
            sig_icon = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(
                d.get("priority_signal"), "")
            state_icon = "✅" if ok else "🚫"
            label = (f"{state_icon} {sig_icon} {d.get('design_id', d.get('gap_id', ''))} "
                     f"· {d.get('experiment_id', '?')} {d.get('experiment_name', '')} "
                     f"· {d.get('pillar') or d.get('signal', '')}")

            with st.expander(label, expanded=d.get("include", False)):
                if not ok:
                    st.error(f"🚫 Manipulation check FAILED — {chk.get('warning')}")
                    st.caption("Running this would produce an uninterpretable result. "
                               "Re-propose, or fix the claims available to it.")
                else:
                    st.success(f"✅ Manipulation verified · isolates "
                               f"**{d.get('isolates')}**")

                st.caption(f"Mechanism `{d.get('mechanism')}` · pillar "
                           f"{d.get('pillar')} · horizon {d.get('horizon')} · covers "
                           f"{len(d.get('gap_ids') or [])} gap(s)")
                st.markdown(f"**Hypothesis** — {d.get('hypothesis')}")
                st.markdown(f"**Question** — “{d.get('question')}”"
                            + ("" if d.get("question_in_audit")
                               else "  ⚠️ *not in the audit set — no measured baseline*"))
                pr = d.get("prediction") or {}
                st.caption(f"Predicts {pr.get('baseline_pp')}% → "
                           f"{pr.get('expected_pp')}% on `{pr.get('metric')}`")
                if d.get("falsification"):
                    st.markdown(f"**Falsification** — {d['falsification']}")
                if chk.get("checks"):
                    st.caption(f"Checks: {chk['checks']}")
                if d.get("claims_allowed"):
                    with st.popover("Claims approved for this design"):
                        for cu in d["claims_allowed"]:
                            st.caption(f"• {cu}")

                v = d.get("variants") or {}
                for k in [x for x in "ABCDE" if f"variant_{x}" in v]:
                    st.markdown(f"**Arm {k}**")
                    st.text((v[f"variant_{k}"].get("text") or "")[:1500])

                d["include"] = st.checkbox("Include", value=d.get("include", False),
                                           key=_wkey("exp_inc", d, i),
                                           disabled=not ok)
                d["mode"] = st.radio(
                    "Mode", ["A/B (fast)", "Progressive (E1)"],
                    index=1 if str(d.get("mode", "")).startswith("Prog") else 0,
                    key=_wkey("exp_mode", d, i))

        # ================= COST ESTIMATE + RUN =================
        active = [d for d in ss["exp_designs"] if d.get("include")]
        calls = sum((5 if d.get("mode", "").startswith("Progressive") else 2)
                    for d in active) * n_runs * len(engines)
        st.info(f"~{calls:,} calls ≈ ${calls * 0.002:.2f} "
                f"({len(active)} experiments × arms × {n_runs} runs × {len(engines)} engines)")

        if st.button("✅ Approve & Run", key="exp_run", disabled=not active):
            raw_path = project_dir(p, t) / "experiment_raw.jsonl"
            resume = st.checkbox("Resume from existing raw file", value=True,
                                 key="exp_resume")
            done = completed_runs(raw_path) if resume else set()
            if not resume:
                raw_path.unlink(missing_ok=True)
            elif done:
                st.info(f"↻ Resuming — {len(done)} run(s) already complete")
            raw_f = open(raw_path, "a", encoding="utf-8")

            raw_path.unlink(missing_ok=True)  # fresh file per run
            raw_f = open(raw_path, "a", encoding="utf-8")

            def raw_writer(rec):
                raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n");
                raw_f.flush()
            prog = st.progress(0.0); status = st.empty(); done = [0]
            total = sum((5 if d.get("mode", "").startswith("Progressive") else 2)
                        for d in active) * n_runs * len(engines)
            total = max(total - len(done), 1)  # ← skipped runs are already done
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
                # ---- authoritative aggregates from the raw log ----
                # run_variant counts only NEW runs, so with resume the in-memory
                # numbers under-report. The JSONL is the single source of truth.
                agg = aggregate_from_raw(raw_path, er["design_id"])
                if agg:
                    for eng in engines:
                        by_arm = agg.get(eng["model"], {})
                        if not by_arm:
                            continue
                        for arm_key, stats in by_arm.items():
                            tgt = er["engines"].setdefault(eng["name"], {})
                            tgt.setdefault(arm_key, {}).update(stats)
                    er["aggregated_from_raw"] = True
                results.append(er)
            raw_f.close()
            ss["exp_results"] = results
            ss["exp_findings"] = assemble_findings(results, t)   # was: [e for e in results if mode=="ab"]   # ← pass target_brand
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
