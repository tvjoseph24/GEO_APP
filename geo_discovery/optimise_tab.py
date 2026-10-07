"""Optimise tab — LLM-driven strategy, engine overlays, constrained roadmap, briefs."""
import json

import pandas as pd
import streamlit as st

from human_gate import HumanGate
from optimise.brief_gen import generate_brief, save_briefs, winning_content_for
from optimise.prioritizer import DEFAULT_WEIGHTS, score_actions, summary
from optimise.roadmap import DEFAULT_CAPACITY, build_roadmap
from optimise.strategy_llm import (PILLARS, assemble_strategy_input, build_strategy_llm,
                                   evidence_summary)
from store import load_artifact, save_artifact
import theme

def render(ctx):
    theme.section("Optimise", "Evidence into strategy")
    p, t = ctx.parent_brand, ctx.target_brand
    gateway = ctx.gateway
    model = ctx.models.get("brief_model") or ctx.models.get("judge", "gemini-2.5-pro")
    ss = st.session_state

    # ---------------- recovery ----------------
    loaded = ss.get("loaded_artifacts", {})
    if "geo_strategy" in loaded and "opt_strategy" not in ss:
        st.success("✅ Existing strategy loaded (read-only). Re-run to regenerate.")
        with st.expander("🎯 Strategy", expanded=True):
            st.json(loaded["geo_strategy"])
        with st.expander("📋 Roadmap"):
            st.json(loaded.get("roadmap", {}))
        st.divider()

    # ---------------- evidence base ----------------
    si = assemble_strategy_input(p, t, ctx.category)
    if not (si["wins"] or si["nulls"] or si["gaps"]):
        st.error("No findings, nulls or gaps. Complete Benchmark and Experiment first.")
        return
    ev = evidence_summary(si)

    st.subheader(" Evidence base")
    k = st.columns(6)
    k[0].metric("Proven findings", ev["n_wins"])
    k[1].metric("Nulls", ev["n_nulls"], f"{ev['n_well_powered_nulls']} well-powered")
    k[2].metric("Diagnosed gaps", ev["n_gaps"])
    k[3].metric("Substantiation backlog", ev["n_substantiation"])
    k[4].metric("Creative bench", ev.get("n_creative_bench", 0),
                f"{len(ev.get('creative_equivalent') or [])} equivalent",
                help="A different instrument: asset-first, no gap or hypothesis. Measures "
                     "which creative a machine reader prefers.")
    k[5].metric("Retrieval gap", f"{ev['retrieval_gap_pp']:.0f}pp",
                help="Audit need-query mention vs ≈100% when the brand's own document "
                     "is in context. The gap is retrieval, not content.")
    _soc = ev.get("social") or {}
    if _soc.get("verdict"):
        _sfn = {"negligible": st.info, "query_triggered": st.warning}.get(
            _soc["verdict"], st.caption)
        _sfn(f"💬 **Community voice — {_soc['verdict'].replace('_', ' ')}** · "
             f"{(_soc.get('overall_social_share') or 0):.1%} of citations"
             + (f" · triggered on {', '.join(_soc['triggered_intents'])}"
                if _soc.get("triggered_intents") else "")
             + (f" · {_soc['n_threads_brand_absent']} cited thread(s) never mention the "
                f"brand" if _soc.get("n_threads_brand_absent") else "")
             + (f" · {_soc['n_threads_negative']} carry negative framing"
                if _soc.get("n_threads_negative") else "")
             + (". Pillar 5 should be empty, and the strategy should say why."
                if _soc["verdict"] == "negligible" else ""))

    if ev.get("creative_authority_constrained"):
        st.warning(f"🔀 On {ev['creative_authority_constrained']} the winning creative "
                   f"raises MENTION but the brand is still never recommended first under "
                   f"any arm. That is an authority constraint, not a creative one — no "
                   f"creative change will move it, and the work belongs in Corroboration.")

    with st.expander("What the evidence says", expanded=True):
        for line in (si["delta"].get("headline") or []):
            st.markdown(f"- {line}")
        c1, c2 = st.columns(2)
        c1.markdown("**✅ Proven**")
        for w in si["wins"]:
            if w.get("is_creative_bench"):
                c1.caption(f"`{w['design_id']}` creative bench → arm "
                           f"**{w.get('winning_creative')}** wins by {w.get('lift')}pp "
                           f"on {w.get('best_metric')}"
                           + (f" · engines differ by {w.get('gap_spread_pp')}pp"
                              if w.get("magnitude_diverges") else ""))
            else:
                c1.caption(f"`{w['design_id']}` "
                           f"{w.get('isolated_variable') or w.get('template')} "
                           f"→ {w.get('lift')} {w.get('lift_unit', 'pp')} on "
                           f"{w.get('best_metric')}"
                           + ("  ⚠ underpowered" if w.get("underpowered") else ""))
        c2.markdown("**❌ Disproven / no effect**")
        for n in si["nulls"]:
            if n.get("is_creative_bench"):
                c2.caption(f"`{n['design_id']}` creative variants are EQUIVALENT to a "
                           f"machine reader — an exclusion, not a failed mechanism. "
                           f"Choose on human grounds.")
                continue
            tag = "" if n.get("underpowered") else " **(well-powered)**"
            c2.caption(f"`{n['design_id']}` "
                       f"{n.get('isolated_variable') or n.get('template')} "
                       f"→ no effect at {n.get('runs_per_cell')} runs/arm{tag}")
        st.caption("Well-powered nulls become explicit exclusions in the strategy — "
                   "stating what not to spend on is as valuable as what to do. A creative "
                   "EQUIVALENCE is also an exclusion, but a narrow one: it rules out those "
                   "variants, not the pillar.")

    # ---------------- config ----------------
    st.subheader("⚙️ Strategy configuration")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Team capacity** (person-days per wave)")
        cap = {}
        for owner, default in DEFAULT_CAPACITY.items():
            cap[owner] = st.number_input(owner, 0, 60, default, key=f"opt_cap_{owner}")
    with c2:
        st.markdown("**Prioritisation weights**")
        weights = {}
        for kk, label in (("evidence", "Evidence strength"), ("effect", "Effect size"),
                          ("reach", "Query reach"), ("effort", "Effort penalty"),
                          ("speed", "Speed preference")):
            weights[kk] = st.slider(label, 0.0, 2.0, DEFAULT_WEIGHTS[kk], 0.1,
                                    key=f"opt_w_{kk}")
        run_critic = st.checkbox("Run adversarial critic (+1 LLM call)", value=True,
                                 key="opt_critic")
        gen_briefs = st.checkbox("Generate content briefs for Wave 0–1", value=True,
                                 key="opt_gen_briefs")

    constraints = {
        "capacity_person_days_per_wave": cap,
        "notes": st.text_area("Constraints the strategy must respect "
                              "(legal, market scope, owned surfaces, tone)",
                              "Regulated category: no claim may be published unless it is "
                              "in the substantiated register. Market scope should be "
                              "confirmed — the audit corpus spans multiple markets.",
                              key="opt_constraints"),
    }

    n_calls = len(PILLARS) + 1 + len(si["metrics"].get("engines") or []) + \
        (1 if run_critic else 0)
    st.caption(f"≈{n_calls} LLM call(s) for the strategy"
               + (" plus one per brief" if gen_briefs else ""))

    # ---------------- build ----------------
    if st.button("🎯 Build strategy + roadmap", key="opt_run"):
        prog = st.progress(0.0, text="Starting…")

        def cb(x, label):
            prog.progress(min(x, 1.0), text=label)

        with st.spinner("Synthesising…"):
            strat = build_strategy_llm(gateway, p, t, ctx.category, model=model,
                                       constraints=constraints, progress_cb=cb,
                                       run_critic=run_critic)
        if strat.get("error"):
            st.error(strat["error"])
            return

        strat["actions"] = score_actions(
            strat["actions"], weights,
            engine_count=len(si["metrics"].get("engines") or []) or 1)

        with st.spinner("Building roadmap…"):
            rm = build_roadmap(gateway, strat, model, capacity=cap)

        briefs = []
        if gen_briefs:
            wave01 = [a for a in strat["actions"]
                      if str(a.get("horizon", "")).startswith(("Wave 0", "Wave 1"))
                      and not a.get("blocked")]
            bp = st.progress(0.0, text="Briefs…")
            for i, a in enumerate(wave01[:12]):
                wc = winning_content_for(a, si["variants"])
                try:
                    briefs.append(generate_brief(gateway, a, wc, si, model=model))
                except Exception as e:
                    st.warning(f"Brief failed for {a.get('action_id')}: {str(e)[:100]}")
                bp.progress((i + 1) / max(len(wave01[:12]), 1), text="Briefs…")

        ss["opt_strategy"] = strat
        ss["opt_roadmap"] = rm
        ss["opt_briefs_data"] = briefs
        # persist immediately — a UI exception after this point no longer loses the build
        try:
            save_artifact(p, t, "geo_strategy_draft",
                          {k: v for k, v in strat.items() if k != "_si"},
                          {"model": model, "status": "draft_pre_signoff"})
            save_artifact(p, t, "roadmap_draft", rm, {"model": model})
        except Exception as e:
            st.caption(f"(draft not persisted: {str(e)[:80]})")
        ss.pop("opt_gate", None)
        st.success(f"Built · {len(strat['actions'])} action(s) · {len(briefs)} brief(s)")

    if "opt_strategy" not in ss:
        st.info("👆 Click **Build strategy + roadmap**.")
        return

    strat, rm = ss["opt_strategy"], ss["opt_roadmap"]
    ov = strat.get("overall_strategy") or {}

    # ---------------- quality gate ----------------
    st.divider()
    blockers = [c for c in strat.get("critic_review", [])
                if c.get("severity") == "blocker"] + \
               [e for e in strat.get("validation_errors", [])
                if e.get("severity") == "blocker"]
    majors = [c for c in strat.get("critic_review", [])
              if c.get("severity") == "major"] + \
             [e for e in strat.get("validation_errors", [])
              if e.get("severity") == "major"]
    icon = "🔴" if blockers else ("🟡" if majors else "✅")
    with st.expander(f"{icon} Strategy quality gate — {len(blockers)} blocker(s), "
                     f"{len(majors)} major", expanded=bool(blockers)):
        for b in blockers:
            st.error(f"**{b.get('target')}** — {b.get('issue') or b.get('error')}"
                     + (f"\n\n➡️ {b.get('fix')}" if b.get("fix") else ""))
        for m in majors:
            st.warning(f"**{m.get('target')}** — {m.get('issue') or m.get('error')}"
                       + (f"\n\n➡️ {m.get('fix')}" if m.get("fix") else ""))
        minors = [x for x in strat.get("critic_review", []) +
                  strat.get("validation_errors", []) if x.get("severity") == "minor"]
        if minors:
            st.caption(f"{len(minors)} minor issue(s)")
            st.json(minors)

    # ---------------- narrative ----------------
    st.subheader("📖 Overall strategy")
    if ov.get("narrative"):
        st.info(ov["narrative"])
    if ov.get("diagnosis"):
        st.markdown(f"**Diagnosis** — {ov['diagnosis']}")
    if strat.get("retrieval_is_primary_constraint"):
        st.warning("🔍 The evidence indicates **retrieval, not content form**, is the "
                   "binding constraint: content-form experiments were null at adequate "
                   "power while unprompted visibility is very low. Visibility gaps have "
                   "been routed to Technical/Structured-Data rather than Content.")

    st.markdown("### 🎲 Big bets")
    for b in (ov.get("big_bets") or []):
        with st.container(border=True):
            st.markdown(f"**{b.get('title')}** · {b.get('evidence_class')} · "
                        f"{', '.join(b.get('pillars') or [])}")
            st.caption(f"{b.get('expected_effect')} — {b.get('why_now')}")
            if b.get("action_ids"):
                st.caption(f"Actions: {', '.join(b['action_ids'])}")

    c1, c2 = st.columns(2)
    c1.markdown("### 🧱 Structural truths (all engines)")
    for s in (ov.get("structural_truths") or []):
        c1.markdown(f"- **{s.get('truth')}**")
        c1.caption(f"  {s.get('evidence')}")
    c2.markdown("### 🚫 Not doing")
    for n in (ov.get("not_doing") or []):
        c2.markdown(f"- **{n.get('item')}**")
        c2.caption(f"  {n.get('reason')} · basis: {n.get('basis')}")

    if ov.get("compounding_plays"):
        with st.expander("🔗 Compounding plays"):
            st.json(ov["compounding_plays"])
    if ov.get("conflicts_resolved"):
        with st.expander("⚖️ Conflicts resolved"):
            st.json(ov["conflicts_resolved"])

    # ---------------- pillars ----------------
    st.subheader("🏛️ Pillars")
    st.caption(f"Actions by pillar: {strat.get('pillar_summary')} · "
               f"by evidence: {strat.get('evidence_summary_by_class')}")
    for pillar in PILLARS:
        pd_ = (strat.get("pillars") or {}).get(pillar) or {}
        acts = [a for a in strat["actions"] if a.get("pillar") == pillar]
        if pd_.get("_skipped"):
            st.caption(f"⚪ **{pillar}** — {pd_.get('thesis')}")
            continue
        with st.expander(f"{pillar} — {len(acts)} action(s) · "
                         f"{pd_.get('objective', '')[:80]}"):
            st.markdown(f"**Thesis** — {pd_.get('thesis')}")
            st.caption(f"Mechanism: `{pd_.get('mechanism')}` · "
                       f"KPI: {json.dumps(pd_.get('pillar_kpi') or {})}")
            for a in sorted(acts, key=lambda x: -x.get("priority_score", 0)):
                flag = "🚫 " if a.get("blocked") else ""
                ec = {"PROVEN": "🟢", "PROVEN_ENGINE_SPECIFIC": "🟠",
                      "PROVEN_WITH_CAVEAT": "🟡", "DIAGNOSED": "🔵",
                      "HYPOTHESIS": "⚪"}.get(a.get("evidence_class"), "⚪")
                with st.container(border=True):
                    st.markdown(f"{flag}{ec} **{a.get('action_id')} · {a.get('title')}** "
                                f"· {a.get('horizon')} · {a.get('clock')} · "
                                f"score {a.get('priority_score')}")
                    st.write(a.get("what"))
                    st.caption(f"**Why** {a.get('why')}")
                    st.caption(f"Effect: {json.dumps(a.get('expected_effect') or {})} · "
                               f"owner {a.get('owner')} · effort {a.get('effort')} · "
                               f"refs {a.get('evidence_refs')}")
                    if a.get("measurement_note"):
                        st.caption(f"📌 {a['measurement_note']}")
                    if a.get("validating_experiment"):
                        st.caption(f"🧪 Needs validation: {a['validating_experiment']}")
                    if a.get("claims_used"):
                        st.caption("Claims: " +
                                   " | ".join(str(x)[:60] for x in a["claims_used"][:3]))
                    st.caption(f"Kill: {a.get('kill_criterion')}")
            if pd_.get("do_not_do"):
                st.markdown("**Do not do**")
                for x in pd_["do_not_do"]:
                    st.caption(f"❌ {x.get('action')} — {x.get('reason')} "
                               f"({x.get('basis')})")

    # ---------------- overlays ----------------
    st.subheader("🔀 Engine overlays")
    for eng, o in (strat.get("engine_overlays") or {}).items():
        with st.expander(f"{eng} — {o.get('behavioural_profile', '')[:90]}"):
            st.markdown(f"**Profile** — {o.get('behavioural_profile')}")
            for x in (o.get("what_changes_here") or []):
                st.markdown(f"- **{x.get('action')}** — {x.get('why_this_engine')}")
                st.caption(f"  {x.get('evidence')}")
            if o.get("deprioritise_here"):
                st.caption("Deprioritise: " +
                           "; ".join(f"{x.get('action')} ({x.get('why')})"
                                     for x in o["deprioritise_here"]))
            if o.get("placement_targets"):
                st.dataframe(pd.DataFrame(o["placement_targets"]),
                             use_container_width=True)
            if o.get("engine_specific_risk"):
                st.warning(o["engine_specific_risk"])

    # ---------------- roadmap ----------------
    st.subheader("📋 Roadmap")
    if rm.get("dependency_cycles"):
        st.error(f"Dependency cycle(s): {rm['dependency_cycles']}")
    idx = {a["action_id"]: a for a in strat["actions"] if a.get("action_id")}
    for w in (rm.get("waves") or []):
        packed = (rm.get("waves_packed") or {}).get(w.get("wave"), {})
        with st.expander(f"**{w.get('wave')}** · {w.get('window')} — {w.get('theme')}",
                         expanded=str(w.get("wave", "")).startswith("Wave 0")):
            st.caption(f"Objective: {w.get('objective')} · "
                       f"Capacity: {packed.get('utilisation')}")
            rows = []
            for a in (w.get("actions") or []):
                full = idx.get(a.get("action_id"), {})
                rows.append({"id": a.get("action_id"), "title": a.get("title"),
                             "pillar": full.get("pillar"), "owner": a.get("owner"),
                             "effort": a.get("effort"),
                             "evidence": a.get("evidence_class"),
                             "depends_on": ", ".join(a.get("depends_on") or [])})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True)
            mc = w.get("measurement_contract") or {}
            st.markdown("**📏 Measurement contract**")
            st.caption(f"Re-audit {len(mc.get('re_audit_queries') or [])} query(ies) × "
                       f"N={mc.get('n_runs')} on {mc.get('engines')} · primary "
                       f"`{mc.get('primary_metric')}` · expect "
                       f"{mc.get('expected_delta')} ({mc.get('basis')}) · "
                       f"by {mc.get('re_audit_date')}")
            st.caption(f"🛑 Kill: {mc.get('kill_criterion')}")
            st.caption(f"Exit: {w.get('exit_criteria')}")
            for r in (w.get("risks") or []):
                st.caption(f"⚠️ {r.get('risk')} → {r.get('mitigation')}")

    if rm.get("projection"):
        st.markdown("**Projected trajectory** (need-query mention rate)")
        pj = pd.DataFrame(rm["projection"])
        st.dataframe(pj, use_container_width=True, hide_index=True)
        st.line_chart(pj.set_index("wave")[["low_pct", "projected_pct", "high_pct"]])
        st.caption(f"📌 {rm.get('projection_note')}")
        if rm.get("multimodal_effects"):
            st.info(" **Multimodal effects, reported separately.** These are discounted "
                    "on the text-answer axis above, because a bake-off measures a "
                    "multimodal read with the asset in context.")
            st.dataframe(pd.DataFrame(rm["multimodal_effects"]),
                         use_container_width=True, hide_index=True)
    if rm.get("assumptions"):
        with st.expander("Assumptions"):
            for a in rm["assumptions"]:
                st.caption(f"- {a}")

    # ---------------- briefs ----------------
    if ss.get("opt_briefs_data"):
        st.subheader(f"📝 Content briefs ({len(ss['opt_briefs_data'])})")
        for b in ss["opt_briefs_data"]:
            with st.expander(f"[{b.get('horizon')}] {b.get('title')} · "
                             f"{b.get('evidence_class')}"):
                st.write("**Objective**", b.get("objective"))
                st.write("**Why it works**", b.get("why_it_works"))
                st.write("**Deliverable**", b.get("deliverable"))
                c1, c2 = st.columns(2)
                c1.markdown("**Must contain**")
                for x in (b.get("must_contain") or []):
                    c1.caption(f"☑ {x}")
                c2.markdown("**Must NOT contain**")
                for x in (b.get("must_not_contain") or []):
                    c2.caption(f"✗ {x}")
                st.write("**Placement**", b.get("placement"))
                st.write("**Corroboration**", b.get("corroboration"))
                st.write("**Success metric**", b.get("success_metric"))
                st.write("**Measurement**", b.get("measurement"))
                for c in (b.get("caveats") or []):
                    st.caption(f"📌 {c}")
                if b.get("proven_content"):
                    st.markdown("**⭐ Tested content pattern**")
                    st.text(b["proven_content"][:2000])

    # ---------------- gate + emit ----------------
    st.divider()
    st.subheader("🚦 Strategy sign-off")
    if "opt_gate" not in ss:
        ss["opt_gate"] = HumanGate("strategy", lambda fb=None: ss["opt_strategy"])
        ss["opt_gate"].generate()
    st.caption(f"Status: **{ss['opt_gate'].state.status}**")
    if blockers:
        st.error(f"🔴 {len(blockers)} blocker(s) unresolved. Approving will emit a "
                 f"strategy the critic rejected.")
    g1, g2 = st.columns(2)
    if g1.button("✅ Approve", key="opt_ok"):
        ss["opt_gate"].approve("analyst")
        st.success("Approved.")
    if g2.button("📦 Emit", key="opt_emit", disabled=not ss["opt_gate"].is_final()):
        clean = {k: v for k, v in strat.items() if k != "_si"}
        save_artifact(p, t, "geo_strategy", clean,
                      {"model": model, "n_actions": len(strat["actions"]),
                       "n_blockers": len(blockers)})
        save_artifact(p, t, "roadmap", rm, {"model": model})
        save_briefs(p, t, ss.get("opt_briefs_data") or [], {"model": model})
        save_artifact(p, t, "action_register",
                      score_actions(strat["actions"], weights), {"weights": weights})
        st.success("Emitted geo_strategy + roadmap + content_briefs + action_register.")

    with st.expander("🧾 Diagnostic dump"):
        st.code(json.dumps({
            "evidence_summary": strat.get("evidence_summary"),
            "retrieval_is_primary_constraint": strat.get("retrieval_is_primary_constraint"),
            "social": (strat.get("evidence_summary") or {}).get("social"),
            "pillar_summary": strat.get("pillar_summary"),
            "horizon_summary": strat.get("horizon_summary"),
            "evidence_classes": strat.get("evidence_summary_by_class"),
            "action_summary": summary(strat["actions"]),
            "n_blockers": len(blockers), "n_majors": len(majors),
            "critic_review": strat.get("critic_review"),
            "validation_errors": strat.get("validation_errors"),
            "projection": rm.get("projection"),
            "dependency_cycles": rm.get("dependency_cycles"),
            "inputs_hash": strat.get("inputs_hash"),
        }, indent=2, default=str), language="json")