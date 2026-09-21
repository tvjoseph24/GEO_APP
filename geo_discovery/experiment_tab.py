############## new Code ################
"""Experiment tab — hypothesis-driven, power-gated, parity-checked, resumable.

Flow: load hypotheses → build designs + frozen competitor field → Gate 1 (manipulation
check) → parity check → pilot (real baseline) → power gate → full run → Gate 2 → emit.
"""
import json

import pandas as pd
import streamlit as st

from config import MODEL_REGISTRY
from human_gate import HumanGate
from store import load_artifact, project_dir

from experiment.competitor_enrich import (build_rich_competitor_context,
                                          freeze_manifest, parity_report,
                                          render_competitor_blocks,
                                          render_competitor_context,
                                          save_competitor_context)
from experiment.design_from_hypothesis import designs_from_hypotheses
from experiment.power import mde, n_for_effect, power_verdict, run_pilot
from experiment.rag_runner import (aggregate_from_raw, completed_runs, run_experiment,
                                   run_progressive_experiment, BudgetStop)
from experiment.results import assemble_findings, save_experiment_results
from experiment.signal_designer import _brand_free_queries, design_from_signals
from experiment.variant_gen import (attach_visual_variant,
                                    generate_progressive_variants,
                                    generate_text_variants,
                                    generate_visual_variants)
import theme
from experiment import visual as VIS


ARM_LETTERS = "ABCDE"
def _limits_for(chk):
    from experiment.results import interpretation_limits
    return interpretation_limits((chk or {}).get("notes"))

def _wkey(prefix, d, i):
    """Stable widget key: identity-based, not position-based."""
    return f"{prefix}_{d.get('design_id') or d.get('gap_id') or i}"

def _arm_keys(variants):
    return [k for k in ARM_LETTERS if f"variant_{k}" in (variants or {})]

def _treatment_text(d):
    """Text of the strongest arm — used for the parity check."""
    v = d.get("variants") or {}
    keys = _arm_keys(v)
    if not keys:
        return ""
    last = keys[-1] if len(keys) > 1 else keys[0]
    return (v.get(f"variant_{last}") or {}).get("text", "") or ""

def _sanitise_designs(designs):
    """store._clean cannot serialise raw bytes, and _hypothesis duplicates the
    gap_hypotheses artifact. Assets are written to blobs separately."""
    return [VIS.strip_for_persistence(d) for d in designs]

def render(ctx):
    theme.section("Experiment",
                  "Proving causation before anything is published")
    p, t = ctx.parent_brand, ctx.target_brand
    gateway = ctx.gateway
    variant_model = ctx.models.get("variant_model", "gemini-2.5-flash")
    sel_engines = ctx.models.get("audit_engines", ["gemini-2.5-flash", "azure/gpt-4o"])
    ss = st.session_state

    # ================= 1. RECOVERY =================
    loaded = ss.get("loaded_artifacts", {})
    if "validated_findings_only" in loaded and "exp_findings" not in ss:
        st.success("✅ Existing Experiment findings loaded (read-only). Re-run to regenerate.")
        with st.expander("🧪 Validated Findings", expanded=True):
            for f in loaded["validated_findings_only"]:
                st.markdown(f"**{f.get('experiment_id')}** "
                            f"({f.get('design_id', f.get('gap_id', ''))}) — "
                            f"{str(f.get('verdict', '')).upper()} · "
                            f"{f.get('pooled_lift_pp')}pp")
                if f.get("per_engine"):
                    st.dataframe(pd.DataFrame(f["per_engine"]), use_container_width=True)
        st.divider()

    # ================= 2. LOAD INPUTS =================
    metrics = load_artifact(p, t, "benchmark_metrics")
    hyps = load_artifact(p, t, "gap_hypotheses") or []
    gaps = load_artifact(p, t, "gap_register_actionable") or []
    claims = load_artifact(p, t, "claims_register") or []
    comps = load_artifact(p, t, "competitive_set") or []
    qm = load_artifact(p, t, "query_master") or []
    hitlist = load_artifact(p, t, "source_hitlist") or []

    if not metrics:
        st.error("No benchmark metrics found. Complete Benchmark first.")
        return

    # competitors, with the role:'self' entry excluded exactly once
    comp_names, alias_ov = [], {}
    for c in comps:
        name = c.get("brand") if isinstance(c, dict) else c
        if not name:
            continue
        is_self = (isinstance(c, dict) and c.get("role") == "self") or \
                  name.strip().lower() == t.strip().lower()
        if isinstance(c, dict) and c.get("aliases"):
            alias_ov[name] = c["aliases"]
        if not is_self:
            comp_names.append(name)

    audit_raw = str(project_dir(p, t) / "audit_raw.jsonl")
    exp_raw = project_dir(p, t) / "experiment_raw.jsonl"

    if hyps:
        n_ready = sum(1 for h in hyps if h.get("status") == "ready")
        st.success(f"🧠 {len(hyps)} hypothesis(es) from Benchmark — {n_ready} ready to test. "
                   f"Competing against {len(comp_names)} competitor(s).")
    else:
        st.warning("No hypotheses found. Re-run Benchmark to generate them "
                   "(will fall back to heuristic signal designs).")

    # ================= 3. WEAKNESS SIGNALS =================
    with st.expander("⚠️ Weakness signals + baselines from Benchmark"):
        for f in metrics.get("weakness_flags", []):
            icon = "🔴" if f.get("severity") == "high" else "🟡"
            st.warning(f"{icon} **{f['type']}**: {f['detail']}")
        st.dataframe(pd.DataFrame(metrics.get("mention_by_intent", [])),
                     use_container_width=True)

    # ================= 4. CONFIG =================
    st.subheader("⚙️ Run configuration")
    c1, c2, c3 = st.columns(3)
    n_runs = c1.number_input("Runs per arm (N)", 20, 400, 100, step=10, key="exp_n")
    temperature = c2.slider("Temperature", 0.5, 1.0, 0.8, key="exp_temp")
    target_pos = c3.selectbox("Target position", ["balanced", "top", "bottom", "shuffle"],
                              key="exp_pos",
                              help="'balanced' counterbalances position WITHIN each arm "
                                   "(top/bottom/shuffle by run index) — the correct "
                                   "default. Fixed positions bake in order bias.")
    _has_visual = any(d.get("needs_visual") for d in ss.get("exp_designs", []))
    comp_mode = st.selectbox(
        "Competitor field weight", ["full", "thin", "none"],
        index=1 if _has_visual else 0, key="exp_compmode",
        help="A visual arm may be 30 words against 900 words of competitor field, which "
             "buries the brand by construction. 'thin' keeps each competitor's opening "
             "line only. Changing this makes runs incomparable with earlier ones.")
    c4, c5, c6 = st.columns(3)
    budget = c4.number_input("Budget cap ($)", 1.0, 500.0, 8.0, key="exp_budget")
    hard_stop = c5.checkbox("Enforce cap mid-run", value=True, key="exp_hard",
                            help="A cap stops spend mid-arm. Untick only to let a run "
                                 "overshoot with a warning — not recommended.")
    min_lift = c6.number_input("Min lift for 'validated' (pp)", 1.0, 30.0, 5.0,
                               key="exp_minlift")

    engines = [{"name": m, "model": m, "provider": MODEL_REGISTRY[m]["provider"],
                "supports": MODEL_REGISTRY[m]["supports"]} for m in sel_engines]
    base_need = next((r["mention_rate"] for r in metrics.get("mention_by_intent", [])
                      if r.get("intent") == "need"), 0.05)
    st.caption(f"At the audit baseline ({base_need:.1%}) and N={n_runs}, the minimum "
               f"detectable effect is ≈{mde(base_need, n_runs)*100:.1f}pp. The pilot "
               f"below measures the TRUE experimental baseline, which is usually far "
               f"higher — and therefore harder to move.")
    neg_ctrl = st.checkbox(
        "Add negative-control arm to E0 (recommended)", value=True, key="exp_negctrl",
        help="Arm A + equal-length irrelevant prose. If it also lifts, you measured "
             "'more text wins', not enrichment. One extra arm; settles the question.")

    # ================= 5. PROPOSE =================
    st.subheader("🎯 Build designs")
    src = st.radio("Design source",
                   ["Hypotheses (Benchmark-derived — recommended)",
                    "Weakness signals (heuristic)"],
                   key="exp_src", horizontal=True)
    target_words = st.slider("Document length per arm (words)", 100, 400, 180,
                             key="exp_words",
                             help="All arms AND all competitor docs are matched to this. "
                                  "Length is otherwise a confound.")

    if st.button("🎯 Build designs + frozen competitor field", key="exp_propose"):
        with st.spinner("Building…"):
            prog = st.progress(0.0, text="Enriching competitors…")
            comp_docs = build_rich_competitor_context(
                gateway, comps, ctx.category, t, audit_raw, model=variant_model,
                target_words=target_words, alias_overrides=alias_ov,
                progress_cb=lambda x: prog.progress(x * 0.35,
                                                    text="Enriching competitors…"))
            ss["exp_comp_docs"] = comp_docs
            ss["exp_comp_blocks"] = render_competitor_blocks(comp_docs)
            ss["exp_comp_ctx"] = render_competitor_context(comp_docs)
            ss["exp_comp_manifest"] = freeze_manifest(comp_docs, ss["exp_comp_ctx"])
            save_competitor_context(p, t, comp_docs, ss["exp_comp_manifest"])

            if src.startswith("Hypotheses") and hyps:
                designs, skipped, not_testable = designs_from_hypotheses(
                    hyps, target_brand=t, query_master=qm)
            else:
                designs = design_from_signals(metrics, qm, claims, t, gap_register=gaps)
                skipped, not_testable = [], []
            ss["exp_skipped"] = skipped
            ss["exp_not_testable"] = not_testable
            if not_testable:
                from store import save_artifact
                save_artifact(p, t, "substantiation_backlog", not_testable,
                              {"source": "gap_hypotheses",
                               "note": "Pillar-6 actions — not experiments"})

            bf = _brand_free_queries(qm)
            for i, d in enumerate(designs):
                prog.progress(0.35 + 0.65 * (i + 1) / max(len(designs), 1),
                              text=f"Drafting arms {i + 1}/{len(designs)}")
                if not d.get("question"):
                    d["question"] = bf[0] if bf else f"best {ctx.category}".strip()
                _tpl = (((d.get("_hypothesis") or {}).get("experiment") or {})
                        .get("template") or d.get("experiment_id") or "E0").upper()
                if _tpl in VIS.VISUAL_TEMPLATES:
                    # identical text across arms, generated once — the image is the variable
                    d["variants"] = generate_visual_variants(
                        gateway, d, claims, t, model=variant_model,
                        hypothesis=d.get("_hypothesis"), target_words=35,
                        category=ctx.category, parent_brand=p)
                    d["needs_visual"] = True
                    d["visual_locked"] = True
                else:
                    d["variants"] = generate_text_variants(
                        gateway, d, claims, t, model=variant_model,
                        hypothesis=d.get("_hypothesis"), target_words=target_words,
                        hitlist=hitlist, audit_raw=audit_raw,
                        parent_brand=p, category=ctx.category,
                        comp_docs=ss.get("exp_comp_docs"),
                        include_negative_control=(neg_ctrl and _tpl == "E0"))
            ss["exp_designs"] = designs
            ss.pop("exp_pilot", None)
            ss.pop("exp_gate", None)
            prog.progress(1.0, text="Done")

        man = ss["exp_comp_manifest"]
        if man.get("contamination_leaks"):
            st.error(f"🚨 Brand contamination in competitor docs: "
                     f"{man['contamination_leaks']} — this biases every arm. Edit the "
                     f"competitor field before running.")
        if man.get("llm_failures"):
            st.warning(f"Competitor structuring failed for {list(man['llm_failures'])}")
        n_ok = sum(1 for d in ss["exp_designs"]
                   if ((d.get("variants") or {}).get("check") or {}).get("ok"))
        st.success(f"{len(ss['exp_designs'])} design(s) built · {n_ok} passed the "
                   f"manipulation check · {len(ss.get('exp_comp_blocks', []))} frozen "
                   f"competitor doc(s).")

    if ss.get("exp_skipped"):
        with st.expander(f"⏭️ {len(ss['exp_skipped'])} hypothesis(es) blocked "
                         f"— review before forcing"):
            st.dataframe(pd.DataFrame(ss["exp_skipped"]), use_container_width=True)
            st.caption("Common blockers: degenerate arms (no testable difference), "
                       "unmeasurable metric, or a target query naming the brand. "
                       "Power warnings no longer block — the pilot decides.")
            if st.checkbox("Include blocked hypotheses anyway (marked FORCED)",
                           key="exp_force_hyps"):
                extra, _, _ = designs_from_hypotheses(hyps, target_brand=t,
                                                   query_master=qm, only_ready=False)
                have = {d["design_id"] for d in ss["exp_designs"]}
                add = [d for d in extra if d["design_id"] not in have]
                for d in add:
                    d["variants"] = generate_text_variants(
                        gateway, d, claims, t,
                        model=variant_model,
                        hypothesis=d.get("_hypothesis"),
                        target_words=target_words,
                        hitlist=hitlist,
                        audit_raw=audit_raw,
                        parent_brand=p,
                        category=ctx.category,
                        comp_docs=ss.get("exp_comp_docs"),
                        include_negative_control=(
                                ss.get("exp_negctrl", True) and
                                ((d.get("_hypothesis") or {}).get("experiment") or {})
                                .get("template", d.get("experiment_id")) == "E0"))
                if add:  # ← see Fix 2
                    ss["exp_designs"] += add
                    st.success(f"Added {len(add)} forced design(s) — check each "
                               f"carefully; they failed validation for a reason.")
                    st.rerun()

    if ss.get("exp_not_testable"):
        nt = ss["exp_not_testable"]
        st.info(f"🔬 **{len(nt)} substantiation gap(s)** — not experiments. The engines "
                f"already assert these facts; the brand has never substantiated them. "
                f"Saved to `substantiation_backlog` → Optimise Pillar 6.")
        with st.expander("Claims-Substantiation backlog"):
            st.dataframe(pd.DataFrame(nt)[["cluster_id", "title", "severity",
                                           "assertion_to_substantiate", "next_step"]],
                         use_container_width=True)
            st.caption("Highest-ROI content work in the framework: facts the engines "
                       "state on the brand's behalf with no substantiation trail. In a "
                       "regulated category this is also a compliance exposure.")


    # ---- guard: nothing below runs until designs exist ----
    if "exp_designs" not in ss:
        st.info("👆 Click **Build designs + frozen competitor field** to begin.")
        return
    # ================= VISION READINESS =================
    _visual_designs = [d for d in ss["exp_designs"] if d.get("needs_visual")]
    if _visual_designs:
        _manifest = load_artifact(p, t, "audit_manifest") or {}
        _ready, _probs = VIS.vision_ready(engines, _manifest)
        with st.expander(f"{'✅' if _ready else '⚠️'} Vision readiness — "
                         f"{len(_visual_designs)} visual design(s)",
                         expanded=not _ready):
            for pr in _probs:
                st.warning(pr)
            if _ready:
                st.success("Vision self-test passed and at least one selected engine "
                           "supports vision.")
            st.caption("A visual experiment where the image payload never arrives "
                       "produces a null that looks like evidence. Image tokens are "
                       "verified on every run and the finding is marked INVALID if "
                       "none are reported.")
            _nonvis = [e["name"] for e in engines
                       if "vision" not in e.get("supports", [])]
            if _nonvis:
                st.info(f"{_nonvis} declare no vision support. Their arms will run "
                        f"TEXT-ONLY and are not comparable to a vision engine's arms.")

    st.divider()

    # ================= 6. FROZEN COMPETITOR FIELD =================
    st.subheader("🏢 Frozen competitor field")
    st.caption("What the target competes against. Identical across all arms and all "
               "experiments — this is the controlled field.")
    edited = st.text_area("Competitor content (markdown)", ss.get("exp_comp_ctx", ""),
                          height=220, key="exp_comp_edit")
    ec1, ec2 = st.columns([1, 3])
    if ec1.button("💾 Update field", key="exp_comp_save"):
        ss["exp_comp_ctx"] = edited
        ss["exp_comp_blocks"] = [b.strip() for b in edited.split("\n\n[") if b.strip()]
        ss["exp_comp_manifest"] = freeze_manifest(ss.get("exp_comp_docs", {}), edited)
        st.success("Updated — applies to the next run.")
    ec2.caption(f"Hash `{(ss.get('exp_comp_manifest') or {}).get('content_hash')}` · "
                f"{(ss.get('exp_comp_manifest') or {}).get('words')} words")

    # ================= 7. GATE 1 — DESIGNS =================
    st.subheader("🚦 Gate 1: Approve designs")
    st.caption("⛔ regulatory = no override · 🚫 no-op = overridable, expect a null · "
               "⚠️ runnable, the finding carries the caveat · ✅ clean")
    for i, d in enumerate(ss["exp_designs"]):
        chk = (d.get("variants") or {}).get("check") or {}
        sev = chk.get("severity", "clean")
        forced_ok = ss.get(f"force_{d.get('design_id')}", False)
        runnable = sev in ("clean", "caveated") or (sev == "invalid" and forced_ok)
        sig_icon = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(
            d.get("priority_signal"), "")
        icon = {"clean": "✅", "caveated": "⚠️", "invalid": "🚫", "legal": "⛔",
                "not_testable": "🔬"}.get(sev, "✅")
        label = (f"{icon}{'FORCED' if d.get('forced') else ''} {sig_icon} "
                 f"{d.get('design_id', '')} · {d.get('experiment_id', '?')} "
                 f"{d.get('experiment_name', '')} · {d.get('pillar') or ''}")

        with st.expander(label, expanded=d.get("include", False)):
            if sev == "legal":
                st.error("⛔ **BLOCKED — regulatory.** Variant text becomes a content "
                         "brief, so this cannot be generated:\n"
                         + "\n".join(f"• {x}" for x in chk.get("legal", [])))
                st.caption("Fix: substantiate the claim and add it to the Claims "
                           "Register, or remove it from the hypothesis. No override.")
            elif sev == "not_testable":
                st.info("🔬 Not an experiment — routed to the Pillar-6 backlog above.")
            elif sev == "invalid":
                st.error("🚫 **No information possible:**\n"
                         + "\n".join(f"• {x}" for x in chk.get("invalid", [])))
                ss[f"force_{d.get('design_id')}"] = st.checkbox(
                    "Run anyway (I accept this will likely be a null)",
                    key=_wkey("exp_forceone", d, i))
            elif sev == "caveated":
                st.warning("⚠️ **Runnable — the finding will carry these caveats:**\n"
                           + "\n".join(f"• {x}" for x in chk.get("caveats", [])))
                for lim in _limits_for(chk):
                    st.caption(f"📌 {lim}")
            else:
                st.success(f"✅ Clean manipulation · isolates **{d.get('isolates')}**")

            cs = (d.get("variants") or {}).get("citation_source")
            cd = (d.get("variants") or {}).get("citation_domains") or []
            if cs == "unavailable":
                st.warning("No verified third-party domains — arms built WITHOUT "
                           "citations. Corroboration is untested here.")
            elif cs:
                st.caption(f"Citation domains (via `{cs}`): {', '.join(cd)}")

            st.caption(f"Mechanism `{d.get('mechanism')}` · pillar {d.get('pillar')} · "
                       f"horizon {d.get('horizon')} · covers "
                       f"{len(d.get('gap_ids') or [])} gap(s)")
            st.markdown(f"**Hypothesis** — {d.get('hypothesis')}")
            st.markdown(f"**Question** — “{d.get('question')}”"
                        + ("" if d.get("question_in_audit")
                           else "  ⚠️ *not in the audit set — no measured baseline*"))
            pr = d.get("prediction") or {}
            st.caption(f"Predicts {pr.get('baseline_pp')}% → {pr.get('expected_pp')}% "
                       f"on `{pr.get('metric')}`")
            if d.get("falsification"):
                st.markdown(f"**Falsification** — {d['falsification']}")
            if d.get("claims_allowed"):
                st.caption("Claims approved: " +
                           " | ".join(str(x)[:70] for x in d["claims_allowed"][:4]))

            v = d.get("variants") or {}
            feats = (chk.get("checks") or {}).get("features") or {}
            if feats:
                st.caption("Structure per arm (the manipulation, made visible)")
                st.dataframe(pd.DataFrame([
                    {"arm": k, "words": (chk["checks"].get("words") or {}).get(k),
                     **{f: ("✓" if f in feats.get(k, []) else "") for f in
                        ["prose", "bullets", "citations", "table", "capsule"]}}
                    for k in (v.get("arm_keys") or feats)]),
                    use_container_width=True, hide_index=True)
            for k in (v.get("arm_keys") or _arm_keys(v)):
                st.markdown(f"**Arm {'N (negative control)' if k == 'N' else k}**")
                st.text((v[f"variant_{k}"].get("text") or "")[:2000])
            if v.get("manipulation_summary"):
                st.caption(f"Difference: {v['manipulation_summary']}")

            gc1, gc2 = st.columns(2)
            d["include"] = gc1.checkbox("Include", value=d.get("include", False),
                                        key=_wkey("exp_inc", d, i),
                                        disabled=not runnable)
            if d.get("visual_locked"):
                d["mode"] = "A/B (fast)"
                gc2.caption(f"Mode: A/B — locked. Progressive rewrites any design to "
                            f"E1 and would replace the visual manipulation with a text "
                            f"structure ladder.")
            else:
                d["mode"] = gc2.radio("Mode", ["A/B (fast)", "Progressive (E1)"],
                                      index=1 if str(d.get("mode", "")).startswith("Prog")
                                      else 0, key=_wkey("exp_mode", d, i),
                                      horizontal=True)
            # ---------------- visual assets, per arm ----------------
            _tpl_here = (d.get("experiment_id") or "").upper()
            if _tpl_here in VIS.VISUAL_TEMPLATES or d.get("needs_visual"):
                st.markdown("**🖼️ Creative assets — one per arm**")
                _plan = VIS.ASSET_PLAN.get(_tpl_here, VIS.ASSET_PLAN["E7"])
                st.caption(f"{_tpl_here}: {_plan['note']}")

                if _tpl_here == "E11":
                    _n = st.slider("Creative variants to test", 2, 5,
                                   int(d.get("n_visual_arms", 3)),
                                   key=_wkey("exp_nvis", d, i),
                                   help="Each arm carries a different creative. Text is "
                                        "byte-identical across all arms, so any lift is "
                                        "attributable to the creative.")
                    if _n != d.get("n_visual_arms"):
                        d["n_visual_arms"] = _n
                        st.caption("Rebuild designs to regenerate the arm set.")

                _arm_keys_here = (d.get("variants") or {}).get("arm_keys") or ["A", "B"]
                _slots = [k for k in _arm_keys_here
                          if not (_tpl_here == "E12" and k == "A")]
                _cols = st.columns(min(len(_slots), 3))
                for _j, _arm in enumerate(_slots):
                    with _cols[_j % len(_cols)]:
                        _req = _arm in _plan["required"]
                        _up = st.file_uploader(
                            f"Arm {_arm}{' *' if _req else ''}",
                            type=["png", "jpg", "jpeg", "webp", "mp4", "mov"],
                            key=_wkey(f"exp_vis_{_arm}", d, i))
                        if _up is not None:
                            VIS.attach_asset(d, _arm, _up.read(),
                                             VIS.mime_for(_up.name), label=_up.name)
                        _a = VIS.assets(d).get(_arm)
                        if _a:
                            if not _a.get("is_video"):
                                st.image(_a["bytes"], use_container_width=True)
                            st.caption(f"{_a.get('label', '')} · "
                                       f"{_a.get('width', '?')}×{_a.get('height', '?')} · "
                                       f"`{_a['digest']}`")
                            if st.button("Remove", key=_wkey(f"exp_visrm_{_arm}", d, i)):
                                VIS.detach_asset(d, _arm)
                                st.rerun()

                if _tpl_here == "E12":
                    st.caption("Arm A deliberately carries NO asset — E12 isolates "
                               "image presence.")

                _vchk = VIS.validate(
                    d, _tpl_here,
                    {k: (d.get("variants") or {}).get(f"variant_{k}", {}).get("text", "")
                     for k in _arm_keys_here})
                d["visual_check"] = _vchk
                if _vchk["invalid"]:
                    st.error("🚫 **Visual design invalid:**\n"
                             + "\n".join(f"• {x}" for x in _vchk["invalid"]))
                elif _vchk["caveats"]:
                    st.warning("⚠️ **Runnable with caveats:**\n"
                               + "\n".join(f"• {x}" for x in _vchk["caveats"]))
                elif VIS.assets(d):
                    st.success("✅ Visual manipulation is clean — text identical across "
                               "arms, assets distinct.")
                if VIS.assets(d):
                    st.dataframe(pd.DataFrame(VIS.summary(d)),
                                 use_container_width=True, hide_index=True)

    active = [d for d in ss["exp_designs"] if d.get("include")]
    _blocked_vis = [d["design_id"] for d in active
                    if d.get("needs_visual")
                    and not (d.get("visual_check") or {}).get("ok", True)]
    if _blocked_vis:
        st.error(f"🚫 {_blocked_vis} have an invalid visual design — attach the required "
                 f"assets, or deselect them.")
        active = [d for d in active if d["design_id"] not in _blocked_vis]
    _missing_assets = [d["design_id"] for d in active
                       if d.get("needs_visual") and not VIS.assets(d)]
    if _missing_assets:
        st.error(f"🚫 {_missing_assets} are visual designs with no creative attached.")
        active = [d for d in active if d["design_id"] not in _missing_assets]
    if not active:
        st.info("Select at least one design above to continue.")
        return

    # ================= 8. PARITY =================
    st.subheader("⚖️ Parity check")
    st.caption("Arms and competitor docs must match on length and structure, or the "
               "experiment measures document quality instead of your variable.")
    _v0 = active[0].get("variants") or {}
    _tpl = _v0.get("experiment_id", active[0].get("experiment_id", "E0"))
    _keys = _v0.get("arm_keys") or ["A", "B"]
    _treat = (_v0.get(f"variant_{[k for k in _keys if k != 'N'][-1]}") or {}).get("text", "")
    pr_df = parity_report(_treat, ss.get("exp_comp_blocks") or [ss.get("exp_comp_ctx", "")],
                          exp_id=_tpl,
                          control_text=(_v0.get("variant_A") or {}).get("text", ""))
    st.dataframe(pr_df, use_container_width=True)
    if pr_df.attrs.get("note"):
        st.caption(pr_df.attrs["note"])
    st.dataframe(pr_df, use_container_width=True)
    parity_warnings = pr_df.attrs.get("warnings") or []
    for w in parity_warnings:
        st.warning(f"⚠️ {w}")

    # ================= 9. PILOT =================
    st.subheader("🔬 Step 1 — Pilot (arm A only)")
    st.caption("The audit baseline is the wrong p0: experiments inject the brand document "
               "with web OFF, so arm-A rates are far higher. The pilot measures the true "
               "baseline and the N you actually need.")
    pc1, pc2, pc3 = st.columns([2, 1, 1])
    pilot_design = pc1.selectbox("Pilot this design", [d["design_id"] for d in active],
                                 key="exp_pilot_d")
    pilot_n = pc2.number_input("Runs/engine", 10, 60, 20, key="exp_pilot_n")
    pc3.caption(f"≈${int(pilot_n) * len(engines) * 0.002:.2f}")

    if st.button("▶️ Run pilot", key="exp_pilot_go"):
        d0 = next((d for d in active if d["design_id"] == pilot_design), None)
        ctrl = ((d0 or {}).get("variants") or {}).get("variant_A", {}).get("text", "")
        if not ctrl:
            st.error("Selected design has no arm-A text — re-build designs.")
        else:
            with st.spinner(f"Pilot: {pilot_n} × {len(engines)} runs…"):
                pl = run_pilot(gateway, ctrl, ss["exp_comp_ctx"], d0["question"], t,
                               comp_names, engines=engines, n=int(pilot_n),
                               temperature=temperature, target_pos=target_pos,
                               alias_overrides=alias_ov)
                pl["design_id"] = pilot_design
                ss["exp_pilot"] = pl

    if ss.get("exp_pilot"):
        pl = ss["exp_pilot"]
        m = st.columns(4)
        m[0].metric("Arm-A mention", f"{pl['p0_mention']:.0%}")
        m[1].metric("Arm-A first-rec", f"{pl['p0_first']:.0%}")
        m[2].metric("Pilot runs", pl["n_total"])
        m[3].metric(f"MDE @ N={n_runs}",
                    "n/a" if pl.get("at_floor")
                    else f"{mde(pl['p0_mention'], n_runs) * 100:.1f}pp")
        if pl["ceiling_risk"]:
            st.error("⬆️ CEILING — the brand is mentioned in ≥90% of arm-A runs, because "
                     "its own document is in context. `mention_rate` cannot move. Switch "
                     "the primary metric to `first_rec_rate`.")
        if pl["floor_risk"]:
            st.error("⬇️ FLOOR — the brand is absent even with its own document injected. "
                     "**Check first that every arm actually names the brand** (Gate 1 → "
                     "arm text). A control with no brand name is not a control.")
            if pl.get("mde_note"):
                st.caption(pl["mde_note"])
        pcol1, pcol2 = st.columns(2)
        pcol1.write("**Per engine**")
        pcol1.dataframe(pd.DataFrame(pl["per_engine"]).T, use_container_width=True)
        pcol2.write("**MDE by N (pp)**")
        pcol2.dataframe(pd.DataFrame(pl["mde_table"]).T, use_container_width=True)

        st.write("**Power verdict per selected design**")
        rows = []
        for d in active:
            pv = power_verdict(pl, d.get("_hypothesis") or
                               {"prediction": d.get("prediction") or {}}, int(n_runs))
            d["_power"] = pv
            rows.append({"design": d["design_id"], **pv})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        bad = [r["design"] for r in rows if r["verdict"] != "powered"]
        if bad:
            st.warning(f"⚠️ {bad} are not adequately powered at N={n_runs}. Options: "
                       f"raise N · switch the metric to first_rec_rate · run one engine "
                       f"at double N (same cost, half the MDE).")
    else:
        st.info("No pilot yet — run one before committing spend, or override below.")

    # ================= 10. FULL RUN =================
    st.subheader("🚀 Step 2 — Full run")
    force = st.checkbox("Override power / parity warnings", key="exp_force")
    if ss.get("exp_pilot") and not force:
        gated = [d for d in active if (d.get("_power") or {}).get("verdict") == "powered"]
    else:
        gated = active
    if parity_warnings and not force:
        st.error("🚫 Parity warnings unresolved. Fix the competitor field or tick override.")
        gated = []

    # ---- real cost per call, measured from the meter (was hardcoded at $0.002,
    # ---- which under-estimated the last run by ~3.5x) ----
    _ev = [e for e in ctx.meter.events if e.get("task") == "rag_run"]
    cost_per_call = (sum(e["cost_usd"] for e in _ev) / len(_ev)) if _ev else 0.007
    st.caption(f"Observed cost per experiment call: ${cost_per_call:.4f} "
               f"(from {len(_ev)} prior call(s))" if _ev else
               "No prior calls — assuming $0.0070/call (200-word doc + competitor field)")

    _eng_cap = getattr(ctx.meter, "budget_usd", None)
    if _eng_cap:
        _left = _eng_cap - ctx.meter.total_cost()
        st.caption(f"Engagement cap: ${ctx.meter.total_cost():.2f} / ${_eng_cap:.2f} "
                   f"— **${_left:.2f} headroom** (always enforced, independent of the "
                   f"per-run cap)")

    def _n_arms(d):
        """Actual arm count. Progressive = 5-arm ladder; A/B may carry a negative
        control, and the check may have dropped an arm — so read arm_keys, never
        assume 2."""
        if str(d.get("mode", "")).startswith("Progressive"):
            return 5
        keys = (d.get("variants") or {}).get("arm_keys")
        return len(keys) if keys else 2

    calls = sum(_n_arms(d) for d in gated) * int(n_runs) * len(engines)
    resume = st.checkbox("Resume from existing raw file", value=True, key="exp_resume")
    _arm_space = sorted({k for d in gated
                         for k in ((d.get("variants") or {}).get("arm_keys") or ["A", "B"])})
    done = completed_runs(exp_raw,
                          design_ids=[d["design_id"] for d in gated],
                          arms=_arm_space,
                          models=[e["model"] for e in engines],
                          modes=["ab", "progressive"]) if resume else set()
    remaining = max(calls - len(done), 0)
    _est = remaining * cost_per_call

    st.info(f"{len(gated)} design(s) · {calls:,} total call(s), {remaining:,} remaining "
            f"≈ **${_est:.2f}**"
            + (f" · ↻ {len(done):,} already complete, will be skipped" if done else ""))

    if budget and _est > budget:
        if hard_stop:
            st.error(f"⛔ Estimated ${_est:.2f} exceeds the ${budget:.2f} cap and the cap "
                     f"IS enforced — the run will stop partway. Reduce N, drop a design, "
                     f"or raise the cap.")
        else:
            st.warning(f"⚠️ Estimated ${_est:.2f} exceeds the ${budget:.2f} cap, but the "
                       f"cap is NOT enforced — the run will overspend.")
    if _eng_cap and _est > (_eng_cap - ctx.meter.total_cost()):
        st.error(f"⛔ Estimated ${_est:.2f} exceeds the engagement headroom "
                 f"(${_eng_cap - ctx.meter.total_cost():.2f}) — the run WILL abort "
                 f"partway regardless of the per-run cap. Raise 'Engagement cap ($)' "
                 f"in the sidebar.")

    if st.button("✅ Approve & Run", key="exp_run", disabled=not gated):
        if not resume:
            exp_raw.unlink(missing_ok=True)
            done = set()
        raw_f = open(exp_raw, "a", encoding="utf-8")

        def raw_writer(rec):
            raw_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            raw_f.flush()

        prog = st.progress(0.0)
        status = st.empty()
        counter = [0]
        total = max(calls - len(done), 1)

        def cb():
            counter[0] += 1
            prog.progress(min(counter[0] / total, 1.0))
            status.write(f"{counter[0]}/{total} · ${ctx.meter.total_cost():.3f}")

        results = []
        try:
            for d in gated:
                spent = ctx.meter.total_cost()
                if budget and spent >= budget:
                    if hard_stop:
                        st.error(f"🛑 Budget ${budget:.2f} reached at ${spent:.2f} — "
                                 f"stopping before {d['design_id']}. Raise the cap and "
                                 f"re-run to resume.")
                        break
                    st.warning(f"⚠️ Over budget (${spent:.2f} / ${budget:.2f}) — "
                               f"continuing because the cap is not enforced.")

                q = (d.get("question") or "").strip()
                if not q:
                    st.warning(f"Skip {d['design_id']} — empty question")
                    continue

                if t.lower() in q.lower():
                    st.warning(f"Skip {d['design_id']} — question names '{t}'; "
                               f"mention rate would be meaningless")
                    continue

                def budget_check():
                    spent = ctx.meter.total_cost()
                    if budget and spent >= budget and hard_stop:
                        return f"Budget ${budget:.2f} reached at ${spent:.2f}"
                    return None

                kw = dict(comp_context=ss["exp_comp_ctx"], question=q, target_brand=t,
                          competitors=comp_names, n_runs=int(n_runs),
                          temperature=temperature, engines=engines,
                          target_pos=target_pos, progress_cb=cb, raw_writer=raw_writer,
                          alias_overrides=alias_ov, done=done,
                          budget_check=budget_check, comp_mode=comp_mode)

                status.write(f"Running {d['design_id']}…")
                if d.get("mode", "").startswith("Progressive"):
                    pv = generate_progressive_variants(
                        gateway, d, claims, t, model=variant_model,
                        hypothesis=d.get("_hypothesis"), target_words=target_words,
                        hitlist=hitlist,audit_raw=audit_raw,
                        parent_brand=p,
                        category=ctx.category,
                        comp_docs=ss.get("exp_comp_docs"))
                    d["progressive_arms"] = pv["arms"]
                    if not pv.get("check", {}).get("ok", True):
                        st.warning(f"{d['design_id']} progressive arms failed the "
                                   f"manipulation check: {pv['quality_warning']}")
                    er = run_progressive_experiment(gateway, d, pv["arms"], **kw)
                    er["mode"] = "progressive"
                else:
                    er = run_experiment(gateway, d, d["variants"], **kw)
                    er["mode"] = "ab"

                er.update({"design_id": d["design_id"], "signal": d.get("signal"),
                           "hypothesis": d.get("hypothesis"),
                           "mechanism": d.get("mechanism"), "pillar": d.get("pillar"),
                           "horizon": d.get("horizon"),
                           "prediction": d.get("prediction"),
                           "power": d.get("_power"), "pilot": ss.get("exp_pilot"),
                           "isolated_variable": d.get("isolates"),
                           "claims_used": d.get("claims_allowed"),
                           "check": (d.get("variants") or {}).get("check"),
                           "treatment_notes": ((d.get("variants") or {})
                                               .get("check") or {}).get("notes"),
                           "competitor_hash": (ss.get("exp_comp_manifest") or {})
                           .get("content_hash")})

                # authoritative aggregates: run_variant counts only NEW runs, so with
                # resume the in-memory numbers under-report. The JSONL is the truth.
                agg, agg_meta = aggregate_from_raw(exp_raw, er["design_id"],
                                                   mode=er["mode"])
                if agg:
                    for eng in engines:
                        for arm_key, stats in (agg.get(eng["model"]) or {}).items():
                            er["engines"].setdefault(eng["name"], {}) \
                                .setdefault(arm_key, {}).update(stats)
                    er["aggregated_from_raw"] = True
                    er["aggregation_meta"] = agg_meta
                    if agg_meta.get("contamination_warning"):
                        st.warning("⚠️ " + agg_meta["contamination_warning"])
                results.append(er)
        except BudgetStop as e:
            st.error(f"🛑 {e} — stopped mid-run. All completed runs are saved; "
                     f"raise the cap and re-run to resume.")
        except RuntimeError as e:  # ← ADD
            if "budget" in str(e).lower():
                st.error(f"🛑 **ENGAGEMENT cap hit** — {e}\n\nThis is the ceiling on the "
                         f"shared Meter (`Engagement cap ($)` in the sidebar), enforced "
                         f"on every LLM call regardless of the per-run cap. Raise it, "
                         f"then re-run — resume skips completed calls.")
            else:
                raise
        finally:
            raw_f.close()

        ss["exp_results"] = results
        try:
            ss["exp_findings"] = assemble_findings(
                results, t, min_lift=float(min_lift), raw_path=str(exp_raw))
        except Exception as e:
            st.error(f"Aggregation failed: {type(e).__name__}: {e}. Raw runs are safe in "
                     f"experiment_raw.jsonl — fix and press Run again (resume will skip "
                     f"all completed calls).")
            ss["exp_findings"] = []
        ss.pop("exp_gate", None)
        st.success(f"Complete — {len(results)} experiment(s).")

    # ================= 11. RESULTS =================
    if "exp_findings" not in ss:
        return

    st.divider()
    st.subheader("📊 Results")
    ICONS = {"validated": "✅", "validated_mention": "📈", "validated_rank": "🎯",
             "engine-specific": "🟡", "engine-specific_mention": "📈",
             "already_winning": "⬆️", "lost": "⬇️", "rejected": "❌", "no_lift": "➖"}

    for f in ss["exp_findings"]:
        v = f.get("verdict", "?")
        ec = f.get("evidence_class", "PROVEN")
        badge = {"PROVEN": "🟢 PROVEN", "PROVEN_WITH_CAVEAT": "🟡 PROVEN (caveated)",
                 "PROVEN_ENGINE_SPECIFIC": "🟠 ENGINE-SPECIFIC",
                 "INVALID": "🔴 INVALID"}.get(ec, ec)
        st.markdown(f"{ICONS.get(v, '?')} **{f.get('experiment_id')}** "
                    f"({f.get('design_id')}) — {str(v).upper()} · {badge} · "
                    f"best `{f.get('best_metric')}` · {f.get('pooled_lift_pp')}pp")
        for lim in (f.get("interpretation_limits") or []):
            st.caption(f"📌 {lim}")
        _vv = f.get("visual_validity")
        if _vv:
            _fn = {"OK": st.success, "PARTIAL": st.warning,
                   "INVALID": st.error}.get(_vv["verdict"], st.info)
            _fn(f"**Vision payload — {_vv['verdict']}.** {_vv['note']}")
        _pel = f.get("per_engine_lift_pp")
        if _pel and len(_pel) > 1:
            _he = f.get("headline_effect") or {}
            if _he.get("pooled_is_misleading"):
                st.warning(f"⚠️ Engines disagree — {_he.get('detail')}. The pooled "
                           f"{f.get('pooled_lift_pp')}pp describes neither engine.")

        # ---- binary metrics ----
        df = pd.DataFrame(f.get("per_engine") or [])
        if not df.empty:
            if f.get("mode") == "progressive":
                st.caption("**Cumulative arms** (A=prose → B=+bullets → C=+citations "
                           "→ D=+table → E=+capsule)")
                st.dataframe(df, use_container_width=True)
            else:
                for metric in ("mention_rate", "first_rec_rate"):
                    sub = df[df.metric == metric] if "metric" in df else pd.DataFrame()
                    if sub.empty:
                        continue
                    st.caption(f"**{metric}**")
                    cols = [c for c in ("engine", "arm", "A_rate", "B_rate", "lift_pp",
                                        "n_A", "n_B", "p_fdr", "significant",
                                        "is_negative_control") if c in sub.columns]
                    st.dataframe(sub[cols], use_container_width=True)
                    if "is_negative_control" in sub and sub.is_negative_control.any():
                        nc = sub[sub.is_negative_control]
                        if nc.significant.any():
                            st.error("⚠️ The NEGATIVE CONTROL also lifted — this measured "
                                     "'more text wins', not enrichment. Subtract its "
                                     "effect from the treatment lift.")
                        else:
                            st.success("✅ Negative control flat — the lift is attributable "
                                       "to enrichment, not volume.")

        # ---- rank: the metric with headroom when mention/first-rec are at ceiling ----
        rk = f.get("rank_analysis") or []
        if rk:
            st.caption("**mean_rank** (lower is better — ordinal position among brands "
                       "present; retains headroom when the binary metrics saturate)")
            rdf = pd.DataFrame(rk)
            st.dataframe(rdf[[c for c in ("engine", "arm", "A_mean_rank", "B_mean_rank",
                                          "improvement", "A_top1_pct", "B_top1_pct",
                                          "n_A", "n_B", "p_value", "significant")
                              if c in rdf.columns]], use_container_width=True)
            if v == "validated_rank":
                st.info("🎯 This finding comes from RANK, not mention rate. The brand was "
                        "already mentioned in nearly every run (its own document is in "
                        "context), so the treatment's effect shows as moving it *earlier* "
                        "among the brands named.")
        elif f.get("rank_error"):
            st.caption(f"Rank analysis unavailable: {f['rank_error']}")

        for w in (f.get("variance_warnings") or []):
            st.warning(f"⚠️ {w}")

        with st.expander(f"View {f.get('design_id')} — arms vs competitor field"):
            d = next((x for x in ss["exp_designs"]
                      if x.get("design_id") == f.get("design_id")), None)
            if d:
                st.caption(f"**Hypothesis** — {d.get('hypothesis')}")
                st.caption(f"**Falsification** — {d.get('falsification')}")
                col1, col2 = st.columns(2)
                col1.markdown("**Treatment arm**")
                col1.markdown(_treatment_text(d)[:1500])
                col2.markdown("**Competitor field**")
                col2.markdown(ss.get("exp_comp_ctx", "")[:1500])

    prog_r = [e for e in ss.get("exp_results", []) if e.get("mode") == "progressive"]
    if prog_r:
        from experiment.exp_stats import analyze_progressive, element_attribution
        st.subheader("🔬 Element attribution (E1)")
        for er in prog_r:
            rows = analyze_progressive(er, t)
            attr = element_attribution(rows)
            marg = pd.Series(attr).reindex(list("BCDE")).dropna()
            st.markdown(f"**{er.get('design_id')}**")
            a_first = sum((ev.get("A") or {}).get("is_first", 0)
                          for ev in er["engines"].values())
            a_n = sum((ev.get("A") or {}).get("n", 0) for ev in er["engines"].values())
            a_rate = a_first / max(a_n, 1)
            if marg.abs().sum() == 0:
                if a_rate >= 0.9:
                    st.info("⬆️ At CEILING — already winning in arm A")
                elif a_rate <= 0.1:
                    st.error("⬇️ At FLOOR — absent in all arms")
                else:
                    st.info("No differentiation between arms")
            else:
                st.bar_chart(marg)


    # ================= 12. GATE 2 — EMIT =================
    st.subheader("🚦 Gate 2: Validate & emit")
    if "exp_gate" not in ss:
        ss["exp_gate"] = HumanGate("results", lambda fb=None: ss["exp_findings"])
        ss["exp_gate"].generate()
    st.caption(f"Status: **{ss['exp_gate'].state.status}**")
    g1, g2 = st.columns(2)
    if g1.button("✅ Approve findings", key="exp_ok"):
        ss["exp_gate"].approve("analyst")
        st.success("Approved.")
    if g2.button("📦 Emit to Optimise", key="exp_emit",
                 disabled=not ss["exp_gate"].is_final()):
        _asset_man = VIS.save_assets(p, t, ss["exp_designs"])
        if _asset_man:
            save_artifact(p, t, "experiment_visual_assets", _asset_man,
                          {"note": "creative assets by design and arm"})
        save_experiment_results(
            p, t, ss["exp_results"], ss["exp_findings"],
            _sanitise_designs(ss["exp_designs"]),
            {"variant_model": variant_model, "n_runs": int(n_runs),
             "target_pos": target_pos, "min_lift": float(min_lift),
             "competitor_hash": (ss.get("exp_comp_manifest") or {}).get("content_hash")},
            competitor_context=ss.get("exp_comp_ctx", ""),
            competitor_docs=ss.get("exp_comp_docs", {}))
        st.success("Emitted → Optimise.")