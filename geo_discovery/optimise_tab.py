"""Optimise tab. Refactored from optimise_app.py → render(ctx)."""
import json
import streamlit as st
import pandas as pd
from store import load_artifact
from optimise.pillar_router import route_all, pillar_summary
from optimise.prioritizer import score_actions
from optimise.brief_gen import generate_brief
from optimise.strategy import build_strategy, build_roadmap, save_strategy
from human_gate import HumanGate

ACTIONABLE = {"validated", "validated_mention",
              "engine-specific", "engine-specific_mention"}

def _key(d):
    if not isinstance(d, dict):
        return None
    return d.get("design_id") or d.get("gap_id") or d.get("experiment_id")

def _variant_records(art):
    """experiment_variants wraps the design list under 'experiments'."""
    if isinstance(art, list):
        return art
    if isinstance(art, dict):
        return art.get("experiments") or []
    return []

def _text(x):
    return x.get("text", "") if isinstance(x, dict) else (x or "")

def _winning_content(v):
    """Records are flat: variant_B is a str (A/B) or arms holds progressive text."""
    if not v:
        return ""
    b = _text(v.get("variant_B"))
    if b:
        return b
    arms = v.get("arms") or {}
    if isinstance(arms, dict) and arms:
        return _text(arms.get("E") or arms[sorted(arms)[-1]])
    return ""

def render(ctx):
    st.header("🎯 Module 5: Optimise (Strategy)")
    #st.warning("⚠️ PROVISIONAL until Phase-2 competitor enrichment.")
    p, t = ctx.parent_brand, ctx.target_brand
    gateway = ctx.gateway
    brief_model = ctx.models.get("brief_model", "gemini-2.5-flash")
    ss = st.session_state
    loaded = ss.get("loaded_artifacts", {})
    if "geo_strategy" in loaded and "opt_strategy" not in ss:
        st.success("✅ Existing Strategy loaded (read-only). Re-run to regenerate.")
        with st.expander("🎯 GEO Strategy", expanded=True):
            st.json(loaded["geo_strategy"])
        with st.expander("📋 Roadmap"):
            rm = loaded.get("roadmap", {})
            for h in ["Now", "Next", "Later"]:
                if rm.get(h): st.write(f"**{h}**"); st.dataframe(pd.DataFrame(rm[h]), use_container_width=True)
        with st.expander("📝 Content Briefs"):
            for b in loaded.get("content_briefs", []):
                st.markdown(f"**{b.get('title', '(brief)')}** — {b.get('pillar')}")
                st.text((b.get("proven_content") or b.get("recommended_content", ""))[:1500])
        st.divider()

    all_findings = load_artifact(p, t, "validated_findings_only") or []
    variants = load_artifact(p, t, "experiment_variants") or []
    gaps = load_artifact(p, t, "gap_register_actionable") or []
    hitlist = load_artifact(p, t, "source_hitlist") or []

    if not all_findings:
        st.error("No validated findings. Complete Experiment first.");
        return

    findings = [f for f in all_findings if f.get("verdict") in ACTIONABLE]
    skipped = [f for f in all_findings if f.get("verdict") not in ACTIONABLE]
    if skipped:
        st.info(f"{len(skipped)} finding(s) excluded from strategy "
                f"({', '.join(sorted({f.get('verdict', '?') for f in skipped}))}).")
    if not findings:
        st.warning("No *actionable* findings — re-run Experiment with different designs.");
        return

    var_records = _variant_records(variants)
    comp_ctx = variants.get("competitor_context_text", "") if isinstance(variants, dict) else ""
    if not var_records:
        st.error("experiment_variants has no 'experiments' records — re-run Experiment.");
        return

    st.success(f"Loaded {len(findings)} actionable finding(s) "
               f"from {len(var_records)} design record(s).")

    gap_lookup = {g["gap_id"]: g for g in gaps if g.get("gap_id")}
    variant_lookup = {k: v for v in var_records if (k := _key(v))}

    # design_ids are S### (signal/claim-driven); gaps are G### — no overlap, so every
    # finding needs a synthetic gap. Recover signal + question from its design record.
    for f in findings:
        k = _key(f)
        if not k:
            continue
        f.setdefault("gap_id", k)
        d = variant_lookup.get(k, {})
        if not f.get("signal") and d.get("signal"):
            f["signal"] = d["signal"]
        f.setdefault("question", d.get("question"))
        gap_lookup.setdefault(k, {
            "gap_id": k,
            "signal": f.get("signal") or d.get("signal"),
            "source": "weakness_signal",
            "description": f"{d.get('signal', 'signal')}: {d.get('question', k)}",
            "question": d.get("question"),
        })
    unmatched = [_key(f) for f in findings if _key(f) not in variant_lookup]
    if unmatched:
        st.warning(f"No design record for {unmatched} — signal unavailable.")

    hit_domains = [h.get("domain") for h in hitlist if h.get("domain")]
    ########### new change #

    if st.button("🎯 Build Strategy + Briefs", key="opt_run"):
        with st.spinner("Building…"):
            f2 = score_actions(route_all(findings, gap_lookup))
            briefs = []
            for f in f2:

                v = variant_lookup.get(_key(f), {})
                winning = _winning_content(v)
                if not winning:
                    st.warning(f"No winning content for {_key(f)} — brief will be thin.")
                ####### End of change #####
            ss["opt_findings"] = f2
            ss["opt_briefs"] = briefs
            ss["opt_strategy"] = build_strategy(f2, briefs, t, p)
            ss["opt_roadmap"] = build_roadmap(f2)
            ss.pop("opt_gate", None)
        st.success("Built.")

    if "opt_strategy" in ss:
        st.subheader("Pillar Summary"); st.json(ss["opt_strategy"]["pillar_summary"])
        st.subheader("📋 Roadmap")
        for h in ["Now","Next","Later"]:
            items = ss["opt_roadmap"].get(h, [])
            if items: st.write(f"**{h}**"); st.dataframe(pd.DataFrame(items), use_container_width=True)
        st.subheader("Engine Overlays"); st.json(ss["opt_strategy"]["engine_overlays"])
        st.subheader("📝 Content Briefs")
        for b in ss["opt_briefs"]:
            with st.expander(f"[{b.get('horizon')}] {b.get('title','(brief)')}"):
                st.write("**Objective:**", b.get("objective"))
                st.write("**Why it works:**", b.get("why_it_works"))
                st.write("**⭐ Proven content:**"); st.text(b.get("proven_content","")[:2000])
                st.write("**Placement:**", b.get("placement"))
                st.write("**Corroboration:**", b.get("corroboration"))
                st.write("**Engines:**", b.get("engines_targeted"))

        st.subheader("🚦 Strategy Sign-off")
        if "opt_gate" not in ss:
            ss["opt_gate"] = HumanGate("strategy", lambda fb=None: ss["opt_strategy"]); ss["opt_gate"].generate()
        edited = st.text_area("Strategy (edit JSON)", json.dumps(ss["opt_strategy"], indent=2),
                              height=250, key="opt_edit")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Approve", key="opt_ok"): ss["opt_gate"].approve("analyst"); st.success("Approved.")
        with c2:
            if st.button("✏️ Save edits", key="opt_edit_btn"):
                try: ss["opt_strategy"] = json.loads(edited); ss["opt_gate"].edit(ss["opt_strategy"]); st.success("Saved.")
                except Exception as e: st.error(f"Invalid JSON: {e}")
        if ss["opt_gate"].is_final() and st.button("📦 Emit", key="opt_emit"):
            save_strategy(p, t, ss["opt_strategy"], ss["opt_roadmap"], ss["opt_briefs"],
                          {"brief_model": brief_model, "provisional": True})
            st.success("Emitted geo_strategy + roadmap + content_briefs.")