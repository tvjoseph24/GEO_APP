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

############ Optimise tab data #############
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
from experiment.results import ACTIONABLE_VERDICTS as ACTIONABLE



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
        _old = loaded["geo_strategy"]
        if not isinstance(_old.get("overall_strategy"), dict):
            st.warning("⚠️ A strategy from an older schema is on disk. Re-run to "
                       "regenerate — it is not displayed.")
        else:
            st.success("✅ Existing strategy loaded (read-only). Re-run to regenerate.")
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
    '''
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
    '''
    if st.button("🎯 Build Strategy + Briefs", key="opt_run"):
        with st.spinner("Building…"):
            f2 = score_actions(route_all(findings, gap_lookup))
            briefs, skipped_briefs = [], []
            prog = st.progress(0.0, text="Generating briefs…")
            for i, f in enumerate(f2):
                v = variant_lookup.get(_key(f), {})
                winning = _winning_content(v)
                if not winning:
                    skipped_briefs.append(_key(f))
                    prog.progress((i + 1) / max(len(f2), 1))
                    continue
                try:
                    b = generate_brief(gateway, f, winning, hit_domains, t,
                                       model=brief_model)
                    b["claims_required"] = v.get("claims_used", [])
                    b["design_id"] = _key(f)
                    b["question"] = f.get("question") or v.get("question")
                    b["competitor_field"] = (comp_ctx or "")[:1500]
                    briefs.append(b)
                except Exception as e:
                    st.warning(f"Brief failed for {_key(f)}: {str(e)[:120]}")
                prog.progress((i + 1) / max(len(f2), 1))
            prog.progress(1.0, text="Done")
            if skipped_briefs:
                st.warning(f"No winning content for {skipped_briefs} — "
                           f"no brief generated (check experiment_variants).")
            ss["opt_findings"] = f2
            ss["opt_briefs"] = briefs
            ss["opt_strategy"] = build_strategy(f2, briefs, t, p)
            ss["opt_roadmap"] = build_roadmap(f2)
            ss.pop("opt_gate", None)
        st.success(f"Built. {len(briefs)} brief(s) from {len(f2)} finding(s).")

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