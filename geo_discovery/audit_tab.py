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
from store import show_audit_samples,save_artifact

def _dom(brand):
    import re
    return re.sub(r"[^a-z0-9]", "", str(brand).lower()) + ".com"


def render(ctx):

    st.header("🔬 Module 2: Audit")
    p, t = ctx.parent_brand, ctx.target_brand
    gateway = ctx.gateway
    ss = st.session_state
    loaded = ss.get("loaded_artifacts", {})

    # ---- PREREQUISITE GUARD (uses `loaded`, previously dead) ----
    missing = [a for a in ("query_master", "competitive_set", "claims_register")
               if not loaded.get(a)]
    if missing:
        st.warning(f"⚠️ Missing Discover artifacts: {missing}. "
                   f"Auditing against an incomplete set wastes spend.")

    baselines_dir = project_dir(p, t) / "baselines"
    existing_baselines = sorted(baselines_dir.glob("*_LOCKED.json")) \
        if baselines_dir.exists() else []
    if existing_baselines:
        st.success(f"✅ {len(existing_baselines)} frozen baseline(s). "
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

    ##### New inserted code
    # ---- ENGINE PREFLIGHT (uses `gateway`, previously dead) ----
    with st.expander("🔌 Engine preflight — test before you spend"):
        if st.button("Test engines", key="aud_preflight"):
            for e in engines:
                try:
                    gateway.generate(e["model"], "Reply with the single word OK.",
                                     module="audit", task="preflight", max_tokens=16)
                    web = "not tested"
                    if "on" in web_modes:
                        try:
                            wr = (gateway.generate_grounded(
                                "What is today's date?", model=e["model"],
                                module="audit", task="preflight_web")
                                  if e["provider"] == "vertex" else
                                  gateway.generate_web_litellm(
                                      e["model"], "What is today's date?",
                                      module="audit", task="preflight_web"))
                            web = f"{len(wr['sources'])} sources"
                        except Exception as ex:
                            web = f"❌ {str(ex)[:70]}"
                    st.write(f"✅ **{e['name']}** · text ok · web: {web} · "
                             f"vision: {'vision' in e['supports']}")
                except Exception as ex:
                    st.error(f"❌ {e['name']}: {str(ex)[:160]}")

    # ---- VISION SELF-TEST: prove images are actually read ----
    if extract_images:
        with st.expander("👁️ Vision self-test (required before trusting V/VT results)",
                         expanded=("aud_vision_check" not in ss)):
            if st.button("Run canary + blank-control", key="aud_vcheck"):
                from audit_runner import vision_selftest
                ss["aud_vision_check"] = vision_selftest(gateway, vision_model)
            vc = ss.get("aud_vision_check")
            if vc:
                if vc.get("vision_live") is True and not vc.get("confabulates"):
                    st.success(f"✅ Vision live · image_tokens="
                               f"{vc.get('image_tokens_reported')}")
                elif vc.get("vision_live") is True and vc.get("confabulates"):
                    st.error("⚠️ Vision live BUT the model invents codes on a blank "
                             "image — E7/E8 results will be untrustworthy.")
                elif vc.get("vision_live") is False:
                    st.error("❌ Vision NOT live — image payload never reached the "
                             "model. Do not interpret visual gaps.")
                else:
                    st.info(vc.get("note", "inconclusive"))
                st.json(vc)

    # ---- RESUME VISIBILITY (uses `ss`, previously dead) ----
    from audit_runner import audit_progress, domains_from_sources
    done_n, planned_n = audit_progress(raw_path, queries, engines, n_runs, web_modes)
    if done_n:
        st.info(f"↻ Resumable: {done_n} run(s) already complete · "
                f"{max(planned_n - done_n, 0)} remaining of {planned_n}")

    brand_domains = domains_from_sources(loaded.get("ingested_sources"))
    if brand_domains:
        st.caption(f"🏷️ Brand-owned domains (for provenance): {', '.join(brand_domains)}")
    else:
        st.caption("🏷️ No brand domains detected — `self_referential_pct` will read 0. "
                   "Ingest brand URLs in Discover.")
    comp_domains_txt = st.text_input(
        "Competitor domains (comma-separated, optional — improves provenance)",
        value=",".join(f"{_dom(c)}" for c in competitors[:5]), key="aud_compdom")
    competitor_domains = [d.strip() for d in comp_domains_txt.split(",") if d.strip()]
    alias_overrides = {c.get("brand"): c.get("aliases", [])
                       for c in (cs or []) if isinstance(c, dict) and c.get("aliases")}

    #####################



    calls, est = estimate_cost(len(queries), len(engines), n_runs, web_modes)
    st.info(f"~{calls:,} calls ≈ ${est}")

    if st.button("🚀 Run Audit", key="aud_run", disabled=not(queries and engines)):
        prog = st.progress(0.0); status = st.empty()
        def cb(done, total, warn=False):
            prog.progress(done/max(total,1))
            status.write(f"{done}/{total} · ${ctx.meter.total_cost():.3f}" + ("  ⚠️ budget breached" if warn else ""))
        try:

            run_audit(gateway, queries, engines, t, competitors,
                      n_runs=n_runs, web_modes=web_modes, out_path=raw_path,
                      use_llm_classify=use_llm_classify, judge_model=judge_model,
                      extract_images=extract_images, vision_model=vision_model,
                      budget=budget, hard_stop=hard_stop, progress_cb=cb,
                      brand_domains=brand_domains,
                      competitor_domains=competitor_domains,
                      alias_overrides=alias_overrides,
                      manifest_cb=lambda man: save_artifact(
                          p, t, "audit_manifest",
                          {**man, "vision_selftest": ss.get("aud_vision_check"),
                           "config": {"intents": sel_intents, "engines": sel_engines,
                                      "n_runs": n_runs, "web_modes": list(web_modes)}},
                          {"metrics_spec_version": 1}))
            st.success("Audit complete.")
        except BudgetExceeded as e:
            st.error(f"🛑 {e}")

    if Path(raw_path).exists():
        @st.cache_data(ttl=60, show_spinner=False)
        def _load_rows(path, mtime):
            import json as _j
            return [_j.loads(l) for l in open(path, encoding="utf-8")]

        rows = _load_rows(raw_path, Path(raw_path).stat().st_mtime)
        att = [i for r in rows for i in r.get("image_insights", [])]
        ok = [i for i in att if i.get("fetched")]
        st.subheader("🖼️ Returned-image interrogation")
        if not att:
            st.caption("No images were attempted (none returned, or option disabled).")
        else:
            st.write(f"{len(ok)} of {len(att)} attempts fetched & interrogated "
                     f"({sum(1 for i in att if i.get('cache_hit'))} cache hits).")
            if ok:
                st.dataframe(pd.DataFrame(ok).astype(str), use_container_width=True)
            else:
                st.error("⚠️ Images were found but NONE were successfully read — "
                         "run the vision self-test above.")

    st.subheader("Visual interrogation (optional)")
    vt = st.file_uploader("Ad(s)/comparison image(s)", type=["png","jpg","jpeg","webp"],
                          accept_multiple_files=True, key="aud_vt")
    vq = st.text_input("Question", "Which product is better and why?", key="aud_vq")
    if st.button("Interrogate visuals", key="aud_vt_btn") and vt:
        imgs = [(f.read(), "image/png" if f.name.lower().endswith("png") else "image/jpeg") for f in vt]
        st.write(visual_interrogation(gateway, imgs, vq, engines, vision_model))
    ######## New code for backfill ########
    # ---- BACKFILL legacy runs to metrics_spec v1 ----
    if Path(raw_path).exists():
        with st.expander("🔧 Backfill legacy audit rows (no LLM cost)"):
            st.caption("Recomputes mention / polarity / rank / refusal / image counts "
                       "from stored response text. Creates a timestamped .bak first.")
            from backfill import backfill_audit
            c1, c2 = st.columns(2)
            if c1.button("🔍 Dry-run (report only)", key="aud_bf_dry"):
                with st.spinner("Scanning…"):
                    ss["aud_bf"] = backfill_audit(p, t, dry_run=True, target_brand=t)
            if c2.button("✍️ Apply backfill", key="aud_bf_go"):
                with st.spinner("Rewriting…"):
                    ss["aud_bf"] = backfill_audit(p, t, dry_run=False, target_brand=t)
                st.cache_data.clear()
                st.success("Backfilled. Re-open Benchmark to see updated metrics.")
            rpt = ss.get("aud_bf")
            if rpt and not rpt.get("error"):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mention rate", f"{rpt['mention_rate_after']:.1%}",
                          f"{(rpt['mention_rate_after'] - rpt['mention_rate_before']) * 100:+.1f}pp")
                m2.metric("First-mention", f"{rpt['first_rate_after']:.1%}",
                          f"{(rpt['first_rate_after'] - rpt['first_rate_before']) * 100:+.1f}pp")
                m3.metric("Negatively framed",
                          rpt["polarity"].get("neg", 0))
                m4.metric("Advisory referral", rpt["advisory_referral"])
                if rpt.get("unknown_brands_in_legacy"):
                    st.warning(f"Brands in old data but not in the current competitive "
                               f"set: {rpt['unknown_brands_in_legacy']} — their mentions "
                               f"are no longer tracked.")
                for label, k in (("✅ Newly detected mentions", "examples_gained"),
                                 ("❌ Mentions removed (false substring match)", "examples_lost"),
                                 ("⚠️ Negatively framed mentions", "examples_negated")):
                    if rpt.get(k):
                        st.markdown(f"**{label}**")
                        for e in rpt[k]:
                            st.caption(f"[{e.get('engine')}] {e.get('query')} — "
                                       f"{e.get('sentence') or e.get('why', '')}")
                st.json(rpt, expanded=False)
            elif rpt:
                st.error(rpt["error"])

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
        st.subheader("🔍 Sample audit responses")
        _t1, _t2 = st.tabs(["Random sample", "❌ Brand absent"])
        with _t1:
            show_audit_samples(raw_path, n=2, key="aud_samp",
                               stratify=("engine", "intent"), target_brand=t)
        with _t2:
            st.caption("Where the brand did NOT surface — the highest-signal reading.")
            show_audit_samples(raw_path, n=2, key="aud_samp_abs",
                               stratify=("engine", "intent"), only_absent=True,
                               target_brand=t)