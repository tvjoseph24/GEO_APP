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
import theme
def _dom(brand):
    import re
    return re.sub(r"[^a-z0-9]", "", str(brand).lower()) + ".com"


def render(ctx):

    theme.section("Audit", "Capturing Machine Reality — what the engines actually say")
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
    # W-78: the role:'self' entry must not be tracked as a competitor, and its domains
    # must never be suggested as COMPETITOR domains — that would classify the brand's
    # own citations as competitor_owned.
    _self_entry = next((c for c in (cs or [])
                        if isinstance(c, dict)
                        and (c.get("role") == "self"
                             or (c.get("brand") or "").strip().lower()
                             == t.strip().lower())), None)
    competitors = [c["brand"] for c in (cs or [])
                   if isinstance(c, dict) and c.get("brand") and c is not _self_entry]
    st.success(f"Loaded {len(qm)} queries, {len(competitors)} competitor(s)"
               + (f" · self entry: {_self_entry.get('brand')}" if _self_entry else
                  " · ⚠️ no role:'self' entry in competitive_set"))

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
    # ---- VISION SELF-TEST: prove images are actually read ----
    # Independent of `extract_images` — visual EXPERIMENTS (E7/E8/E11/E12) require this
    # regardless of whether returned-image interrogation is enabled during the audit.
    with st.expander("Vision self-test — required before any visual experiment",
                     expanded=("aud_vision_check" not in ss)):
        st.caption("Sends a synthetic image containing a known code, plus a BLANK "
                   "control. Confirms the model read the real image and did not invent "
                   "content on the blank one. Two calls, a fraction of a cent. A visual "
                   "experiment run without this can produce a null that looks like "
                   "evidence.")
        _vc1, _vc2 = st.columns([1, 2])
        if _vc1.button("Run canary + blank control", key="aud_vcheck"):
            from audit_runner import vision_selftest
            with st.spinner("Testing vision…"):
                ss["aud_vision_check"] = vision_selftest(gateway, vision_model)
            # persist immediately — vision_ready() reads audit_manifest, and waiting for
            # a full audit run to write it means the Experiment tab reports "no self-test
            # on record" even after one has passed
            _man = load_artifact(p, t, "audit_manifest") or {}
            _man["vision_selftest"] = ss["aud_vision_check"]
            save_artifact(p, t, "audit_manifest", _man,
                          {"metrics_spec_version": 1,
                           "note": "vision self-test recorded standalone"})
            st.success("Result saved to audit_manifest — the Experiment tab will read it.")

        vc = ss.get("aud_vision_check") or (
            (loaded.get("audit_manifest") or {}).get("vision_selftest"))
        if vc:
            _vc2.caption(f"Model: `{vc.get('model')}` · checked "
                         f"{str(vc.get('checked_utc', ''))[:19]}")
            if vc.get("vision_live") is True and not vc.get("confabulates"):
                st.success(f"Vision live · image_tokens={vc.get('image_tokens_reported')} "
                           f"· blank control clean. Visual experiments are safe to run.")
            elif vc.get("vision_live") is True and vc.get("confabulates"):
                st.error("Vision live BUT the model invents codes on a blank image. "
                         "E7/E8 results would be untrustworthy — it will 'read' claims "
                         "that are not on the pack.")
            elif vc.get("vision_live") is False:
                st.error("Vision NOT live — the image payload never reached the model. "
                         "Do not interpret any visual result until this is fixed.")
            else:
                st.info(vc.get("note", "inconclusive"))
            if st.toggle("Show self-test detail", key="aud_vc_detail"):
                st.json(vc)
        else:
            st.warning("Not yet run. `vision_ready()` in the Experiment tab will block "
                       "visual designs until it passes.")

    # ---- RESUME VISIBILITY (uses `ss`, previously dead) ----
    from audit_runner import audit_progress, domains_from_sources
    done_n, planned_n = audit_progress(raw_path, queries, engines, n_runs, web_modes)
    if done_n:
        st.info(f"↻ Resumable: {done_n} run(s) already complete · "
                f"{max(planned_n - done_n, 0)} remaining of {planned_n}")

    brand_domains = sorted(set(domains_from_sources(loaded.get("ingested_sources")))
                           | {str(d).lower() for d in
                              ((_self_entry or {}).get("domains") or [])})
    if brand_domains:
        st.caption(f"Brand-owned domains (for provenance): {', '.join(brand_domains)}")
    else:
        st.caption("No brand domains detected — `self_referential_pct` will read 0. "
                   "Add a role:'self' entry with domains to competitive_set.")
    _declared = [d for c in (cs or []) if isinstance(c, dict) and c is not _self_entry
                 for d in (c.get("domains") or [])]
    comp_domains_txt = st.text_input(
        "Competitor domains (comma-separated — improves provenance)",
        value=",".join(sorted(set(_declared)))
              or ",".join(_dom(c) for c in competitors[:5]),
        key="aud_compdom")
    competitor_domains = [d.strip().lower() for d in comp_domains_txt.split(",")
                          if d.strip() and d.strip().lower() not in brand_domains]
    alias_overrides = {c.get("brand"): c.get("aliases", [])
                       for c in (cs or []) if isinstance(c, dict) and c.get("aliases")}

    # ---- SOURCE PACK (W-56): per-category classification of the long tail ----
    from source_profile import (SOURCE_TYPES, build_source_pack, load_source_pack,
                                pack_summary, save_source_pack, unclassified_domains)
    from audit_runner import build_source_classifier

    _pack = ss.get("aud_pack") or load_source_pack(p, t)
    with st.expander("🗂️ Source classification pack — whose voice are the engines citing?",
                     expanded=False):
        st.caption("Provenance drives the Corroboration pillar and the source hit-list. An "
                   "unclassified citation is invisible to both, and an affiliate site "
                   "counted as `other` understates how much of the brand's narrative is "
                   "written by parties it cannot correct. This pack is generated per "
                   "CATEGORY, reviewed by you, and re-applied to existing runs with "
                   "`backfill.py` — so it can be regenerated freely without re-auditing.")

        if Path(raw_path).exists():
            _fn_now = build_source_classifier(brand_domains, competitor_domains, _pack)
            _unc, _meta = unclassified_domains(raw_path, _fn_now)
            _u = _meta["unclassified_citations"]
            _tot = max(_meta["total_citations"], 1)
            m1, m2, m3 = st.columns(3)
            m1.metric("Citations", f"{_tot:,}")
            m2.metric("Unclassified", f"{_u / _tot:.0%}",
                      f"{_meta['distinct_unclassified']:,} domains")
            m3.metric("Pack", f"{(_pack or {}).get('n_domains', 0)} domains",
                      f"v{(_pack or {}).get('pack_version', '—')}")
            if _u / _tot > 0.25:
                st.warning(f"⚠️ {_u / _tot:.0%} of citations are unclassified. Every "
                           f"provenance percentage — including `voice_control_ratio`, "
                           f"which decides whether the headline reads *citation control is "
                           f"adequate* or *third parties outweigh the brand* — is "
                           f"unreliable until this is below about 15%.")
            if _meta["excluded_titles"]:
                st.caption(f"{_meta['excluded_titles']:,} citation(s) carry an unresolved "
                           f"page title rather than a domain. That is a redirect failure "
                           f"and no pack can classify it.")

            _top = st.slider("Domains to classify (by citation count)", 40, 400, 200,
                             step=20, key="aud_pack_n")
            _cover = sum(r["citations"] for r in _unc[:_top])
            st.caption(f"The top {min(_top, len(_unc))} domains cover {_cover:,} of "
                       f"{_u:,} unclassified citations "
                       f"({_cover / max(_u, 1):.0%}) · ≈{-(-min(_top, len(_unc)) // 50)} "
                       f"LLM call(s)")

            if _pack:
                st.caption("↻ Generating again ADDS to the existing pack. The domain list "
                           "is citation-sorted, so a second pass sees only the tail — "
                           "replacing rather than merging would discard the "
                           "high-citation classifications already made.")
            if st.button("🗂️ Generate pack", key="aud_pack_go", disabled=not _unc):
                _pb = st.progress(0.0, text="Classifying…")
                with st.spinner("Classifying sources…"):
                    ss["aud_pack"] = build_source_pack(
                        gateway, _unc[:_top], ctx.category, model=judge_model,
                        existing=_pack,
                        progress_cb=lambda x: _pb.progress(x, text="Classifying…"))
                _pack = ss["aud_pack"]
                _pb.progress(1.0, text="Done")
                ctx.meter.flush()
                st.success(
                    f"{_pack['n_domains']} domain(s) in the pack — "
                    f"{_pack.get('n_new_this_pass', 0)} new this pass, "
                    f"{_pack.get('n_carried_forward', 0)} carried forward "
                    f"({_pack.get('n_reviewed_preserved', 0)} human-reviewed preserved) · "
                    f"{_pack['n_structural']} classified structurally at no cost · "
                    f"{_pack['n_needs_review']} need review")

        if _pack:
            _ps = pack_summary(_pack)
            st.write("**Citations by classification** — this is what changes the provenance "
                     "figures")
            st.dataframe(pd.DataFrame(
                [{"Source type": k, "Domains": _pack["by_type"].get(k, 0),
                  "Citations": v,
                  "Definition": SOURCE_TYPES.get(k, "")[:90]}
                 for k, v in sorted((_pack.get("citations_by_type") or {}).items(),
                                    key=lambda x: -x[1])]),
                use_container_width=True, hide_index=True)

            _need = [e for e in _pack["entries"].values() if e.get("needs_review")]
            if _need:
                st.warning(f"{len(_need)} low-confidence classification(s) — review these, "
                           f"because a misclassified high-citation domain moves the "
                           f"headline.")
            _edit = st.data_editor(
                pd.DataFrame(sorted(_pack["entries"].values(),
                                    key=lambda e: -e.get("citations", 0)))
                [["domain", "citations", "source_type", "confidence", "method",
                  "reason", "reviewed"]],
                column_config={
                    "source_type": st.column_config.SelectboxColumn(
                        options=list(SOURCE_TYPES), required=True),
                    "reviewed": st.column_config.CheckboxColumn(
                        help="Tick to mark as human-confirmed. A reviewed entry is "
                             "trusted regardless of the model's confidence."),
                    "citations": st.column_config.NumberColumn(disabled=True),
                    "domain": st.column_config.TextColumn(disabled=True),
                    "confidence": st.column_config.NumberColumn(disabled=True),
                    "method": st.column_config.TextColumn(disabled=True)},
                hide_index=True, use_container_width=True, height=420,
                key="aud_pack_edit")

            pc1, pc2 = st.columns(2)
            if pc1.button("💾 Save pack", key="aud_pack_save"):
                for _r in _edit.to_dict("records"):
                    _e = _pack["entries"].get(_r["domain"])
                    if _e:
                        _e["source_type"] = _r["source_type"]
                        _e["reviewed"] = bool(_r["reviewed"])
                        if _e["reviewed"]:
                            _e.pop("needs_review", None)
                from collections import Counter as _C
                _pack["by_type"] = dict(_C(e["source_type"]
                                           for e in _pack["entries"].values()))
                _cb = _C()
                for _e in _pack["entries"].values():
                    _cb[_e["source_type"]] += _e.get("citations", 0)
                _pack["citations_by_type"] = dict(_cb)
                _pack["n_needs_review"] = sum(1 for e in _pack["entries"].values()
                                              if e.get("needs_review"))
                save_source_pack(p, t, _pack)
                ss["aud_pack"] = _pack
                st.success("Saved. Run `python backfill.py --parent "
                           f"{p} --target {t}` to re-classify existing runs, then re-run "
                           f"Benchmark — provenance percentages will change.")
            pc2.caption("The pack applies to NEW runs immediately. Existing runs need a "
                        "backfill, which is free and resumable.")
        else:
            st.info("No pack yet. Without one, the long tail of cited domains stays "
                    "unclassified and provenance is unreliable.")

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
                      source_pack=_pack,
                      manifest_cb=lambda man: save_artifact(
                          p, t, "audit_manifest",
                          {**man, "vision_selftest": ss.get("aud_vision_check"),
                           "config": {"intents": sel_intents, "engines": sel_engines,
                                      "n_runs": n_runs, "web_modes": list(web_modes)}},
                          {"metrics_spec_version": 1}))
            ctx.meter.flush()
            st.success("Audit complete.")
        except BudgetExceeded as e:
            ctx.meter.flush()  # a stopped run's spend is still real
            st.error(f"🛑 {e}")

    # ---- W-57: READ THE IMAGES THE ENGINES RETURNED ----
    if Path(raw_path).exists():
        import visual_enrich as VE
        _vplan = VE.plan(p, t, raw_path,
                         page_limit=ss.get("aud_ve_pages", 60),
                         allow_video_frames=ss.get("aud_ve_video", False))
        with st.expander(
                f"🖼️ Read the images the engines returned — "
                f"{_vplan['distinct_pages']} page(s), {_vplan['fetchable']} fetchable",
                expanded=bool(_vplan["fetchable"] and not _vplan["cached_images"])):
            st.caption("Engines return page URLs far more often than direct image URLs — "
                       "a retailer product page, an editorial review — so the audit "
                       "records them and this pass resolves them to images and reads "
                       "them. It answers whether the brand's packaging is legible to a "
                       "machine as it appears IN THE WILD, which is a different question "
                       "from whether an asset you upload is legible.")
            st.info(_vplan["note"])

            _c1, _c2, _c3 = st.columns(3)
            _c1.metric("Returned pages", _vplan["distinct_pages"],
                       f"{_vplan['total_returns']:,} returns")
            _c2.metric("Fetchable", _vplan["fetchable"],
                       f"{_vplan['unfetchable']} not")
            _c3.metric("Already cached", _vplan["cached_images"],
                       help="Engines return the same product shot repeatedly, so the "
                            "cache is shared across the whole corpus.")

            if _vplan["unfetchable_by_reason"]:
                with st.expander(f"Why {_vplan['unfetchable']} page(s) cannot be read"):
                    st.dataframe([{"Reason": k, "Pages": v} for k, v in
                                  _vplan["unfetchable_by_reason"].items()],
                                 use_container_width=True, hide_index=True)
                    st.caption("Declared rather than retried. Social platforms serve a "
                               "login wall to any unauthenticated fetch, and a video "
                               "platform's preview image is a frame rather than a "
                               "product photograph — so neither tells you anything "
                               "about pack legibility. This is a limit of the medium, "
                               "and the report states it.")
                    if _vplan["unfetchable_by_host"]:
                        st.caption("By host: " + " · ".join(
                            f"{k} ({v})" for k, v in
                            _vplan["unfetchable_by_host"].items()))

            _v1, _v2, _v3 = st.columns(3)
            _pl = _v1.slider("Pages to consider", 20, 300, 60, step=20,
                             key="aud_ve_pages",
                             help="Most-returned first. An image surfaced ninety times "
                                  "matters more than one surfaced twice.")
            _mi = _v2.slider("Images to read (vision calls)", 5, 120, 40, step=5,
                             key="aud_ve_max")
            _v3.checkbox("Include video previews", value=False, key="aud_ve_video",
                         help="A video platform's og:image is a single frame. Fetchable, "
                              "but it says little about pack legibility — so it is "
                              "excluded by default.")
            st.caption(f"Page fetching costs nothing. ≈{_mi} vision call(s) ≈ "
                       f"${_mi * 0.0004:.2f}, less the "
                       f"{_vplan['cached_images']} already cached.")

            if st.button("🖼️ Fetch pages and read the images",
                         key="aud_ve_go", type="primary",
                         disabled=not _vplan["fetchable"]):
                _vb = st.progress(0.0, text="Starting…")
                with st.spinner("Reading…"):
                    ss["aud_ve"] = VE.run(
                        gateway, p, t, raw_path, vision_model,
                        page_limit=int(_pl), max_images=int(_mi),
                        allow_video_frames=ss.get("aud_ve_video", False),
                        budget_ok=lambda: ctx.meter.total_cost() < budget,
                        progress_cb=lambda x, m="": _vb.progress(x, text=m or "Reading…"))
                _vb.progress(1.0, text="Done")
                ctx.meter.flush()
                st.cache_data.clear()
                st.rerun()

            _ve = ss.get("aud_ve") or load_artifact(p, t, "visual_enrichment")
            if _ve:
                _ext = _ve.get("extraction") or {}
                _m = st.columns(4)
                _m[0].metric("Pages fetched", _ext.get("pages_fetched", 0),
                             f"of {_ext.get('pages_seen', 0)}")
                _m[1].metric("Images found", _ext.get("images_found", 0))
                _m[2].metric("Read successfully",
                             _ve.get("images_read_successfully", 0),
                             f"{_ve.get('cache_hits', 0)} from cache")
                _m[3].metric("Rows enriched", _ve.get("rows_written", 0))
                st.caption(f"📌 {_ve.get('coverage_note')}")
                if _ve.get("warning"):
                    st.error(f"⚠️ {_ve['warning']}")
                if _ext.get("yield_by_host"):
                    st.caption("Images recovered by host: " + " · ".join(
                        f"{k} ({v})" for k, v in
                        list(_ext["yield_by_host"].items())[:8]))
                if _ext.get("failed_by_reason"):
                    with st.expander("Pages fetched that carried no usable image"):
                        st.dataframe([{"Reason": k, "Pages": v} for k, v in
                                      _ext["failed_by_reason"].items()],
                                     use_container_width=True, hide_index=True)
                _ok = [i for i in (ss.get("aud_ve") or {}).get("insights", [])
                       if i.get("fetched") and not i.get("error")]
                if _ok:
                    with st.expander(f"What was read from {len(_ok)} image(s)"):
                        st.dataframe(pd.DataFrame([
                            {"brand": i.get("brand_visible"),
                             "product": i.get("product_type"),
                             "role": i.get("image_role"),
                             "legibility": i.get("text_legibility"),
                             "claims": "; ".join(str(x) for x in
                                                 (i.get("visible_claims") or []))[:90],
                             "unreadable": str(i.get("what_is_unreadable") or "")[:90],
                             "tokens": i.get("image_tokens")}
                            for i in _ok]).astype(str),
                                     use_container_width=True, hide_index=True)
                st.caption("Re-run Benchmark to fold this into the visual findings.")

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