"""Discover tab. Refactored from app.py → render(ctx). (Largest module.)"""
import json
from collections import Counter
import streamlit as st
import pandas as pd
from ingestion import ingest_doc, ingest_url_smart, ingest_image, ingest_video, estimate_video_cost
from extraction import (extract_positioning, extract_claims, dedupe_claims,
                        draft_query_master, extract_creative)
from grounded_mining import discover_competitors, mine_real_demand, merge_query_masters
from kpi import suggest_targets
from entity_check import check_entity_alignment
from human_gate import HumanGate
from store import save_artifact, save_audit_log, list_brands
from config import PRICING
import theme

def render(ctx):
    theme.section("Discover",
                  "Establishing Ground Truth — what the brand can substantiate")
    p, t = ctx.parent_brand, ctx.target_brand
    cat = ctx.category
    gateway = ctx.gateway
    ext_model = ctx.models.get("extraction","gemini-2.5-pro")
    q_model = ctx.models.get("query_mining","gemini-2.5-flash")
    mining_model = ctx.models.get("query_mining","gemini-2.5-flash")
    vision_model = ctx.models.get("vision","gemini-2.5-flash")
    brand_level = st.radio("GEO level", ["product","corporate"], key="disc_level")
    ss = st.session_state
    loaded = ss.get("loaded_artifacts", {})
    if "ingested_sources" in loaded and "disc_gates" not in ss:
        st.success("✅ Existing Discovery loaded.")
        with st.expander("📄 Previously ingested sources"):
            st.dataframe(pd.DataFrame([{k: v for k, v in s.items() if k != "text"}
                                       for s in loaded["ingested_sources"]]),
                         use_container_width=True)
    if "positioning_layers" in loaded and "disc_gates" not in ss:
        st.success("✅ Existing Discovery artifacts loaded (read-only). Re-run any step to regenerate & edit.")
        with st.expander("📄 Positioning (3-layer)"):
            st.json(loaded["positioning_layers"])
        with st.expander("📄 Claims Register"):
            st.dataframe(pd.DataFrame(loaded.get("claims_register", [])), use_container_width=True)
        with st.expander("📄 Query Master"):
            st.dataframe(pd.DataFrame(loaded.get("query_master", [])), use_container_width=True)
        with st.expander("📄 Competitive Set"):
            _cs = loaded.get("competitive_set", [])
            _self = [c for c in _cs if isinstance(c, dict) and c.get("role") == "self"]
            _comp = [c for c in _cs if not (isinstance(c, dict) and c.get("role") == "self")]
            if _self:
                st.caption(f"Self: {_self[0].get('brand')} · "
                           f"domains {_self[0].get('domains') or '(none)'} · "
                           f"aliases {_self[0].get('aliases') or '(derived)'}")
            st.dataframe(pd.DataFrame(_comp), use_container_width=True)
        with st.expander("📄 KPI Targets"):
            st.dataframe(pd.DataFrame(loaded.get("kpi_targets", [])), use_container_width=True)
        st.divider()
    ss.setdefault("disc_sources", []); ss.setdefault("disc_gates", {})

    st.info("💡 Upload material specific to the TARGET brand; point URLs at the brand site.")

    # ---- ingest ----
    st.subheader("1 · Ingest sources")
    c1, c2 = st.columns(2)
    with c1:
        up = st.file_uploader("Docs (PDF/DOCX/PPTX/TXT)", type=["pdf","docx","pptx","txt","md"],
                              accept_multiple_files=True, key="disc_docs")
        if st.button("Ingest docs", key="disc_ing_doc") and up:
            for f in up: ss["disc_sources"].append(ingest_doc(f.read(), name=f.name))
            st.success(f"{len(up)} docs.")
    with c2:
        urls = st.text_area("Brand URLs (one per line)", key="disc_urls")
        if st.button("Ingest URLs", key="disc_ing_url") and urls.strip():
            for u in [x.strip() for x in urls.splitlines() if x.strip()]:
                ss["disc_sources"].append(ingest_url_smart(u))
            st.success("URLs ingested.")

    imgs = st.file_uploader("Static ads (PNG/JPG)", type=["png","jpg","jpeg","webp"],
                            accept_multiple_files=True, key="disc_imgs")
    if st.button("Ingest images", key="disc_ing_img") and imgs:
        for f in imgs: ss["disc_sources"].append(ingest_image(f.read(), name=f.name))
        st.success(f"{len(imgs)} images.")

    vids = st.file_uploader("Video ads (MP4/MOV)", type=["mp4","mov","webm"],
                            accept_multiple_files=True, key="disc_vids")
    if vids:
        dur = st.number_input("Est. total video seconds", 5, 600, 60, key="disc_dur")
        toks, cost = estimate_video_cost(dur, vision_model, PRICING)
        st.warning(f"⚠️ ~{toks:,} tokens ≈ ${cost}")
        if st.button("💰 Confirm & ingest videos", key="disc_ing_vid"):
            for f in vids: ss["disc_sources"].append(ingest_video(f.read(), name=f.name))
            st.success(f"{len(vids)} videos.")

    if ss["disc_sources"]:
        df = pd.DataFrame([{"type":s.source_type,"origin":s.origin,"chars":s.meta.get("chars",0),
                            "warning":s.meta.get("warning",""),"error":s.meta.get("error","")}
                           for s in ss["disc_sources"]]).astype(str)
        st.dataframe(df, use_container_width=True)

    # ---- competitors + KPIs ----
    st.subheader("2 · Competitors + KPIs")
    def regen_comp(fb=None):
        comps, srcs = discover_competitors(gateway, t, p, cat, model=mining_model)
        ss["disc_comp_sources"] = srcs; ss["disc_competitors"] = [c["brand"] for c in comps]
        return comps
    def regen_kpi(fb=None): return suggest_targets(gateway, cat, model=mining_model)
    if st.button("🔎 Discover competitors + KPIs", key="disc_comp"):
        with st.spinner("Grounded search…"):
            ss["disc_gates"]["competitors"] = HumanGate("competitors", regen_comp); ss["disc_gates"]["competitors"].generate()
            ss["disc_gates"]["kpis"] = HumanGate("kpis", regen_kpi); ss["disc_gates"]["kpis"].generate()
        st.success("Competitors + KPIs drafted.")

    # ---- extraction ----
    st.subheader("3 · Extract")
    def regen_pos(fb=None):
        layers = {"aspirational":[],"expressed":[],"substantiated":[]}
        prog = st.progress(0.0, text="Extracting…")
        n = len(ss["disc_sources"])
        for i, s in enumerate(ss["disc_sources"]):
            if s.source_type in ("doc","web") and s.text:
                pr = extract_positioning(gateway, s, ext_model)
                for pil in pr.get("pillars",[]):
                    lay = pil.get("layer","aspirational"); layers.setdefault(lay if lay in layers else "aspirational",[]).append(pil)
            elif s.source_type in ("image","video") and s.media:
                cr = extract_creative(gateway, s, vision_model)
                for pil in cr.get("positioning_pillars",[]):
                    layers.setdefault(pil.get("layer","expressed"),[]).append(pil)
            if fb:  # ← PORT THIS
                r = gateway.generate(ext_model,
                                     f"Revise this positioning per feedback: {fb}\n\n{json.dumps(layers)[:8000]}",
                                     json_mode=True, module="discover", task="positioning_revise")
                layers = r["parsed"] or layers
            prog.progress((i + 1) / n, text=f"Source {i + 1}/{n}")
        return layers

    def regen_claims(fb=None):
        cl=[]
        for s in ss["disc_sources"]:
            if s.source_type in ("doc","web") and s.text: cl += extract_claims(gateway, s, ext_model)
            elif s.source_type in ("image","video") and s.media: cl += extract_creative(gateway, s, vision_model).get("claims",[])
        cl = dedupe_claims(cl)
        if fb:  # ← PORT THIS
            r = gateway.generate(ext_model,
                                 f"Revise this claims register per feedback: {fb}\n\n{json.dumps(cl)[:8000]}",
                                 json_mode=True, module="discover", task="claims_revise")
            cl = r["parsed"] if isinstance(r["parsed"], list) else cl
        return cl

    def regen_q(fb=None):
        pos = ss["disc_gates"]["positioning"].state.artifact if "positioning" in ss["disc_gates"] else {}
        claims = ss["disc_gates"]["claims"].state.artifact if "claims" in ss["disc_gates"] else []
        # pass claims → attribute queries derive from them
        bd = draft_query_master(gateway, pos, ss.get("disc_competitors", []), t, p, cat,
                                q_model, claims=claims)
        try:
            mined, _ = mine_real_demand(gateway, t, p, cat, model=mining_model)
        except Exception:
            mined = []
        q = merge_query_masters(bd, mined)
        # VALIDATE brand leakage
        from query_validator import validate_queries
        _alias_ov = {c.get("brand"): c.get("aliases", [])
                     for c in (ss.get("disc_gates", {})
                               .get("competitors", type("x", (), {"state": type("y", (), {"artifact": []})})())
                               .state.artifact or [])
                     if isinstance(c, dict) and c.get("aliases")}
        q, warns, errs = validate_queries(q, t, p, alias_overrides=_alias_ov)
        ss["disc_query_warnings"] = warns
        ss["disc_query_errors"] = errs
        if fb:
            r = gateway.generate(q_model,
                                 f"Revise this query master per feedback: {fb}\n\n{json.dumps(q)[:8000]}",
                                 json_mode=True, module="discover", task="query_revise")
            q = r["parsed"] if isinstance(r["parsed"], list) else q
        return q

    if ss.get("disc_query_warnings"):
        st.warning(f"🔍 {len(ss['disc_query_warnings'])} brand-leakage issue(s) — "
                   f"non-branded queries naming the brand:")
        for w in ss["disc_query_warnings"][:10]:
            st.caption(w)
        st.info("Edit these in the Query Master gate to remove the brand name, "
                "or approve if intentional.")
    if ss.get("disc_query_errors"):
        st.error("🛑 Query Master has structural errors:\n"
                 + "\n".join(f"• {e}" for e in ss["disc_query_errors"][:6]))

    if st.button("🚀 Run extraction", key="disc_ext"):
        if not ss["disc_sources"]: st.error("Ingest sources first.")
        else:
            with st.spinner("Extracting…"):
                for k, fn in [("positioning",regen_pos),("claims",regen_claims),("queries",regen_q)]:
                    ss["disc_gates"][k] = HumanGate(k, fn); ss["disc_gates"][k].generate()
            st.success("Drafts generated.")

    # thin-content banner (#2)
    thin = [s for s in ss["disc_sources"]
            if s.meta.get("thin_content") or s.meta.get("likely_js_wall")]
    if thin:
        st.warning(f"⚠️ {len(thin)} web source(s) thin/blocked — will contribute little.")

    # origin coverage (#1 — Fix 3)
    if "positioning" in ss["disc_gates"]:
        from collections import Counter
        cov = Counter()
        for items in ss["disc_gates"]["positioning"].state.artifact.values():
            for pil in items: cov[pil.get("_origin", "?")] += 1
        st.info("📊 Positioning pillars by source origin:")
        st.dataframe(pd.DataFrame([{"source": k, "pillars": v} for k, v in cov.items()]),
                     use_container_width=True)
        web_origins = [s.origin for s in ss["disc_sources"] if s.source_type == "web"]
        if web_origins and sum(cov.get(o, 0) for o in web_origins) == 0:
            st.warning("⚠️ Web sources contributed 0 pillars — extraction is doc-only.")

    # ---- gates ----
    def gate_ui(key, title):
        if key not in ss["disc_gates"]: return
        g = ss["disc_gates"][key]
        st.markdown(f"**🚦 {title}** — status: {g.state.status}")
        edited = st.text_area(f"{title} (JSON)", json.dumps(g.state.artifact, indent=2, ensure_ascii=False),
                              height=220, key=f"disc_ta_{key}")
        c1,c2,c3,c4 = st.columns([1,1,2,1])
        with c1:
            if st.button("✅", key=f"disc_ok_{key}"): g.approve("analyst")
        with c2:
            if st.button("✏️", key=f"disc_ed_{key}"):
                try: g.edit(json.loads(edited)); st.success("Saved")
                except Exception as e: st.error(e)
        with c3: fb = st.text_input("Reject+feedback", key=f"disc_fb_{key}")
        with c4:
            if st.button("🔄", key=f"disc_rg_{key}") and fb: g.reject_and_regenerate(fb); st.rerun()

    if ss["disc_gates"]:
        st.subheader("4 · Gates")
        for k,title in [("positioning","Positioning"),("claims","Claims"),("queries","Query Master"),
                        ("competitors","Competitors"),("kpis","KPIs")]:
            gate_ui(k, title)

    # ---- finalize ----
    st.subheader("5 · Finalise")
    REQ = ["positioning","claims","queries","competitors","kpis"]
    present = all(k in ss["disc_gates"] for k in REQ)
    final = present and all(ss["disc_gates"][k].is_final() for k in REQ)
    if not present:
        st.info(f"Missing gates: {[k for k in REQ if k not in ss['disc_gates']]}")
    elif not final:
        st.info(f"Approve: {[k for k in REQ if not ss['disc_gates'][k].is_final()]}")
    else:
        src_meta_for_check = [{"text": (s.text or "")} for s in ss["disc_sources"]]
        warns, errs = check_entity_alignment(
            t, ss["disc_gates"]["queries"].state.artifact,
            ss["disc_gates"]["competitors"].state.artifact,
            ss["disc_gates"]["positioning"].state.artifact,
            parent_brand=p, brand_level=brand_level,
            ingested_sources=src_meta_for_check)
        errs = errs + (ss.get("disc_query_errors") or [])

        if errs:
            st.error("🛑 **Blocking issues — resolve before emitting.** These indicate "
                     "the engagement is scoped to the wrong entity, which would make "
                     "every downstream measurement invalid.")
            for e in errs:
                st.error(f"• {e}")
            _override = st.checkbox(
                "I have verified the brand name is correct and wish to proceed anyway",
                key="disc_override_entity")
        else:
            _override = True
            for w in warns:
                st.warning(w)
            if not warns:
                st.success("✅ Entity check passed.")

        if st.button("📦 Emit artifacts", key="disc_emit", disabled=not _override):
            mu = {"extraction":ext_model,"query_mining":q_model}
            for k,name in [("positioning","positioning_layers"),("claims","claims_register"),
                           ("queries","query_master"),("competitors","competitive_set"),("kpis","kpi_targets")]:
                save_artifact(p, t, name, ss["disc_gates"][k].state.artifact, mu, scope=brand_level)
            # persist source metadata + extracted text (A-4) — not raw bytes
            src_meta = [{"type": s.source_type, "origin": s.origin,
                         "chars": s.meta.get("chars", 0),
                         "text": (s.text or "")[:5000],  # extracted text, capped
                         "warning": s.meta.get("warning", "")}
                        for s in ss["disc_sources"]]
            save_artifact(p, t, "ingested_sources", src_meta, {"note": "metadata+text"})
            save_audit_log(p, t, [g.audit_record() for g in ss["disc_gates"].values()])
            st.success(f"Discover complete: {p} → {t}")