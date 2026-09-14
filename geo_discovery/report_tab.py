"""Report tab — assemble artifacts into a client-ready document."""
import pandas as pd
import streamlit as st

from report.builders import build_report_model
from report.contract import audit_contract, coverage_declaration
from report.render_docx import render_docx
from store import project_dir
from human_gate import HumanGate

def _df(rows):
    """Stringify every cell. Report tables intentionally mix ints and strings in a
    single column (e.g. a Field/Detail layout), which Arrow cannot serialise."""
    return pd.DataFrame([{k: ("" if v is None else str(v)) for k, v in r.items()}
                         for r in rows if isinstance(r, dict)])
def _as_text_safe(payload):
    if isinstance(payload, list):
        return "\n".join(str(x) for x in payload)
    return str(payload or "")

def _preview_block(b):
    """Read-only rendering of one block in the Streamlit preview."""
    if b.kind == "heading":
        st.markdown(f"{'#' * min(b.level + 2, 6)} {b.payload}")
    elif b.kind == "prose":
        st.markdown(b.payload + ("  _[edited]_" if getattr(b, "edited", False) else ""))
    elif b.kind == "bullets":
        for it in (b.payload or []):
            st.markdown(f"- {it}")
    elif b.kind == "callout":
        {"info": st.info, "warn": st.warning, "danger": st.error,
         "good": st.success}.get(b.tone, st.info)(b.payload)
    elif b.kind == "kpi_row":
        cols = st.columns(len(b.payload))
        for col, kp in zip(cols, b.payload):
            col.metric(kp["label"], kp["value"], help=kp.get("note"))
    elif b.kind in ("table", "coverage", "evidence_key"):
        data = ([{"Parameter": kk, "Value": vv} for kk, vv in (b.payload or {}).items()]
                if b.kind == "coverage" else b.payload)
        if data:
            st.dataframe(_df(data), use_container_width=True, hide_index=True)
        if b.caption:
            st.caption(b.caption)
    elif b.kind == "chart" and b.payload:
        st.image(b.payload, use_container_width=True)
        if b.caption:
            st.caption(b.caption)
    elif b.kind == "quote":
        st.code(str(b.payload)[:1500])

def render(ctx):
    st.header("📄 Module 6: Report")
    p, t = ctx.parent_brand, ctx.target_brand
    ss = st.session_state

    st.caption("Assembles the platform's artifacts into a client-ready report. No new "
               "analysis is performed — every number is traceable to its source artifact.")

    # ---------------- readiness ----------------
    rows = audit_contract(p, t)
    df = pd.DataFrame(rows)
    blockers = df[(df.required) & (df.state != "OK")]
    with st.expander(f"{'🔴' if len(blockers) else '✅'} Data readiness — "
                     f"{len(blockers)} required field(s) missing",
                     expanded=bool(len(blockers))):
        st.dataframe(df.groupby(["section", "state"]).size().unstack(fill_value=0),
                     use_container_width=True)
        if len(blockers):
            st.dataframe(_df(blockers.to_dict("records"))[
                             ["section", "label", "artifact", "path", "state", "note"]],
                         use_container_width=True, hide_index=True)
            st.caption("Sections whose data is missing are omitted and declared, never "
                       "rendered blank.")

    cov = coverage_declaration(p, t)
    with st.expander("📐 Coverage declaration — what was and was not assessed",
                     expanded=True):
        c1, c2 = st.columns(2)
        c1.metric("Audit runs", f"{cov.get('n_runs') or 0:,}")
        c1.metric("Query categories",
                  f"{len(cov.get('intents_audited') or [])} of "
                  f"{len(cov.get('intents_planned') or [])}")
        c2.metric("Proven findings", cov.get("n_proven") or 0,
                  f"{cov.get('n_disproven') or 0} disproven")
        c2.metric("Engines", len(cov.get("engines") or []))
        st.json(cov)

    # ---------------- options ----------------
    st.subheader("⚙️ Report options")
    o1, o2, o3 = st.columns(3)
    profile = o1.selectbox("Redaction profile", ["internal", "client"], key="rep_profile",
                           help="`client` masks unauthorised-reseller and affiliate "
                                "domain names while retaining the aggregate percentages.")
    audience = o2.selectbox("Audience", ["both", "exec", "practitioner"],
                            key="rep_audience",
                            help="`exec` keeps the narrative, KPIs and priorities. "
                                 "`practitioner` adds full tables, briefs and "
                                 "methodology. `both` is the comprehensive report.")
    extra = o3.text_area("Additional provisional caveats (one per line)", "",
                         key="rep_caveats", height=90)

    if st.button("📄 Build report", key="rep_build"):
        with st.spinner("Assembling…"):
            rm = build_report_model(
                p, t, ctx.category, profile=profile,
                provisional_reasons=[x.strip() for x in extra.splitlines() if x.strip()])
            ss["rep_model"] = rm
        st.success("Assembled.")

    if "rep_model" not in ss:
        st.info("👆 Click **Build report**.")
        return

    rm = ss["rep_model"]

    # ---------------- apply saved edits ----------------
    from report.edits import (EDITABLE_KINDS, apply_edits, block_key, edit_summary,
                              load_edits, save_edits)
    if "rep_edits" not in ss:
        ss["rep_edits"] = load_edits(p, t)
    edit_report = apply_edits(rm, ss["rep_edits"])

    st.subheader("📊 Report contents")
    k = st.columns(6)
    stt = rm.stats()
    k[0].metric("Sections", stt["sections"])
    k[1].metric("Blocks", stt["blocks"])
    k[2].metric("Traceable facts", stt["facts"])
    k[3].metric("Charts", stt["charts"])
    k[4].metric("Tables", stt["tables"])
    k[5].metric("Analyst edits", edit_report["applied"],
                f"{len(edit_report['stale'])} stale" if edit_report["stale"] else None)
    if rm.provisional:
        st.error("**PROVISIONAL** — " + " ".join(rm.provisional_reasons))
    for w in rm.warnings:
        st.warning(w)
    if edit_report["stale"]:
        with st.expander(f"⚠️ {len(edit_report['stale'])} edit(s) no longer apply"):
            st.caption("The underlying analysis changed beneath these edits, so they were "
                       "not applied. Re-enter them against the current text.")
            st.dataframe(_df(edit_report["stale"]), use_container_width=True,
                         hide_index=True)

    # ---------------- editable preview ----------------
    st.subheader("✏️ Review and edit")
    st.caption("Narrative, bullets, callouts and headings are editable. Numbers, tables "
               "and charts are generated from artifacts and are not editable — change the "
               "analysis, not the report.")
    edit_mode = st.toggle("Enable editing", value=False, key="rep_edit_mode")

    dirty = False
    for s in rm.filtered(audience, profile):
        n_ed = sum(1 for b in s.blocks if getattr(b, "edited", False))
        with st.expander(f"{s.title} ({len(s.blocks)} block(s)"
                         + (f", {n_ed} edited)" if n_ed else ")")):
            for i, b in enumerate(s.blocks):
                key = block_key(s.id, i, b) if b.kind in EDITABLE_KINDS else None
                if edit_mode and key:
                    saved = ss["rep_edits"].get(key, {})
                    original = saved.get("original_text") or _as_text_safe(b.payload)
                    label = {"prose": "Paragraph", "bullets": "Bullets",
                             "callout": f"Callout ({b.tone})",
                             "heading": f"Heading (level {b.level})"}[b.kind]
                    new = st.text_area(f"{label} · {key}",
                                       value=_as_text_safe(b.payload),
                                       key=f"ta_{key}",
                                       height=90 if b.kind == "heading" else 150)
                    if new.strip() != original.strip():
                        ss["rep_edits"][key] = {
                            "original_text": original, "edited_text": new,
                            "kind": b.kind, "section": s.id,
                            "approver": "analyst",
                            "at": __import__("datetime").datetime.now(
                                __import__("datetime").timezone.utc).isoformat()}
                        dirty = True
                    elif key in ss["rep_edits"]:
                        ss["rep_edits"].pop(key)
                        dirty = True
                    if saved.get("edited_text"):
                        st.caption(f"_Original:_ {original[:200]}")
                else:
                    _preview_block(b)

    e1, e2, e3 = st.columns(3)
    if e1.button("💾 Save edits", key="rep_save_edits", type="primary" if dirty else
    "secondary"):
        save_edits(p, t, ss["rep_edits"])
        st.success(f"Saved {sum(1 for v in ss['rep_edits'].values() if v.get('edited_text'))}"
                   f" edit(s). They will be re-applied on every rebuild.")
    if e2.button("↩️ Discard all edits", key="rep_clear_edits"):
        ss["rep_edits"] = {}
        save_edits(p, t, {})
        ss.pop("rep_model", None)
        st.rerun()
    e3.caption(f"{sum(1 for v in ss['rep_edits'].values() if v.get('edited_text'))} "
               f"edit(s) stored")
    if ss["rep_edits"]:
        with st.expander("📝 Edit log"):
            st.dataframe(_df(edit_summary(ss["rep_edits"])),
                         use_container_width=True, hide_index=True)

    # ---------------- sign-off gate ----------------
    st.subheader("🚦 Report sign-off")
    if "rep_gate" not in ss:
        ss["rep_gate"] = HumanGate("report", lambda fb=None: rm.meta)
        ss["rep_gate"].generate()
    st.caption(f"Status: **{ss['rep_gate'].state.status}**")
    if rm.provisional:
        st.warning("This report is marked PROVISIONAL. Approving does not remove that "
                   "banner — resolve the underlying gaps first.")
    g1, g2 = st.columns(2)
    if g1.button("✅ Approve for release", key="rep_ok"):
        ss["rep_gate"].approve("analyst")
        st.success("Approved.")
    g2.caption("Export is available before approval, but exported files record the "
               "sign-off state.")

    # ---------------- export ----------------
    st.subheader("⬇️ Export")
    approved = ss["rep_gate"].is_final()
    rm.meta["signoff"] = "approved" if approved else "draft"
    rm.meta["n_analyst_edits"] = edit_report["applied"]
    stamp = "APPROVED" if approved else "DRAFT"

    from report.render_pdf import available_routes
    _routes = available_routes()
    if _routes:
        st.caption("PDF route: " + " → ".join(f"`{n}` ({d})" for n, d in _routes))
    else:
        st.caption("⚠️ No PDF route installed. Run `pip install docx2pdf` (uses Word) or "
                   "`pip install reportlab`, or install LibreOffice from "
                   "libreoffice.org/download. Word export works regardless.")
    x1, x2, x3 = st.columns(3)
    if x1.button("📄 Word (.docx)", key="rep_gen_docx"):
        with st.spinner("Rendering…"):
            data = render_docx(rm, audience=audience, profile=profile)
            path = project_dir(p, t) / f"GEO_Report_{t}_{audience}_{profile}_{stamp}.docx"
            path.write_bytes(data)
            ss["rep_docx_data"] = (data, path.name)
        st.success(f"{path.name} ({len(data):,} bytes)")
    if x2.button("📕 PDF", key="rep_gen_pdf"):
        with st.spinner("Converting…"):
            from report.render_pdf import render_pdf
            data, how = render_pdf(rm, audience=audience, profile=profile)
            if data:
                path = project_dir(p, t) / f"GEO_Report_{t}_{audience}_{profile}_{stamp}.pdf"
                path.write_bytes(data)
                ss["rep_pdf_data"] = (data, path.name)
                st.success(f"{path.name} via {how} ({len(data):,} bytes)")
            else:
                st.error(f"PDF conversion unavailable: {how}")
    if x3.button("📊 PowerPoint", key="rep_gen_pptx"):
        with st.spinner("Building deck…"):
            from report.render_pptx import render_pptx
            data = render_pptx(rm, profile=profile)
            path = project_dir(p, t) / f"GEO_Deck_{t}_{profile}_{stamp}.pptx"
            path.write_bytes(data)
            ss["rep_pptx_data"] = (data, path.name)
        st.success(f"{path.name} ({len(data):,} bytes)")

    d1, d2, d3 = st.columns(3)
    if ss.get("rep_docx_data"):
        data, name = ss["rep_docx_data"]
        d1.download_button("⬇️ .docx", data, file_name=name, key="rep_dl_docx",
                           mime="application/vnd.openxmlformats-officedocument."
                                "wordprocessingml.document")
    if ss.get("rep_pdf_data"):
        data, name = ss["rep_pdf_data"]
        d2.download_button("⬇️ .pdf", data, file_name=name, key="rep_dl_pdf",
                           mime="application/pdf")
    if ss.get("rep_pptx_data"):
        data, name = ss["rep_pptx_data"]
        d3.download_button("⬇️ .pptx", data, file_name=name, key="rep_dl_pptx",
                           mime="application/vnd.openxmlformats-officedocument."
                                "presentationml.presentation")

    with st.expander("🧾 Fact ledger (traceability)"):
        st.dataframe(_df(rm.ledger.table()), use_container_width=True, hide_index=True)