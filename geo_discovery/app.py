"""GEO Discover — Module 1 PoC front-end.
Ingest (docs/urls) → model selection → extract → human gates → emit artifacts.
Now with: thin-content warnings (Fix1), headless-render toggle (Fix2),
source-origin coverage display (Fix3)."""
import json
from collections import Counter
import streamlit as st
import pandas as pd

from config import MODEL_REGISTRY, DEFAULT_TASK_MODELS
from meter import Meter
from llm_gateway import LLMGateway
from ingestion import ingest_doc, ingest_url, ingest_url_smart
from extraction import (extract_positioning, extract_claims, dedupe_claims,
                        draft_query_master)
from human_gate import HumanGate
from store import save_artifact, save_audit_log, list_brands
from entity_check import check_entity_alignment

st.set_page_config(page_title="GEO · Discover", layout="wide")
st.title("🧭 GEO Framework — Module 1: Discover")

ss = st.session_state
ss.setdefault("meter", Meter(budget_usd=25.0))
ss.setdefault("sources", [])
ss.setdefault("gates", {})
gateway = LLMGateway(ss.meter)

# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙️ Configuration")
    brand = st.text_input("Brand name", "Nestle")
    ss.meter.budget_usd = st.number_input("Budget cap ($)", 1.0, 500.0, 25.0)

    st.subheader("Brand scope")
    parent_brand = st.text_input("Parent / Company (context)", "Nestlé")
    target_brand = st.text_input("TARGET brand (what we audit)", "Nescafé",
                                 help="The specific PRODUCT brand shoppers search for — not the parent company.")
    category = st.text_input("Category", "instant coffee")
    brand_level = st.radio("GEO level", ["product", "corporate"],
                           format_func=lambda x: "Product brand (recommended)" if x == "product"
                           else "Corporate / Masterbrand",
                           help="Most GEO = product level. Corporate only for reputation/ESG.")
    if list_brands(parent_brand):
        st.caption(f"Existing brands under {parent_brand}: "
                   f"{', '.join(list_brands(parent_brand))}")

    st.info("💡 Upload material specific to the TARGET brand (not corporate overviews) "
            "and point URLs at the brand site (e.g., nescafe.com).")

    st.subheader("Models per task")
    models = list(MODEL_REGISTRY.keys())
    def _idx(key):
        d = DEFAULT_TASK_MODELS.get(key)
        return models.index(d) if d in models else 0
    extraction_model = st.selectbox("Extraction model", models, index=_idx("extraction"))
    query_model      = st.selectbox("Query-mining model", models, index=_idx("query_mining"))

    st.subheader("Web scraping")
    allow_render = st.checkbox("Enable headless render fallback (Playwright)", value=True,
                               help="For JS-heavy brand sites. Slower; needs "
                                    "`playwright install chromium`.")

    category = st.text_input("Category", "food, nutrition & beverages")
    st.subheader("Mining / creative models")
    mining_model = st.selectbox("Grounded-mining model",
                                [m for m in models if MODEL_REGISTRY[m]["provider"] == "vertex"], index=0)
    vision_model = st.selectbox("Vision/creative model",
                                [m for m in models if "vision" in MODEL_REGISTRY[m]["supports"]], index=0)
    max_video_sec = st.number_input("Max video duration (s)", 5, 300, 60)

    st.divider()
    st.subheader("💰 Live cost / time")
    st.metric("Total cost", f"${ss.meter.total_cost():.4f}")
    st.metric("LLM calls", ss.meter.calls)
    st.metric("Time (s)", f"{ss.meter.elapsed:.1f}")
    if ss.meter.events:
        st.dataframe(ss.meter.summary_by_task(), use_container_width=True, height=200)

# ================= STEP 1: INGEST =================
st.header("1 · Ingest brand sources")
c1, c2 = st.columns(2)
with c1:
    up = st.file_uploader("Upload brand docs (PDF/DOCX/PPTX/TXT)",
                          type=["pdf", "docx", "pptx", "txt", "md"],
                          accept_multiple_files=True)
    if st.button("Ingest uploaded docs") and up:
        for f in up:
            ss.sources.append(ingest_doc(f.read(), name=f.name))
        st.success(f"Ingested {len(up)} doc(s).")
with c2:
    urls = st.text_area("Brand website URLs (one per line)")
    if st.button("Ingest URLs") and urls.strip():
        with st.spinner("Scraping (with render fallback if needed)…"):
            for u in [x.strip() for x in urls.splitlines() if x.strip()]:
                ss.sources.append(ingest_url_smart(u, allow_render=allow_render))
        st.success("URLs ingested.")

st.markdown("**Creative assets (ads)**")
imgs = st.file_uploader("Upload static ads (PNG/JPG/WEBP)",
                        type=["png","jpg","jpeg","webp"], accept_multiple_files=True)
if st.button("Ingest images") and imgs:
    from ingestion import ingest_image
    for f in imgs: ss.sources.append(ingest_image(f.read(), name=f.name))
    st.success(f"Ingested {len(imgs)} image(s).")

vids = st.file_uploader("Upload video ads (MP4/MOV/WEBM)",
                        type=["mp4","mov","webm"], accept_multiple_files=True)
if vids:
    from ingestion import ingest_video, estimate_video_cost
    from config import PRICING
    # cost estimate + confirm gate
    est_dur = st.number_input("Est. total video seconds", 5, 600, 60)
    toks, cost = estimate_video_cost(est_dur, vision_model, PRICING)
    st.warning(f"⚠️ Video processing est: ~{toks:,} tokens ≈ **${cost}** "
               f"(capped at {max_video_sec}s each). Confirm to ingest.")
    if st.button("💰 Confirm & ingest videos"):
        for f in vids:
            ss.sources.append(ingest_video(f.read(), name=f.name,
                                           max_duration_sec=max_video_sec))
        st.success(f"Ingested {len(vids)} video(s).")

# ---- source table with WARNING column (Fix 1) ----
if ss.sources:
    st.caption("Ingested sources:")
    df_src = pd.DataFrame([{"type": s.source_type, "origin": s.origin,
                            "chars": s.meta.get("chars", 0),
                            "rendered": str(s.meta.get("rendered", "")),        # ← str()
                            "boilerplate_dropped": s.meta.get("boilerplate_dropped", 0),
                            "warning": str(s.meta.get("warning", "")),
                            "error": str(s.meta.get("error", ""))} for s in ss.sources])
    df_src = df_src.astype(str)          # ← belt: cast everything to string for Arrow
    st.dataframe(df_src, use_container_width=True)
    # loud banner if any web source is thin
    thin = [s for s in ss.sources if s.meta.get("thin_content") or s.meta.get("likely_js_wall")]
    if thin:
        st.warning(f"⚠️ {len(thin)} web source(s) returned thin/blocked content and will "
                   f"contribute little. Consider enabling render fallback, using lighter "
                   f"pages (newsroom/about), or relying on uploaded docs.")

# ================= STEP 2: EXTRACT =================
st.header("2 · Extract → draft artifacts")

def regen_positioning(feedback=None):
    layers = {"aspirational": [], "expressed": [], "substantiated": []}
    for s in ss.sources:
        if s.source_type in ("doc", "web") and s.text:
            p = extract_positioning(gateway, s, extraction_model)
            for pillar in p.get("pillars", []):
                layer = pillar.get("layer", "aspirational")
                if layer not in layers: layer = "aspirational"
                pillar["_origin"] = p.get("_origin");
                layers[layer].append(pillar)
        elif s.source_type in ("image", "video") and s.media:
            from extraction import extract_creative
            cr = extract_creative(gateway, s, vision_model)
            for pillar in cr.get("positioning_pillars", []):
                layers.setdefault(pillar.get("layer", "expressed"), []).append(pillar)
    if feedback:
        prompt = (f"Revise this layered positioning JSON per feedback: {feedback}\n\n"
                  f"{json.dumps(layers)[:8000]}")
        r = gateway.generate(extraction_model, prompt, json_mode=True,
                             module="discover", task="positioning_revise")
        layers = r["parsed"] or layers
    return layers

def regen_claims(feedback=None):
    claims = []
    for s in ss.sources:
        if s.source_type in ("doc", "web") and s.text:
            claims += extract_claims(gateway, s, extraction_model)
        elif s.source_type in ("image", "video") and s.media:
            from extraction import extract_creative
            claims += extract_creative(gateway, s, vision_model).get("claims", [])
    claims = dedupe_claims(claims)
    if feedback:
        prompt = (f"Revise this claims register per feedback: {feedback}\n\n"
                  f"{json.dumps(claims)[:8000]}")
        r = gateway.generate(extraction_model, prompt, json_mode=True,
                             module="discover", task="claims_revise")
        claims = r["parsed"] if isinstance(r["parsed"], list) else claims
    return claims

def regen_queries(feedback=None):
    # 1. brand-derived queries (LLM-guessed from positioning) — as before
    pos = ss.gates["positioning"].state.artifact if "positioning" in ss.gates else {}
    brand_derived = draft_query_master(gateway, pos, ss.get("competitors", []),
                                       target_brand, parent_brand, category, query_model)
    from grounded_mining import mine_real_demand, merge_query_masters
    try:
        mined, msrc = mine_real_demand(gateway, target_brand, parent_brand,
                                       category, model=mining_model)  # model= keyword
        ss["demand_sources"] = msrc
    except Exception as e:
        mined = [];
        st.warning(f"Mining failed (brand-derived only): {e}")
        st.write(f"DEBUG: mining_model = {mining_model!r}")  # ← ADD temporarily
    q = merge_query_masters(brand_derived, mined)

    # 3. feedback loop — now operates on the MERGED set
    if feedback:
        prompt = (f"Revise this query master per feedback: {feedback}\n\n"
                  f"{json.dumps(q)[:8000]}")
        r = gateway.generate(query_model, prompt, json_mode=True,
                             module="discover", task="query_revise")
        q = r["parsed"] if isinstance(r["parsed"], list) else q

    return q

st.header("2b · Competitors & KPIs")

def regen_competitors(feedback=None):
    from grounded_mining import discover_competitors
    comps, srcs = discover_competitors(gateway, target_brand, parent_brand,
                                       category, model=mining_model)   # target/parent, not brand
    ss["competitor_sources"] = srcs
    ss["competitors"] = [c["brand"] for c in comps]
    return comps

def regen_kpis(feedback=None):
    from kpi import suggest_targets
    return suggest_targets(gateway, category, model=mining_model)

if st.button("🔎 Discover competitors + suggest KPIs"):
    with st.spinner("Grounded search…"):
        ss.gates["competitors"] = HumanGate("competitors", regen_competitors)
        ss.gates["competitors"].generate()
        ss.gates["kpis"] = HumanGate("kpis", regen_kpis)
        ss.gates["kpis"].generate()
    st.success("Competitors + KPIs drafted — review in gates below.")


if st.button("🚀 Run extraction (positioning + claims + queries)"):
    if not ss.sources:
        st.error("Ingest at least one source first.")
    else:
        with st.spinner("Extracting…"):
            ss.gates["positioning"] = HumanGate("positioning", regen_positioning)
            ss.gates["positioning"].generate()
            ss.gates["claims"] = HumanGate("claims", regen_claims)
            ss.gates["claims"].generate()
            ss.gates["queries"] = HumanGate("queries", regen_queries)
            ss.gates["queries"].generate()
        st.success("Drafts generated — review below.")

# ---- SOURCE-ORIGIN COVERAGE (Fix 3) ----
def origin_coverage(layers):
    c = Counter()
    for items in layers.values():
        for p in items:
            c[p.get("_origin", "?")] += 1
    return dict(c)

if "positioning" in ss.gates:
    cov = origin_coverage(ss.gates["positioning"].state.artifact)
    st.info("📊 **Positioning pillars by source origin** "
            "(confirms which sources actually contributed):")
    cov_df = pd.DataFrame([{"source": k, "pillars_contributed": v} for k, v in cov.items()]
                          ).sort_values("pillars_contributed", ascending=False)
    st.dataframe(cov_df, use_container_width=True)
    web_origins = [s.origin for s in ss.sources if s.source_type == "web"]
    web_contrib = sum(cov.get(o, 0) for o in web_origins)
    if web_origins and web_contrib == 0:
        st.warning("⚠️ Web sources contributed **0** positioning pillars — "
                   "extraction is doc-only (likely blocked scrapes).")

# ================= STEP 3: HUMAN GATES =================
def render_gate(gate_key, title):
    if gate_key not in ss.gates: return
    gate = ss.gates[gate_key]
    st.subheader(f"🚦 Gate: {title}")
    edited = st.text_area(f"{title} (JSON — edit inline if needed)",
                          json.dumps(gate.state.artifact, indent=2),
                          height=280, key=f"ta_{gate_key}")
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
    with col1:
        if st.button("✅ Approve", key=f"ap_{gate_key}"):
            gate.approve(); st.success(f"{title} approved.")
    with col2:
        if st.button("✏️ Save edit", key=f"ed_{gate_key}"):
            try: gate.edit(json.loads(edited)); st.success("Edited & accepted.")
            except Exception as e: st.error(f"Invalid JSON: {e}")
    with col3:
        fb = st.text_input("Reject + feedback → regenerate", key=f"fb_{gate_key}")
    with col4:
        if st.button("🔄 Regenerate", key=f"rg_{gate_key}") and fb:
            with st.spinner("Regenerating with feedback…"):
                gate.reject_and_regenerate(fb)
            st.rerun()
    st.caption(f"Status: **{gate.state.status}** · "
               f"feedback rounds: {len(gate.state.feedback_history)}")

if ss.gates:
    st.header("3 · Human validation gates")
    render_gate("positioning", "Positioning (3-layer)")
    render_gate("claims", "Substantiated Claims Register")
    render_gate("queries", "Query Master (draft)")
    render_gate("competitors", "Competitive Set")
    render_gate("kpis", "KPI Targets")


if ss.gates and all(g.is_final() for g in ss.gates.values()):
    warns = check_entity_alignment(
        target_brand,
        ss.gates["queries"].state.artifact,  # query_master
        ss.gates["competitors"].state.artifact,  # competitors
        ss.gates["positioning"].state.artifact,  # positioning_layers
        parent_brand=parent_brand,
        brand_level=brand_level)

    if warns:
        st.error("🔍 Entity-level check flagged issues (advisory):")
        for w in warns:
            st.warning(w)
        st.info("If GEO is intentionally corporate-level, proceed. "
                "Otherwise, re-scope to the product brand and re-run.")
    else:
        st.success("✅ Entity check passed — coherently brand-scoped.")

# ================= STEP 4: EMIT =================
REQUIRED_GATES = ["positioning", "claims", "queries", "competitors", "kpis"]

st.header("4 · Finalise Discover")

gates_present = all(k in ss.gates for k in REQUIRED_GATES)
gates_final = gates_present and all(ss.gates[k].is_final() for k in REQUIRED_GATES)

if not gates_present:
    missing = [k for k in REQUIRED_GATES if k not in ss.gates]
    st.info(f"⏳ Missing gates: {', '.join(missing)}. Run both "
            f"'🔎 Discover competitors + KPIs' and '🚀 Run extraction'.")
elif not gates_final:
    pending = [k for k in REQUIRED_GATES if not ss.gates[k].is_final()]
    st.info(f"⏳ Approve/edit before finalising: {', '.join(pending)}")
else:
    from entity_check import check_entity_alignment
    warns = check_entity_alignment(
        target_brand,
        ss.gates["queries"].state.artifact,
        ss.gates["competitors"].state.artifact,
        ss.gates["positioning"].state.artifact,
        parent_brand=parent_brand,
        brand_level=brand_level)
    if warns:
        st.error("🔍 Entity-level check flagged issues (advisory):")
        for w in warns: st.warning(w)
        st.info("If GEO is intentionally corporate-level, proceed. "
                "Otherwise re-scope to the product brand and re-run.")
    else:
        st.success("✅ Entity check passed — coherently brand-scoped.")

    if st.button("📦 Emit versioned artifacts"):
        models_used = {"extraction": extraction_model, "query_mining": query_model}
        paths = []
        for key, name in [("positioning","positioning_layers"),
                          ("claims","claims_register"),
                          ("queries","query_master"),
                          ("competitors","competitive_set"),
                          ("kpis","kpi_targets")]:
            paths.append(save_artifact(parent_brand, target_brand, name,
                                       ss.gates[key].state.artifact, models_used,
                                       scope=brand_level))
        save_audit_log(parent_brand, target_brand,
                       [g.audit_record() for g in ss.gates.values()])
        st.success(f"Discover complete: {parent_brand} → {target_brand}")
        st.code("\n".join(paths))
