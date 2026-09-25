"""report/builders.py — artifacts → ReportModel. No new analysis.

Every number becomes a Fact with a source path. Every claim carries its evidence class.
Sections with no data DECLARE that rather than rendering blank.
"""
from __future__ import annotations

import json
from collections import Counter

from report import charts as C
from report.contract import coverage_declaration, input_hashes
from report.model import (Block, EVIDENCE_GLOSS, EVIDENCE_LABEL, FactLedger, ReportModel,
                          redact_domain)
from store import load_artifact

ARTIFACTS = ["positioning_layers", "claims_register", "query_master", "competitive_set",
             "kpi_targets", "benchmark_metrics", "delta_scorecard", "source_hitlist",
             "gap_register_actionable", "validated_findings", "experiment_variants",
             "substantiation_backlog", "geo_strategy", "roadmap", "content_briefs",
             "action_register", "audit_manifest", "ingested_sources",
             "experiment_visual_assets", "bench_designs", "bench_legibility"]


def _load(parent, target):
    return {n: load_artifact(parent, target, n) for n in ARTIFACTS}

def H(t, level=1, audience="both"):
    return Block("heading", t, level=level, audience=audience)

def P(t, audience="both", redaction="internal"):
    return Block("prose", t, audience=audience, redaction=redaction)

def BUL(items, audience="both"):
    return Block("bullets", list(items), audience=audience)

def TBL(rows, caption="", audience="practitioner", redaction="internal"):
    clean = [{k: ("" if v is None else v if isinstance(v, str) else str(v))
              for k, v in r.items()}
             for r in (rows or []) if isinstance(r, dict)]
    return Block("table", clean, caption=caption, audience=audience,
                 redaction=redaction)

def CH(png, caption="", audience="both"):
    return Block("chart", png, caption=caption, audience=audience) if png else None

def CALL(t, tone="info", audience="both"):
    return Block("callout", t, tone=tone, audience=audience)

def _add(section, block):
    if block is not None:
        section.add(block)

# ====================================================================== front matter
def build_front_matter(rm, A, parent, target, category):
    cov = rm.coverage
    s = rm.section("front", "Scope and coverage")
    _add(s, P(f"This report presents a point-in-time measurement of how generative AI "
              f"engines describe and recommend **{target}** in the {category} category. "
              f"It is assembled entirely from the platform's analytical artifacts; no "
              f"new analysis is performed at report time."))
    _add(s, Block("coverage", {
        "Brand": f"{target} ({parent})",
        "Category": category,
        "Engines tested": ", ".join(cov.get("engines") or []) or "—",
        "Audit runs": f"{cov.get('n_runs') or 0:,}",
        "Query categories audited": f"{len(cov.get('intents_audited') or [])} of "
                                    f"{len(cov.get('intents_planned') or [])} "
                                    f"({', '.join(cov.get('intents_audited') or [])})",
        "Not audited this cycle": ", ".join(cov.get("intents_not_audited") or []) or "none",
        "Modalities": ", ".join(cov.get("modalities_assessed") or []),
        "Modalities not assessed": ", ".join(cov.get("modalities_not_assessed") or []),
        "Retrieval mode": ", ".join(cov.get("web_modes") or []),
        "Memory vs retrieval separated": "Yes" if cov.get(
            "memory_vs_retrieval_separated") else "No — retrieval mode only",
        "Experiments": f"{cov.get('n_experiments') or 0} "
                       f"({cov.get('n_proven') or 0} proven, "
                       f"{cov.get('n_proven_caveated') or 0} caveated, "
                       f"{cov.get('n_disproven') or 0} disproven)",
        "Report generated": rm.meta.get("generated_utc", "")[:19].replace("T", " ") + " UTC",
        "Claims assessed": (f"{cov.get('claims_assessed') or 0} of "
                            f"{cov.get('claims_in_register') or 0} "
                            f"({cov.get('claims_assessed_pct') or 0}%)"),
    }))
    gaps = []
    if cov.get("intents_not_audited"):
        gaps.append(f"{len(cov['intents_not_audited'])} of "
                    f"{len(cov.get('intents_planned') or [])} query categories were not "
                    f"audited ({', '.join(cov['intents_not_audited'])}). Findings do not "
                    f"extend to them.")
    if not cov.get("memory_vs_retrieval_separated"):
        gaps.append("Only retrieval mode (web enabled) was audited. The distinction "
                    "between what engines *retrieve* and what they *remember* was not "
                    "measured, which limits confidence in fast-vs-slow sequencing.")
    if cov.get("vision_verified") is not True:
        gaps.append("Visual and multimodal behaviour was not assessed. No conclusions "
                    "are drawn about packaging legibility to vision models.")
    if cov.get("claim_sampling_bias"):
        gaps.append(cov["claim_sampling_bias"])
    if not cov.get("query_volume_data"):
        gaps.append("Search-volume data was not available, so query importance is "
                    "weighted by category structure rather than measured demand.")
    if gaps:
        _add(s, CALL("**What this report does not cover.** " + " ".join(gaps),
                     tone="warn"))
    if rm.provisional:
        _add(s, CALL("**PROVISIONAL ANALYSIS.** " + " ".join(rm.provisional_reasons)
                     + " Findings and scores may change once the outstanding work "
                       "completes. Do not treat these as final client findings.",
                     tone="danger"))
    return s

# ====================================================================== 1 · exec summary
def build_exec_summary(rm, A, target):
    L, s = rm.ledger, rm.section("exec", "Executive summary")
    m = A.get("benchmark_metrics") or {}
    d = A.get("delta_scorecard") or {}
    strat = A.get("geo_strategy") or {}
    ov = strat.get("overall_strategy") or {}
    findings = A.get("validated_findings") or []
    prov = d.get("provenance") or {}
    rd = m.get("retrieval_delta") or {}

    if ov.get("narrative"):
        _add(s, CALL(ov["narrative"], tone="info"))
    if ov.get("diagnosis"):
        _add(s, P(f"**Diagnosis.** {ov['diagnosis']}"))

    # ---- the KPI row ----
    kpis, need = [], next((r for r in m.get("mention_by_intent", [])
                           if r.get("intent") == "need"), None)
    if need:
        f = L.add(value=need["mention_rate"], label="Unprompted visibility (need queries)",
                  unit="share", source_artifact="benchmark_metrics",
                  source_path="mention_by_intent[need].mention_rate",
                  ci=(need.get("ci_low"), need.get("ci_high")), n=need.get("runs"),
                  evidence_class="DIAGNOSED")
        kpis.append({"label": "Unprompted visibility", "value": f.rendered(),
                     "note": "Share of unbranded need queries naming the brand",
                     "fact": f.fact_id})
    if rd.get("gap_pp") is not None:
        f = L.add(value=rd["gap_pp"], label="Retrieval gap", unit="pp",
                  source_artifact="benchmark_metrics",
                  source_path="retrieval_delta.gap_pp",
                  evidence_class="DIAGNOSED",
                  limits=[f"Best case: with substantive brand content in context, mention "
                          f"reaches {rd.get('in_context_best', 0):.0%}. Worst case: with "
                          f"generic brand copy, {rd.get('in_context_worst', 0):.0%}. "
                          f"Audit and experiment use different query samples."])
        kpis.append({"label": "Retrieval gap", "value": f.rendered(),
                     "note": "Visibility when own content is present, minus visibility "
                             "in live retrieval",
                     "fact": f.fact_id})
    if prov.get("self_referential_pct") is not None:
        f = L.add(value=prov["self_referential_pct"], label="Citation control", unit="share",
                  source_artifact="delta_scorecard",
                  source_path="provenance.self_referential_pct",
                  evidence_class="DIAGNOSED")
        kpis.append({"label": "Citation control", "value": f.rendered(),
                     "note": "Share of cited sources the brand owns", "fact": f.fact_id})
    if prov.get("voice_control_ratio") is not None:
        f = L.add(value=prov["voice_control_ratio"], label="Voice-control ratio", unit="x",
                  source_artifact="delta_scorecard",
                  source_path="provenance.voice_control_ratio",
                  evidence_class="DIAGNOSED")
        kpis.append({"label": "Voice control", "value": f.rendered(),
                     "note": "Brand-owned citations ÷ unauthorised + affiliate citations. "
                             "Below 1.0 means third parties dominate.",
                     "fact": f.fact_id})
    if kpis:
        _add(s, Block("kpi_row", kpis))

    # ---- headline findings ----
    if d.get("headline"):
        _add(s, H("Principal findings", 2))
        _add(s, BUL(d["headline"]))

    if rd.get("gap_pp") and rd["gap_pp"] > 50:
        _add(s, CH(C.retrieval_gap(rd.get("unprompted_mention", 0),
                                   rd.get("in_context_best", 1)),
                   caption=rd.get("interpretation", "")))

    # ---- big bets ----
    if ov.get("big_bets"):
        _add(s, H("Where to act", 2))
        rows = []
        for b in ov["big_bets"]:
            rows.append({"Priority": b.get("title"),
                         "Pillars": ", ".join(b.get("pillars") or []),
                         "Evidence": EVIDENCE_LABEL.get(b.get("evidence_class"),
                                                        b.get("evidence_class") or "—"),
                         "Expected effect": b.get("expected_effect") or "—",
                         "Why now": b.get("why_now") or ""})
        _add(s, TBL(rows, caption="Priority moves, with the evidence class behind each.",
                    audience="both"))

    # ---- what the evidence rules out — the credibility section ----
    disproven = [f for f in findings if f.get("evidence_class") == "DISPROVEN"]
    if ov.get("not_doing") or disproven:
        _add(s, H("What the evidence rules out", 2))
        _add(s, P("Stating what not to spend on is as valuable as the recommendations. "
                  "The following are excluded on measured grounds."))
        rows, cited = [], set()
        for n in (ov.get("not_doing") or []):
            basis = str(n.get("basis") or "")
            cited |= {f["design_id"] for f in findings
                      if f.get("design_id") and f["design_id"] in basis}
            rows.append({"Excluded": n.get("item"), "Reason": n.get("reason"),
                         "Basis": basis or "—"})
        for dn in disproven:
            if dn.get("design_id") in cited:
                continue  # already covered by a not_doing entry
            rows.append({
                "Excluded": dn.get("isolated_variable")
                            or f"{dn.get('experiment_id')} treatment",
                "Reason": f"No measurable effect at {int(dn.get('runs_per_cell') or 0)} "
                          f"runs per arm per engine.",
                "Basis": f"Experiment {dn.get('design_id')}"})
        _add(s, TBL(rows, audience="both"))

    # ---- compliance ----
    backlog = A.get("substantiation_backlog") or []
    crit = [b for b in backlog if b.get("severity") == "critical"]
    if backlog:
        sub_fact = L.add(value=len(backlog),
                         label="Unsubstantiated assertions engines already make",
                         unit="count", source_artifact="substantiation_backlog",
                         source_path="", evidence_class="DIAGNOSED")
        n_assert = (d.get("hallucination_summary") or {}).get("counts", {}).get(
            "unsubstantiated")
        span = f" covering {n_assert} distinct assertions" if n_assert else ""
        _add(s, CALL(
         f"**Substantiation exposure ({sub_fact.fact_id}).** The reverse scan found "
            f"assertions the engines make about the brand that are absent from the "
            f"substantiated claims register. These cluster into {len(backlog)} "
            f"substantiation items{span} — {len(crit)} rated critical. Because such "
            f"claims cannot be published or defended, they cannot be tested; the remedy "
            f"is substantiation, not content. In a regulated category this is a "
            f"compliance matter as well as the least expensive high-value work "
            f"available. See §6.", tone="danger"))

    _add(s, Block("evidence_key", [
        {"Class": EVIDENCE_LABEL[k], "Means": v} for k, v in EVIDENCE_GLOSS.items()],
        audience="both"))
    return s

# ====================================================================== 2 · discovery
def build_discovery(rm, A, target):
    L, s = rm.ledger, rm.section("discover", "Ground truth: what the brand actually claims")
    pos = A.get("positioning_layers") or {}
    claims = A.get("claims_register") or []
    comps = A.get("competitive_set") or []
    qm = A.get("query_master") or []

    _add(s, P("The Ground Truth is the benchmark against which machine reality is "
              "measured. It is extracted from the brand's own documents, packaging and "
              "creative, and is the only permitted source of publishable facts."))

    _add(s, H("Positioning", 2))
    for layer, label, gloss in (
            ("aspirational", "Aspirational", "Benefit promise and brand intent"),
            ("expressed", "Expressed", "Current in-market campaign messaging"),
            ("substantiated", "Substantiated", "Provable, citable fact")):
        items = pos.get(layer) or []
        if not items:
            continue
        _add(s, P(f"**{label}** — {gloss} ({len(items)} pillar(s))"))
        _add(s, BUL([f"{p.get('pillar')}: {p.get('statement')}" for p in items[:8]]))

    _add(s, H("Substantiated claims register", 2))
    f = L.add(value=len(claims), label="Registered claims", unit="count",
              source_artifact="claims_register", source_path="",
              evidence_class="DIAGNOSED")
    by_cat = Counter(c.get("category") for c in claims)
    by_proof = Counter(c.get("proof_type") for c in claims)
    _add(s, P(f"{len(claims)} claims are registered ({f.fact_id}). "
              f"By category: {dict(by_cat)}. By proof type: {dict(by_proof)}."))
    _add(s, TBL([{"Claim": (c.get("claim") or "")[:150],
                  "Category": c.get("category"),
                  "Proof": c.get("proof_in_source") or "—",
                  "Proof type": c.get("proof_type"),
                  "Legal": c.get("legal_ok")} for c in claims[:60]],
                caption="Claims register (first 60). The full register is the definitive "
                        "source and no content action may exceed it.",
                audience="practitioner"))

    _add(s, H("Competitive set", 2))
    _add(s, TBL([{"Brand": c.get("brand"), "Role": c.get("role"),
                  "Known strength": c.get("known_strength") or "—",
                  "Evidence": c.get("evidence") or "—"}
                 for c in comps if isinstance(c, dict) and c.get("role") != "self"],
                caption="Competitive set, product-level.", audience="both"))

    _add(s, H("Demand landscape", 2))
    ic = Counter(q.get("intent") for q in qm)
    _add(s, TBL([{"Query type": k, "Queries": v} for k, v in sorted(ic.items())],
                caption=f"{len(qm)} queries across {len(ic)} intent categories. "
                        f"Volume data was not available; weighting is structural.",
                audience="practitioner"))
    _add(s, TBL([{"Query": q.get("query"), "Type": q.get("intent"),
                  "Hypothesis": q.get("winnability_hypothesis")} for q in qm[:40]],
                caption="Query Master (first 40).", audience="practitioner"))
    return s

# ====================================================================== 3 · audit
def build_audit(rm, A, target):
    L, s = rm.ledger, rm.section("audit", "Machine reality: how AI describes the brand")
    m = A.get("benchmark_metrics") or {}
    d = A.get("delta_scorecard") or {}
    fr = d.get("framing") or {}

    _add(s, P(f"Every query in the Query Master was put to each engine repeatedly, and "
              f"the responses parsed for brand presence, position, framing and citation. "
              f"{m.get('n_runs', 0):,} successful runs form the basis of this section."))

    _add(s, H("Visibility", 2))
    _add(s, CH(C.mention_by_intent(m.get("mention_by_intent") or [], target),
               caption="Mention rate by query type, with 95% Wilson confidence intervals. "
                       "Branded and comparative queries name the brand in the prompt, so "
                       "their rates measure description accuracy rather than visibility."))
    rows = []
    fr_by = {r["intent"]: r for r in m.get("first_mention_by_intent", [])}
    t3_by = {r["intent"]: r for r in m.get("top3_by_intent", [])}
    rc_by = {r["intent"]: r for r in m.get("recommendation_by_intent", [])}
    rk_by = {r["intent"]: r for r in m.get("rank_when_present_by_intent", [])}
    rf_by = {r["intent"]: r for r in m.get("refusal_by_intent", [])}
    ad_by = {r["intent"]: r for r in m.get("advisory_rate_by_intent", [])}
    for r in m.get("mention_by_intent", []):
        i = r["intent"]
        rows.append({
            "Query type": i, "Runs": f"{r.get('runs', 0):,}",
            "Mention": f"{r.get('mention_rate', 0):.1%}",
            "95% CI": f"{r.get('ci_low', 0):.1%}–{r.get('ci_high', 0):.1%}",
            "First mention": f"{(fr_by.get(i) or {}).get('first_rate', 0):.1%}",
            "Recommended": f"{(rc_by.get(i) or {}).get('recommendation_rate', 0):.1%}"
                           if rc_by.get(i) else "—",
            "Top-3": f"{(t3_by.get(i) or {}).get('top3_rate', 0):.1%}"
                     if t3_by.get(i) else "—",
            "Refusal": f"{(rf_by.get(i) or {}).get('refusal_rate', 0):.1%}",
            "Defers to professional": f"{(ad_by.get(i) or {}).get('advisory_rate', 0):.1%}"
                                      if ad_by.get(i) else "—"})
    _add(s, TBL(rows, caption="Core visibility metrics by query type.", audience="both"))

    _add(s, H("Engine differences", 2))
    _add(s, CH(C.engine_compare(m.get("mention_by_engine_intent") or []),
               caption="Engines behave differently. These differences drive the "
                       "engine-specific overlays in §7."))
    if fr.get("by_engine"):
        _add(s, TBL([{"Engine": e,
                      "Positive": f"{v.get('pos', 0):.0%}",
                      "Neutral": f"{v.get('neu', 0):.0%}",
                      "Negative": f"{v.get('neg', 0):.0%}",
                      "Defers to professional": f"{(v.get('advisory') or 0):.0%}"}
                     for e, v in fr["by_engine"].items()],
                    caption="Framing of brand mentions by engine.", audience="both"))
    if fr.get("mentions"):
        f = L.add(value=fr.get("advisory_share") or 0,
                  label="Mentions inside a professional-deferral answer", unit="share",
                  source_artifact="delta_scorecard", source_path="framing.advisory_share",
                  n=fr.get("mentions"), evidence_class="DIAGNOSED")
        _add(s, CALL(
            f"Of {fr['mentions']} brand mentions, {fr.get('pos_share', 0):.0%} are framed "
            f"positively and {fr.get('neu_share', 0):.0%} neutrally, with "
            f"{fr.get('hedged_share', 0):.0%} hedged. {f.rendered()} sit inside an answer "
            f"that also defers to a health professional ({f.fact_id}). The engines are not "
            f"refusing to recommend — they are recommending while deferring. The objective "
            f"is therefore to be the brand named inside that deferral.", tone="info"))

    _add(s, H("Who the engines listen to", 2))
    _add(s, CH(C.provenance_bar(d.get("provenance") or {}, rm.meta.get("profile")),
               caption="Citation mix across all answers."))
    return s

# ====================================================================== 4 · delta
def build_delta(rm, A, target):
    L, s = rm.ledger, rm.section("delta", "The Delta: ground truth versus machine reality")
    d = A.get("delta_scorecard") or {}
    m = A.get("benchmark_metrics") or {}
    prov = d.get("provenance") or {}
    profile = rm.meta.get("profile", "internal")

    _add(s, P("The Delta is the gap between what the brand can substantiate and what the "
              "machines actually say — assessed per claim, per engine, with the verbatim "
              "evidence retained."))

    # ---- scorecard ----
    dims = []
    rd = m.get("retrieval_delta") or {}
    if rd.get("gap_pp") is not None:
        dims.append({"Dimension": "Retrieval", "Measure": f"{rd['gap_pp']:.0f}pp gap",
                     "Reading": "Brand wins when present, is absent when not."})
    if prov:
        dims.append({"Dimension": "Citation",
                     "Measure": f"{prov.get('self_referential_pct', 0):.1%} brand-owned "
                                f"vs {prov.get('grey_market_pct', 0):.1%} unauthorised",
                     "Reading": f"Voice-control ratio "
                                f"{prov.get('voice_control_ratio')}× — below 1.0 means "
                                f"third parties dominate."})
    fl = d.get("flatness") or {}
    if fl.get("richness_ratio_vs_leader"):
        dims.append({"Dimension": "Content",
                     "Measure": f"{fl['richness_ratio_vs_leader']}× leader density",
                     "Reading": f"{fl.get('shallow_share', 0):.0%} of mentions carry at "
                                f"most one substantive reason."})
    ev = d.get("engine_veracity") or {}
    if ev:
        avg = sum(v.get("mean_coverage", 0) for v in ev.values()) / max(len(ev), 1)
        dims.append({"Dimension": "Veracity",
                     "Measure": f"{avg:.0%} mean claim coverage",
                     "Reading": "Share of substantiated claims the engines state "
                                "accurately."})
    mbi = {r["intent"]: r["mention_rate"] for r in m.get("mention_by_intent", [])}
    if "need" in mbi:
        dims.append({"Dimension": "Intent",
                     "Measure": f"{mbi['need']:.1%} on need queries",
                     "Reading": "Visibility where purchase intent is expressed."})
    if dims:
        _add(s, TBL(dims, caption="Delta scorecard.", audience="both"))

    # ---- competitive leaderboard ----
    _add(s, H("Competitive position", 2))
    rr = d.get("reason_richness") or []
    _add(s, CH(C.richness_leaderboard(rr, target),
               caption="Argument density per 100 tokens. Presence alone does not win a "
                       "recommendation — the strength of the reasons attached does."))
    lb = {r.get("brand"): r for r in (d.get("leaderboard") or [])}
    _add(s, TBL([{"Brand": r.get("brand"),
                  "First-mention share": f"{(lb.get(r['brand']) or {}).get('first_share', 0):.1%}"
                                         if lb.get(r["brand"]) else "—",
                  "Mentions analysed": r.get("spans"),
                  "Reason density /100t": r.get("richness_per_100t"),
                  "Reason types": r.get("reason_types"),
                  "Clinical": r.get("clinical"), "Certification": r.get("certification"),
                  "Hedging": r.get("hedge")} for r in rr],
                caption="Competitive leaderboard. Density is normalised per 100 tokens, "
                        "so it is not inflated by verbosity.", audience="both"))
    rg = d.get("reason_gap") or []
    if rg:
        _add(s, TBL([{"Query type": r.get("intent"), "Dimension": r.get("dimension"),
                      "Leader": r.get("leader"),
                      "Leader /100t": r.get("leader_per_100t"),
                      "Brand /100t": r.get("brand_per_100t"),
                      "Deficit": r.get("deficit_per_100t"),
                      "Confidence": r.get("evidence_strength")} for r in rg],
                    caption="Where competitors argue harder, per 100 tokens.",
                    audience="practitioner"))

    # ---- veracity ----
    _add(s, H("Claim accuracy", 2))
    _add(s, CH(C.veracity_stack(d.get("per_claim") or []),
               caption="Status of each substantiated claim, per engine."))
    if ev:
        _add(s, TBL([{"Engine": e, "Claims assessed": v.get("claims_evaluated"),
                      "Mean coverage": f"{v.get('mean_coverage', 0):.0%}",
                      "Consistently correct": v.get("locked_in"),
                      "Fragile": v.get("fragile"), "Absent": v.get("absent"),
                      "Distorted": v.get("distorted"),
                      "Hallucination rate": f"{v.get('hallucination_rate', 0):.0%}"}
                     for e, v in ev.items()],
                    caption="Veracity by engine.", audience="both"))
    crit = [c for c in (d.get("claim_deltas") or [])
            if c.get("severity") in ("critical", "high")][:20]
    if crit:
        _add(s, TBL([{"Claim": (c.get("claim") or "")[:110],
                      "Engine": c.get("engine"), "Status": c.get("machine_status"),
                      "Coverage": f"{(c.get('coverage') or 0):.0%}",
                      "Discrepancy": (c.get("delta_note") or "")[:130]} for c in crit],
                    caption="Claims where machine reality diverges most from ground truth.",
                    audience="both"))
    rr_ = d.get("risk_register") or []
    if rr_:
        _add(s, CALL(
            "**Assertions the engines make that the brand has not substantiated.** These "
            "are not errors — they are largely plausible and specific. But they carry no "
            "substantiation trail, which in a regulated category is an exposure. Each is "
            "also a low-cost content opportunity: the engines are already making the "
            "argument.", tone="warn"))
        _add(s, TBL([{"Type": r.get("delta_type"), "Severity": r.get("severity"),
                      "Engine": r.get("engine"),
                      "Assertion": (r.get("assertion") or "")[:150],
                      "Verbatim": (r.get("evidence_quote") or "")[:150]}
                     for r in rr_[:20]],
                    caption="Risk register.", audience="both"))

    # ---- placement targets ----
    _add(s, H("Placement targets", 2))
    hl = A.get("source_hitlist") or []
    _add(s, TBL([{"Source": redact_domain(h.get("domain"), profile, h.get("source_type")),
                  "Type": h.get("source_type"), "Times cited": h.get("times_cited"),
                  "Engines": h.get("n_engines"),
                  "Brand present": f"{(h.get('brand_present_rate') or 0):.0%}",
                  "Influence × absence": h.get("brand_absence_index"),
                  "Action": h.get("bucket"), "Route": h.get("access_route")}
                 for h in hl[:25]],
                caption="Ranked by influence weighted by brand absence — the pages that "
                        "shape answers while not mentioning the brand.",
                audience="both"))

    # ---- winnability ----
    ws = d.get("winnability_summary") or {}
    if ws:
        _add(s, H("Winnability", 2))
        _add(s, TBL([{"Assessment": k, "Queries": v} for k, v in ws.items()],
                    caption="Per-query winnability triage. Unwinnable queries are "
                            "explicitly excluded from the roadmap.", audience="both"))
    return s

# ====================================================================== 5 · experiments
def build_experiments(rm, A, target):
    L, s = rm.ledger, rm.section("experiment", "Experimental evidence")
    findings = A.get("validated_findings") or []
    variants = A.get("experiment_variants") or {}
    recs = variants.get("experiments") if isinstance(variants, dict) else (variants or [])
    vmap = {(r.get("design_id") or r.get("gap_id")): r for r in (recs or [])}

    _add(s, P("Diagnosis identifies correlation. These experiments test causation: brand "
              "and competitor documents are injected into context with live retrieval "
              "disabled, one variable is changed at a time, and the outcome is measured "
              "over repeated runs at high temperature. This isolates content effects from "
              "retrieval noise — and, importantly, it cannot test retrieval itself."))

    counts = Counter(f.get("evidence_class") for f in findings)
    _add(s, TBL([{"Evidence class": EVIDENCE_LABEL.get(k, k), "Findings": v,
                  "Meaning": EVIDENCE_GLOSS.get(k, "")} for k, v in counts.items()],
                caption="Outcome of the experimental programme.", audience="both"))

    forest = []
    for f in findings:
        he = f.get("headline_effect") or {}
        val = he.get("value") if he else f.get("pooled_lift_pp")
        forest.append({"design_id": f.get("design_id"),
                       "label": (f.get("isolated_variable") or f.get("experiment_id")
                                 or "")[:34],
                       "value": val or 0, "unit": he.get("unit") or "pp",
                       "evidence_class": f.get("evidence_class")})
    _add(s, CH(C.experiment_forest(forest),
               caption="Measured effect per design. Bars at zero are well-powered nulls — "
                       "these are results, not missing data."))

    for f in findings:
        d = vmap.get(f.get("design_id")) or {}
        ec = f.get("evidence_class")
        _add(s, H(f"{f.get('design_id')} · {f.get('experiment_id')} — "
                  f"{EVIDENCE_LABEL.get(ec, ec)}", 2))
        _add(s, TBL([
            {"Field": "Question tested", "Detail": f.get("question") or d.get("question")},
            {"Field": "Hypothesis", "Detail": f.get("hypothesis") or d.get("hypothesis")
                                              or "—"},
            {"Field": "Variable isolated",
             "Detail": f.get("isolated_variable") or d.get("isolates") or "—"},
            {"Field": "Design", "Detail": f"{f.get('mode')} · "
                                          f"{f.get('runs_per_cell')} runs per arm per "
                                          f"engine"},
            {"Field": "Primary metric", "Detail": f.get("best_metric")},
            {"Field": "Measured effect",
             "Detail": (f.get("headline_effect") or {}).get("detail")
                       or f"{f.get('pooled_lift_pp')} "
                          f"{f.get('lift_unit', 'pp')}"},
            {"Field": "Verdict", "Detail": f"{f.get('verdict')} — "
                                           f"{EVIDENCE_LABEL.get(ec, ec)}"},
        ], caption="Design and outcome.", audience="both"))

        rows = [r for r in (f.get("per_engine") or []) if r.get("A_rate") is not None]
        if rows:
            _add(s, TBL([{"Engine": r.get("engine"), "Arm": r.get("arm"),
                          "Metric": r.get("metric"),
                          "Control": f"{(r.get('A_rate') or 0):.0%}",
                          "Treatment": f"{(r.get('B_rate') or 0):.0%}",
                          "Effect": f"{(r.get('lift_pp') or 0):+.1f}pp",
                          "n": f"{r.get('n_A')}/{r.get('n_B')}",
                          "p (FDR)": r.get("p_fdr"),
                          "Significant": "Yes" if r.get("significant") else "No",
                          "Negative control": "Yes" if r.get("is_negative_control") else ""}
                         for r in rows],
                        caption="Arm-by-arm results, FDR-corrected across comparisons.",
                        audience="practitioner"))
        rk = f.get("rank_analysis") or []
        if rk:
            _add(s, TBL([{"Engine": r.get("engine"), "Arm": r.get("arm"),
                          "Mean rank (control)": r.get("A_mean_rank"),
                          "Mean rank (treatment)": r.get("B_mean_rank"),
                          "Improvement": r.get("improvement"),
                          "Top-1 control": f"{(r.get('A_top1_pct') or 0):.0f}%",
                          "Top-1 treatment": f"{(r.get('B_top1_pct') or 0):.0f}%",
                          "p": r.get("p_value"),
                          "Significant": "Yes" if r.get("significant") else "No"}
                         for r in rk],
                        caption="Rank analysis. Where the binary metrics saturate, ordinal "
                                "rank retains sensitivity.", audience="practitioner"))
        ea = f.get("element_attribution")
        if ea:
            _add(s, TBL([{"Element added": k, "Marginal effect": f"{v:+.1f}pp"}
                         for k, v in ea.items()],
                        caption="Element attribution from the cumulative ladder. This is "
                                "what changed; causal attribution requires the ladder to "
                                "have been strictly cumulative.", audience="practitioner"))
        for lim in (f.get("interpretation_limits") or []):
            _add(s, CALL(f"**Interpretation limit.** {lim}", tone="warn"))
        for w in (f.get("variance_warnings") or []):
            _add(s, P(f"_Note: {w}_", audience="practitioner"))

    # ---- cross-engine ----
    eng_specific = [f for f in findings
                    if f.get("evidence_class") == "PROVEN_ENGINE_SPECIFIC"]
    universal = [f for f in findings if f.get("evidence_class") == "PROVEN"]
    if eng_specific or universal:
        _add(s, H("Universal versus engine-specific", 2))
        rows = [{"Finding": f.get("design_id"),
                 "Transfers": "All engines tested" if f in universal
                              else ", ".join(f.get("rank_engines") or ["one engine"]),
                 "Implication": "Belongs in the overall strategy" if f in universal
                                else "Belongs in an engine overlay"}
                for f in universal + eng_specific]
        _add(s, TBL(rows, audience="both"))

    _add(s, CALL(
        "**A boundary of this method.** Documents are injected into context, which "
        "presupposes that retrieval has already succeeded. These experiments therefore "
        "measure how content performs *once found*; they cannot measure whether it gets "
        "found. Retrieval conclusions in this report rest on the audit's citation "
        "evidence, not on experiment.", tone="info"))
    return s

# ====================================================================== 6 · substantiation
def build_substantiation(rm, A, target):
    backlog = A.get("substantiation_backlog") or []
    if not backlog:
        return None
    L, s = rm.ledger, rm.section("substantiation",
                                 "Claims the engines make that the brand has not "
                                 "substantiated")
    crit = [b for b in backlog if b.get("severity") == "critical"]
    _add(s, CALL(
        f"AI engines assert {len(backlog)} factual claims about the brand that are absent "
        f"from the substantiated claims register — {len(crit)} rated critical. Because "
        f"these claims cannot be published or defended, they cannot be tested; the "
        f"remedy is substantiation, not content. This is simultaneously the lowest-cost "
        f"content backlog available and, in a regulated category, a compliance matter.",
        tone="danger"))
    _add(s, TBL([{"Ref": b.get("cluster_id"), "Severity": b.get("severity"),
                  "Assertion to substantiate":
                      (b.get("assertion_to_substantiate") or "")[:260],
                  "Why it cannot be tested": (b.get("reason") or "")[:160],
                  "Next step": b.get("next_step")} for b in backlog],
                caption="Substantiation backlog. Route each to regulatory review before "
                        "any content action depends on it.", audience="both"))
    return s

# ====================================================================== 6b · visual
# ====================================================================== 6b · visual
def build_visual(rm, A, target):
    """Creative and multimodal findings. Without this section a visual result renders as
    a table of arm letters with no images, which is not a creative finding."""
    from store import load_bytes

    findings = [f for f in (A.get("validated_findings") or [])
                if f.get("mode") == "creative" or f.get("visual_validity")]
    man = A.get("experiment_visual_assets") or {}
    leg = A.get("bench_legibility") or {}
    if not findings and not man:
        return None

    L, s = rm.ledger, rm.section("visual", "Creative and multimodal evidence")
    _add(s, P("Creative variants were tested against an identical competitive field, with "
              "the accompanying text held constant so that any difference is attributable "
              "to the creative. Media payload delivery is verified on every run: a visual "
              "result where the image never reached the model is reported as invalid, not "
              "as a null."))

    # Declares WHICH question each bench answered. The text mode and attribution are the
    # two choices that determine what the finding can claim, so they belong in the prose
    # rather than only in a table.
    _bench = [f for f in findings if f.get("mode") == "creative"]
    if _bench:
        _tp = sorted({f.get("text_presence") for f in _bench if f.get("text_presence")})
        _at = sorted({f.get("attribution") for f in _bench if f.get("attribution")})
        _add(s, P(f"Text mode: {', '.join(_tp) or 'not recorded'} · attribution: "
                  f"{', '.join(_at) or 'not recorded'}. "
                  + ("In *isolation* attribution the accompanying text is byte-identical "
                     "across arms, so the creative is the only variable. "
                     if "isolation" in _at else "")
                  + ("In *bundle* attribution the copy varies with the creative, so the "
                     "finding describes the shipped package and no part of the effect is "
                     "attributable to the creative alone. "
                     if "bundle" in _at else "")
                  + "These results describe how machines read the creative. They say "
                    "nothing about human response, and are not a substitute for "
                    "conventional creative testing."))

    for f in findings:
        did = f.get("design_id")
        _add(s, H(f"{did} · {f.get('experiment_id')}", 2))
        bs = f.get("bakeoff_summary") or {}
        _add(s, TBL([
            {"Field": "Question tested", "Detail": f.get("question") or "—"},
            {"Field": "Variable isolated", "Detail": f.get("isolated_variable") or "—"},
            {"Field": "Accompanying text", "Detail": f.get("text_presence") or "—"},
            {"Field": "Attribution", "Detail": f.get("attribution") or "—"},
            {"Field": "Arms", "Detail": f.get("n_visual_arms") or len(man.get(did, {}))},
            {"Field": "Runs per arm per engine", "Detail": f.get("runs_per_cell")},
            {"Field": "Primary metric", "Detail": f.get("best_metric")},
            {"Field": "Outcome", "Detail": bs.get("decision") or f.get("verdict")},
        ], caption="Design and outcome.", audience="both"))

        # ---- the assets actually tested ----
        for arm, meta in sorted((man.get(did) or {}).items()):
            cap = (f"Arm {arm} — {meta.get('label') or ''} · "
                   f"{meta.get('width', '?')}×{meta.get('height', '?')} · "
                   f"`{meta.get('digest')}`")
            if meta.get("is_video"):
                _add(s, P(f"**Arm {arm}** — video asset, "
                          f"{round((meta.get('n_bytes') or 0) / 1e6, 1)} MB "
                          f"(`{meta.get('digest')}`). Not reproduced here."))
                continue
            data = load_bytes(rm.meta.get("parent"), target, meta.get("path", ""))
            _add(s, CH(data, caption=cap) if data else P(f"_{cap} — asset unavailable_"))

        lg = leg.get(did) or f.get("legibility") or {}
        if lg.get("comparison"):
            _add(s, H("What a machine can read off each asset", 3))
            _add(s, P("Read by a vision model before the runs. This is the material the "
                      "engine actually has to work with — anything not listed here is "
                      "invisible to it, however prominent it is to a person."))
            _add(s, TBL([{"Arm": r.get("arm"), "Brand read": r.get("brand_read") or "—",
                          "Legibility": r.get("legibility") or "—",
                          "Subject": r.get("subject") or "—",
                          "Claims read": r.get("claims") or "—",
                          "Not readable": r.get("unreadable") or "—"}
                         for r in lg["comparison"]],
                        caption="Machine legibility per arm.", audience="both"))
            for c in (lg.get("cautions") or []):
                _add(s, CALL(c, tone="warn"))
            for b in (lg.get("blockers") or []):
                _add(s, CALL(f"**Blocked before the run.** {b}", tone="danger"))

        for r in (f.get("bakeoff") or []):
            _add(s, H(f"{r['engine']} · {r['metric']}", 3))
            # mde_pp is None at both clamps: a floored set has no rate to detect a
            # departure from, a saturated one has no headroom. Formatting it as a number
            # crashes on None.
            _mde = (f"detectable effect at this sample size ≥{r['mde_pp']}pp"
                    if r.get("mde_pp") is not None
                    else f"minimum detectable effect {r.get('mde_note', 'undefined')}")
            _add(s, TBL([{"Rank": x["rank"], "Arm": x["arm"],
                          "Rate": f"{x['rate']:.1%}",
                          "95% CI": f"{x['ci_lo']:.1%}–{x['ci_hi']:.1%}",
                          "n": x["n"]} for x in (r.get("ranking") or [])],
                        caption=f"Ranking. Omnibus {r['omnibus_test']} "
                                f"p={r['omnibus_p']:.3f}"
                                + (f" — {r['omnibus_note']}" if r.get("omnibus_note")
                                   else "") + f"; {_mde}.",
                        audience="both"))
            if r.get("pairwise"):
                _add(s, TBL([{"A": x["arm_a"], "B": x["arm_b"],
                              "Difference": f"{x['diff_pp']:+.1f}pp",
                              "95% CI": f"{x['ci_lo_pp']:+.1f} to {x['ci_hi_pp']:+.1f}pp",
                              "p (FDR)": x.get("p_fdr"),
                              "Distinguishable": "Yes" if x.get("significant") else "No"}
                             for x in r["pairwise"]],
                            caption="Pairwise comparisons, FDR-corrected across all pairs.",
                            audience="practitioner"))
            _add(s, P(f"**Reading.** {r['decision']}"))

        # ---- the two readings a ranking table cannot convey ----
        if bs.get("magnitude_diverges"):
            _add(s, CALL(
                f"**Engines agree on the creative, not on the size of the effect.** The "
                f"gap between arms differs by {bs['gap_spread_pp']:.0f} percentage points "
                f"between engines ({bs.get('winners_by_engine')}), so the pooled figure "
                f"describes neither. Quote the per-engine numbers.", tone="warn"))
        if bs.get("gap_is_lower_bound_on"):
            _add(s, CALL(
                f"**The advantage is a lower bound on {bs['gap_is_lower_bound_on']}.** The "
                f"winning arm saturated the metric, so the true difference may be larger — "
                f"and no further creative could score higher on it.", tone="info"))
        for sec in (bs.get("secondary_results") or []):
            _add(s, CALL(
                f"**A second validated result.** On {sec['engine']}, "
                f"`{sec['metric']}` moved from {sec['from_pct']:.0f}% to "
                f"{sec['to_pct']:.0f}% ({sec['gap_pp']:+.0f}pp).", tone="good"))
        if bs.get("mention_without_recommendation_note"):
            _add(s, CALL(f"**This is not a creative problem.** "
                         f"{bs['mention_without_recommendation_note']}", tone="warn"))

        vv = f.get("visual_validity")
        if vv:
            _add(s, CALL(f"**Media payload — {vv['verdict']}.** {vv['note']}",
                         tone={"OK": "good", "PARTIAL": "warn",
                               "INVALID": "danger"}.get(vv["verdict"], "info")))
        for lim in (f.get("interpretation_limits") or []):
            _add(s, CALL(f"**Interpretation limit.** {lim}", tone="warn"))

    if _bench:
        _add(s, CALL(
            "**Where these results apply.** Creative was tested with the asset in "
            "context, so these findings apply directly to multimodal surfaces — image "
            "search, shopping, an assistant shown a photograph. On a text-only answer a "
            "creative acts only where the pack copy is transcribed into machine-readable "
            "text, which is a content and technical dependency rather than a creative "
            "one. The roadmap projection discounts these effects on its text-visibility "
            "axis for that reason, and reports the multimodal effect separately.",
            tone="info"))
    return s

# ====================================================================== 7 · strategy
def build_strategy(rm, A, target):
    strat = A.get("geo_strategy") or {}
    if not strat:
        return None
    L, s = rm.ledger, rm.section("strategy", "GEO strategy")
    ov = strat.get("overall_strategy") or {}

    if ov.get("narrative"):
        _add(s, CALL(ov["narrative"], tone="info"))
    if strat.get("retrieval_is_primary_constraint"):
        _add(s, CALL(
            "**Strategic reframing.** Content-form experiments returned null results at "
            "adequate power while unprompted visibility remains very low. The evidence "
            "therefore points to retrieval and citation — not messaging quality — as the "
            "binding constraint. Visibility gaps are routed to the Technical and "
            "Corroboration pillars accordingly.", tone="warn"))

    if ov.get("structural_truths"):
        _add(s, H("Structural truths", 2))
        _add(s, TBL([{"Holds across engines": t.get("truth"),
                      "Evidence": t.get("evidence")} for t in ov["structural_truths"]],
                    audience="both"))

    _add(s, H("Pillars", 2))
    actions = strat.get("actions") or []
    for pillar, pdat in (strat.get("pillars") or {}).items():
        acts = [a for a in actions if a.get("pillar") == pillar]
        if pdat.get("_skipped"):
            _add(s, P(f"**{pillar}** — not addressed. {pdat.get('thesis')}",
                      audience="practitioner"))
            continue
        stance = (pdat.get("_evidence_stance") or {}).get("stance", "")
        _add(s, H(f"{pillar}", 3))
        _add(s, P(f"**Objective.** {pdat.get('objective')}"))
        _add(s, P(f"**Thesis.** {pdat.get('thesis')}"))
        _add(s, P(f"_Mechanism: {pdat.get('mechanism')} · evidence standing: "
                  f"{stance.replace('_', ' ').lower()}_", audience="practitioner"))
        if acts:
            _add(s, TBL([{"Ref": a.get("action_id"), "Action": a.get("title"),
                          "Evidence": EVIDENCE_LABEL.get(a.get("evidence_class"),
                                                         a.get("evidence_class")),
                          "Horizon": str(a.get("horizon", "")).split(" (")[0],
                          "Clock": a.get("clock"), "Owner": a.get("owner"),
                          "Effort": a.get("effort"),
                          "Expected effect": _fmt_effect(a.get("expected_effect"))}
                         for a in sorted(acts,
                                         key=lambda x: -(x.get("priority_score") or 0))],
                        caption=f"{pillar} actions.", audience="both"))
            detail = []
            for a in sorted(acts, key=lambda x: -(x.get("priority_score") or 0)):
                notes = " ".join(x for x in (a.get("sequencing_note"),
                                             a.get("measurement_note")) if x)
                detail.append({
                    "Ref": a.get("action_id"),
                    "What to do": a.get("what"),
                    "Rationale": a.get("why"),
                    "Measure": f"{a.get('leading_indicator')} → "
                               f"{a.get('lagging_indicator')}",
                    "Stop if": a.get("kill_criterion"),
                    "Validation needed": a.get("validating_experiment") or "—",
                    "Sequencing": notes or "—"})
            _add(s, TBL(detail, caption=f"{pillar} — action detail.",
                        audience="practitioner"))
        if pdat.get("do_not_do"):
            _add(s, TBL([{"Not doing": x.get("action"), "Reason": x.get("reason"),
                          "Basis": x.get("basis")} for x in pdat["do_not_do"]],
                        caption=f"{pillar} — explicitly excluded.", audience="both"))

    if strat.get("engine_overlays"):
        _add(s, H("Engine overlays", 2))
        _add(s, P("Engines differ in what they cite and how they frame recommendations. "
                  "These overlays are derived from this engagement's own measurements, "
                  "not from general assumptions about each engine."))
        for eng, o in strat["engine_overlays"].items():
            _add(s, H(eng, 3))
            _add(s, P(f"**Behaviour.** {o.get('behavioural_profile')}"))
            if o.get("what_changes_here"):
                _add(s, TBL([{"Change": x.get("action"),
                              "Why this engine": x.get("why_this_engine"),
                              "Evidence": x.get("evidence")}
                             for x in o["what_changes_here"]], audience="both"))
            if o.get("deprioritise_here"):
                _add(s, BUL([f"Deprioritise: {x.get('action')} — {x.get('why')}"
                             for x in o["deprioritise_here"]], audience="practitioner"))
            if o.get("placement_targets"):
                _add(s, TBL([{"Target": redact_domain(x.get("domain"),
                                                      rm.meta.get("profile")),
                              "Why": x.get("why")} for x in o["placement_targets"]],
                            audience="both"))
            if o.get("engine_specific_risk"):
                _add(s, CALL(o["engine_specific_risk"], tone="warn",
                             audience="practitioner"))

    briefs = A.get("content_briefs") or []
    if briefs:
        _add(s, H("Content briefs", 2))
        _add(s, P("Each brief traces a chain: observed gap → hypothesis → experiment → "
                  "validated finding → production requirement. Briefs derived from "
                  "caveated evidence carry that caveat."))
        for b in briefs:
            _add(s, H(b.get("title") or b.get("action_id"), 3))
            _add(s, TBL([
                {"Field": "Reference", "Detail": f"{b.get('action_id')} · "
                                                 f"{b.get('pillar')} · "
                                                 f"{b.get('horizon')}"},
                {"Field": "Evidence",
                 "Detail": f"{EVIDENCE_LABEL.get(b.get('evidence_class'), '')} "
                           f"({', '.join(b.get('evidence_refs') or []) or 'diagnostic'})"},
                {"Field": "Objective", "Detail": b.get("objective")},
                {"Field": "Why it works", "Detail": b.get("why_it_works")},
                {"Field": "Deliverable", "Detail": b.get("deliverable")},
                {"Field": "Placement",
                 "Detail": "; ".join(b.get("placement") or [])},
                {"Field": "Corroboration required",
                 "Detail": "; ".join(b.get("corroboration") or [])},
                {"Field": "Success metric", "Detail": b.get("success_metric")},
                {"Field": "Measurement", "Detail": b.get("measurement")},
            ], audience="practitioner"))
            if b.get("must_contain"):
                _add(s, BUL([f"Must contain: {x}" for x in b["must_contain"]],
                            audience="practitioner"))
            if b.get("must_not_contain"):
                _add(s, BUL([f"Must NOT contain: {x}" for x in b["must_not_contain"]],
                            audience="practitioner"))
            if b.get("claims_required"):
                _add(s, BUL([f"Approved claim: {x}" for x in b["claims_required"]],
                            audience="practitioner"))
            for c in (b.get("caveats") or []):
                _add(s, CALL(f"**Caveat.** {c}", tone="warn", audience="practitioner"))
            if b.get("proven_content"):
                _add(s, Block("quote", b["proven_content"][:2400],
                              caption="Tested content pattern — publish this structure.",
                              audience="practitioner"))
    return s

def _fmt_effect(eff):
    eff = eff or {}
    basis = eff.get("basis")
    if basis == "none" or eff.get("value") in (None, ""):
        return f"{eff.get('qualitative', 'unknown')} (no measured basis)"
    try:
        v = float(eff.get("value"))
    except Exception:
        return str(eff.get("value"))
    tag = {"measured": "measured", "inferred": "inferred"}.get(basis, basis or "")
    src = f" · {eff.get('source')}" if eff.get("source") else ""
    return f"{v:+.1f}{eff.get('unit', 'pp')} ({tag}{src})"

# ====================================================================== 8 · roadmap
def build_roadmap(rm, A, target):
    rmap = A.get("roadmap") or {}
    strat = A.get("geo_strategy") or {}
    if not rmap:
        return None
    L, s = rm.ledger, rm.section("roadmap", "Implementation roadmap")
    idx = {a.get("action_id"): a for a in (strat.get("actions") or [])}

    _add(s, P("Actions are sequenced by dependency and packed against stated team "
              "capacity. Each wave carries a measurement contract so that progress is "
              "verifiable rather than asserted."))
    if rmap.get("sequencing_rationale"):
        _add(s, P(f"**Sequencing.** {rmap['sequencing_rationale']}"))

    for w in (rmap.get("waves") or []):
        packed = (rmap.get("waves_packed") or {}).get(w.get("wave"), {})
        _add(s, H(f"{w.get('wave')} — {w.get('theme')}", 2))
        _add(s, P(f"**Objective.** {w.get('objective')}"))
        rows = []
        for a in (w.get("actions") or []):
            full = idx.get(a.get("action_id"), {})
            rows.append({"Ref": a.get("action_id"), "Action": a.get("title"),
                         "Pillar": full.get("pillar"), "Owner": a.get("owner"),
                         "Effort": a.get("effort"),
                         "Evidence": EVIDENCE_LABEL.get(a.get("evidence_class"),
                                                        a.get("evidence_class")),
                         "Expected effect": _fmt_effect(full.get("expected_effect")),
                         "Depends on": ", ".join(a.get("depends_on") or []) or "—",
                         "KPI": full.get("lagging_indicator") or "—"})
        if rows:
            _add(s, TBL(rows, caption=f"{w.get('wave')} · capacity "
                                      f"{packed.get('utilisation')}", audience="both"))
        mc = w.get("measurement_contract") or {}
        _add(s, TBL([
            {"Field": "Queries to re-audit",
             "Detail": f"{len(mc.get('re_audit_queries') or [])} query(ies)"},
            {"Field": "Runs per query", "Detail": mc.get("n_runs")},
            {"Field": "Engines", "Detail": ", ".join(mc.get("engines") or [])},
            {"Field": "Primary metric", "Detail": mc.get("primary_metric")},
            {"Field": "Expected change",
             "Detail": f"{mc.get('expected_delta')} ({mc.get('basis')})"},
            {"Field": "Re-audit date", "Detail": mc.get("re_audit_date")},
            {"Field": "Stop if", "Detail": mc.get("kill_criterion")},
            {"Field": "Exit criteria", "Detail": w.get("exit_criteria")},
        ], caption="Measurement contract.", audience="both"))
        if w.get("risks"):
            _add(s, TBL([{"Risk": r.get("risk"), "Mitigation": r.get("mitigation"),
                          "Rollback": r.get("rollback")} for r in w["risks"]],
                        audience="practitioner"))

    if rmap.get("projection"):
        _add(s, H("Projected trajectory", 2))
        _add(s, CH(C.projection(rmap["projection"], rmap.get("baseline_pct", 0)),
                   caption="Illustrative projection, not a forecast."))
        _add(s, TBL([{"Wave": str(t["wave"]).split(" (")[0],
                      "Added": f"{t['added_pp']:+.1f}pp",
                      "Projected": f"{t['projected_pct']:.1f}%",
                      "Range": f"{t['low_pct']:.1f}–{t['high_pct']:.1f}%",
                      "Dominant evidence":
                          EVIDENCE_LABEL.get(t.get("dominant_evidence"), "—")}
                     for t in rmap["projection"]], audience="both"))
        _add(s, CALL(rmap.get("projection_note", ""), tone="info"))
        if rmap.get("projection_warning"):
            _add(s, CALL(rmap["projection_warning"], tone="warn"))
    if rmap.get("critical_path"):
        _add(s, P(f"**Critical path.** {' → '.join(rmap['critical_path'])}",
                  audience="practitioner"))
    if rmap.get("assumptions"):
        _add(s, BUL([f"Assumption: {a}" for a in rmap["assumptions"]],
                    audience="practitioner"))
    return s

# ====================================================================== 9 · methodology
def build_methodology(rm, A, target):
    L, s = rm.ledger, rm.section("method", "Methodology, limitations and traceability",
                                 audience="practitioner")
    m = A.get("benchmark_metrics") or {}
    d = A.get("delta_scorecard") or {}
    cov = rm.coverage

    _add(s, H("Framework", 2))
    _add(s, BUL([
        "**Discover** — extract brand positioning and every substantiable claim from the "
        "brand's own material; mine real consumer query language; define the competitive "
        "set and KPI targets.",
        "**Audit** — put every query to each engine repeatedly, logging brand presence, "
        "ordinal position, framing, refusals and cited sources.",
        "**Benchmark** — compare machine output against the claims register per claim per "
        "engine, with verbatim quote evidence; quantify citation provenance and "
        "argument density against competitors.",
        "**Experiment** — inject brand and frozen competitor documents into context with "
        "retrieval disabled, vary one element at a time, and measure over repeated runs.",
        "**Optimise** — route diagnosed gaps to strategy pillars, back each action with "
        "its evidence class, and sequence into capacity-bounded waves.",
    ]))

    _add(s, H("Measurement", 2))
    _add(s, BUL([
        f"Engines: {', '.join(cov.get('engines') or [])}.",
        f"Successful audit runs: {m.get('n_runs', 0):,}.",
        f"Query categories audited: {', '.join(cov.get('intents_audited') or [])}.",
        "Brand detection uses word-boundary matching over a derived alias set, with "
        "negation handling — an anti-recommendation counts as a mention but is recorded "
        "as negatively framed and excluded from share-of-voice numerators.",
        "Confidence intervals are Wilson intervals on run proportions.",
        "Claim veracity is assessed by a language model constrained to verbatim quotation: "
        "any judgement whose quote is not a literal substring of the source is discarded.",
        "Experiment significance uses two-proportion tests with Benjamini–Hochberg FDR "
        "correction across all comparisons; rank effects use Mann–Whitney U.",
        "Document position is counterbalanced within each arm to remove order bias.",
    ]))
    cal = d.get("calibration") or {}
    if cal.get("agreement") is not None:
        _add(s, P(f"**Judge calibration.** Agreement between the language-model judge and "
                  f"an independent keyword method was {cal['agreement']:.0%} across "
                  f"{cal.get('n_compared')} claims, with {cal.get('disagreements')} "
                  f"disagreements reviewed."))

    _add(s, H("Limitations", 2))
    lims = [
        "AI engines and their retrieval indexes change continuously. These findings are a "
        "point-in-time measurement and will drift.",
        "Confidence intervals treat repeated runs of the same query as independent "
        "observations. Because runs within a query are correlated, the true intervals are "
        "wider than those reported.",
        "Experiments inject documents into context, which presupposes retrieval. They "
        "measure how content performs once found, not whether it is found.",
        "Competitor documents are constructed from this audit's own evidence and from the "
        "pages the engines cited. They are representative, not the competitors' actual "
        "published content.",
    ]
    if cov.get("intents_not_audited"):
        lims.append(f"Query categories not audited: "
                    f"{', '.join(cov['intents_not_audited'])}. No conclusions extend to "
                    f"them.")
    if not cov.get("memory_vs_retrieval_separated"):
        lims.append("Only retrieval mode was audited, so memory-driven and "
                    "retrieval-driven behaviour cannot be distinguished.")
    if cov.get("vision_verified") is not True:
        lims.append("Visual and multimodal behaviour was not assessed.")
    if cov.get("n_proven", 0) <= 1:
        lims.append(f"The experimental programme yielded {cov.get('n_proven', 0)} finding "
                    f"proven at adequate power. Recommendations are correspondingly "
                    f"weighted towards diagnosed evidence, and each action states its "
                    f"class.")
    _add(s, BUL(lims))

    _add(s, H("Data provenance", 2))
    _add(s, TBL([{"Artifact": k, "Content hash": v}
                 for k, v in (rm.meta.get("input_hashes") or {}).items()],
                caption="Input artifact hashes. The report is reproducible from these; a "
                        "changed hash means the underlying analysis has been re-run.",
                audience="practitioner"))

    _add(s, H("Traceability", 2))
    _add(s, P("Every quantitative statement in this report carries a fact reference. The "
              "table below maps each to its source artifact and path."))
    _add(s, TBL(rm.ledger.table(), caption="Fact ledger.", audience="practitioner"))
    return s

# ====================================================================== orchestrator
def build_report_model(parent, target, category, profile="internal",
                       provisional_reasons=None) -> ReportModel:
    from datetime import datetime, timezone
    A = _load(parent, target)
    cov = coverage_declaration(parent, target)

    reasons = list(provisional_reasons or [])
    if not A.get("geo_strategy"):
        reasons.append("The strategy phase has not been completed.")
    if not A.get("validated_findings"):
        reasons.append("No experimental findings are available.")
    if (cov.get("n_proven") or 0) == 0 and (cov.get("n_experiments") or 0) > 0:
        reasons.append("No experiment reached adequate power with a positive result; "
                       "strategy rests on diagnosed evidence.")

    rm = ReportModel(meta={
        "brand": target, "parent": parent, "category": category, "profile": profile,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator_version": "1.0.0",
        "input_hashes": input_hashes(parent, target),
    }, ledger=FactLedger(), coverage=cov,
        provisional=bool(reasons), provisional_reasons=reasons)

    build_front_matter(rm, A, parent, target, category)
    build_exec_summary(rm, A, target)
    build_discovery(rm, A, target)
    build_audit(rm, A, target)
    build_delta(rm, A, target)
    build_experiments(rm, A, target)
    build_visual(rm, A, target)
    build_substantiation(rm, A, target)
    build_strategy(rm, A, target)
    build_roadmap(rm, A, target)
    build_methodology(rm, A, target)

    for name in ("geo_strategy", "roadmap", "content_briefs", "validated_findings"):
        if not A.get(name):
            rm.warnings.append(f"Artifact '{name}' is missing — the corresponding section "
                               f"is omitted.")
    return rm