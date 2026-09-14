"""LLM-driven GEO strategy synthesis.

S0 assemble → S1 route to pillars → S2 per-pillar strategy → S3 cross-engine synthesis
→ S4 engine overlays → S5 horizon split → S6 adversarial critic + deterministic validation

Design rules
------------
* NULLS ARE EVIDENCE. A well-powered null generates a "do not do" item, which is often
  more valuable than another positive. Nulls are passed into every pillar prompt.
* Every action carries an EVIDENCE CLASS. Unevidenced actions are capped and cannot enter
  Wave 1.
* Claims are hard-constrained to the substantiated register (regulatory).
* Retrieval plays (publishable now) are separated from Memory plays (brand fame, slow).
* Engine overlays are DERIVED from this engagement's own Tier-A data — nothing about any
  engine is hardcoded.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

from metrics_spec import fold


PILLARS = ["Content", "Technical/Structured-Data", "Visual/Multimodal",
           "Corroboration", "Social/Community", "Claims-Substantiation"]

PILLAR_SCOPE = {
    "Content": "Messaging, claim specificity, quantified and extractable facts, "
               "reason density. Fed by content-gap and reason-richness analysis.",
    "Technical/Structured-Data": "Schema, answer capsules, machine-readable PDPs, "
                                 "comparison tables, crawlability and indexability of "
                                 "brand surfaces. Fed by experiment proof and by whether "
                                 "brand content is retrievable at all.",
    "Visual/Multimodal": "On-pack claims, alt-text, image legibility to vision models. "
                         "Fed by visual-gap analysis.",
    "Corroboration": "Editorial placement, PR, expert quotes, third-party echo, "
                     "correcting third-party pages. Fed by citation provenance and the "
                     "source hit-list.",
    "Social/Community": "Forums and community sources where engines weight UGC. "
                        "Fed by source-mix analysis.",
    "Claims-Substantiation": "Certifying and substantiating claims so they become "
                             "citable — including claims engines already assert on the "
                             "brand's behalf. Fed by the veracity audit and reverse scan.",
}

EVIDENCE_WEIGHT = {
    "PROVEN": 1.00,                    # experiment significant, transfers across engines
    "PROVEN_ENGINE_SPECIFIC": 0.70,    # significant on >=1 engine
    "PROVEN_WITH_CAVEAT": 0.60,        # significant but confounded / underpowered
    "DIAGNOSED": 0.45,                 # strong benchmark evidence, no experiment
    "DISPROVEN": 0.00,                 # tested and failed — must not be recommended
    "HYPOTHESIS": 0.20,                # neither
}
HYPOTHESIS_EFFORT_CAP = 0.15           # max share of roadmap effort on unevidenced work

MEMORY_PROXY = {
    "Corroboration": "count of third-party editorial citations naming the brand",
    "Social/Community": "count of community threads where the brand is recommended",
    "Claims-Substantiation": "share of register claims with citable third-party proof",
    "Content": "share of brand mentions carrying >=2 substantive reasons",
    "Technical/Structured-Data": "share of brand surfaces with valid structured data",
    "Visual/Multimodal": "share of on-pack claims machine-readable from imagery",
}

PILLAR_TEST_TEMPLATE = {
    "Content": "E2 (specificity) or E4 (gap-closure)",
    "Technical/Structured-Data": "E1 (structure isolation) or E5 (format)",
    "Corroboration": "E3 (third-party corroboration)",
    "Visual/Multimodal": "E7 (visual extraction)",
    "Claims-Substantiation": "E4 (gap-closure), once the claim is substantiated",
    "Social/Community": "E3 (community-sourced corroboration)",
}

NULL_VERDICTS = {"no_lift", "rejected", "already_winning", "lost"}
WIN_VERDICTS = {"validated", "validated_mention", "validated_rank",
                "engine-specific", "engine-specific_mention"}

DISPROVEN_PILLARS = {
    # template → the pillar whose mechanism the null disproves
    "E0": ["Content", "Technical/Structured-Data"],
    "E1": ["Technical/Structured-Data", "Content"],
    "E5": ["Technical/Structured-Data"],
    "E2": ["Content"], "E4": ["Content"], "E3": ["Corroboration"],
}

def pillar_evidence_verdict(si, pillar):
    """Was this pillar's own mechanism tested, and what happened?

    Evidence hierarchy: a WELL-POWERED null outweighs an UNDERPOWERED win on the same
    mechanism. Without this, one n=20 result cancels two N=100 nulls.
    """
    tested, disproven, proven, weak_proven = [], [], [], []
    for n in si["nulls"]:
        if pillar in DISPROVEN_PILLARS.get(str(n.get("template")), []):
            tested.append(n["design_id"])
            if not n.get("underpowered"):
                disproven.append(n)
    for w in si["wins"]:
        if pillar in DISPROVEN_PILLARS.get(str(w.get("template")), []) \
                or w.get("pillar") == pillar:
            (weak_proven if w.get("underpowered") else proven).append(w)

    if disproven and not proven:
        stance = "DISPROVEN_MECHANISM"
        guidance = (
            f"This pillar's primary mechanism was TESTED AND FAILED at adequate power "
            f"({', '.join(d['design_id'] for d in disproven)}). You must NOT recommend "
            f"more of the same.")
        if weak_proven:
            guidance += (
                f" Note {', '.join(w['design_id'] for w in weak_proven)} suggested an "
                f"effect but was UNDERPOWERED — it does not override the well-powered "
                f"nulls. If you build on it, label the action DIAGNOSED, cap the expected "
                f"effect, and name the replication experiment.")
        guidance += (" Either return a minimal actions list confined to a DIFFERENT "
                     "mechanism within this pillar, stating the difference explicitly, or "
                     "return an empty actions list and explain in the thesis why this "
                     "pillar is not the lever. An honest empty pillar is correct here.")
    elif proven:
        stance = "PROVEN_MECHANISM"
        guidance = (f"This pillar has PROVEN evidence at adequate power "
                    f"({', '.join(p['design_id'] for p in proven)}). Lead with it and "
                    f"prioritise scaling what worked.")
    elif weak_proven:
        stance = "INCONCLUSIVE"
        guidance = (f"The only signal for this pillar "
                    f"({', '.join(w['design_id'] for w in weak_proven)}) was UNDERPOWERED. "
                    f"All actions must be DIAGNOSED with a named replication experiment, "
                    f"and expected effects must use basis='none' or a capped 'inferred' "
                    f"value.")
    elif tested:
        stance = "INCONCLUSIVE"
        guidance = ("Tested but underpowered. Treat actions as DIAGNOSED and name the "
                    "validating experiment.")
    else:
        stance = "UNTESTED"
        guidance = ("No experimental evidence. All actions must be DIAGNOSED with a named "
                    "validating experiment.")
    return {"stance": stance, "guidance": guidance,
            "disproven_ids": [d["design_id"] for d in disproven],
            "proven_ids": [p["design_id"] for p in proven],
            "weak_proven_ids": [w["design_id"] for w in weak_proven],
            "tested_ids": tested}

def pillar_budget(si, pillar, stance):
    """Action budget per pillar. Evidence strength — not gap count — determines how much
    of the strategy a pillar owns. Without this, the pillar with the largest gap pile wins
    regardless of whether its mechanism was disproven."""
    base = {"PROVEN_MECHANISM": 6, "UNTESTED": 4,
            "INCONCLUSIVE": 3, "DISPROVEN_MECHANISM": 2}.get(stance, 4)
    if pillar == "Claims-Substantiation" and si.get("substantiation_backlog"):
        base = max(base, 4)
    pv = (si.get("delta") or {}).get("provenance") or {}
    if pillar == "Corroboration" and (pv.get("voice_control_ratio") or 1) < 0.5:
        base += 2                      # citation control is broken — carries more weight
    if pillar == "Technical/Structured-Data" and si.get("_retrieval_constraint"):
        base += 2                      # retrieval is the binding constraint
    return base

# ====================================================================== S0
def assemble_strategy_input(parent, target, category, constraints=None):
    """Collect every artifact the strategy depends on, and normalise the experiment
    evidence into wins / nulls / not-testable."""
    from store import load_artifact
    A = lambda n: load_artifact(parent, target, n)      # noqa: E731

    findings = A("validated_findings") or []
    wins, nulls = [], []
    for f in findings:
        row = {
            "design_id": f.get("design_id"), "template": f.get("experiment_id"),
            "mode": f.get("mode"), "verdict": f.get("verdict"),
            "evidence_class": f.get("evidence_class"),
            "question": f.get("question"),
            "isolated_variable": f.get("isolated_variable"),
            "mechanism": f.get("mechanism"), "pillar": f.get("pillar"),
            "best_metric": f.get("best_metric"),
            "lift": f.get("pooled_lift_pp"), "lift_unit": f.get("lift_unit", "pp"),
            "headline_effect": f.get("headline_effect"),          # ← ADD
            "runs_per_cell": f.get("runs_per_cell"),
            "underpowered": f.get("underpowered"),
            "binary_at_ceiling": f.get("binary_at_ceiling"),
            "element_attribution": f.get("element_attribution"),
            "best_element": f.get("best_element"),
            "rank_engines": f.get("rank_engines"),
            "significant_rows": [
                {k: r.get(k) for k in ("engine", "arm", "metric", "A_rate", "B_rate",
                                       "lift_pp", "p_fdr")}
                for r in (f.get("per_engine") or []) if r.get("significant")][:8],
            "interpretation_limits": f.get("interpretation_limits") or [],
            "control_profile": f.get("control_profile"),
            "variance_warnings": f.get("variance_warnings") or [],
        }
        if row.get("evidence_class") == "PROVEN_WITH_CAVEAT" and not row.get("headline_effect"):
            row["headline_effect"] = {
                "value": row.get("lift"), "unit": row.get("lift_unit", "pp"),
                "metric": row.get("best_metric"),
                "detail": f"{row.get('lift')}{row.get('lift_unit', 'pp')} measured at only "
                          f"{row.get('runs_per_cell')} runs per arm per engine — the "
                          f"direction is credible but the magnitude is not. Do not quote "
                          f"this figure without the caveat.",
                "engines": sorted({r.get("engine") for r in
                                   (row.get("significant_rows") or []) if r.get("engine")}),
                "caveated": True}
        (wins if f.get("verdict") in WIN_VERDICTS else nulls).append(row)

    return {
        "brand": target, "parent": parent, "category": category or "this category",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "wins": wins, "nulls": nulls,
        "gaps": A("gap_register_actionable") or [],
        "clusters": A("gap_clusters") or [],
        "hypotheses": A("gap_hypotheses") or [],
        "substantiation_backlog": A("substantiation_backlog") or [],
        "delta": A("delta_scorecard") or {},
        "metrics": A("benchmark_metrics") or {},
        "hitlist": A("source_hitlist") or [],
        "claims": A("claims_register") or [],
        "kpis": A("kpi_targets") or [],
        "variants": A("experiment_variants") or {},
        "constraints": constraints or {},
    }

def evidence_summary(si):
    """One-screen honest view of what the engagement actually proved."""
    m = si["metrics"]
    d = si["delta"]
    mbi = {r["intent"]: r["mention_rate"] for r in m.get("mention_by_intent", [])}
    fbi = {r["intent"]: r.get("first_rate", 0)
           for r in m.get("first_mention_by_intent", [])}
    pv = d.get("provenance") or {}
    ctrl_ceiling = [w["design_id"] for w in si["wins"] + si["nulls"]
                    if w.get("binary_at_ceiling")]
    return {
        "audit_mention_by_intent": mbi,
        "audit_first_by_intent": fbi,
        "in_context_mention": "≈100% (experiments: the brand's own document in context)",
        "retrieval_gap_pp": round((1.0 - mbi.get("need", 0)) * 100, 1),
        "n_wins": len(si["wins"]), "n_nulls": len(si["nulls"]),
        "n_well_powered_nulls": sum(1 for n in si["nulls"] if not n.get("underpowered")),
        "n_gaps": len(si["gaps"]),
        "n_substantiation": len(si["substantiation_backlog"]),
        "designs_at_ceiling": ctrl_ceiling,
        "provenance": {k: pv.get(k) for k in
                       ("self_referential_pct", "corroborated_pct", "grey_market_pct",
                        "affiliate_pct", "competitor_owned_pct", "voice_control_ratio")},
        "framing": d.get("framing") or {},
        "engine_veracity": d.get("engine_veracity") or {},
    }

# ====================================================================== S1
GAP_TYPE_PILLAR = {
    "CLAIM_ABSENT": "Content", "CLAIM_FRAGILE": "Corroboration",
    "CLAIM_DILUTED": "Content", "CLAIM_DISTORTED": "Claims-Substantiation",
    "HALLUCINATION": "Claims-Substantiation",
    "UNSUBSTANTIATED_HELP": "Claims-Substantiation",
    "STALE_CLAIM": "Claims-Substantiation", "COMPETITOR_BLEED": "Content",
    "REASON_DEFICIT": "Content", "FLAT_MENTION": "Content",
    "PROVENANCE_THIN": "Corroboration", "GREY_MARKET_VOICE": "Corroboration",
    "SOURCE_ABSENCE": "Corroboration", "NEGATIVE_FRAMING": "Content",
    "ADVISORY_DEFERRAL": "Corroboration",
    "VISIBILITY_GAP": "Technical/Structured-Data",   # retrieval, not messaging
    "RANK_GAP": "Technical/Structured-Data",
    "ENGINE_DIVERGENCE": "Technical/Structured-Data",
    "REFUSAL": "Content", "FRESHNESS_STALE": "Content",
    "VISUAL_UNREADABLE": "Visual/Multimodal",
}

def route_gaps_to_pillars(si, retrieval_is_constraint=None):
    """Deterministic routing with ONE evidence-driven override: when the experiments show
    content form has no effect while the audit shows the brand absent, VISIBILITY_GAP is a
    retrieval problem (Pillar 2), not a messaging problem (Pillar 1)."""
    if retrieval_is_constraint is None:
        well_powered_form_nulls = [
            n for n in si["nulls"]
            if not n.get("underpowered")
            and str(n.get("template", "")) in ("E0", "E1", "E5")]
        mbi = {r["intent"]: r["mention_rate"]
               for r in si["metrics"].get("mention_by_intent", [])}
        retrieval_is_constraint = bool(well_powered_form_nulls) and \
            mbi.get("need", 1.0) < 0.20

    routed = defaultdict(list)
    for g in si["gaps"]:
        pillar = GAP_TYPE_PILLAR.get(g.get("type")) or g.get("pillar") or "Content"
        if not retrieval_is_constraint and g.get("type") in ("VISIBILITY_GAP",):
            pillar = "Content"
        routed[pillar].append(g)

    # the substantiation backlog is Pillar 6 by construction
    for b in si["substantiation_backlog"]:
        routed["Claims-Substantiation"].append({
            "gap_id": b.get("cluster_id"), "type": "UNSUBSTANTIATED_HELP",
            "severity": b.get("severity", "high"),
            "gap": b.get("assertion_to_substantiate"),
            "evidence": {"reason": b.get("reason"),
                         "engines": b.get("engines_asserting")},
            "pillar": "Claims-Substantiation", "source": "substantiation_backlog"})

    return dict(routed), bool(retrieval_is_constraint)

# ====================================================================== S2
PILLAR_PROMPT = """You are the GEO (Generative Engine Optimization) strategy lead for
"{brand}" in the {category} category.

Build the **{pillar}** pillar. Scope: {scope}

Ground EVERY action in the evidence below. Never invent an effect size — quote the measured
lift, and where there is no experiment say so and name the test that would validate it.

=== THIS PILLAR'S OWN EVIDENCE STANDING: {stance} ===
{stance_guidance}

ACTION BUDGET for this pillar: {budget} action(s). Evidence strength, not gap count,
determines how much of the strategy a pillar owns.

=== WHAT THE EXPERIMENTS PROVED ===
{wins}

When an action builds directly on a PROVEN or PROVEN_ENGINE_SPECIFIC finding, its
evidence_class MUST match that finding's class — not DIAGNOSED. Quote the headline_effect
verbatim, including the engines it applies to. Downgrading proven evidence to DIAGNOSED
understates the strongest thing this engagement established.

=== WHAT THE EXPERIMENTS DISPROVED — read this carefully ===
{nulls}
A well-powered null is EVIDENCE. You must NOT recommend anything a null disproved. Where a
null applies to this pillar, put it in "do_not_do" with the measured basis.

=== MEASURED POSITION (audit: {n_runs} runs, engines {engines}) ===
mention rate by intent: {mention}
first-mention by intent: {first}
per-engine mention: {by_engine}
refusal by intent: {refusal}
advisory-deferral by intent: {advisory}
source-type mix per engine (what each engine cites): {source_mix}
answer consistency — locked-in / fragile / absent: {consistency}

=== THE DELTA (ground truth vs machine reality) ===
claim deltas (sample): {claim_deltas}
assertions engines make that are NOT in the register: {unregistered}
citation provenance: {provenance}
framing (positive / neutral / hedged / advisory): {framing}
reason-density deficits vs the leader, per 100 tokens: {reason_gap}
substantiveness of brand mentions: {flatness}

=== GAPS ASSIGNED TO THIS PILLAR ===
{gaps}

=== SUBSTANTIATED CLAIMS AVAILABLE (the ONLY publishable facts) ===
{claims}

=== SOURCE HIT-LIST (placement targets, bucketed) ===
{hitlist}

=== CONSTRAINTS ===
{constraints}

Return STRICT JSON:
{{
 "pillar": "{pillar}",
 "objective": "one sentence — the measurable outcome this pillar delivers",
 "thesis": "2-3 sentences — WHY this works for THIS brand on THESE engines, citing the evidence",
 "mechanism": "how the change propagates to an AI recommendation: retrieval, memory, extraction or corroboration",
 "actions": [
   {{"action_id":"{prefix}-1", "title":"", "what":"specific and executable — name the claims, pages, schema or domains",
     "why":"the evidence that justifies it, quoted",
     "evidence_class":"PROVEN|PROVEN_ENGINE_SPECIFIC|PROVEN_WITH_CAVEAT|DIAGNOSED|HYPOTHESIS",
     "evidence_refs":["design_id or gap_id"],
     "expected_effect":{{"metric":"","value":null,"unit":"pp|rank_positions|share",
                         "engines":[],"basis":"measured|inferred|none",
                         "qualitative":"large|moderate|small|unknown",
                         "source":"design_id, or the diagnostic that implies it"}},
     "claims_used":[], "target_queries":[], "target_domains":[],
     "clock":"retrieval|memory",
     "effort":"low|medium|high", "owner":"content|dev|PR|design|legal|community",
     "leading_indicator":"measurable in 2-4 weeks",
     "lagging_indicator":"the KPI it moves",
     "validating_experiment":"only if evidence_class is DIAGNOSED or HYPOTHESIS",
     "dependencies":[], "risk":"", "kill_criterion":""}}
 ],
 "do_not_do": [{{"action":"","reason":"","basis":"design_id of the null, or the diagnostic"}}],
 "pillar_kpi": {{"metric":"","baseline":0,"target":0,"horizon_weeks":0}}
}}

RULES
- Every claims_used entry must appear VERBATIM in the substantiated claims list. Never
  invent a fact — this is a regulated category and these actions become content briefs.
- clock="retrieval" for anything publishable on a surface the brand controls.
  clock="memory" for anything requiring third-party behaviour change or category authority.
- If this pillar has no credible evidence, return an EMPTY actions list and say so in the
  thesis. An honest empty pillar is more valuable than an invented one.
- expected_effect: use basis="measured" ONLY when quoting a real experiment result, and
  then quote it exactly with its engines. Use basis="inferred" with a value only if a
  measured result on a comparable mechanism supports it — name the source. Otherwise use
  basis="none", value=null, and a qualitative band. NEVER write value=0 for an action you
  are recommending — zero means "no effect", which contradicts recommending it.
- When quoting a measured effect from an UNDERPOWERED experiment, you MUST state the
  runs-per-arm and the caveat inside expected_effect.source.
- Actions must be specific enough that a practitioner could start on Monday."""

def _slim_claims(claims, k=40):
    return [c.get("claim") for c in (claims or []) if c.get("claim")][:k]

def _hitlist_slim(hitlist, k=14):
    rows = (hitlist.to_dict("records") if hasattr(hitlist, "to_dict") else (hitlist or []))
    keys = ("domain", "source_type", "times_cited", "bucket", "brand_present_rate",
            "brand_absence_index", "access_route", "top_url")
    return [{k2: r.get(k2) for k2 in keys if r.get(k2) is not None} for r in rows[:k]]

def build_pillar_strategies(gateway, si, routed, model, module="optimise",
                            progress_cb=None):
    m, d = si["metrics"], si["delta"]
    cons = si["consistency"] = (d.get("consistency") or {}).get("overall", {})
    # ---- derive metric guidance from THIS engagement's control arms ----
    ctrl = [r["rate"] for w in (si["wins"] + si["nulls"])
            for r in (w.get("rate_table") or [])
            if r.get("arm") == "A" and r.get("metric") == "mention"]
    at_ceiling = bool(ctrl) and (sum(ctrl) / len(ctrl)) >= 0.9
    metric_hint = (
        f"Control-arm mention rate measured ~{sum(ctrl) / len(ctrl):.0%} in prior "
        f"experiments, so mention_rate has no headroom — use first_rec_rate or mean rank "
        f"as the primary metric." if at_ceiling else
        "Use mention_rate as the primary metric, with first_rec_rate secondary.")
    out = {}
    for i, pillar in enumerate(PILLARS):
        gaps = routed.get(pillar, [])
        pillar_wins = [w for w in si["wins"] if _relevant_to(w, pillar)]
        pillar_nulls = [n for n in si["nulls"] if _relevant_to(n, pillar)]

        # ---------- INSERT 1: evidence stance + action budget ----------
        pv = pillar_evidence_verdict(si, pillar)
        budget = pillar_budget(si, pillar, pv["stance"])

        if not gaps and not pillar_wins:
            out[pillar] = {"pillar": pillar, "objective": "No evidence for this pillar.",
                           "thesis": "No gaps were routed here and no experiment "
                                     "addressed it. Excluded from the strategy.",
                           "actions": [], "do_not_do": [], "_skipped": True,
                           "_evidence_stance": pv}
            if progress_cb:
                progress_cb((i + 1) / len(PILLARS))
            continue

        res = gateway.generate_json(model, PILLAR_PROMPT.format(
            brand=si["brand"], category=si["category"], pillar=pillar,
            scope=PILLAR_SCOPE[pillar], prefix=_prefix(pillar),
            # ---------- INSERT 2: the two new prompt fields ----------
            stance=pv["stance"], stance_guidance=pv["guidance"], budget=budget,
            wins=json.dumps(pillar_wins, indent=1)[:4500],
            nulls=json.dumps(pillar_nulls, indent=1)[:3500],
            n_runs=m.get("n_runs"), engines=json.dumps(m.get("engines") or []),
            mention=json.dumps(m.get("mention_by_intent") or [])[:900],
            first=json.dumps(m.get("first_mention_by_intent") or [])[:700],
            by_engine=json.dumps(m.get("mention_by_engine_intent") or [])[:1200],
            refusal=json.dumps(m.get("refusal_by_intent") or [])[:500],
            advisory=json.dumps(m.get("advisory_rate_by_intent") or [])[:500],
            source_mix=json.dumps(m.get("source_mix") or {})[:1200],
            consistency=json.dumps({k: len(v) if isinstance(v, list) else v
                                    for k, v in cons.items()}),
            claim_deltas=json.dumps((d.get("claim_deltas") or [])[:10])[:2200],
            unregistered=json.dumps(
                (d.get("hallucination_summary") or {}).get("high_risk") or [])[:1200],
            provenance=json.dumps(d.get("provenance") or {})[:900],
            framing=json.dumps(d.get("framing") or {})[:700],
            reason_gap=json.dumps(d.get("reason_gap") or [])[:1200],
            flatness=json.dumps(d.get("flatness") or {})[:700],
            gaps=json.dumps([{k: g.get(k) for k in
                              ("gap_id", "type", "severity", "gap", "metric_baseline",
                               "engines", "intent", "domain")}
                             for g in gaps[:14]], indent=1)[:4000],
            claims=json.dumps(_slim_claims(si["claims"]), indent=1)[:3500],
            hitlist=json.dumps(_hitlist_slim(si["hitlist"]), indent=1)[:2200],
            constraints=json.dumps(si["constraints"])),
            expect="dict", module=module, task=f"pillar_{_prefix(pillar)}",
            temperature=0.4, max_tokens=5000)

        p = res.get("data") or {}
        p.setdefault("pillar", pillar)
        p["_generated_ok"] = bool(res.get("ok"))
        p["_failure"] = res.get("failure")
        p["_n_gaps"] = len(gaps)

        # ---------- INSERT 3: stance + budget enforcement + kill-criterion autofill ----------
        p["_evidence_stance"] = pv
        p["_action_budget"] = budget
        acts = p.get("actions") or []
        if len(acts) > budget:
            p["_budget_trimmed"] = len(acts) - budget
            acts = acts[:budget]
            p["actions"] = acts
        for a in acts:
            if not a.get("kill_criterion"):
                if pillar == "Claims-Substantiation":
                    a["kill_criterion"] = ("Stop if legal or regulatory review rejects "
                                           "the claim as unsubstantiable — then remove it "
                                           "from all downstream content actions.")
                elif a.get("clock") == "memory":
                    a["kill_criterion"] = (
                        f"Stop if the leading indicator "
                        f"({MEMORY_PROXY.get(pillar, 'proxy')}) has not moved after two "
                        f"quarters of sustained effort.")
                else:
                    a["kill_criterion"] = ("Stop if the wave's measurement contract shows "
                                           "no movement on the primary metric at the "
                                           "scheduled re-audit.")
                a["_kill_autofilled"] = True
            if a.get("evidence_class") in ("DIAGNOSED", "HYPOTHESIS") \
                    and not a.get("validating_experiment"):
                a["validating_experiment"] = (
                    f"PLACEHOLDER — {PILLAR_TEST_TEMPLATE.get(pillar, 'E4 (gap-closure)')} "
                    f"at N>=100 on all engines tested: control document without this "
                    f"change vs treatment with it, on a brand-free target query. "
                    f"{metric_hint} Design this properly before committing spend.")
                a["_validating_autofilled"] = True

        out[pillar] = p
        if progress_cb:
            progress_cb((i + 1) / len(PILLARS))
    return out

def _prefix(pillar):
    return {"Content": "P1", "Technical/Structured-Data": "P2",
            "Visual/Multimodal": "P3", "Corroboration": "P4",
            "Social/Community": "P5", "Claims-Substantiation": "P6"}[pillar]

TEMPLATE_PILLAR = {"E0": ["Content", "Technical/Structured-Data"],
                   "E1": ["Technical/Structured-Data", "Content"],
                   "E2": ["Content"], "E3": ["Corroboration"],
                   "E4": ["Content", "Claims-Substantiation"],
                   "E5": ["Technical/Structured-Data"],
                   "E7": ["Visual/Multimodal"], "E8": ["Visual/Multimodal"],
                   "E9": ["Content"]}

def _relevant_to(row, pillar):
    if row.get("pillar") == pillar:
        return True
    return pillar in TEMPLATE_PILLAR.get(str(row.get("template") or ""), [])

# ====================================================================== S3
SYNTHESIS_PROMPT = """You are the GEO strategy lead for "{brand}". Six pillar strategies were
drafted independently. Synthesise them into ONE coherent Overall GEO Strategy.

Your job:
1. Name the 3 BIG BETS — highest-leverage moves, each backed by named evidence.
2. Separate STRUCTURAL TRUTHS (hold across all engines → the overall strategy) from
   ENGINE-SPECIFIC signals (→ overlays). Use the cross-engine data.
3. Find COMPOUNDING combinations, and note where effects will NOT be additive.
4. Remove DUPLICATED work; merge overlapping actions.
5. Resolve CONFLICTS — e.g. Content wants to publish a claim that Claims-Substantiation
   says is not yet substantiated. Sequence, do not publish.
6. Write the STRATEGIC NARRATIVE a CMO can repeat in one paragraph.
7. Consolidate the "not doing" list. Where the experiments DISPROVED something, say so
   with the measured basis — a credible strategy states what it will not spend on.For each not_doing item derived from a null, state the SCOPE of what was disproven with
    precision. A ceiling-limited null does NOT mean the mechanism never works — it means the
    mechanism has no headroom UNDER THE TESTED CONDITION. Write it as:
      "<mechanism> produced no measurable gain when <condition> (design_id, N per arm).
       This rules out <specific spend>. It does not rule out <adjacent case>."
    Example of the right precision: "Reformatting content that already contains extractable
    product specifics produced no gain (C20, 5 arms, N=100, both engines). This rules out
    reformatting-led content programmes. It does not rule out adding specifics to content
    that lacks them."
- Every big_bet MUST list at least one action_id drawn from the pillar strategies above.
  A big bet with no actions is an aspiration, not a strategy — if you cannot find actions
  for a theme, do not make it a big bet.
- Big bet expected_effect must name its source design_id and, if that design was
  underpowered, say so. Never sum effects across different metrics (pp and rank positions
  are not comparable).
- If a null disproved content FORM (structure, formatting, reformatting), the narrative
  must NOT use language a reader could hear as reformatting — "machine-readable",
  "structured", "well-organised" — unless you explicitly distinguish RETRIEVABILITY
  (schema, indexability, crawlable surfaces: not tested, plausibly the constraint) from
  PRESENTATION (bullets, tables, capsules: tested and disproven). State the distinction
  in the same sentence.
  
BIG BET SELECTION — mandatory:
1. If any finding is PROVEN or PROVEN_ENGINE_SPECIFIC, at least one big bet MUST be built
   on it and MUST quote its headline_effect verbatim with the engines named. Scaling a
   proven mechanism outranks a larger claim resting on an underpowered one.
2. The pillar holding the most actions MUST appear in at least one big bet. A pillar with
   the largest action count and no big bet is a synthesis failure.
3. Order the big bets by evidence strength, not by ambition. A caveated result must never
   lead the document.

=== EVIDENCE HEADLINE ===
{headline}

=== WHAT WAS PROVEN ===
{wins}

=== WHAT WAS DISPROVEN (well-powered nulls) ===
{nulls}

=== CROSS-ENGINE AGREEMENT ===
{cross_engine}

=== ENGINE SOURCE-MIX PROFILES ===
{source_mix}

=== PILLAR STRATEGIES ===
{pillars}

=== KPI TARGETS ===
{kpis}

Return STRICT JSON:
{{
 "narrative": "one paragraph a CMO can repeat verbatim",
 "diagnosis": "3 sentences: where the brand actually stands in machine reality",
 "big_bets": [{{"title":"","pillars":[],"evidence_class":"","expected_effect":"","why_now":"","action_ids":[]}}],
 "structural_truths": [{{"truth":"","evidence":"","applies_to_engines":["all"]}}],
 "engine_specific_signals": [{{"engine":"","signal":"","implication":""}}],
 "compounding_plays": [{{"combine":["action_id","action_id"],"rationale":"","caution":""}}],
 "conflicts_resolved": [{{"conflict":"","resolution":"","sequencing":""}}],
 "merged_actions": [{{"kept":"action_id","absorbed":["action_id"],"reason":""}}],
 "not_doing": [{{"item":"","reason":"","basis":"disproven by design_id | unwinnable | unsubstantiable"}}],
 "north_star_kpi": {{"metric":"","baseline":0,"target":0,"by_when":""}}
}}"""

def synthesise_overall(gateway, si, pillar_strategies, model, module="optimise"):
    res = gateway.generate_json(model, SYNTHESIS_PROMPT.format(
        brand=si["brand"],
        headline=json.dumps(si["delta"].get("headline") or [], indent=1)[:2000],
        wins=json.dumps(si["wins"], indent=1)[:4000],
        nulls=json.dumps([n for n in si["nulls"] if not n.get("underpowered")],
                         indent=1)[:3000],
        cross_engine=json.dumps(si["delta"].get("consistency", {})
                                .get("divergence") or [])[:1200],
        source_mix=json.dumps(si["metrics"].get("source_mix") or {})[:1200],
        pillars=json.dumps({k: {kk: v.get(kk) for kk in
                                ("objective", "thesis", "mechanism", "actions",
                                 "do_not_do", "pillar_kpi")}
                            for k, v in pillar_strategies.items()
                            if not v.get("_skipped")}, indent=1)[:11000],
        kpis=json.dumps(si["kpis"])[:1200]),
        expect="dict", module=module, task="strategy_synthesis",
        temperature=0.4, max_tokens=5000)
    s = res.get("data") or {}
    s["_generated_ok"] = bool(res.get("ok"))
    s["_failure"] = res.get("failure")
    return s

# ====================================================================== S4
OVERLAY_PROMPT = """Build the GEO overlay for the engine "{engine}" for brand "{brand}".

Derive EVERYTHING from this engine's own measured behaviour below. Do not rely on general
reputation or assumptions about the engine.

THIS ENGINE'S MEASURED BEHAVIOUR
- mention rate by intent: {mention}
- source-type mix (what it actually cites): {source_mix}
- refusal / search-invocation / avg sources: {ops}
- framing of brand mentions (positive / neutral / advisory): {framing}
- claim veracity: {veracity}
- experiments that worked ON THIS ENGINE: {wins_here}
- experiments that worked elsewhere but NOT here: {failed_here}

OVERALL STRATEGY (already decided — do not repeat it)
{overall}

Return STRICT JSON:
{{
 "engine": "{engine}",
 "behavioural_profile": "2 sentences — how this engine forms recommendations, from the data above",
 "what_changes_here": [{{"action":"","why_this_engine":"","evidence":""}}],
 "deprioritise_here": [{{"action":"","why":""}}],
 "placement_targets": [{{"domain":"","why":"this engine cites it heavily"}}],
 "engine_specific_risk": ""
}}"""

def build_engine_overlays(gateway, si, overall, model, module="optimise"):
    m, d = si["metrics"], si["delta"]
    engines = m.get("engines") or []
    mei = defaultdict(list)
    for r in m.get("mention_by_engine_intent", []):
        mei[r.get("engine")].append({k: r.get(k) for k in
                                     ("intent", "mention_rate", "runs")})
    pv_eng = (d.get("provenance") or {}).get("per_engine") or {}
    fr_eng = (d.get("framing") or {}).get("by_engine") or {}
    ver_eng = d.get("engine_veracity") or {}

    overlays = {}
    for eng in engines:
        wins_here = [{"design_id": w["design_id"], "variable": w.get("isolated_variable"),
                      "rows": [r for r in (w.get("significant_rows") or [])
                               if r.get("engine") == eng]}
                     for w in si["wins"]
                     if any(r.get("engine") == eng
                            for r in (w.get("significant_rows") or []))]
        failed_here = [{"design_id": w["design_id"], "variable": w.get("isolated_variable"),
                        "worked_on": sorted({r.get("engine")
                                             for r in (w.get("significant_rows") or [])})}
                       for w in si["wins"]
                       if not any(r.get("engine") == eng
                                  for r in (w.get("significant_rows") or []))]
        res = gateway.generate_json(model, OVERLAY_PROMPT.format(
            engine=eng, brand=si["brand"],
            mention=json.dumps(mei.get(eng) or [])[:900],
            source_mix=json.dumps((m.get("source_mix") or {}).get(eng) or {})[:800],
            ops=json.dumps(pv_eng.get(eng) or {})[:600],
            framing=json.dumps(fr_eng.get(eng) or {})[:400],
            veracity=json.dumps(ver_eng.get(eng) or {})[:500],
            wins_here=json.dumps(wins_here, indent=1)[:2000],
            failed_here=json.dumps(failed_here, indent=1)[:1200],
            overall=json.dumps({k: overall.get(k) for k in
                                ("narrative", "big_bets", "structural_truths")},
                               indent=1)[:3000]),
            expect="dict", module=module, task="engine_overlay",
            temperature=0.4, max_tokens=2500)
        o = res.get("data") or {"engine": eng}
        o["_generated_ok"] = bool(res.get("ok"))
        overlays[eng] = o
    return overlays

# ====================================================================== S5
def assign_horizon(action, unblocks=0, critical=False, in_wave0=False):
    clock = action.get("clock", "retrieval")
    eff = {"low": 1, "medium": 2, "high": 3}.get(action.get("effort", "medium"), 2)
    ew = EVIDENCE_WEIGHT.get(action.get("evidence_class", "HYPOTHESIS"), 0.2)
    if critical or in_wave0:
        return "Wave 0 (0-2 weeks)"
    if clock == "memory":
        return "Wave 3 (2-4 quarters)"
    if eff <= 2 and ew >= 0.5:
        return "Wave 1 (2-8 weeks)"
    if unblocks >= 1 or ew >= 0.45:
        return "Wave 1 (2-8 weeks)" if eff <= 2 else "Wave 2 (2-4 months)"
    return "Wave 2 (2-4 months)"

def apply_horizons(pillar_strategies, si=None, wave0_cap=6):
    """Wave 0 is for TRUE root dependencies and compliance only. An action that unblocks
    others but is itself blocked cannot start first, and Wave 0 must fit in two weeks."""
    from optimise.roadmap import build_dag, EFFORT_DAYS
    actions = all_actions(pillar_strategies)
    dag = build_dag(actions)

    downstream = Counter()
    for aid, deps in dag.items():
        for d in deps:
            downstream[d] += 1
    blocked_by_something = {aid for aid, deps in dag.items() if deps}

    crit_ids = {b.get("cluster_id") for b in (si or {}).get("substantiation_backlog", [])
                if b.get("severity") == "critical"}

    # rank candidates: unblocks work, has no upstream of its own
    cands = [a for a in actions
             if downstream.get(a.get("action_id"), 0) > 0
             and a.get("action_id") not in blocked_by_something]
    cands.sort(key=lambda a: (-downstream.get(a.get("action_id"), 0),
                              EFFORT_DAYS.get(a.get("effort", "medium"), 5)))

    wave0, days = set(), 0.0
    for a in actions:                                # compliance first, uncapped
        if a.get("pillar") == "Claims-Substantiation" and \
                set(a.get("evidence_refs") or []) & crit_ids:
            wave0.add(a.get("action_id"))
            days += EFFORT_DAYS.get(a.get("effort", "medium"), 5)
    for a in cands:
        if len(wave0) >= wave0_cap or days > 20:     # ~2 weeks of team time
            break
        wave0.add(a.get("action_id"))
        days += EFFORT_DAYS.get(a.get("effort", "medium"), 5)

    for pillar, p in pillar_strategies.items():
        for a in (p.get("actions") or []):
            aid = a.get("action_id")
            a["pillar"] = pillar
            n_un = downstream.get(aid, 0)
            is_crit = (pillar == "Claims-Substantiation"
                       and bool(set(a.get("evidence_refs") or []) & crit_ids))
            a["unblocks_count"] = n_un
            a["is_root_dependency"] = aid not in blocked_by_something and n_un > 0
            a["is_critical_dependency"] = is_crit
            a["horizon"] = assign_horizon(a, n_un, is_crit, in_wave0=aid in wave0)
            if is_crit:
                a["sequencing_note"] = ("CRITICAL: the engines already assert this claim "
                                        "on the brand's behalf with no substantiation "
                                        "trail. Regulatory exposure, not a content gap.")
            elif aid in wave0:
                a["sequencing_note"] = (f"Root dependency — unblocks {n_un} downstream "
                                        f"action(s) and has no upstream blocker.")
            elif n_un and aid in blocked_by_something:
                a["sequencing_note"] = (f"Unblocks {n_un} action(s) but is itself blocked "
                                        f"upstream — cannot start first.")
            if a.get("clock") == "memory":
                a["measurement_note"] = (
                    "Memory play — no mention-rate target this quarter. Track the proxy: "
                    f"{MEMORY_PROXY.get(pillar, 'category authority')}.")
    return pillar_strategies

# ====================================================================== S6
CRITIC_PROMPT = """You are a skeptical GEO auditor. Your job is to stop an indefensible
recommendation reaching a client. Be harsh.

Check for:
1. Actions with NO evidence, or evidence that does not support the stated effect size.
2. Effect sizes not traceable to a measured result.
3. Actions that CONTRADICT a well-powered null — the single worst failure.
4. Claims used that are not in the substantiated register (regulatory risk).
5. Memory plays given retrieval-speed timelines, or vice versa.
6. Unstated dependencies — e.g. publish before substantiate.
7. Over-claiming: summed effects that ignore overlap between actions.
8. Diagnosed gaps left unaddressed by any pillar.
9. Actions so vague a practitioner could not start them.
10. Findings quoted without their interpretation limits (underpowered, ceiling, confounded).

NOTE: the SUBSTANTIATED REGISTER below is the complete register ({n_claims} claims). If a
claim appears in it, it is registered — do not flag it.

STRATEGY: {strategy}
PROVEN: {wins}
DISPROVEN (well-powered nulls): {nulls}
SUBSTANTIATED REGISTER — this is the COMPLETE register ({n_claims} claims). If a claim
appears below, it IS registered. Do not flag it as unregistered. A deterministic validator
already checks every claim against the full register, so trust it over your own scan and
raise claim issues only for claims genuinely absent from this list:
{claims}
UNADDRESSED GAP TYPES: {unaddressed}

DETERMINISTIC VALIDATION ERRORS ALREADY DETECTED — treat these as established facts, not
opinions. Do not re-litigate them, but DO raise any consequence the strategy has not
addressed:
{prior_errors}

Return STRICT JSON:
[{{"severity":"blocker|major|minor","target":"action_id or section","issue":"","fix":""}}]"""

def critic_pass(gateway, si, strategy, pillar_strategies, model, module="optimise",
                prior_errors=None):
    actions = all_actions(pillar_strategies)
    addressed = {g for p in pillar_strategies.values()
                 for a in (p.get("actions") or [])
                 for g in (a.get("evidence_refs") or [])}
    unaddressed = sorted({g.get("type") for g in si["gaps"]
                          if g.get("gap_id") not in addressed})
    res = gateway.generate_json(model, CRITIC_PROMPT.format(
        strategy=json.dumps({"overall": {k: strategy.get(k) for k in
                                         ("narrative", "big_bets", "not_doing")},
                             "actions": actions}, indent=1)[:11000],
        wins=json.dumps(si["wins"], indent=1)[:2500],
        nulls=json.dumps([n for n in si["nulls"] if not n.get("underpowered")],
                         indent=1)[:2500],
        n_claims=len(si["claims"] or []),
        claims=json.dumps(_slim_claims(si["claims"], 250))[:14000],  # was 30
        prior_errors=json.dumps([{k: e.get(k) for k in
                                  ("severity", "target", "error", "detail", "fix")}
                                 for e in (prior_errors or [])], indent=1)[:3000],
        unaddressed=json.dumps(unaddressed[:20])),
                                expect="list", module=module, task="strategy_critic",
                                temperature=0.2, max_tokens=3000)
    return res.get("data") or []

def deterministic_validate(si, pillar_strategies, strategy):
    """Rules the LLM cannot be trusted to enforce."""
    reg = {fold(c.get("claim", "")) for c in (si["claims"] or [])}
    disproven_vars = {}
    for n in si["nulls"]:
        if n.get("underpowered"):
            continue
        var = fold(n.get("isolated_variable") or "")
        if not var:
            continue
        disproven_vars[var] = {
            "design_id": n["design_id"],
            "runs": n.get("runs_per_cell"),
            "pillars": set(DISPROVEN_PILLARS.get(str(n.get("template")), []))}
    wins_by_id = {w["design_id"]: w for w in si["wins"] if w.get("design_id")}
    actions = all_actions(pillar_strategies)
    errs = []

    for a in actions:
        aid = a.get("action_id")
        if not aid:
            errs.append({"severity": "major", "target": f"{a.get('pillar')}/unnamed",
                         "error": "action_missing_id", "detail": a.get("title"),
                         "fix": "Every action needs an action_id so the roadmap and "
                                "briefs can reference it."})
            continue

        # ---- claims must exist in the substantiated register (regulatory) ----
        bad = [c for c in (a.get("claims_used") or []) if fold(c) not in reg]
        if bad:
            errs.append({"severity": "blocker", "target": aid,
                         "error": "unregistered_claims", "detail": bad[:3],
                         "fix": "Substantiate and add to the register, or remove."})

        ec = a.get("evidence_class")
        if ec in ("PROVEN", "PROVEN_ENGINE_SPECIFIC", "PROVEN_WITH_CAVEAT") \
                and not a.get("evidence_refs"):
            errs.append({"severity": "blocker", "target": aid,
                         "error": "claims_proven_without_reference",
                         "fix": "Cite the design_id, or downgrade to DIAGNOSED."})
        if ec in ("DIAGNOSED", "HYPOTHESIS"):
            if not a.get("validating_experiment"):
                errs.append({"severity": "major", "target": aid,
                             "error": "unevidenced_without_test",
                             "fix": "Name the experiment that would validate it."})
            elif a.get("_validating_autofilled"):
                errs.append({"severity": "minor", "target": aid,
                             "error": "validating_experiment_is_placeholder",
                             "fix": "A generic test template was inserted automatically. "
                                    "Replace with a specific design — arms, target query, "
                                    "N and primary metric — before this action is "
                                    "committed to a wave."})
        if not a.get("kill_criterion"):
            errs.append({"severity": "minor", "target": aid, "error": "no_kill_criterion",
                         "fix": "State what result would stop this work."})
        elif a.get("_kill_autofilled"):
            errs.append({"severity": "minor", "target": aid,
                         "error": "kill_criterion_is_default",
                         "fix": "Default criterion applied — confirm it is the right "
                                "stopping condition for this action."})

        # ---- effect-size integrity ----
        eff = a.get("expected_effect") or {}
        basis = eff.get("basis")
        try:
            val = float(eff.get("value")) if eff.get("value") is not None else None
        except Exception:
            val = None

        if val == 0.0:
            errs.append({"severity": "major", "target": aid,
                         "error": "zero_effect_action",
                         "fix": 'Use basis="none" with value=null and a qualitative band, '
                                'or state a plausible effect. Zero means "no effect", '
                                'which contradicts recommending the action.'})
        if basis == "measured":
            for r in (set(a.get("evidence_refs") or []) & set(wins_by_id)):
                w = wins_by_id[r]
                if w.get("underpowered") and "run" not in str(eff.get("source", "")).lower():
                    errs.append({
                        "severity": "major", "target": aid,
                        "error": "measured_effect_from_underpowered_experiment",
                        "detail": f"{r} ran at ~{w.get('runs_per_cell')} runs/arm",
                        "fix": "State the runs-per-arm and the caveat in "
                               "expected_effect.source, or set basis='inferred'."})
        if basis not in ("measured", "inferred", "none"):
            errs.append({"severity": "minor", "target": aid, "error": "no_effect_basis",
                         "fix": 'Declare basis as "measured", "inferred" or "none".'})

        # ---------- INSERT (F-6) ----------
        refs = set(a.get("evidence_refs") or [])
        exp_refs = refs & (set(wins_by_id) | {n["design_id"] for n in si["nulls"]})
        if ec in ("DIAGNOSED", "HYPOTHESIS") and val and exp_refs \
                and basis == "measured":
            src_class = next((wins_by_id[r].get("evidence_class")
                              for r in exp_refs if r in wins_by_id), None)
            errs.append({
                "severity": "blocker", "target": aid,
                "error": "diagnosed_action_quotes_measured_effect",
                "detail": f"class={ec} but quotes {val}{eff.get('unit', 'pp')} measured in "
                          f"{sorted(exp_refs)} (which is {src_class})",
                "fix": f"Either set evidence_class to {src_class} and carry its caveats, "
                       f"or set basis='inferred' with a value no greater than half the "
                       f"measured effect, or basis='none'."})
        if ec in ("DIAGNOSED", "HYPOTHESIS") and (val or 0) > 25:
            errs.append({"severity": "major", "target": aid,
                         "error": "implausible_effect_for_unevidenced_action",
                         "detail": f"{val}pp with no experimental backing",
                         "fix": "Cap inferred effects at <=25pp or use basis='none'."})

        # ---- must not contradict a well-powered null ----
        # Substring matching on a free-text field false-positives badly ("cumulative
        # structure" appears in unrelated prose), so require substantial TOKEN overlap
        # AND the same pillar the null actually tested.
        a_text = fold(f"{a.get('title', '')} {a.get('what', '')}")
        a_tokens = {w for w in a_text.split() if len(w) > 3}
        for var, meta in disproven_vars.items():
            v_tokens = {w for w in var.split() if len(w) > 3}
            if len(v_tokens) < 2 or a.get("pillar") not in meta["pillars"]:
                continue
            if v_tokens <= a_tokens:  # every content word of the null present
                errs.append({
                    "severity": "blocker", "target": aid,
                    "error": "contradicts_well_powered_null",
                    "detail": f"'{var}' showed no effect in {meta['design_id']} "
                              f"({meta['runs']} runs/arm) and this action is in the same "
                              f"pillar",
                    "fix": "Remove, or re-scope to a mechanism the null did not test and "
                           "state the difference explicitly."})
                break

    # ---- roadmap-level checks ----
    hyp = [a for a in actions if a.get("evidence_class") == "HYPOTHESIS"]
    if actions and len(hyp) / len(actions) > HYPOTHESIS_EFFORT_CAP:
        errs.append({"severity": "major", "target": "roadmap",
                     "error": "too_many_unevidenced_actions",
                     "detail": f"{len(hyp)} of {len(actions)} are HYPOTHESIS "
                               f"(cap {HYPOTHESIS_EFFORT_CAP:.0%})",
                     "fix": "Downgrade to a test-first plan or drop."})
    if not (strategy.get("not_doing") or []):
        errs.append({"severity": "major", "target": "overall",
                     "error": "no_not_doing_list",
                     "detail": f"{sum(1 for n in si['nulls'] if not n.get('underpowered'))}"
                               f" well-powered null(s) exist but nothing was excluded",
                     "fix": "State explicitly what the evidence rules out."})

    # ---- F-F: a big bet with no actions is not a strategy ----
    action_ids = {a.get("action_id") for a in actions}
    for i, b in enumerate(strategy.get("big_bets") or []):
        if not [x for x in (b.get("action_ids") or []) if x in action_ids]:
            errs.append({"severity": "blocker", "target": f"big_bets[{i}]",
                         "error": "big_bet_without_actions",
                         "detail": b.get("title"),
                         "fix": "Assign concrete action_ids, or remove."})
    # ---- proven evidence must reach the big bets ----
    proven = [w for w in si["wins"]
              if str(w.get("evidence_class", "")).startswith("PROVEN")
              and not w.get("underpowered")]
    bet_refs = {r for b in (strategy.get("big_bets") or [])
                for aid in (b.get("action_ids") or [])
                for r in ((next((a for a in actions if a.get("action_id") == aid), {})
                           ).get("evidence_refs") or [])}
    for w in proven:
        if w["design_id"] not in bet_refs:
            errs.append({
                "severity": "blocker", "target": "big_bets",
                "error": "proven_finding_absent_from_big_bets",
                "detail": f"{w['design_id']} is {w.get('evidence_class')} "
                          f"({w.get('headline_effect', {}).get('detail')}) but no big bet "
                          f"builds on it",
                "fix": "Add a big bet scaling this proven mechanism, or explain in the "
                       "narrative why proven evidence is being deprioritised."})

    counts = Counter(a.get("pillar") for a in actions)
    if counts:
        top_pillar = counts.most_common(1)[0][0]
        bet_pillars = {p for b in (strategy.get("big_bets") or [])
                       for p in (b.get("pillars") or [])}
        if top_pillar not in bet_pillars:
            errs.append({"severity": "major", "target": "big_bets",
                         "error": "largest_pillar_has_no_big_bet",
                         "detail": f"{top_pillar} holds {counts[top_pillar]} action(s) "
                                   f"but appears in no big bet",
                         "fix": "Add a big bet for it, or reduce its action count."})

    return errs

def all_actions(pillar_strategies):
    out = []
    for pillar, p in (pillar_strategies or {}).items():
        for a in (p.get("actions") or []):
            a = dict(a)
            a.setdefault("pillar", pillar)
            out.append(a)
    return out

# ====================================================================== orchestrator
def build_strategy_llm(gateway, parent, target, category, *, model,
                       constraints=None, progress_cb=None, run_critic=True):
    si = assemble_strategy_input(parent, target, category, constraints)
    if not (si["wins"] or si["nulls"] or si["gaps"]):
        return {"error": "No findings, nulls or gaps found. Complete Benchmark and "
                         "Experiment first."}

    def step(x, label):
        if progress_cb:
            progress_cb(x, label)

    step(0.05, "Routing gaps to pillars…")
    routed, retrieval_constraint = route_gaps_to_pillars(si)
    si["_retrieval_constraint"] = retrieval_constraint

    step(0.10, "Drafting pillar strategies…")
    pillars = build_pillar_strategies(
        gateway, si, routed, model,
        progress_cb=lambda x: step(0.10 + 0.45 * x, "Drafting pillar strategies…"))

    step(0.58, "Assigning horizons…")
    pillars = apply_horizons(pillars,si)

    step(0.62, "Cross-engine synthesis…")
    overall = synthesise_overall(gateway, si, pillars, model)

    # auto-link big bets that named pillars but no actions
    _acts = all_actions(pillars)
    _by_pillar = defaultdict(list)
    for a in _acts:
        _by_pillar[a.get("pillar")].append(a)
    for b in (overall.get("big_bets") or []):
        if b.get("action_ids"):
            continue
        cand = [a for pl in (b.get("pillars") or []) for a in _by_pillar.get(pl, [])]
        cand.sort(key=lambda a: -EVIDENCE_WEIGHT.get(a.get("evidence_class"), 0.2))
        if cand:
            b["action_ids"] = [a["action_id"] for a in cand[:4]]
            b["_auto_linked"] = True

    from optimise.roadmap import build_dag
    _dag = build_dag(_acts)
    _byid = {a["action_id"]: a for a in _acts if a.get("action_id")}
    for b in (overall.get("big_bets") or []):
        ids = list(b.get("action_ids") or [])
        upstream = {d for i in ids for d in _dag.get(i, []) if d not in ids}
        if upstream:
            b["action_ids"] = sorted(upstream) + ids
            b["prerequisites"] = sorted(upstream)
            b["_dependency_note"] = (f"{len(upstream)} upstream action(s) must complete "
                                     f"first: {', '.join(sorted(upstream))}.")
        for i in ids:
            a = _byid.get(i)
            if a and not a.get("dependencies"):
                a["dependencies"] = _dag.get(i, [])

    # ---- order big bets by evidence strength, not by ambition ----
    # An LLM reliably gravitates to the largest number available regardless of its
    # evidence class. A caveated result must never lead the document.
    _RANK = {"PROVEN": 0, "PROVEN_ENGINE_SPECIFIC": 1, "PROVEN_WITH_CAVEAT": 2,
             "DIAGNOSED": 3, "HYPOTHESIS": 4}

    def _bet_rank(b):
        # take the BEST evidence available: the bet's own declared class OR any of its
        # actions'. Previously the bet's own class was ignored whenever it listed actions,
        # so a PROVEN_ENGINE_SPECIFIC bet sank below a caveated one.
        classes = [(_byid.get(aid) or {}).get("evidence_class")
                   for aid in (b.get("action_ids") or [])]
        classes.append(b.get("evidence_class"))
        classes = [c for c in classes if c]
        return min((_RANK.get(c, 5) for c in classes), default=5)

    if overall.get("big_bets"):
        overall["big_bets"] = sorted(overall["big_bets"], key=_bet_rank)
        for b in overall["big_bets"]:
            b["_evidence_rank"] = _bet_rank(b)

    step(0.78, "Engine overlays…")
    overlays = build_engine_overlays(gateway, si, overall, model)

    # deterministic validation FIRST, so its blockers reach the critic as evidence
    det = deterministic_validate(si, pillars, overall)
    critique = []
    if run_critic:
        step(0.90, "Adversarial critic…")
        critique = critic_pass(gateway, si, overall, pillars, model,
                               prior_errors=det)
    critique = [c for c in critique
                if isinstance(c, dict) and (c.get("issue") or c.get("error"))]
    step(1.0, "Complete")

    actions = all_actions(pillars)
    blockers = {c.get("target") for c in critique
                if c.get("severity") == "blocker"} | \
               {e.get("target") for e in det if e.get("severity") == "blocker"}
    for a in actions:
        a["blocked"] = a.get("action_id") in blockers

    return {
        "PROVISIONAL": False,
        "target_brand": target, "parent_brand": parent, "category": category,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_summary": evidence_summary(si),
        "retrieval_is_primary_constraint": retrieval_constraint,
        "overall_strategy": overall,
        "pillars": pillars,
        "engine_overlays": overlays,
        "actions": actions,
        "pillar_summary": dict(Counter(a["pillar"] for a in actions)),
        "horizon_summary": dict(Counter(a.get("horizon") for a in actions)),
        "evidence_summary_by_class": dict(Counter(a.get("evidence_class")
                                                 for a in actions)),
        "critic_review": critique,
        "validation_errors": det,
        "n_blockers": len(blockers),
        "inputs_hash": {"n_wins": len(si["wins"]), "n_nulls": len(si["nulls"]),
                        "n_gaps": len(si["gaps"]),
                        "n_substantiation": len(si["substantiation_backlog"]),
                        "n_claims": len(si["claims"])},
        "_si": si,
    }