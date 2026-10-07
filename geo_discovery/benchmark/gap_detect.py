"""Deterministic, typed, evidence-carrying gap detection.

Stage 1 of the gap pipeline. Detection MUST be reproducible, so it stays rule-based —
Batch 3 adds the LLM reasoning layer (root cause, mechanism, prediction, falsification,
experiment mapping) on top of these typed gaps.

Replaces the old rule-based build_gap_register, which read only `claim_deltas` with the
v1 verdict vocabulary and emitted one identical format-string hypothesis for every gap.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from metrics_spec import fold

# ---------------------------------------------------------------- taxonomy
GAP_TYPES = {
    "CLAIM_ABSENT":         ("Content", "high"),
    "CLAIM_FRAGILE":        ("Corroboration", "medium"),
    "CLAIM_DILUTED":        ("Content", "medium"),
    "CLAIM_DISTORTED":      ("Claims-Substantiation", "critical"),
    "HALLUCINATION":        ("Claims-Substantiation", "critical"),
    "COMPETITOR_BLEED":     ("Content", "critical"),
    "STALE_CLAIM":          ("Claims-Substantiation", "high"),
    "UNSUBSTANTIATED_HELP": ("Claims-Substantiation", "medium"),
    "REASON_DEFICIT":       ("Content", "high"),
    "FLAT_MENTION":         ("Content", "high"),
    "PROVENANCE_THIN":      ("Corroboration", "high"),
    "GREY_MARKET_VOICE":    ("Corroboration", "high"),
    "SOURCE_ABSENCE":       ("Corroboration", "medium"),
    "NEGATIVE_FRAMING":     ("Content", "high"),
    "ADVISORY_DEFERRAL":    ("Corroboration", "medium"),
    "VISIBILITY_GAP":       ("Content", "high"),
    "RANK_GAP":             ("Technical/Structured-Data", "medium"),
    "ENGINE_DIVERGENCE":    ("Technical/Structured-Data", "medium"),
    "REFUSAL":              ("Content", "medium"),
    "FRESHNESS_STALE":      ("Content", "medium"),
    "CLAIM_NO_EVIDENCE":    ("Content", "high"),
    "OBJECTION_EXPOSURE":   ("Content", "high"),
    "DEFEND_POSITION":      ("Corroboration", "medium"),
    "VISUAL_UNREADABLE":    ("Visual/Multimodal", "high"),
    "VISUAL_UNASSESSED":    ("Visual/Multimodal", "high"),   # was medium
    # W-55: Pillar 5 was structurally unreachable — no gap type routed to it, so the
    # pillar prompt was never called. These two are the entry point.
    "SOCIAL_ABSENCE":       ("Social/Community", "medium"),
    "SOCIAL_NEGATIVE":      ("Social/Community", "high"),
}
SEV_PRIORITY = {"critical": 5, "high": 4, "medium": 3, "low": 2, "none": 1}

class _GapBuilder:
    def __init__(self):
        self.gaps = []
        self._n = 0

    def add(self, gtype, gap_text, *, severity=None, pillar=None, evidence=None,
            engines=(), intent=None, claim=None, claim_id=None, dimension=None,
            domain=None, query=None, hypothesis=None, metric_baseline=None):
        default_pillar, default_sev = GAP_TYPES.get(gtype, ("Content", "medium"))
        sev = severity or default_sev
        self._n += 1
        self.gaps.append({
            "gap_id": f"G{self._n:03d}",
            "type": gtype,
            "gap": gap_text,
            # legacy keys kept so gap_mapper / signal_designer / optimise keep working
            "delta_type": _legacy_delta_type(gtype),
            "severity": sev,
            "pillar": pillar or default_pillar,
            "priority": SEV_PRIORITY.get(sev, 3),
            "engines": sorted(set(e for e in engines if e and e != "*")),
            "intent": intent, "claim": claim, "claim_id": claim_id,
            "dimension": dimension, "domain": domain, "query": query,
            "metric_baseline": metric_baseline,
            "evidence": evidence or {},
            # placeholder — Batch 3 replaces this with an LLM-reasoned hypothesis
            "hypothesis": hypothesis or _stub_hypothesis(gtype, gap_text),
            "hypothesis_source": "rule_stub",
        })

def _legacy_delta_type(gtype):
    return {
        "CLAIM_ABSENT": "missing", "CLAIM_FRAGILE": "fragile",
        "CLAIM_DILUTED": "diluted", "CLAIM_DISTORTED": "hallucination",
        "HALLUCINATION": "hallucination", "COMPETITOR_BLEED": "hallucination",
        "STALE_CLAIM": "hallucination", "UNSUBSTANTIATED_HELP": "missing",
        "REASON_DEFICIT": "under-surfaced", "FLAT_MENTION": "under-surfaced",
        "PROVENANCE_THIN": "under-surfaced", "GREY_MARKET_VOICE": "under-surfaced",
        "SOURCE_ABSENCE": "under-surfaced", "NEGATIVE_FRAMING": "negative",
        "ADVISORY_DEFERRAL": "negative", "VISIBILITY_GAP": "under-surfaced",
        "RANK_GAP": "under-surfaced", "ENGINE_DIVERGENCE": "under-surfaced",
        "REFUSAL": "negative", "FRESHNESS_STALE": "missing",
        "CLAIM_NO_EVIDENCE": "missing",
        "OBJECTION_EXPOSURE": "negative", "DEFEND_POSITION": "under-surfaced",
        "VISUAL_UNREADABLE": "missing", "VISUAL_UNASSESSED": "under-surfaced",
        "SOCIAL_ABSENCE": "under-surfaced", "SOCIAL_NEGATIVE": "negative",
    }.get(gtype, "missing")

def _stub_hypothesis(gtype, gap_text):
    return {
        "CLAIM_ABSENT": "Publishing this claim in machine-readable form will increase "
                        "its appearance in answers about the brand.",
        "CLAIM_FRAGILE": "Third-party corroboration will stabilise this claim across runs.",
        "CLAIM_DILUTED": "Restating the claim with its quantified specifics will preserve "
                         "the specific detail in answers.",
        "CLAIM_DISTORTED": "Publishing an authoritative, citable version of this claim "
                           "will displace the distorted version.",
        "HALLUCINATION": "Publishing a substantiated correction will reduce the "
                         "unsupported assertion.",
        "COMPETITOR_BLEED": "Explicit brand-attribute association will stop the "
                            "competitor attribute being attributed to us.",
        "REASON_DEFICIT": "Adding this reason type will close the richness gap to the leader.",
        "FLAT_MENTION": "Attaching quantified reasons to brand mentions will move the "
                        "brand from named to recommended.",
        "PROVENANCE_THIN": "Increasing brand-owned and editorially corroborated citations "
                           "will increase citation control.",
        "GREY_MARKET_VOICE": "Publishing authoritative brand content will displace "
                             "third-party resellers as the cited source.",
        "SOURCE_ABSENCE": "Presence on this cited domain will increase brand appearance "
                          "in answers that cite it.",
        "NEGATIVE_FRAMING": "A substantiated rebuttal will reduce negative framing.",
        "ADVISORY_DEFERRAL": "Clinical/professional corroboration will make the brand the "
                             "one named inside professional-deferral answers.",
        "VISIBILITY_GAP": "Enriched, relevant content will increase unprompted mention on "
                          "these queries.",
        "RANK_GAP": "Structured, answer-first content will move the brand from mentioned "
                    "to first-recommended.",
        "ENGINE_DIVERGENCE": "The winning engine's content pattern will transfer to the "
                             "lagging engine.",
        "REFUSAL": "Reframing content around substantiated, non-advisory facts will "
                   "reduce refusals.",
        "CLAIM_NO_EVIDENCE": "No answer in the corpus discussed this claim's topic. "
                             "Publishing it where the topic arises will create the "
                             "opportunity for it to be stated.",
        "OBJECTION_EXPOSURE": "A substantiated rebuttal published where the objection "
                              "is discussed will improve how the brand is framed when "
                              "named on this query.",
        "DEFEND_POSITION": "Third-party corroboration will protect a leading position "
                           "against competitor displacement.",
        "VISUAL_UNREADABLE": "Printing this claim legibly on-pack will let vision "
                             "models extract it, so it appears in answers where the "
                             "product image is the evidence.",
        "VISUAL_UNASSESSED": "Interrogating the images and social pages the engines "
                             "return will reveal whether packaging is machine-readable.",
        "SOCIAL_ABSENCE": "Earning presence in the community threads the engines cite "
                          "will increase brand mention on the queries that trigger them.",
        "SOCIAL_NEGATIVE": "Correcting the factual errors in the community threads the "
                           "engines cite will reduce negative framing where those threads "
                           "are the evidence base.",
    }.get(gtype, f"Closing this gap will improve brand visibility: {gap_text[:80]}")

# ---------------------------------------------------------------- detectors
def detect_gaps(delta, winnability=None, metrics=None, hitlist=None,
                target_brand="", max_source_gaps=8, max_reason_gaps=8,
                max_claim_gaps=25):
    """Read the v2 Delta Scorecard + metrics bundle → typed, evidenced gaps."""
    B = _GapBuilder()
    d = delta or {}
    m = metrics or {}

    # ---- 1. CLAIM DELTAS, clustered per claim across engines ----
    per_claim = defaultdict(list)
    for cd in d.get("claim_deltas", []):
        if cd.get("delta_type") == "correct":
            continue
        per_claim[cd.get("claim")].append(cd)

    ordered = sorted(per_claim.items(),
                     key=lambda kv: min(SEV_PRIORITY.get(x["severity"], 0)
                                        for x in kv[1]), reverse=True)
    for claim, rows in ordered[:max_claim_gaps]:
        worst = max(rows, key=lambda r: SEV_PRIORITY.get(r["severity"], 0))
        gtype = {"missing": "CLAIM_ABSENT", "fragile": "CLAIM_FRAGILE",
                 "diluted": "CLAIM_DILUTED", "distortion": "CLAIM_DISTORTED",
                 "no_evidence": "CLAIM_NO_EVIDENCE",
                 }.get(worst["delta_type"], "CLAIM_ABSENT")
        # suppress under-sampled mid-band "fragile" — it is a sampling artifact
        if gtype == "CLAIM_FRAGILE":
            covs_all = [r.get("coverage") or 0 for r in rows]
            if max(covs_all) >= 0.7 and all((r.get("n_spans") or 0) < 8 for r in rows):
                continue

        engs = [r.get("engine") for r in rows]
        covs = [r.get("coverage") for r in rows if r.get("coverage") is not None]
        note = next((r.get("delta_note") for r in rows if r.get("delta_note")), "")
        quote = next((r.get("evidence_quote") for r in rows if r.get("evidence_quote")), "")

        # ---- W-14: never claim "all engines" when engines disagree ----
        real_engs = sorted({e for e in engs if e and e != "*"})
        statuses = {r.get("engine"): r.get("machine_status") for r in rows}
        stated = sorted({e for e, s in statuses.items()
                         if s in ("locked_in", "mostly_supported") and e != "*"})
        divergent = bool(stated) and len(real_engs) > 1 and len(stated) < len(real_engs)
        if "*" in engs:
            where = "no engine retrieved this claim"
        elif divergent:
            missing_on = [e for e in real_engs if e not in stated]
            where = (f"{', '.join(missing_on)} (stated correctly on "
                     f"{', '.join(stated)} — engine divergence)")
        elif len(real_engs) > 1:
            where = f"all engines tested ({', '.join(real_engs)})"
        else:
            where = real_engs[0] if real_engs else "unknown engine"

        # ---- W-17: low-n evidence must not carry full severity ----
        sev = worst["severity"]
        strengths = {r.get("evidence_strength") for r in rows}
        if strengths and strengths <= {"low_n", "none", None} and sev != "critical":
            sev = {"high": "medium", "medium": "low"}.get(sev, sev)
            note = (note + " [low sample — severity downgraded]").strip()

        B.add(gtype,
              f"Claim '{_short(claim)}' is {worst['delta_type']} on {where}"
              + (f" — {note}" if note else ""),
              severity=sev, engines=engs,
              claim=claim, claim_id=worst.get("claim_id"),
              metric_baseline=(round(sum(covs) / len(covs), 3) if covs else 0.0),
              evidence={"statuses": statuses,
                        "coverage": {r.get("engine"): r.get("coverage") for r in rows},
                        "ci": worst.get("ci"), "n_spans": worst.get("n_spans"),
                        "evidence_strength": worst.get("evidence_strength"),
                        "engine_divergent": divergent,
                        "stated_correctly_on": stated,
                        "delta_note": note, "quote": quote,
                        "claim_category": worst.get("claim_category")})

    # ---- 2. RISK REGISTER (reverse hallucination scan) ----
    risk_map = {"hallucination": "HALLUCINATION", "misattribution": "COMPETITOR_BLEED",
                "stale": "STALE_CLAIM", "unsubstantiated": "UNSUBSTANTIATED_HELP"}
    seen_assert = set()
    for r in d.get("risk_register", []):
        a = fold(r.get("assertion", ""))[:90]
        if not a or a in seen_assert:
            continue
        seen_assert.add(a)
        gtype = risk_map.get(r.get("delta_type"), "UNSUBSTANTIATED_HELP")
        B.add(gtype,
              f"{r.get('delta_type','risk').title()}: \"{_short(r.get('assertion'))}\" "
              f"({r.get('engine')})",
              severity=r.get("severity"), engines=[r.get("engine")],
              query=r.get("query"),
              evidence={"quote": r.get("evidence_quote"), "why": r.get("why"),
                        "risk": r.get("risk")})

    # ---- 3. REASON DEFICITS vs the leader ----
    for rg in (d.get("reason_gap") or [])[:max_reason_gaps]:
        # tolerate both schemas: per-100-token (current) and raw per-span (pre-4b)
        brand_v = rg.get("brand_per_100t", rg.get("brand_avg"))
        leader_v = rg.get("leader_per_100t", rg.get("leader_avg"))
        deficit = rg.get("deficit_per_100t", rg.get("deficit"))
        unit = "per 100 tokens" if "brand_per_100t" in rg else "per span"
        if brand_v is None or leader_v is None:
            continue
        strength = rg.get("evidence_strength")
        rsev = "medium" if strength == "low_n" else None
        B.add("REASON_DEFICIT",
              f"'{rg.get('dimension')}' deficit vs {rg.get('leader')} on "
              f"{rg.get('intent')} queries ({brand_v} vs {leader_v} {unit})"
              + (" [low sample]" if strength == "low_n" else ""),
              severity=rsev,
              intent=rg.get("intent"), dimension=rg.get("dimension"),
              metric_baseline=brand_v,
              evidence={"leader": rg.get("leader"),
                        "leader_value": leader_v, "brand_value": brand_v,
                        "deficit": deficit, "unit": unit,
                        "evidence_strength": strength,
                        "leader_spans": rg.get("leader_spans"),
                        "brand_spans": rg.get("brand_spans")})

    # ---- 4. FLATNESS: named but not argued for ----
    fl = d.get("flatness") or {}
    if fl.get("spans", 0) >= 10:
        if fl.get("no_substantive_share", 0) >= 0.30:
            B.add("FLAT_MENTION",
                  f"{fl['no_substantive_share']:.0%} of brand mentions carry no "
                  f"substantive reason (no dose, study, certification, mechanism or "
                  f"cited source)",
                  metric_baseline=fl.get("no_substantive_share"), evidence=fl)
        elif fl.get("shallow_share", 0) >= 0.55:
            B.add("FLAT_MENTION",
                  f"{fl['shallow_share']:.0%} of brand mentions carry at most ONE "
                  f"substantive reason — thin argumentation",
                  severity="medium", metric_baseline=fl.get("shallow_share"),
                  evidence=fl)
        if (fl.get("richness_ratio_vs_leader") or 1) < 0.75:
            B.add("REASON_DEFICIT",
                  f"Reason density is {fl['richness_ratio_vs_leader']}× "
                  f"{fl.get('leader')}'s per 100 tokens — competitors argue harder in "
                  f"the same space",
                  dimension="overall_density",
                  metric_baseline=fl.get("mean_richness_per_100t"), evidence=fl)

    # ---- 5. PROVENANCE / VOICE CONTROL ----
    pv = d.get("provenance") or {}
    _rel = (pv.get("reliability") or {})
    if pv and not _rel.get("reliable", True):
        B.add("PROVENANCE_THIN",
              f"Provenance cannot be assessed: {pv.get('unclassified_pct', 0):.0%} of "
              f"citations are unclassified. Every citation-control figure is a lower "
              f"bound and the voice-control ratio is withheld.",
              severity="high",
              metric_baseline=pv.get("unclassified_pct"),
              evidence={"reliability": _rel,
                        "provisional_ratio": pv.get("voice_control_ratio_provisional"),
                        "remedy": "Generate a source pack in the Audit tab, then run "
                                  "backfill.py and re-run Benchmark."})
    elif pv:
        if pv.get("self_referential_pct", 0) < 0.05:
            B.add("PROVENANCE_THIN",
                  f"Brand-owned sources are only "
                  f"{pv.get('self_referential_pct',0):.1%} of citations — the brand has "
                  f"almost no machine-readable voice of its own",
                  metric_baseline=pv.get("self_referential_pct"),
                  evidence={k: pv.get(k) for k in
                            ("self_referential_pct", "corroborated_pct",
                             "total_citations", "voice_control_ratio")})
        grey = pv.get("grey_market_pct", 0) + pv.get("affiliate_pct", 0)
        if pv.get("practitioner_pct", 0) >= 0.10:
            B.add("SOURCE_ABSENCE",
                  f"{pv['practitioner_pct']:.0%} of citations come from independent "
                  f"practitioners in this category — a corroboration channel that is "
                  f"engaged professionally, not pitched editorially",
                  severity="medium",
                  metric_baseline=pv["practitioner_pct"],
                  evidence={"source_type": "practitioner",
                            "practitioner_pct": pv["practitioner_pct"],
                            "note": pv.get("reliability", {}).get("practitioner_note")})
        if grey >= 0.12:
            B.add("GREY_MARKET_VOICE",
                  f"{grey:.0%} of citations come from grey-market resellers and "
                  f"SEO/affiliate sites (voice-control ratio "
                  f"{pv.get('voice_control_ratio')}) — uncontrolled parties describe "
                  f"the brand in a regulated category",
                  severity="critical" if grey >= 0.25 else "high",
                  metric_baseline=grey,
                  evidence={k: pv.get(k) for k in
                            ("grey_market_pct", "affiliate_pct",
                             "competitor_owned_pct", "voice_control_ratio")})
        if pv.get("competitor_owned_pct", 0) >= 0.10:
            B.add("SOURCE_ABSENCE",
                  f"{pv['competitor_owned_pct']:.0%} of citations are competitor-owned "
                  f"properties — competitors supply the evidence base",
                  evidence={"competitor_owned_pct": pv["competitor_owned_pct"]})

    # ---- 6. FRAMING / ADVISORY ----
    fr = d.get("framing") or {}
    if fr.get("mentions"):
        if fr.get("neg_share", 0) >= 0.12:
            B.add("NEGATIVE_FRAMING",
                  f"{fr['neg_share']:.0%} of brand mentions are negatively framed",
                  metric_baseline=fr["neg_share"], evidence=fr)
        if (fr.get("advisory_share") or 0) >= 0.5:
            B.add("ADVISORY_DEFERRAL",
                  f"{fr['advisory_share']:.0%} of brand mentions sit inside an answer "
                  f"that defers to a health professional — recommendations are hedged, "
                  f"not absent",
                  metric_baseline=fr["advisory_share"],
                  evidence={"advisory_share": fr.get("advisory_share"),
                            "hedged_share": fr.get("hedged_share")})
        if fr.get("neu_share", 0) >= 0.5:
            B.add("FLAT_MENTION",
                  f"{fr['neu_share']:.0%} of mentions are neutral — the brand is listed, "
                  f"not advocated",
                  metric_baseline=fr["neu_share"], evidence=fr)

    # ---- 7. WEAKNESS FLAGS from the metrics bundle ----
    flag_map = {"visibility_gap": "VISIBILITY_GAP",
                "need_visibility_gap": "VISIBILITY_GAP",
                "intent_gap": "VISIBILITY_GAP", "need_first_low": "VISIBILITY_GAP",
                "rank_gap": "RANK_GAP", "richness_deficit": "REASON_DEFICIT",
                "refusal_heavy": "REFUSAL", "advisory_heavy": "ADVISORY_DEFERRAL",
                "negative_framing": "NEGATIVE_FRAMING", "veracity_gap": "CLAIM_ABSENT"}
    seen_flags = set()
    for f in m.get("weakness_flags", []):
        gt = flag_map.get(f.get("type"))
        key = (gt, f.get("target_intent"))
        if not gt or key in seen_flags:
            continue
        seen_flags.add(key)
        B.add(gt, f.get("detail", f.get("type", "weakness")),
              severity=f.get("severity"), intent=f.get("target_intent"),
              evidence={"flag": f.get("type"), "detail": f.get("detail")})

    # ---- 8. FRAGILE / ABSENT QUERIES + ENGINE DIVERGENCE ----
    cons = d.get("consistency") or {}
    ov = cons.get("overall", {})
    for q in (ov.get("fragile") or [])[:6]:
        B.add("CLAIM_FRAGILE",
              f"Brand appears in under 50% of runs for '{_short(q)}' — unstable presence",
              query=q, evidence={"note": "fragile across runs"})
    # W-16: the 50-90% band previously fell into no bucket and generated no gap
    for q in (ov.get("emerging") or [])[:6]:
        B.add("CLAIM_FRAGILE",
              f"Brand appears in 50-90% of runs for '{_short(q)}' — present but not "
              f"locked in; the most actionable band",
              severity="medium", query=q,
              evidence={"note": "emerging — inconsistent but established"})
    for dv in (cons.get("divergence") or [])[:6]:
        B.add("ENGINE_DIVERGENCE",
              f"'{_short(dv['query'])}': {dv['spread_pp']}pp gap between "
              f"{dv['best_engine']} and {dv['worst_engine']}",
              engines=[dv.get("best_engine"), dv.get("worst_engine")],
              query=dv.get("query"), evidence=dv)

    # ---- 9. WINNABILITY-DERIVED GAPS ----
    if winnability is not None and len(winnability):
        try:
            w = winnability
            # W-13: rank by HEADROOM × REACH, not by `confidence`, which was a symmetric
            # distance from 50% and therefore preferred 100%-mention queries with no room.
            win = w[w["winnability"] == "winnable"].sort_values(
                "opportunity", ascending=False).head(6)
            for _, r in win.iterrows():
                sig = r.get("signals") or {}
                B.add("VISIBILITY_GAP",
                      f"Winnable query under-served: '{_short(r['query'])}' "
                      f"(mention {sig.get('mention_rate', 0):.0%}, "
                      f"first {sig.get('first_rate', 0):.0%}, "
                      f"opportunity {r.get('opportunity')})",
                      query=r["query"], intent=r.get("intent"),
                      metric_baseline=sig.get("first_rate"),
                      evidence={"signals": sig, "rule_reason": r.get("rule_reason"),
                                "opportunity": r.get("opportunity")})

            # negative-intent queries are an OBJECTION problem, not a visibility one
            # negative-intent queries: FRAMING problem, not a visibility one. The gap
            # text must adapt — at 0% mention this is absence, not exposure.
            obj = w[w["winnability"] == "objection_defence"].sort_values(
                "opportunity", ascending=False).head(4)
            for _, r in obj.iterrows():
                sig = r.get("signals") or {}
                mr = sig.get("mention_rate", 0)
                ns = sig.get("neg_share", 0)
                if mr < 0.10 and ns < 0.05:
                    # the brand barely appears and is not framed negatively — this is
                    # not an exposure. Record as a low-severity absence instead.
                    B.add("VISIBILITY_GAP",
                          f"Brand is largely absent from the objection query "
                          f"'{_short(r['query'])}' ({mr:.0%} mention) — no negative "
                          f"framing observed, so there is nothing to rebut yet",
                          severity="low", query=r["query"], intent=r.get("intent"),
                          metric_baseline=mr,
                          evidence={"signals": sig, "note": "objection absence"})
                    continue
                if ns >= 0.15:
                    desc = (f"with {ns:.0%} of those mentions negatively framed — "
                            f"reliably retrieved as a cautionary example")
                    sev = "critical" if (mr >= 0.60 and ns >= 0.25) else "high"
                elif mr >= 0.30:
                    desc = ("framed neutrally, so the brand is listed among products "
                            "being questioned without any rebuttal present")
                    sev = "high"
                else:
                    desc = "framed neutrally — monitor, no rebuttal case yet"
                    sev = "medium"
                B.add("OBJECTION_EXPOSURE",
                      f"Brand surfaces on the objection query '{_short(r['query'])}' "
                      f"in {mr:.0%} of runs, {desc}",
                      severity=sev, query=r["query"], intent=r.get("intent"),
                      metric_baseline=ns if ns else mr,
                      evidence={"signals": sig, "neg_share": ns, "mention_rate": mr,
                                "rule_reason": r.get("rule_reason")})

            # leading positions worth defending
            win = w[w["winnability"] == "winnable"].sort_values(
                "opportunity", ascending=False).head(8)
            for _, r in win.iterrows():
                sig = r.get("signals") or {}
                mr, fr = sig.get("mention_rate", 0), sig.get("first_rate", 0)
                # W-37: mentioned but never first is a RANK problem with a different
                # remedy and a different metric from an absence problem.
                if mr >= 0.25 and fr <= 0.10:
                    B.add("RANK_GAP",
                          f"Mentioned but buried on '{_short(r['query'])}' "
                          f"(mention {mr:.0%}, first {fr:.1%}) — the brand is retrieved "
                          f"and then not chosen",
                          severity="high", query=r["query"], intent=r.get("intent"),
                          metric_baseline=fr,
                          evidence={"signals": sig, "rule_reason": r.get("rule_reason"),
                                    "opportunity": r.get("opportunity")})
                else:
                    B.add("VISIBILITY_GAP",
                          f"Winnable query under-served: '{_short(r['query'])}' "
                          f"(mention {mr:.0%}, first {fr:.0%}, "
                          f"opportunity {r.get('opportunity')})",
                          query=r["query"], intent=r.get("intent"),
                          metric_baseline=mr,
                          evidence={"signals": sig, "rule_reason": r.get("rule_reason"),
                                    "opportunity": r.get("opportunity")})
        except Exception as e:
            B.add("VISIBILITY_GAP", f"winnability parsing failed: {str(e)[:80]}",
                  severity="low")

    # ---- 10. SOURCE HIT-LIST targets ----
    rows = (hitlist.to_dict("records") if hasattr(hitlist, "to_dict")
            else (hitlist or []))
    ranked = sorted(rows,
                    key=lambda h: -(h.get("brand_absence_index")
                                    or h.get("times_cited") or 0))
    for h in ranked[:max_source_gaps]:
        st = h.get("source_type", "other")
        # competitor_owned added: tier_c._bucket already returns "Ignore (competitor
        # property)", but the detector was still creating placement gaps for them —
        # producing a hypothesis proposing we cite a direct competitor's website.
        if st in ("brand_owned", "cdn_asset", "competitor_owned"):
            continue
        bpr = h.get("brand_present_rate")
        desc = (f"absent on {(1-bpr):.0%} of its citing runs" if bpr is not None
                else f"cited {h.get('times_cited')}× — brand presence unverified")
        B.add("SOURCE_ABSENCE",
              f"High-influence source '{h.get('domain')}' ({st}) — {desc}",
              domain=h.get("domain"),
              evidence={k: h.get(k) for k in
                        ("times_cited", "source_type", "brand_present_rate",
                         "brand_absence_index", "n_engines", "top_url", "bucket",
                         "access_route") if h.get(k) is not None})
    # ---- 11. VISUAL: claims whose only proof is an image, absent from machine reality ----
    # The pipeline could not previously produce a visual gap at all, so no visual cluster,
    # hypothesis or experiment was ever generated.
    _img_origins = {"image", "video", ".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov"}

    def _from_creative(cd):
        o = str((cd.get("evidence") or {}).get("origin")
                or cd.get("_origin") or "").lower()
        return any(x in o for x in _img_origins)

    creative_absent = [cd for cd in d.get("claim_deltas", [])
                       if cd.get("delta_type") in ("missing", "no_evidence")
                       and _from_creative(cd)]
    seen_c = set()
    for cd in creative_absent[:6]:
        key = fold(cd.get("claim", ""))[:60]
        if key in seen_c:
            continue
        seen_c.add(key)
        B.add("VISUAL_UNREADABLE",
              f"Claim '{_short(cd.get('claim'))}' exists only in creative or packaging "
              f"and is absent from machine reality — the engines cannot read it off the "
              f"image",
              severity="high", claim=cd.get("claim"), claim_id=cd.get("claim_id"),
              engines=[cd.get("engine")],
              evidence={"origin": "creative/packaging", "coverage": cd.get("coverage"),
                        "quote": cd.get("evidence_quote"),
                        "why": "claim sourced from an image or video asset"})

    # ---- visual instrumentation gap ----
    man = m.get("_manifest") or {}
    n_vis_pages = m.get("visual_pages_total")
    n_imgs = m.get("images_interrogated")
    vs = (man.get("vision_selftest") or {})
    if n_vis_pages and not n_imgs:
        B.add("VISUAL_UNASSESSED",
              f"The engines returned {n_vis_pages} social or video PAGE(s) but zero "
              f"direct image URLs, so no packaging was interrogated. Visual legibility "
              f"is unmeasured, not confirmed good.",
              severity="medium",
              evidence={"visual_pages": n_vis_pages, "images_interrogated": n_imgs,
                        "vision_selftest": vs.get("vision_live")})
    elif vs.get("vision_live") is False:
        B.add("VISUAL_UNASSESSED",
              "The vision self-test failed — the image payload never reached the model. "
              "No visual conclusion can be drawn until this is fixed.",
              severity="high", evidence={"vision_selftest": vs})

        # ---- 12. SOCIAL / COMMUNITY (W-55) ----
        # Without these, no gap routes to Social/Community and the pillar is never drafted.
    soc = m.get("social") or (d.get("social") or {})
    if soc and soc.get("verdict") == "negligible":
        # Say it explicitly rather than leaving the pillar silently empty. Recommending
        # community work where engines do not listen is the most expensive mistake
        # available in this pillar.
        B.add("SOCIAL_ABSENCE",
              f"Community and social sources are only "
              f"{soc['overall_social_share']:.1%} of all citations — the engines do not "
              f"listen to community voice in this category, so community work will not "
              f"move machine visibility",
              severity="low", metric_baseline=soc["overall_social_share"],
              evidence={"verdict": "negligible", "why": soc.get("why"),
                        "by_engine": {k: v for k, v in
                                      (soc.get("by_engine") or {}).items()
                                      if not k.startswith("_")}})
    elif soc:
        for it in (soc.get("by_intent") or []):
            if it["social_trigger_rate"] >= 0.20 and it.get("community_over_editorial"):
                B.add("SOCIAL_ABSENCE",
                      f"'{it['intent']}' queries cite a community source in "
                      f"{it['social_trigger_rate']:.0%} of runs — more often than an "
                      f"editorial one. On these queries consensus outweighs authority",
                      severity="high", intent=it["intent"],
                      metric_baseline=it["social_trigger_rate"],
                      evidence={k: it.get(k) for k in
                                ("runs", "runs_citing_social", "social_trigger_rate",
                                 "community_reviews", "editorial_authority",
                                 "ci_low", "ci_high")}
                               | {"source_type": "community_reviews"})
                break

        _th = soc.get("threads") or []
        for t in [x for x in _th if x["brand_present_rate"] <= 0.10][:5]:
            B.add("SOCIAL_ABSENCE",
                  f"Cited thread with no brand presence: \"{_short(t['title'], 60)}\" on "
                  f"{t['domain']} — cited {t['times_cited']}× across "
                  f"{t['n_engines']} engine(s)"
                  + (f", recommending {t['top_competitor']}"
                     if t.get("top_competitor") else ""),
                  severity="medium", domain=t["domain"], intent=t.get("top_intent"),
                  metric_baseline=t["brand_present_rate"],
                  evidence={k: t.get(k) for k in
                            ("url", "title", "times_cited", "n_engines", "n_queries",
                             "brand_present_rate", "brand_absence_index",
                             "top_competitor", "competitor_mentions", "bucket", "route",
                             "participatory", "source_type")})

        for t in [x for x in _th if x["negative_rate"] >= 0.15][:4]:
            B.add("SOCIAL_NEGATIVE",
                  f"Cited thread carrying negative framing "
                  f"({t['negative_rate']:.0%} of its citing runs): "
                  f"\"{_short(t['title'], 60)}\" on {t['domain']}",
                  severity="critical" if t["negative_rate"] >= 0.30 else "high",
                  domain=t["domain"], metric_baseline=t["negative_rate"],
                  evidence={k: t.get(k) for k in
                            ("url", "title", "times_cited", "negative_rate",
                             "hedged_rate", "brand_present_rate", "bucket", "route",
                             "participatory", "source_type")})

        _fr = soc.get("framing") or {}
        _d = _fr.get("delta_with_minus_without") or {}
        if _d.get("hedged", 0) >= 0.10 or _d.get("neg", 0) >= 0.05:
            B.add("SOCIAL_NEGATIVE",
                  (soc["framing"]["interpretation"] or
                   [f"Answers citing a community source are framed less favourably"])[0],
                  severity="high",
                  metric_baseline=max(_d.get("hedged", 0), _d.get("neg", 0)),
                  evidence={"delta": _d,
                            "with_social": _fr.get("with_social_citation"),
                            "without_social": _fr.get("without_social_citation"),
                            "caveat": _fr.get("negative_detection_caveat"),
                            "source_type": "community_reviews"})

    return _dedupe(B.gaps)

def _short(s, n=70):
    s = str(s or "")
    return s if len(s) <= n else s[:n - 1] + "…"

def _dedupe(gaps):
    """Collapse near-duplicates on (type, claim|dimension|domain|query|intent)."""
    seen, out = {}, []
    for g in gaps:
        key = (g["type"], fold(g.get("claim") or g.get("dimension") or
                               g.get("domain") or g.get("query") or
                               g.get("intent") or g["gap"])[:80])
        if key in seen:
            prev = seen[key]
            prev["engines"] = sorted(set(prev["engines"]) | set(g["engines"]))
            if SEV_PRIORITY.get(g["severity"], 0) > SEV_PRIORITY.get(prev["severity"], 0):
                prev["severity"] = g["severity"]
                prev["priority"] = g["priority"]
            continue
        seen[key] = g
        out.append(g)
    for i, g in enumerate(out, 1):        # renumber after dedupe
        g["gap_id"] = f"G{i:03d}"
    return sorted(out, key=lambda x: (-x["priority"], x["type"]))

# ---------------------------------------------------------------- public API
def build_gap_register(delta_scorecard, winnability=None, reason_rich=None,
                       target_brand="", metrics=None, hitlist=None):
    """Backward-compatible signature. `reason_rich` is unused (reason gaps now come
    from delta['reason_gap'], which is normalised per 100 tokens)."""
    return detect_gaps(delta_scorecard, winnability=winnability, metrics=metrics,
                       hitlist=hitlist, target_brand=target_brand)

def gap_summary(gaps):
    by_type = Counter(g["type"] for g in gaps)
    by_pillar = Counter(g["pillar"] for g in gaps)
    by_sev = Counter(g["severity"] for g in gaps)
    return {"total": len(gaps), "by_type": dict(by_type),
            "by_pillar": dict(by_pillar), "by_severity": dict(by_sev)}