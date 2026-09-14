"""Assemble validated findings for Optimise handoff."""
from datetime import datetime, timezone
from store import save_artifact
from experiment.exp_stats import analyze_experiment, verdict, variance_check

# Single source of truth — optimise_tab imports this instead of redefining it.
ACTIONABLE_VERDICTS = {"validated", "validated_mention",
                       "engine-specific", "engine-specific_mention"}

EVIDENCE_CLASS = {"clean": "PROVEN", "caveated": "PROVEN_WITH_CAVEAT",
                  "invalid": "INVALID", "legal": "BLOCKED",
                  "not_testable": "NOT_TESTABLE"}

NULL_VERDICTS = {"no_lift", "rejected", "already_winning", "lost"}

WIN_VERDICTS = {"validated", "validated_mention", "validated_rank",
                "engine-specific", "engine-specific_mention"}

CARRY = ("mode", "signal", "hypothesis", "mechanism", "pillar", "horizon",
         "prediction", "isolated_variable", "claims_used", "power", "pilot",
         "competitor_hash", "position_mode", "aggregation_meta", "resumed",
         "treatment_notes")

_LIMITS = {
    "specificity_untestable":
        "The approved claim set supplies no quantified values, so this tested STRUCTURE "
        "and CORROBORATION only — do not attribute the lift to specificity.",
    "specificity_not_applied":
        "The quantification manipulation did not land — treat this as a structure test.",
    "neg_control_dropped":
        "No negative control, but enrichment compressed the text (B ≤ A), so a "
        "'more text wins' explanation is already excluded.",
    "attribution_unreliable":
        "The E1 ladder was not strictly cumulative — treat element attribution as "
        "indicative, not causal.",
    "no_brand_in_arm":
        "An arm did not name the brand, so its baseline is near zero by construction. "
        "The lift measures brand presence, not the intended variable.",
    "structure_confound":
        "Structure differed between arms alongside the intended variable — the lift is "
        "attributable to the combination, not the variable alone.",
    "control_partly_structured":
        "The control carried some structure, so this tests more-vs-less enrichment "
        "rather than none-vs-full. The true effect of full enrichment is likely larger.",
    "treatment_incomplete":
        "The treatment arm lacked part of the GEO package — this is a lower bound on "
        "the full treatment effect.",
    "facts_lost_in_treatment":
        "Restructuring dropped some approved claims, so content varies alongside "
        "structure.",
    "length_asymmetry":
        "Arm length diverged from the competitor field, depressing absolute rates. "
        "The A-vs-B comparison remains internally valid.",
    "low_relevance":
        "The control was weakly on-topic for the target query, lowering the baseline.",
    "below_field_parity":
        "Both arms sat below the competitor field's structural standard, so absolute "
        "rates understate real-world performance.",
    "addition_oversized":
        "The treatment added more than the isolated claim, so the delta is not purely "
        "attributable to that claim.",
    "binary_at_ceiling":
            "The control arm was already at ceiling. A null here means the variable had no "
            "headroom, not that it has no effect under other conditions.",
}

def interpretation_limits(notes):
    return [_LIMITS[n] for n in (notes or []) if n in _LIMITS]

def classify_evidence(finding, check=None):
    """Evidence class travels with the finding into Optimise. A NULL is never PROVEN —
    it is DISPROVEN (well-powered, an actionable exclusion) or INCONCLUSIVE
    (underpowered, proves nothing)."""
    chk = check or {}
    sev = chk.get("severity", "clean")
    verdict = str(finding.get("verdict", ""))
    limits = interpretation_limits(chk.get("notes"))

    if sev in ("invalid", "legal", "not_testable"):
        cls = EVIDENCE_CLASS[sev]
    elif verdict in NULL_VERDICTS:
        if finding.get("underpowered"):
            cls = "INCONCLUSIVE"
            limits = limits + [
                f"No effect detected, but at ~{finding.get('runs_per_cell')} runs per arm "
                f"this does not rule one out. Not evidence either way."]
        else:
            cls = "DISPROVEN"
            extra = (" The control arm was already at ceiling, so this variable had no "
                     "room to act." if finding.get("binary_at_ceiling") else "")
            limits = limits + [
                f"No effect at {finding.get('runs_per_cell')} runs per arm per engine — "
                f"an actionable exclusion, not a missing result.{extra}"]
    elif verdict in WIN_VERDICTS:
        cls = EVIDENCE_CLASS.get(sev, "PROVEN_WITH_CAVEAT")
        if finding.get("underpowered") and cls == "PROVEN":
            cls = "PROVEN_WITH_CAVEAT"
        if verdict.startswith("engine-specific") and cls == "PROVEN":
            cls = "PROVEN_ENGINE_SPECIFIC"
        if verdict == "validated_rank" and cls == "PROVEN":
            cls = "PROVEN_ENGINE_SPECIFIC" if finding.get("rank_engines") else cls
    else:
        cls = "INCONCLUSIVE"

    return {"evidence_class": cls, "caveats": chk.get("caveats") or [],
            "design_notes": chk.get("notes") or [], "interpretation_limits": limits}


def assemble_findings(exp_results, target_brand, min_lift=5.0, raw_path=None):
    from experiment.exp_stats import analyze_progressive, element_attribution,analyze_rank
    findings = []
    for er in exp_results:
        if er.get("mode") == "progressive":
            rows = analyze_progressive(er, target_brand)
            attr = element_attribution(rows)
            best_arm = max(attr, key=attr.get) if attr else None
            findings.append({
                "design_id": er.get("design_id", er.get("gap_id")),
                "experiment_id": er.get("experiment_id", "E1"),
                "mode": "progressive",
                "verdict": ("validated" if attr and max(attr.values()) >= min_lift
                            else "no_lift"),
                "best_metric": "mention_rate",
                "pooled_lift_pp": round(max(attr.values()), 1) if attr else 0.0,
                "element_attribution": attr, "best_element": best_arm,
                "per_engine": rows, "variance_warnings": variance_check(er)})
            findings[-1].update({k: er.get(k) for k in CARRY if er.get(k) is not None})
            findings[-1].update(classify_evidence(findings[-1], er.get("check")))
            if raw_path:
                try:
                    rk = analyze_rank(raw_path, findings[-1]["design_id"],
                                      mode=findings[-1].get("mode"))
                    findings[-1]["rank_analysis"] = rk
                    sig = [r for r in rk if r.get("significant") and r["improvement"] > 0]
                    if sig and findings[-1]["verdict"] in ("no_lift", "rejected",
                                                           "already_winning"):
                        findings[-1]["verdict"] = "validated_rank"
                        findings[-1]["best_metric"] = "mean_rank"
                        findings[-1]["pooled_lift_pp"] = round(
                            sum(r["improvement"] for r in sig) / len(sig), 2)
                        findings[-1].setdefault("interpretation_limits", []).append(
                            "Binary mention/first-rec were at ceiling; the effect is measured "
                            "as an improvement in ordinal RANK among brands present.")
                except Exception as e:
                    findings[-1]["rank_error"] = str(e)[:120]
            continue
        analysis = analyze_experiment(er, target_brand)
        v = verdict(analysis, min_lift)
        best = v.get("best_metric") or "mention_rate"
        best_rows = [r for r in analysis if r["metric"] == best]
        pooled = round(sum(r["lift_pp"] for r in best_rows) / max(len(best_rows), 1), 1)
        findings.append({
            "design_id": er.get("design_id", er.get("gap_id")),
            "experiment_id": er.get("experiment_id"), "mode": "ab",
            "verdict": v["verdict"], "best_metric": best,
            "pooled_lift_pp": pooled, "per_engine": analysis,
            "variance_warnings": variance_check(er)})
        findings[-1].update(classify_evidence(findings[-1], er.get("check")))
        if raw_path:
            try:
                rk = analyze_rank(raw_path, findings[-1]["design_id"],
                                  mode=findings[-1].get("mode"))
                findings[-1]["rank_analysis"] = rk
                sig = [r for r in rk if r.get("significant") and r["improvement"] > 0]
                if sig and findings[-1]["verdict"] in ("no_lift", "rejected",
                                                       "already_winning"):
                    findings[-1]["verdict"] = "validated_rank"
                    findings[-1]["best_metric"] = "mean_rank"
                    findings[-1]["pooled_lift_pp"] = round(
                        sum(r["improvement"] for r in sig) / len(sig), 2)
                    findings[-1].setdefault("interpretation_limits", []).append(
                        "Binary mention/first-rec were at ceiling; the effect is measured "
                        "as an improvement in ordinal RANK among brands present.")
            except Exception as e:
                findings[-1]["rank_error"] = str(e)[:120]
    return findings

def save_experiment_results(parent_brand, target_brand, exp_results, findings,
                            designs, models_used, competitor_context=None,
                            competitor_docs=None):     # ← NEW params
    save_artifact(parent_brand, target_brand, "experiment_results", exp_results, models_used)
    save_artifact(parent_brand, target_brand, "validated_findings", findings, models_used)

    variant_log = []
    for d in designs:
        entry = {"design_id": d.get("design_id", d.get("gap_id")),
                 "experiment_id": d.get("experiment_id"),
                 "signal": d.get("signal"),
                 "question": d.get("question"),
                 "mode": d.get("mode", "ab"),
                 "quality_warning": d.get("variants", {}).get("quality_warning", "")}
        if "variants" in d:
            entry["variant_A"] = d["variants"]["variant_A"].get("text", "")
            entry["variant_B"] = d["variants"]["variant_B"].get("text", "")
        if "progressive_arms" in d:
            entry["arms"] = {k: a.get("text", "") for k, a in d["progressive_arms"].items()}
        variant_log.append(entry)

    # ── NEW: attach the FROZEN competitor content (what target competed against) ──
    bundle = {
        "target_brand": target_brand,
        "competitor_context_text": competitor_context or "",   # the injected text block
        "competitor_docs": competitor_docs or {},              # structured competitor docs
        "experiments": variant_log,                            # target variants per experiment
    }
    save_artifact(parent_brand, target_brand, "experiment_variants", bundle, models_used)

    validated = [f for f in findings if f["verdict"] in ACTIONABLE_VERDICTS]
    save_artifact(parent_brand, target_brand, "validated_findings_only", validated,
                  {"generated": datetime.now(timezone.utc).isoformat(),
                   "actionable_verdicts": sorted(ACTIONABLE_VERDICTS)})
    # keep the FULL set — null results are required evidence in Optimise
    save_artifact(parent_brand, target_brand, "validated_findings", findings, models_used)