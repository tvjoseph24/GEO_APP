"""Assemble validated findings for Optimise handoff."""
from datetime import datetime, timezone
from store import save_artifact
from experiment.exp_stats import analyze_experiment, verdict, variance_check

def assemble_findings(exp_results, target_brand, min_lift=5.0):
    findings = []
    for er in exp_results:
        analysis = analyze_experiment(er, target_brand)
        v = verdict(analysis, min_lift)
        # best lift for the winning metric
        best = v.get("best_metric") or "mention_rate"
        best_rows = [r for r in analysis if r["metric"] == best]
        pooled = round(sum(r["lift_pp"] for r in best_rows)/max(len(best_rows),1), 1)
        findings.append({
            "design_id": er.get("design_id", er.get("gap_id")),
            "experiment_id": er.get("experiment_id"),
            "verdict": v["verdict"], "best_metric": best,
            "pooled_lift_pp": pooled, "per_engine": analysis,
            "variance_warnings": variance_check(er)})
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

    validated = [f for f in findings if f["verdict"] in ("validated","engine-specific")]
    save_artifact(parent_brand, target_brand, "validated_findings_only", validated,
                  {"generated": datetime.now(timezone.utc).isoformat()})