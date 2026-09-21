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

# Weakest-wins ordering. When several downgrades apply, the most conservative class is
# correct — a caveated engine-specific win is not PROVEN_ENGINE_SPECIFIC (0.70 weight in
# Optimise), it is PROVEN_WITH_CAVEAT (0.60).
_CLASS_RANK = {"PROVEN": 0, "PROVEN_ENGINE_SPECIFIC": 1, "PROVEN_WITH_CAVEAT": 2,
               "DISPROVEN": 3, "INCONCLUSIVE": 4, "NOT_TESTABLE": 5, "BLOCKED": 6,
               "INVALID": 7}

def _weakest(*classes):
    cs = [c for c in classes if c]
    return max(cs, key=lambda c: _CLASS_RANK.get(c, 99)) if cs else "INCONCLUSIVE"

def _control_at_ceiling(finding, thresh=0.95):
    """Was the control arm already saturated?

    `binary_at_ceiling` is set only by reanalyse.py, so the live path never explained a
    null caused by having no headroom — and "the variable failed" reads very differently
    from "the variable had no room to act".
    """
    if finding.get("binary_at_ceiling") is not None:
        return bool(finding["binary_at_ceiling"])
    rows = finding.get("per_engine") or []
    ctrl = [r.get("A_rate") for r in rows
            if r.get("metric") == finding.get("best_metric")
            and isinstance(r.get("A_rate"), (int, float))]
    if not ctrl:
        ctrl = [r.get("rate") for r in rows
                if r.get("arm") == "A" and isinstance(r.get("rate"), (int, float))]
    return bool(ctrl) and min(ctrl) >= thresh

def _engine_split(finding, min_lift=5.0):
    """Did the effect hold across engines, or only on some?

    Detected from measured per-engine lift, not from the verdict string — a 'validated'
    verdict can still hide one engine at +66pp and another at +3pp.
    """
    per = finding.get("per_engine_lift_pp") or {}
    sig = finding.get("significant_engines") or []
    if len(per) < 2:
        return False, per, sig
    spread = max(per.values()) - min(per.values())
    partial_sig = bool(sig) and len(sig) < len(per)
    return (partial_sig or spread >= min_lift), per, sig

def classify_evidence(finding, check=None, min_lift=5.0):
    """Evidence class travels with the finding into Optimise, where it gates action
    weighting, blocks contradicting actions and can disprove a whole pillar mechanism.

    A NULL is never PROVEN — it is DISPROVEN (well-powered, an actionable exclusion),
    INCONCLUSIVE (underpowered, proves nothing) or INVALID (instrumentation failure).
    """
    chk = check or {}
    sev = chk.get("severity", "clean")
    verdict = str(finding.get("verdict", ""))
    limits = list(interpretation_limits(chk.get("notes")))
    vv = finding.get("visual_validity") or {}
    rpc = finding.get("runs_per_cell")
    rpc_txt = f"{rpc}" if rpc is not None else "an unknown number of"

    def _out(cls, extra_limits=()):
        return {"evidence_class": cls,
                "caveats": chk.get("caveats") or [],
                "design_notes": chk.get("notes") or [],
                "interpretation_limits": limits + list(extra_limits)}

    # ---- 1. instrumentation failure: not a result at all ----
    if verdict == "invalid_visual" or vv.get("verdict") == "INVALID":
        return _out("INVALID", [
            "Vision payload never reached the model — this is an instrumentation "
            "failure, not a null. Re-run after the vision self-test passes."])

    # ---- 2. design blocked before it could produce information ----
    if sev in ("invalid", "legal", "not_testable"):
        return _out(EVIDENCE_CLASS[sev])

    # ---- 3. no data ----
    if finding.get("n_cells") == 0 or (rpc is None and not finding.get("per_engine")):
        return _out("INCONCLUSIVE", [
            "No completed runs were found for this design, so nothing was measured. "
            "This is an absence of evidence, not a negative result."])

    # ---- 4. NULL ----
    if verdict in NULL_VERDICTS:
        # `underpowered` is None when power could not be established. None is falsy, so
        # the previous code fell through to DISPROVEN — classifying "we don't know" as
        # an actionable exclusion, which is the failure W-22 exists to prevent.
        up = finding.get("underpowered")
        if up is None:
            return _out("INCONCLUSIVE", [
                f"No effect detected, but runs-per-arm could not be established, so "
                f"statistical power is unknown. This does not rule an effect out."])
        if up:
            return _out("INCONCLUSIVE", [
                f"No effect detected, but at ~{rpc_txt} runs per arm per engine this "
                f"does not rule one out. Not evidence either way."])
        extra = (" The control arm was already at ceiling, so the variable had no room "
                 "to act — this rules out the variable UNDER THIS CONDITION, not in "
                 "general." if _control_at_ceiling(finding) else "")
        return _out("DISPROVEN", [
            f"No effect at {rpc_txt} runs per arm per engine — an actionable exclusion, "
            f"not a missing result.{extra}"])

    # ---- 5. WIN ----
    if verdict in WIN_VERDICTS:
        cls = EVIDENCE_CLASS.get(sev, "PROVEN_WITH_CAVEAT")
        extra = []

        if finding.get("underpowered"):
            cls = _weakest(cls, "PROVEN_WITH_CAVEAT")
            extra.append(f"Measured at only ~{rpc_txt} runs per arm per engine — the "
                         f"direction is credible, the magnitude is not. Do not quote "
                         f"the effect size without this caveat.")

        # engine-specificity from measured data, not the verdict string
        split, per, sig = _engine_split(finding, min_lift)
        if split or verdict.startswith("engine-specific"):
            cls = _weakest(cls, "PROVEN_ENGINE_SPECIFIC")
            if per:
                extra.append(
                    "Effect is ENGINE-SPECIFIC: "
                    + " · ".join(f"{e}: {l:+.1f}pp" for e, l in sorted(per.items()))
                    + (f" — significant on {', '.join(sig)} only." if sig else ".")
                    + " The pooled figure describes neither engine; belongs in an "
                      "engine overlay, not the overall strategy.")

        if verdict == "validated_rank":
            cls = _weakest(cls, "PROVEN_ENGINE_SPECIFIC"
                           if finding.get("rank_engines") else cls)
            extra.append(
                "Measured on ordinal RANK rather than a binary metric. Rank is computed "
                "only on runs where the brand appeared, so it is conditional on presence "
                "and not directly comparable across arms with different mention rates.")

        if vv.get("verdict") == "PARTIAL":
            cls = _weakest(cls, "PROVEN_WITH_CAVEAT")
            extra.append(f"Vision payload arrived on only "
                         f"{vv.get('delivery_rate', 0):.0%} of image-bearing runs, so "
                         f"some arms effectively ran text-only.")

        if not sig and (finding.get("per_engine_lift_pp") or {}):
            extra.append("Verdict is a win but no engine reached significance on the "
                         "primary metric — verify before quoting.")
            cls = _weakest(cls, "PROVEN_WITH_CAVEAT")

        return _out(cls, extra)

    # ---- 6. anything else ----
    return _out("INCONCLUSIVE", [
        f"Verdict '{verdict}' is not a recognised win or null, so no evidence class can "
        f"be assigned."])

def _cell_stats(exp_result):
    """Runs per arm per engine — the basis of the power verdict.

    W-22: assemble_findings never set runs_per_cell or underpowered, only
    reanalyse.py did. Consequence: interpretation limits rendered "No effect at None
    runs per arm", and EVERY live null was classified DISPROVEN with no power check,
    making INCONCLUSIVE unreachable. DISPROVEN then generates client-facing exclusions
    and can disprove a whole pillar mechanism in Optimise.
    """
    ns = [a.get("n", 0) for ev in (exp_result.get("engines") or {}).values()
          for k, a in (ev or {}).items() if isinstance(a, dict) and "n" in a]
    if not ns:
        return {"runs_per_cell": None, "n_cells": 0, "n_rows": 0, "underpowered": None}
    return {"runs_per_cell": round(sum(ns) / len(ns), 1),
            "min_runs_per_cell": min(ns), "n_cells": len(ns), "n_rows": sum(ns),
            "underpowered": (sum(ns) / len(ns)) < MIN_PER_CELL}

MIN_PER_CELL = 30

def _visual_validity(exp_result):
    """Did the image payload actually reach the model on every visual arm?"""
    vis = [a["visual"] for ev in (exp_result.get("engines") or {}).values()
           for a in (ev or {}).values()
           if isinstance(a, dict) and isinstance(a.get("visual"), dict)]
    if not vis:
        return None
    runs = sum(v["runs_with_image"] for v in vis)
    tok_runs = sum(v["runs_with_image_tokens"] for v in vis)
    toks = sum(v["image_tokens_total"] for v in vis)
    rate = tok_runs / max(runs, 1)
    out = {"runs_with_image": runs, "runs_with_image_tokens": tok_runs,
           "image_tokens_total": toks, "delivery_rate": round(rate, 3),
           "mean_image_tokens": round(toks / max(runs, 1), 1),
           "payload_delivered": tok_runs > 0}
    if tok_runs == 0:
        out["verdict"] = "INVALID"
        out["note"] = (f"{runs} run(s) attached an image and NONE reported image "
                       f"tokens. The payload never reached the model, so every arm ran "
                       f"text-only. This result is an artifact, not a null.")
    elif rate < 0.9:
        out["verdict"] = "PARTIAL"
        out["note"] = (f"Only {rate:.0%} of image-bearing runs reported image tokens — "
                       f"the payload intermittently failed to arrive.")
    else:
        out["verdict"] = "OK"
        out["note"] = (f"Image payload confirmed on {rate:.0%} of runs "
                       f"({out['mean_image_tokens']} mean image tokens).")
    return out

MIN_PER_CELL = 30

def _promote_rank(f, raw_path):
    """Rank promotion, shared by both branches.

    Rank is only consulted when the binary verdict is already null — so a marginal
    binary 'win' prevents rank from ever being examined, despite rank being the correct
    test once the brand's own document is in context (W-89, deferred).

    Also sets lift_unit and headline_effect, which assemble_findings never did — only
    reanalyse.py. Without them a rank improvement renders as "0.08pp" when the same
    result expressed as top-1 share is the number a reader can act on.
    """
    if not raw_path:
        return
    from experiment.exp_stats import analyze_rank
    try:
        rk = analyze_rank(raw_path, f["design_id"], mode=f.get("mode"))
    except Exception as e:
        f["rank_error"] = str(e)[:120]
        return
    f["rank_analysis"] = rk
    sig = [r for r in rk if r.get("significant") and (r.get("improvement") or 0) > 0]
    if not sig or f["verdict"] not in ("no_lift", "rejected", "already_winning"):
        return
    f["verdict"] = "validated_rank"
    f["best_metric"] = "mean_rank"
    f["pooled_lift_pp"] = round(sum(r["improvement"] for r in sig) / len(sig), 2)
    f["lift_unit"] = "rank_positions"
    f["rank_engines"] = sorted({r["engine"] for r in sig})
    top1 = round(sum((r.get("B_top1_pct") or 0) - (r.get("A_top1_pct") or 0)
                     for r in sig) / len(sig), 1)
    f["rank_top1_lift_pp"] = top1
    f["headline_effect"] = {
        "value": top1, "unit": "pp", "metric": "top-1 recommendation rate",
        "detail": (f"top-1 recommendation share rose {top1:+.0f}pp; mean rank improved "
                   f"{f['pooled_lift_pp']} positions"),
        "engines": f["rank_engines"]}
    f.setdefault("interpretation_limits", []).append(
        "Binary mention/first-rec were at ceiling; the effect is measured as an "
        "improvement in ordinal RANK among brands present. Note that rank is computed "
        "only on runs where the brand appeared, so it is conditional on presence.")

def _apply_visual(f, er):
    """Attach vision-payload validity and invalidate the finding if it never arrived."""
    vv = _visual_validity(er)
    if not vv:
        return
    f["visual_validity"] = vv
    if vv["verdict"] == "INVALID":
        f["verdict"] = "invalid_visual"
        f.setdefault("interpretation_limits", []).append(vv["note"])
    elif vv["verdict"] == "PARTIAL":
        f.setdefault("interpretation_limits", []).append(vv["note"])
def assemble_findings(exp_results, target_brand, min_lift=5.0, raw_path=None):
    """Assemble findings. Order matters:
         stats -> visual validity -> rank promotion -> classify_evidence
       classify_evidence must run LAST, or a promoted verdict carries the class of the
       verdict it replaced (W-22: produced "VALIDATED_RANK · DISPROVEN").
    """
    from experiment.exp_stats import (analyze_experiment, analyze_progressive,
                                      element_attribution, variance_check, verdict)
    findings = []
    for er in exp_results:

        # ================= PROGRESSIVE (E1 ladder) =================
        if er.get("mode") == "progressive":
            rows = analyze_progressive(er, target_brand)
            attr = element_attribution(rows)
            best_arm = max(attr, key=attr.get) if attr else None

            # W-23: element_attribution averages marginals across engines, so a real
            # single-engine gradient is diluted below min_lift and reported as no_lift.
            per_eng_attr = {}
            for r in rows:
                if r.get("metric") == "mention_rate" and r.get("arm") != "A":
                    per_eng_attr.setdefault(r["engine"], {})[r["arm"]] = r.get("marginal_pp")
            best_per_engine = {e: (max(a, key=lambda k: a[k] or 0) if a else None)
                               for e, a in per_eng_attr.items()}
            engine_best_lift = {e: round(max((v or 0) for v in a.values()), 1)
                                for e, a in per_eng_attr.items() if a}

            pooled_best = round(max(attr.values()), 1) if attr else 0.0
            any_engine_best = max(engine_best_lift.values(), default=0.0)
            v = ("validated" if pooled_best >= min_lift else
                 "engine-specific" if any_engine_best >= min_lift else "no_lift")

            f = {
                "design_id": er.get("design_id", er.get("gap_id")),
                "experiment_id": er.get("experiment_id", "E1"),
                "mode": "progressive",
                "verdict": v,
                "best_metric": "mention_rate",
                "pooled_lift_pp": pooled_best,
                "lift_unit": "pp",
                "element_attribution": attr,
                "element_attribution_by_engine": per_eng_attr,
                "best_element": best_arm,
                "best_element_by_engine": best_per_engine,
                "per_engine_lift_pp": engine_best_lift,
                "significant_engines": sorted(
                    {r["engine"] for r in rows if r.get("significant")}),
                "headline_effect": ({
                    "value": any_engine_best or pooled_best, "unit": "pp",
                    "metric": "mention_rate",
                    "engines": sorted(e for e, l in engine_best_lift.items()
                                      if l >= min_lift),
                    "detail": " · ".join(f"{e}: {l:+.1f}pp (best arm "
                                         f"{best_per_engine.get(e)})"
                                         for e, l in sorted(engine_best_lift.items())),
                    "pooled_is_misleading": (
                        len(engine_best_lift) > 1
                        and max(engine_best_lift.values())
                        - min(engine_best_lift.values()) >= min_lift)}
                    if engine_best_lift else None),
                "per_engine": rows,
                "variance_warnings": variance_check(er),
            }
            findings.append(f)
            f.update({k: er.get(k) for k in CARRY if er.get(k) is not None})
            f.update(_cell_stats(er))
            _apply_visual(f, er)
            _promote_rank(f, raw_path)
            f.update(classify_evidence(f, er.get("check")))
            continue

        # ================= A/B =================
        analysis = analyze_experiment(er, target_brand)
        v = verdict(analysis, min_lift)
        best = v.get("best_metric") or "mention_rate"
        best_rows = [r for r in analysis if r.get("metric") == best]
        sig_rows = [r for r in best_rows if r.get("significant")]
        pooled = round(sum(r["lift_pp"] for r in best_rows) / max(len(best_rows), 1), 1)

        # W-23: pooling a 66pp effect with a 3pp null yields 34.5pp, which describes
        # neither engine — and that pooled figure is what the report and strategy quote.
        per_engine_lift = {}
        for r in best_rows:
            e = r.get("engine")
            if e is not None:
                per_engine_lift[e] = max(per_engine_lift.get(e, r["lift_pp"]),
                                         r["lift_pp"])
        sig_engines = sorted({r["engine"] for r in sig_rows})

        f = {
            "design_id": er.get("design_id", er.get("gap_id")),
            "experiment_id": er.get("experiment_id"), "mode": "ab",
            "verdict": v["verdict"], "best_metric": best,
            "pooled_lift_pp": pooled,
            "lift_unit": "pp",
            "per_engine_lift_pp": per_engine_lift,
            "significant_engines": sig_engines,
            "headline_effect": ({
                "value": max((r["lift_pp"] for r in sig_rows), default=pooled),
                "unit": "pp", "metric": best,
                "engines": sig_engines,
                "detail": (" · ".join(f"{e}: {l:+.1f}pp"
                                      for e, l in sorted(per_engine_lift.items()))
                           + (f" — significant on {', '.join(sig_engines)}"
                              if sig_engines else " — no engine significant")),
                "pooled_is_misleading": (
                    len(per_engine_lift) > 1
                    and max(per_engine_lift.values())
                    - min(per_engine_lift.values()) >= min_lift)}
                if per_engine_lift else None),
            "per_engine": analysis,
            "variance_warnings": variance_check(er),
        }
        findings.append(f)
        # CARRY was applied ONLY in the progressive branch, so every A/B finding lost
        # mechanism, pillar, horizon, prediction, isolated_variable, claims_used, pilot
        # and power. Observable in the Range Rover output: C01 (A/B) displayed its
        # template "E4" where C02 (progressive) displayed the real isolated variable.
        f.update({k: er.get(k) for k in CARRY if er.get(k) is not None})
        f.update(_cell_stats(er))
        _apply_visual(f, er)
        _promote_rank(f, raw_path)
        f.update(classify_evidence(f, er.get("check")))

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