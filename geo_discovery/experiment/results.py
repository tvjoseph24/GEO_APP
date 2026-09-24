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
         "treatment_notes", "primary_metric", "mode_override", "visual_check",
         # creative bench
         "source", "text_presence", "attribution", "visual_arms",
         "attribution_warning", "legibility", "payload_estimate", "n_visual_arms",
         # F: without this the finding cannot be traced to the creative it tested —
         # competitor_hash is carried, the thing actually under test was not.
         "asset_digests")

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
    INCONCLUSIVE (underpowered or incomplete, proves nothing) or INVALID (instrumentation
    failure). The distinction is load-bearing: DISPROVEN becomes a client-facing "we will
    not spend on this", so anything short of a real comparison at real power must not
    reach it.
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

    # ---- 3. no data at all ----
    if finding.get("n_cells") == 0 or (rpc is None and not finding.get("per_engine")):
        return _out("INCONCLUSIVE", [
            "No completed runs were found for this design, so nothing was measured. "
            "This is an absence of evidence, not a negative result."])

    # ---- 4. data, but no COMPARISON was made ----
    # Observed: an interrupted five-arm ladder left ONE cell of 76 runs. No engine had two
    # arms, so the analyser returned no rows and the verdict fell through to `no_lift` —
    # then 76 >= MIN_PER_CELL made `underpowered` False, and this function issued
    # DISPROVEN. A client-facing exclusion built on a single arm with nothing to compare
    # it to. Same shape as the video-token and empty-`live` bugs: a confident answer
    # returned because the guard was handed no data.
    if verdict == "insufficient_data":
        return _out("INCONCLUSIVE", [
            f"No comparison was made — no engine recorded two or more arms "
            f"({finding.get('arms_per_engine')}). The run is incomplete; resume it. This "
            f"is not a null and must not be quoted as an exclusion."])

    # ---- 5. NULL ----
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

    # ---- 6. WIN ----
    if verdict in WIN_VERDICTS:
        cls = EVIDENCE_CLASS.get(sev, "PROVEN_WITH_CAVEAT")
        extra = []

        if finding.get("underpowered"):
            cls = _weakest(cls, "PROVEN_WITH_CAVEAT")
            extra.append(f"Measured at only ~{rpc_txt} runs per arm per engine — the "
                         f"direction is credible, the magnitude is not. Do not quote "
                         f"the effect size without this caveat.")
        if finding.get("mode_override"):
            cls = _weakest(cls, "PROVEN_WITH_CAVEAT")
            extra.append("TEMPLATE OVERRIDE: "
                         + finding["mode_override"]["note"]
                         + " The isolated_variable recorded on this finding describes the "
                           "hypothesis, not what ran.")

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
                         f"{vv.get('measurable_delivery_rate', vv.get('delivery_rate', 0)):.0%} "
                         f"of verifiable image-bearing runs, so some arms effectively ran "
                         f"text-only.")

        if not sig and (finding.get("per_engine_lift_pp") or {}):
            extra.append("Verdict is a win but no engine reached significance on the "
                         "primary metric — verify before quoting.")
            cls = _weakest(cls, "PROVEN_WITH_CAVEAT")

        return _out(cls, extra)

    # ---- 7. anything else ----
    return _out("INCONCLUSIVE", [
        f"Verdict '{verdict}' is not a recognised win or null, so no evidence class can "
        f"be assigned."])

def _comparable(exp_result):
    """Did ANY engine have two or more arms? Without that no comparison was made.

    Observed: an interrupted progressive run left ONE cell of 76 runs. `analyze_progressive`
    skipped the engine, returned no rows, and the verdict fell through to `no_lift` — then
    76 >= MIN_PER_CELL so `underpowered=False`, and classify_evidence issued DISPROVEN: a
    client-facing exclusion built on a single arm with nothing to compare it to.
    """
    for ev in (exp_result.get("engines") or {}).values():
        arms = [k for k, v in (ev or {}).items() if isinstance(v, dict) and v.get("n")]
        if len(arms) >= 2:
            return True
    return False
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
    """Did the media payload reach the model?

    Two things had to be separated. A provider that does not PUBLISH a media-token
    breakdown (OpenAI folds image tokens into prompt_tokens) reports zero on every vision
    call, so a token-only test marks every gpt-4o visual finding as caveated regardless of
    what happened — observed live as "50% delivery" on a run where the gemini half was at
    100% and the gpt-4o half was unmeasurable.

    Where token evidence is unavailable, BEHAVIOURAL evidence substitutes: in isolation
    mode the arms carry byte-identical text, so a significant difference between them can
    only have come from the image.

    Messages name the ENGINE, not the provider. "litellm publishes no breakdown" names a
    gateway; an operator needs to know it was gpt-4o.
    """
    vis, by_prov = [], {}
    for ev in (exp_result.get("engines") or {}).values():
        for a in (ev or {}).values():
            if isinstance(a, dict) and isinstance(a.get("visual"), dict):
                v = a["visual"]
                vis.append(v)
                p = v.get("provider", "unknown")
                d = by_prov.setdefault(p, {"runs": 0, "tok_runs": 0, "toks": 0,
                                           "reports": v.get("reports_media_tokens", True),
                                           "models": set()})
                d["runs"] += v.get("runs_with_image", 0)
                d["tok_runs"] += v.get("runs_with_image_tokens", 0)
                d["toks"] += v.get("image_tokens_total", 0)
                if v.get("model"):
                    d["models"].add(v["model"])
    if not vis:
        return None

    runs = sum(v.get("runs_with_image", 0) for v in vis)
    vid_skipped = sum(v.get("video_runs_skipped", 0) for v in vis)

    # Only providers that publish a breakdown can be assessed on tokens at all.
    measurable = {p: d for p, d in by_prov.items() if d["reports"]}
    silent = {p: d for p, d in by_prov.items() if not d["reports"]}
    m_runs = sum(d["runs"] for d in measurable.values())
    m_tok = sum(d["tok_runs"] for d in measurable.values())
    m_rate = m_tok / max(m_runs, 1)
    silent_engines = sorted({m for d in silent.values() for m in d["models"]})
    silent_label = ", ".join(silent_engines or sorted(silent))

    out = {"runs_with_image": runs,
           "runs_with_image_tokens": sum(d["tok_runs"] for d in by_prov.values()),
           "image_tokens_total": sum(d["toks"] for d in by_prov.values()),
           "measurable_runs": m_runs,
           "measurable_delivery_rate": round(m_rate, 3),
           "providers_not_reporting_tokens": sorted(silent),
           "engines_not_reporting_tokens": silent_engines,
           "runs_on_silent_providers": sum(d["runs"] for d in silent.values()),
           "video_runs_skipped": vid_skipped,
           # Divided by MEASURABLE runs, not all runs. Dividing by all runs reported 903
           # mean tokens on a run whose delivering half averaged 1,806.
           "mean_image_tokens": round(sum(d["toks"] for d in measurable.values())
                                      / max(m_runs, 1), 1),
           "payload_delivered": m_tok > 0}

    # ---- behavioural proof, where tokens cannot speak ----
    behavioural = None
    if silent and exp_result.get("attribution") == "isolation":
        for eng, ev in (exp_result.get("engines") or {}).items():
            rates = [(a.get("mentioned", 0) / max(a.get("n", 1), 1))
                     for a in (ev or {}).values()
                     if isinstance(a, dict) and a.get("n")]
            if len(rates) >= 2 and (max(rates) - min(rates)) >= 0.15:
                behavioural = (
                    f"{eng}: arms differ by {(max(rates) - min(rates)) * 100:.0f}pp on "
                    f"mention with BYTE-IDENTICAL text. In isolation mode the image is "
                    f"the only variable, so the payload demonstrably arrived.")
                break
    out["behavioural_evidence"] = behavioural

    if runs == 0 and vid_skipped:
        out["verdict"] = "INVALID"
        out["note"] = (f"{vid_skipped} run(s) carried a VIDEO payload no selected engine "
                       f"could accept, so every arm ran text-only. Select a video-capable "
                       f"engine — this is not a null.")
    elif m_runs == 0 and behavioural:
        out["verdict"] = "OK"
        out["note"] = (f"{silent_label} publishes no media-token breakdown, so delivery "
                       f"cannot be verified from tokens. It is verified BEHAVIOURALLY "
                       f"instead — {behavioural}")
    elif m_runs == 0:
        out["verdict"] = "PARTIAL"
        out["note"] = (f"{silent_label} publishes no media-token breakdown, and the arms "
                       f"did not differ enough to infer delivery behaviourally. Add a "
                       f"Vertex engine to verify the payload directly.")
    elif m_tok == 0:
        out["verdict"] = "INVALID"
        out["note"] = (f"{m_runs} run(s) on a token-reporting provider attached media and "
                       f"NONE reported media tokens. The payload never reached the model, "
                       f"so every arm ran text-only. This is an artifact, not a null.")
    elif m_rate < 0.9:
        out["verdict"] = "PARTIAL"
        out["note"] = (f"Only {m_rate:.0%} of verifiable media-bearing runs reported media "
                       f"tokens — the payload intermittently failed to arrive.")
    else:
        out["verdict"] = "OK"
        note = (f"Media payload confirmed on {m_rate:.0%} of the {m_runs} verifiable "
                f"run(s) ({out['mean_image_tokens']} mean media tokens).")
        if silent:
            note += (f" A further {out['runs_on_silent_providers']} run(s) ran on "
                     f"{silent_label}, which publishes no media-token breakdown"
                     + (f"; delivery there is confirmed behaviourally — {behavioural}"
                        if behavioural else "; delivery there is unverified"))
        if vid_skipped:
            note += f" {vid_skipped} video run(s) skipped on a non-video engine."
        out["note"] = note
    return out

MIN_PER_CELL = 30

def _promote_rank(f, raw_path):
    """Rank promotion, shared by both branches.

    The headline is now UNCONDITIONAL top-1 lift. Mean-rank movement reads as trivially
    small (0.08 positions) while the same result expressed as top-1 share is the number a
    reader can act on — and when presence rates differ between arms, conditional mean
    rank is not comparable at all (W-24).
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
    meta = (rk[0].get("_meta") if rk else None) or {}
    if meta.get("engines_excluded"):
        f.setdefault("interpretation_limits", []).append(
            "Rank analysis excluded "
            + "; ".join(f"{x.get('engine')}/{x.get('arm', 'A')}: {x.get('why')}"
                        for x in meta["engines_excluded"][:3])
            + ". Rank is only defined where the brand appeared.")

    sig = [r for r in rk if r.get("significant")
           and (r.get("top1_uncond_lift_pp") or 0) > 0]
    if not sig or f["verdict"] not in ("no_lift", "rejected", "already_winning"):
        return

    top1 = round(sum(r["top1_uncond_lift_pp"] for r in sig) / len(sig), 1)
    mean_rank_gain = round(sum(r["improvement"] for r in sig) / len(sig), 2)
    f["verdict"] = "validated_rank"
    f["best_metric"] = "top1_unconditional"
    f["pooled_lift_pp"] = top1
    f["lift_unit"] = "pp"
    f["rank_engines"] = sorted({r["engine"] for r in sig})
    f["mean_rank_improvement"] = mean_rank_gain
    f["per_engine_lift_pp"] = {r["engine"]: r["top1_uncond_lift_pp"] for r in sig}
    f["significant_engines"] = f["rank_engines"]
    f["headline_effect"] = {
        "value": top1, "unit": "pp",
        "metric": "top-1 recommendation rate (all runs)",
        "detail": (f"top-1 recommendation share rose {top1:+.1f}pp across all runs; "
                   f"mean rank among brands present improved {mean_rank_gain} positions"),
        "engines": f["rank_engines"],
        "pooled_is_misleading": len(set(
            round(r["top1_uncond_lift_pp"], 0) for r in sig)) > 1}
    f.setdefault("interpretation_limits", []).append(
        "The binary metrics were at ceiling, so the effect is measured as an improvement "
        "in top-1 recommendation share over ALL runs. Mean rank among brands present is "
        "reported alongside as a diagnostic, not as the effect.")
    warn = [r["conditioning_warning"] for r in sig if r.get("conditioning_warning")]
    if warn:
        f.setdefault("interpretation_limits", []).append(warn[0])

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

        # ---- no comparison possible: not a null, an incomplete run ----
        if not _comparable(er):
            _cells = {e: sorted(k for k, v in (ev or {}).items()
                                if isinstance(v, dict) and v.get("n"))
                      for e, ev in (er.get("engines") or {}).items()}
            f = {"design_id": er.get("design_id", er.get("gap_id")),
                 "experiment_id": er.get("experiment_id"), "mode": er.get("mode"),
                 "verdict": "insufficient_data", "best_metric": None,
                 "pooled_lift_pp": None, "lift_unit": "pp", "per_engine": [],
                 "arms_per_engine": _cells,
                 "variance_warnings": [
                     f"No engine recorded two or more arms ({_cells}), so no comparison "
                     f"was made. The run was interrupted or only partially executed — "
                     f"resume it rather than reading this as a result."]}
            findings.append(f)
            f.update({k: er.get(k) for k in CARRY if er.get(k) is not None})
            f.update(_cell_stats(er))
            f["underpowered"] = True  # never DISPROVEN on an incomplete run
            f.update(classify_evidence(f, er.get("check")))
            continue

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
        # ================= CREATIVE BENCH (N-way) =================
        if er.get("mode") == "creative":
            from experiment.bench_stats import analyze_bakeoff, bakeoff_summary
            rows = analyze_bakeoff(er, target_brand)
            prefer = ("top1_unconditional"
                      if er.get("text_presence") == "full" else "mention_rate")
            summ = bakeoff_summary(rows, prefer=prefer)

            _V = {"winner": "validated", "engine_specific": "engine-specific",
                  "leader_not_separated": "no_lift", "equivalent": "no_lift",
                  "no_decision": "no_lift", "no_variance": "no_lift"}
            f = {
                "design_id": er.get("design_id", er.get("gap_id")),
                "experiment_id": er.get("experiment_id", "VBENCH"),
                "mode": "creative",
                "verdict": _V.get(summ.get("verdict"), "no_lift"),
                "best_metric": summ.get("metric") or prefer,
                "pooled_lift_pp": summ.get("mean_gap_pp") or 0.0,
                "lift_unit": "pp",
                "bakeoff": rows,
                "bakeoff_summary": summ,
                "winning_arm": summ.get("winner"),
                "winners_by_engine": summ.get("winners_by_engine"),
                "per_engine_lift_pp": {r["engine"]: r["gap_pp"] for r in rows
                                       if r["metric"] == (summ.get("metric") or prefer)},
                "significant_engines": sorted({r["engine"] for r in rows
                                               if r.get("winner_distinguishable")}),
                "headline_effect": ({
                                        "value": summ.get("mean_gap_pp"), "unit": "pp",
                                        "metric": summ.get("metric"),
                                        "engines": sorted(summ.get("winners_by_engine") or {}),
                                        "detail": summ.get("decision"),
                                        "pooled_is_misleading": not summ.get("engines_agree", True)}
                                    if summ.get("verdict") in ("winner", "engine_specific") else None),
                "per_engine": [{"engine": r["engine"], "arm": x["arm"],
                                "metric": r["metric"], "rate": x["rate"],
                                "n": x["n"], "rank": x["rank"],
                                "ci": [x["ci_lo"], x["ci_hi"]],
                                "significant": r.get("winner_distinguishable")
                                               and x["rank"] == 1}
                               for r in rows for x in r["ranking"]],
                "variance_warnings": variance_check(er),
            }
            # An N-way test with an omnibus gate still promotes a noise winner sometimes.
            if summ.get("verdict") == "leader_not_separated":
                f.setdefault("interpretation_limits", []).append(
                    "The omnibus test found a difference among the arms, but the leader is "
                    "not distinguishable from the runner-up. The weakest arms can be "
                    "dropped; choosing between the top two needs a confirmation run of "
                    "winner-versus-second at higher N.")
            if not summ.get("engines_agree", True):
                f.setdefault("interpretation_limits", []).append(
                    f"Engines disagree on the winning creative "
                    f"({summ.get('winners_by_engine')}). There is no single best "
                    f"creative — this belongs in an engine overlay.")
            if er.get("attribution") == "bundle":
                f.setdefault("interpretation_limits", []).append(
                    "BUNDLE mode: the copy varied with the creative, so this finding "
                    "describes the shipped package. No part of the effect is attributable "
                    "to the creative alone.")
            if er.get("text_presence") == "none" and all(
                    x["rate"] <= 0.02 for r in rows for x in r["ranking"]):
                f.setdefault("interpretation_limits", []).append(
                    "Every arm floored with no accompanying text. That is a LEGIBILITY "
                    "finding — the engine cannot identify the brand from the creative — "
                    "not a comparison between creatives.")
            findings.append(f)
            f.update({k: er.get(k) for k in CARRY if er.get(k) is not None})
            f.update(_cell_stats(er))
            _apply_visual(f, er)
            _promote_rank(f, raw_path)
            f.update(classify_evidence(f, er.get("check")))
            continue

        # ================= A/B =================
        analysis = analyze_experiment(er, target_brand)
        v = verdict(analysis, min_lift, primary_metric=er.get("primary_metric"))
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