"""Pilot-based power estimation.

The audit baseline (e.g. 4.4% on need queries) is the WRONG p0 for an experiment: the
runner injects the brand document with web OFF, so arm-A mention rates are typically
40-70%. MDE scales with sqrt(p0(1-p0)), so using the audit baseline understates the
required N by 2-4x. Run a cheap arm-A pilot, then gate on the real number.
"""
from __future__ import annotations

import math

from experiment.rag_runner import run_variant

def mde(p0, n, alpha=0.05, power=0.80):
    z_a, z_b = 1.959964, 0.841621
    p0 = min(max(float(p0 or 0.0), 0.01), 0.99)
    return (z_a + z_b) * math.sqrt(2 * p0 * (1 - p0) / max(int(n or 1), 1))

def n_for_effect(p0, effect_pp, alpha=0.05, power=0.80):
    z_a, z_b = 1.959964, 0.841621
    p0 = min(max(float(p0 or 0.0), 0.01), 0.99)
    e = max(float(effect_pp or 1) / 100.0, 0.005)
    return int(math.ceil(2 * p0 * (1 - p0) * ((z_a + z_b) / e) ** 2))

def run_pilot(gateway, control_text, comp_context, question, target_brand, competitors,
              *, engines, n=20, temperature=0.8, target_pos="balanced",
              alias_overrides=None, progress_cb=None, budget_check=None,
              design_id="_pilot", raw_writer=None, comp_mode="full",
              visual=None, asset_store=None, design=None):
    """Arm A only. Returns the REAL experimental baseline per engine plus pooled.

    W-20: pilot rows are now WRITTEN to the raw log. Previously run_pilot passed no
    raw_writer, so the baseline that gates all downstream spend was never persisted and
    could not be audited.
    """
    per_engine, hits, tot, fhits = {}, 0, 0, 0
    # A visual design's arm A is TEXT PLUS A CREATIVE. Piloting the text alone measures a
    # different arm — observed: a text-only pilot read 0% while the real arm A reached 26%
    # on one engine, because the creative was doing the work.
    if visual is None and asset_store is not None and design is not None:
        from experiment.visual import asset_for_arm
        visual = asset_for_arm(asset_store, design, "A")
    for e in engines:
        _vis = visual if "vision" in e.get("supports", []) else None
        r = run_variant(gateway, control_text, comp_context, question, target_brand,
                        competitors, n_runs=n, temperature=temperature,
                        target_pos=target_pos, model=e["model"], visual=_vis,
                        progress_cb=progress_cb, design_id=f"_pilot_{design_id}",
                        arm="A", alias_overrides=alias_overrides,
                        budget_check=budget_check, mode="pilot",
                        raw_writer=raw_writer, comp_mode=comp_mode,
                        video_supported="video" in e.get("supports", []))
        nv = max(r["n"], 1)
        per_engine[e["name"]] = {
            "n": r["n"], "errors": r["n_errors"],
            "mention_rate": round(r["mentioned"] / nv, 3),
            "first_rec_rate": round(r["is_first"] / nv, 3),
            "refusals": r["refusals"]}
        hits += r["mentioned"]; fhits += r["is_first"]; tot += nv

    p0_m = hits / max(tot, 1)
    p0_f = fhits / max(tot, 1)
    at_floor, at_ceiling = p0_m <= 0.02, p0_m >= 0.98
    return {"design_id": design_id, "question": question,
            "piloted_with_asset": visual is not None,
            "per_engine": per_engine, "n_total": tot,
            "p0_mention": round(p0_m, 3), "p0_first": round(p0_f, 3),
            "ceiling_risk": p0_m >= 0.90, "floor_risk": p0_m <= 0.05,
            "at_floor": at_floor, "at_ceiling": at_ceiling,
            # W-21: mde() clamps p0 to [0.01, 0.99], so BOTH extremes yield the same
            # ~3.9pp at N=100 — an identical number for "no evidence at all" and
            # "already at 100%". Suppress the table rather than print an artifact.
            "mde_table": ({} if (at_floor or at_ceiling) else
                          {str(N): {"mention_pp": round(mde(p0_m, N) * 100, 1),
                                    "first_pp": round(mde(p0_f, N) * 100, 1)}
                           for N in (30, 50, 100, 150, 200, 300)}),
            "mde_note": ("MDE is undefined at a 0% baseline — fix the control document "
                         "first" if at_floor else
                         "MDE is undefined at a 100% baseline — there is no headroom on "
                         "this metric" if at_ceiling else "")}

def run_pilots(gateway, designs, comp_context, target_brand, competitors, *, engines,
               n=20, temperature=0.8, target_pos="balanced", alias_overrides=None,
               progress_cb=None, budget_check=None, raw_writer=None, comp_mode="full",
               asset_store=None):
    """Pilot EVERY selected design's own control arm.

    W-20: one pilot's p0 was applied to all designs. Measured consequence: a design whose
    real arm-A mention rate was 34% inherited a pilot reading of 100%, was gated as
    'ceiling', and a 66pp effect was declared undetectable. The gate produced a false
    negative and the override was accidentally correct — the worst outcome, because it
    teaches distrust of the gate.
    """
    out, done = {}, 0
    total = max(len(designs), 1)
    for d in designs:
        did = d.get("design_id") or d.get("gap_id")
        ctrl = ((d.get("variants") or {}).get("variant_A") or {}).get("text", "")
        q = (d.get("question") or "").strip()
        _bench = d.get("source") == "bench"
        if not q or (not ctrl and not _bench):
            out[did] = {"design_id": did,
                        "error": "no question" if not q else "no arm-A text"}
            done += 1
            continue
        out[did] = run_pilot(gateway, ctrl, comp_context, q, target_brand, competitors,
                             engines=engines, n=n, temperature=temperature,
                             asset_store=asset_store, design=d,
                             target_pos=target_pos, alias_overrides=alias_overrides,
                             budget_check=budget_check, design_id=did,
                             raw_writer=raw_writer, comp_mode=comp_mode)
        out[did]["mode_note"] = (
            "This design runs PROGRESSIVE, which regenerates its arms at run time — the "
            "piloted arm A will be replaced and this baseline may not hold."
            if str(d.get("mode", "")).startswith("Prog") else
            "Bench design: arm A is the first CANDIDATE creative, not a control. This "
            "measures the baseline rate with a creative present, which is the right "
            "reference for sizing an N-way comparison."
            if _bench else None)
        done += 1
        if progress_cb:
            progress_cb(done / total)
    return out

def power_verdict(pilot, hypothesis, n_planned, metric_override=None):
    """Re-gate a hypothesis against ITS OWN pilot baseline, not the audit baseline."""
    if not pilot or pilot.get("error"):
        return {"verdict": "no_pilot", "metric": metric_override or "mention_rate",
                "note": f"No usable pilot for this design ({(pilot or {}).get('error')}). "
                        f"Power is unknown — the audit baseline is not a substitute, "
                        f"because experiments inject the brand's own document with web "
                        f"off.",
                "pilot_p0": None, "mde_pp": None, "wanted_pp": None, "n_required": None}

    pred = (hypothesis or {}).get("prediction") or {}
    metric = metric_override or pred.get("metric", "mention_rate")
    p0 = pilot["p0_first"] if metric in ("first_rec_rate", "top1_unconditional") \
        else pilot["p0_mention"]
    try:
        want = float(pred.get("lift_pp")
                     or (float(pred.get("expected_pp", 0))
                         - float(pred.get("baseline_pp", 0))))
    except Exception:
        want = 0.0

    achievable = mde(p0, n_planned) * 100
    need_n = n_for_effect(p0, want) if want > 0 else None
    audit_base = (pred.get("baseline_pp") or 0) / 100.0

    # A bake-off is EXPLORATORY: it predicts no specific lift, so the confirmatory power
    # question does not apply. What matters is whether the baseline leaves headroom, and
    # what effect is detectable given the arm count.
    if (hypothesis or {}).get("_bench") or (hypothesis or {}).get("source") == "bench":
        k = int((hypothesis or {}).get("n_visual_arms") or 2)
        from experiment.bench_stats import mde_nway
        det = mde_nway(p0, n_planned, k) * 100
        if p0 >= 0.95:
            v, note = "ceiling", (
                f"Arm-A baseline is {p0:.0%} on `{metric}` — no headroom for another arm "
                f"to beat it. Switch to top1_unconditional, or thin the competitor field.")
        elif p0 <= 0.02:
            v, note = "floor", (
                f"Arm-A baseline is {p0:.0%}: the brand is near-absent even with its own "
                f"creative and document in context. A comparison is still valid — you are "
                f"measuring whether ANY creative makes the brand visible — but check the "
                f"competitor field weight first. At 9 words against a full field, all arms "
                f"may floor.")
        else:
            v, note = "exploratory", (
                f"Bake-off: no predicted lift, so there is nothing to power FOR. At "
                f"N={n_planned} with {k} arm(s) an effect of ≥{det:.1f}pp is detectable "
                f"from a {p0:.0%} baseline. Smaller creative differences are real but "
                f"unmeasurable at this sample size.")
        return {"verdict": v, "note": note, "metric": metric,
                "pilot_p0": round(p0, 3), "mde_pp": round(det, 1),
                "wanted_pp": None, "n_required": None, "exploratory": True,
                "piloted_with_asset": (pilot or {}).get("piloted_with_asset")}

    if p0 >= 0.98:
        verdict = "ceiling"
        note = (f"Arm-A baseline on `{metric}` is {p0:.0%} — no headroom at all. "
                f"Switch the primary metric, or strengthen the competitor field.")
    elif p0 >= 0.90:
        verdict = "ceiling"
        note = (f"Arm-A baseline is {p0:.0%}, leaving {(1 - p0) * 100:.0f}pp of "
                f"headroom — less than the {achievable:.1f}pp detectable at "
                f"N={n_planned}. Switch metric or raise N substantially.")
    elif p0 <= 0.05:
        verdict = "floor"
        note = (f"Arm-A baseline is {p0:.0%} — the brand is near-absent even with its own "
                f"document injected. Check first that arm A actually names the brand.")
    elif want <= 0:
        verdict, note = "no_prediction", "Hypothesis predicts no lift — nothing to detect."
    elif want < achievable:
        verdict = "underpowered"
        note = (f"Predicted +{want:.1f}pp is below the {achievable:.1f}pp detectable at "
                f"N={n_planned} (pilot p0={p0:.0%}). Needs N≈{need_n}.")
    else:
        verdict = "powered"
        note = (f"+{want:.1f}pp is detectable at N={n_planned} "
                f"(MDE {achievable:.1f}pp, pilot p0={p0:.0%}).")

    out = {"verdict": verdict, "note": note, "metric": metric,
           "pilot_p0": round(p0, 3), "mde_pp": round(achievable, 1),
           "wanted_pp": round(want, 1), "n_required": need_n}
    # the audit baseline is measured in LIVE retrieval; the pilot injects the document
    if audit_base and abs(audit_base - p0) > 0.15:
        out["baseline_shift"] = (
            f"The audit baseline for this metric was {audit_base:.1%} but the pilot "
            f"measures {p0:.1%}. The hypothesis was sized against the audit figure, so "
            f"its prediction should be re-read against the pilot.")
    return out