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
              alias_overrides=None, progress_cb=None,budget_check=None):
    """Arm A only. Returns the REAL experimental baseline per engine + pooled."""
    per_engine, hits, tot, fhits = {}, 0, 0, 0
    for e in engines:
        r = run_variant(gateway, control_text, comp_context, question, target_brand,
                        competitors, n_runs=n, temperature=temperature,
                        target_pos=target_pos, model=e["model"],
                        progress_cb=progress_cb, design_id="_pilot", arm="A",
                        alias_overrides=alias_overrides,budget_check=budget_check)
        nv = max(r["n"], 1)
        per_engine[e["name"]] = {
            "n": r["n"], "errors": r["n_errors"],
            "mention_rate": round(r["mentioned"] / nv, 3),
            "first_rec_rate": round(r["is_first"] / nv, 3),
            "refusals": r["refusals"]}
        hits += r["mentioned"]; fhits += r["is_first"]; tot += nv
    p0_m = hits / max(tot, 1)
    p0_f = fhits / max(tot, 1)
    at_floor = p0_m <= 0.02
    return {"per_engine": per_engine, "n_total": tot,
            "p0_mention": round(p0_m, 3), "p0_first": round(p0_f, 3),
            "ceiling_risk": p0_m >= 0.90, "floor_risk": p0_m <= 0.05,
            "at_floor": at_floor,
            "mde_table": ({} if at_floor else
                          {str(N): {"mention_pp": round(mde(p0_m, N) * 100, 1),
                                    "first_pp": round(mde(p0_f, N) * 100, 1)}
                           for N in (30, 50, 100, 150, 200, 300)}),
            "mde_note": ("MDE undefined at a 0% baseline — fix the control document first"
                         if at_floor else "")}

def power_verdict(pilot, hypothesis, n_planned):
    """Re-gate a hypothesis against the PILOT baseline, not the audit baseline."""
    pred = (hypothesis or {}).get("prediction") or {}
    metric = pred.get("metric", "mention_rate")
    p0 = pilot["p0_first"] if metric == "first_rec_rate" else pilot["p0_mention"]
    try:
        want = float(pred.get("expected_pp", 0)) - float(pred.get("baseline_pp", 0))
    except Exception:
        want = 0.0
    achievable = mde(p0, n_planned) * 100
    need_n = n_for_effect(p0, want) if want > 0 else None
    if pilot["ceiling_risk"]:
        verdict, note = "ceiling", (f"Arm-A baseline is already {p0:.0%} — no headroom. "
                                    f"Switch the primary metric to first_rec_rate, or "
                                    f"strengthen the competitor field.")
    elif pilot["floor_risk"]:
        verdict, note = "floor", (f"Arm-A baseline is {p0:.0%} — the brand is absent even "
                                  f"with its own document injected. Check the competitor "
                                  f"field and the target query.")
    elif want <= 0:
        verdict, note = "no_prediction", "Hypothesis predicts no lift — nothing to detect."
    elif want < achievable:
        verdict, note = "underpowered", (f"Predicted +{want:.1f}pp is below the "
                                         f"{achievable:.1f}pp detectable at N={n_planned} "
                                         f"(pilot p0={p0:.0%}). Need N≈{need_n}.")
    else:
        verdict, note = "powered", (f"+{want:.1f}pp is detectable at N={n_planned} "
                                    f"(MDE {achievable:.1f}pp, pilot p0={p0:.0%}).")
    return {"verdict": verdict, "note": note, "pilot_p0": round(p0, 3),
            "mde_pp": round(achievable, 1), "wanted_pp": round(want, 1),
            "n_required": need_n, "metric": metric}