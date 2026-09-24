"""N-way statistics for a creative bake-off.

analyze_experiment compares every arm to arm A, which assumes A is a control. In a
four-way creative test there may be no control — all four are candidates. Two consequences
the two-arm path gets wrong:

  * Six pairwise comparisons under FDR cost real power, so the two-arm MDE materially
    overstates sensitivity.
  * "Is B better than A" is not the question. "Which of the four wins, and is the winner
    distinguishable from second place" is — because a winner indistinguishable from the
    runner-up is not a decision.

Omnibus gate → pairwise under FDR → ranked table with a CI on winner-versus-runner-up.
"""
from __future__ import annotations

import itertools
import math

from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

METRICS = (("mentioned", "mention_rate"), ("is_first", "top1_unconditional"))

def mde_nway(p0, n, k_arms=2, alpha=0.05, power=0.80):
    """MDE accounting for multiplicity. k=4 means six comparisons, so the per-comparison
    alpha is Bonferroni-adjusted and the detectable effect is materially larger."""
    from scipy.stats import norm
    m = max(1, k_arms * (k_arms - 1) // 2)
    z_a = norm.ppf(1 - alpha / (2 * m))
    z_b = norm.ppf(power)
    p0 = min(max(float(p0 or 0.0), 0.01), 0.99)
    return (z_a + z_b) * math.sqrt(2 * p0 * (1 - p0) / max(int(n or 1), 1))

def _omnibus(cells):
    """cells: [(hits, n)]. Returns (p, test, note). Guards the degenerate cases that
    make chi-square raise or return a meaningless value."""
    if len(cells) < 2:
        return 1.0, "none", "fewer than two arms"
    hits = [h for h, _ in cells]
    ns = [n for _, n in cells]
    if all(h == 0 for h in hits):
        return 1.0, "degenerate", "no arm produced a single hit — floor on every arm"
    if all(h == n for h, n in cells):
        return 1.0, "degenerate", "every arm is at 100% — ceiling, no headroom"
    if len({round(h / max(n, 1), 6) for h, n in cells}) == 1:
        return 1.0, "degenerate", "every arm has an identical rate"
    table = [[h, n - h] for h, n in cells]
    try:
        if len(cells) == 2:
            from scipy.stats import fisher_exact
            _, p = fisher_exact(table)
            return float(p), "fisher_exact", ""
        from scipy.stats import chi2_contingency
        _, p, _, exp = chi2_contingency(table)
        low = sum(1 for row in exp for v in row if v < 5)
        note = (f"{low} expected cell(s) below 5 — the chi-square approximation is weak, "
                f"treat the omnibus p as indicative" if low else "")
        return float(p), "chi2_homogeneity", note
    except Exception as e:
        return 1.0, "failed", f"{type(e).__name__}: {str(e)[:80]}"

def _diff_ci(h1, n1, h2, n2, alpha=0.05):
    """CI on the difference of two independent proportions."""
    try:
        from statsmodels.stats.proportion import confint_proportions_2indep
        lo, hi = confint_proportions_2indep(h1, n1, h2, n2, method="wald",
                                            compare="diff", alpha=alpha)
        return float(lo), float(hi)
    except Exception:
        p1, p2 = h1 / max(n1, 1), h2 / max(n2, 1)
        se = math.sqrt(p1 * (1 - p1) / max(n1, 1) + p2 * (1 - p2) / max(n2, 1))
        d = p1 - p2
        return d - 1.96 * se, d + 1.96 * se
def _mde_block(ranking, k_arms):
    """W-21, reproduced here and now fixed. mde_nway clamps p0 to [0.01, 0.99], so a
    winner at 0% and a winner at 100% both returned ~3.9pp — an identical number for
    'no evidence at all' and 'already saturated'. Observed both in one run.

    Computed on the POOLED rate, which is what a departure is detected from, not on the
    winner's rate.
    """
    hits = sum(r["hits"] for r in ranking)
    n = sum(r["n"] for r in ranking)
    pooled = hits / max(n, 1)
    if pooled <= 0.01:
        return {"mde_pp": None, "pooled_rate": round(pooled, 4),
                "mde_note": "Undefined: every arm floored, so there is no rate to detect "
                            "a departure from."}
    if pooled >= 0.99:
        return {"mde_pp": None, "pooled_rate": round(pooled, 4),
                "mde_note": "Undefined: every arm is saturated, so there is no headroom "
                            "on this metric."}
    per_arm = n // max(len(ranking), 1)
    return {"mde_pp": round(mde_nway(pooled, per_arm, k_arms) * 100, 1),
            "pooled_rate": round(pooled, 4),
            "mde_note": (f"From a pooled rate of {pooled:.0%} at ~{per_arm} runs per arm, "
                         f"across {max(1, k_arms * (k_arms - 1) // 2)} comparison(s).")}


def analyze_bakeoff(exp_result, target_brand=None, alpha=0.05, min_gap_pp=3.0):
    """Per engine, per metric: omnibus → pairwise (FDR) → ranking → decision.

    INDENTATION IS LOAD-BEARING. `return out` sits at FUNCTION level. Pulled one level
    in, the function returns after the first engine and every subsequent engine is
    silently discarded — which reads as unanimous agreement rather than as missing data,
    and is therefore invisible in the output.
    """
    from experiment.exp_stats import arm_keys
    out = []

    for eng, ev in (exp_result.get("engines") or {}).items():
        keys = [k for k in arm_keys(ev) if not str(k).endswith("N")]
        if len(keys) < 2:
            continue

        for field, metric in METRICS:
            cells = [(int(ev[k].get(field, 0) or 0), max(int(ev[k].get("n", 0) or 0), 1))
                     for k in keys]
            p_omni, test, onote = _omnibus(cells)

            # ---- ranking, with Wilson intervals ----
            ranking = []
            for k, (h, n) in zip(keys, cells):
                lo, hi = proportion_confint(h, n, alpha=alpha, method="wilson")
                ranking.append({"arm": str(k).replace("variant_", ""),
                                "rate": round(h / n, 4), "hits": h, "n": n,
                                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                "ci_width_pp": round((hi - lo) * 100, 1)})
            ranking.sort(key=lambda r: -r["rate"])
            for i, r in enumerate(ranking, 1):
                r["rank"] = i

            # ---- pairwise, ONLY if the omnibus gate passes ----
            pairs, pvals = [], []
            if p_omni < alpha:
                idx = {str(k).replace("variant_", ""): (h, n)
                       for k, (h, n) in zip(keys, cells)}
                for a, b in itertools.combinations([r["arm"] for r in ranking], 2):
                    (ha, na), (hb, nb) = idx[a], idx[b]
                    ra, rb = ha / na, hb / nb
                    try:
                        _, pv = proportions_ztest([ha, hb], [na, nb])
                    except Exception:
                        pv = 1.0
                    lo, hi = _diff_ci(ha, na, hb, nb, alpha)
                    pairs.append({"arm_a": a, "arm_b": b,
                                  "rate_a": round(ra, 4), "rate_b": round(rb, 4),
                                  "diff_pp": round((ra - rb) * 100, 1),
                                  "ci_lo_pp": round(lo * 100, 1),
                                  "ci_hi_pp": round(hi * 100, 1),
                                  "p_value": round(float(pv), 4)})
                    pvals.append(pv)
                if pvals:
                    fdr = multipletests(pvals, alpha=alpha, method="fdr_bh")[1]
                    for pr, pf in zip(pairs, fdr):
                        pr["p_fdr"] = round(float(pf), 4)
                        pr["significant"] = bool(pf < alpha)

            # ---- is the leader separated from second place? ----
            win, runner = ranking[0], ranking[1]
            head = next((pr for pr in pairs
                         if {pr["arm_a"], pr["arm_b"]} == {win["arm"], runner["arm"]}),
                        None)
            gap = round((win["rate"] - runner["rate"]) * 100, 1)
            distinguishable = bool(head and head.get("significant")) and gap >= min_gap_pp

            if test == "degenerate":
                verdict, decision = "no_variance", (
                    f"No decision available: {onote}. "
                    + ("Mention rate has no headroom here — read top-1 instead."
                       if "ceiling" in onote else
                       "Either the assets are not legible to the model, or the target "
                       "query does not surface this category at all." if "floor" in onote
                       else "The arms performed identically."))
            elif p_omni >= alpha:
                verdict, decision = "equivalent", (
                    f"The arms are statistically EQUIVALENT (omnibus p={p_omni:.3f}). "
                    f"Observed spread {ranking[0]['rate'] - ranking[-1]['rate']:+.1%} is "
                    f"within noise at this sample size. Choose between these creatives on "
                    f"human grounds — a machine reader does not distinguish them.")
            elif distinguishable:
                verdict, decision = "winner", (
                    f"Arm {win['arm']} wins at {win['rate']:.1%}, ahead of arm "
                    f"{runner['arm']} by {gap:+.1f}pp "
                    f"(95% CI {head['ci_lo_pp']:+.1f} to {head['ci_hi_pp']:+.1f}pp, "
                    f"FDR p={head['p_fdr']:.3f}). This is a decision.")
            else:
                verdict, decision = "leader_not_separated", (
                    f"Arm {win['arm']} leads at {win['rate']:.1%}, but is NOT "
                    f"distinguishable from arm {runner['arm']} ({gap:+.1f}pp"
                    + (f", CI {head['ci_lo_pp']:+.1f} to {head['ci_hi_pp']:+.1f}pp"
                       if head else "") + "). The omnibus test found a difference "
                    f"SOMEWHERE among the arms, so the weakest arms can be dropped — but "
                    f"picking a winner from the top two needs a confirmation run at "
                    f"higher N.")

            # A stable sort on equal rates leaves the first arm at index 0, so a
            # floor-on-every-arm test previously reported `winner: "A"` at a 0.0% rate.
            # `winner` now exists only on a decisive verdict; `leader` is always the
            # top-ranked arm.
            decisive = verdict in ("winner", "leader_not_separated")

            # A saturated winner means the gap is a LOWER BOUND, and no further creative
            # could score higher on this metric.
            at_ceiling = win["rate"] >= 0.99
            if at_ceiling and decisive:
                decision += (
                    f" Arm {win['arm']} is SATURATED at {win['rate']:.0%}, so "
                    f"{gap:+.1f}pp is a LOWER BOUND on its advantage — the true "
                    f"difference may be larger, and no additional creative could score "
                    f"higher on this metric. Read top1_unconditional for headroom.")

            out.append({
                "engine": eng, "metric": metric, "n_arms": len(keys),
                "omnibus_p": round(p_omni, 4), "omnibus_test": test,
                "omnibus_note": onote,
                "omnibus_significant": bool(p_omni < alpha and test != "degenerate"),
                "ranking": ranking, "pairwise": pairs,
                "winner": win["arm"] if decisive else None,
                "winner_rate": win["rate"] if decisive else None,
                "leader": win["arm"], "leader_rate": win["rate"],
                "runner_up": runner["arm"], "gap_pp": gap,
                "gap_ci_pp": [head["ci_lo_pp"], head["ci_hi_pp"]] if head else None,
                "gap_is_lower_bound": bool(at_ceiling and decisive),
                "winner_at_ceiling": at_ceiling,
                "winner_distinguishable": distinguishable,
                "verdict": verdict, "decision": decision,
                "eliminated": [r["arm"] for r in ranking
                               if any(pr.get("significant") and
                                      pr["arm_b"] == r["arm"] and
                                      pr["arm_a"] == win["arm"] for pr in pairs)],
                **_mde_block(ranking, len(keys)),
            })

    # ---- FUNCTION level. One level in and only the first engine is ever analysed. ----
    return out

def bakeoff_summary(rows, prefer=None, magnitude_spread_pp=15.0):
    """One decision across engines.

    Three things the earlier version dropped:
      * a SECOND validated result on another metric never surfaced — observed: top-1
        recommendation moved 8% to 47% on one engine, FDR p<0.001, and the summary
        reported mention_rate only
      * `pooled_is_misleading` tested winner DISAGREEMENT only, so gaps of 51pp and 74pp
        passed as agreement while the pooled 62.5pp described neither engine
      * an engine that MENTIONS the brand more but never RECOMMENDS it first is a
        corroboration constraint, not a creative one, and no creative change will move it
    """
    if not rows:
        return {}
    pool = [r for r in rows if r["metric"] == prefer] if prefer else rows
    usable = [r for r in pool if r["verdict"] in ("winner", "leader_not_separated")]
    if not usable:
        pool2 = pool or rows
        return {"verdict": "no_decision",
                "metric": (pool2[0]["metric"] if pool2 else None),
                "decision": "; ".join(sorted({r["decision"] for r in pool2}))[:600],
                "per_engine": pool2}

    by_metric = {}
    for r in usable:
        by_metric.setdefault(r["metric"], []).append(r)
    metric = max(by_metric, key=lambda m: sum(r["verdict"] == "winner"
                                              for r in by_metric[m]))
    rows_m = by_metric[metric]
    winners = {r["winner"] for r in rows_m if r["winner"]}
    agree = len(winners) == 1
    clean = [r for r in rows_m if r["verdict"] == "winner"]

    # ---- C: engines can agree on WHICH creative wins and disagree on HOW MUCH ----
    gaps = [r["gap_pp"] for r in rows_m]
    spread = round(max(gaps) - min(gaps), 1) if len(gaps) > 1 else 0.0
    magnitude_diverges = spread >= magnitude_spread_pp

    # ---- D: a decisive result on ANOTHER metric is a second finding, not noise ----
    secondary = [{"engine": r["engine"], "metric": r["metric"],
                  "winner": r["winner"], "gap_pp": r["gap_pp"],
                  "from_pct": round(min(x["rate"] for x in r["ranking"]) * 100, 1),
                  "to_pct": round(max(x["rate"] for x in r["ranking"]) * 100, 1),
                  "decision": r["decision"]}
                 for r in rows if r["metric"] != metric and r["verdict"] == "winner"]

    # ---- E: mentioned more, still never recommended first ----
    floored = {r["engine"] for r in rows
               if r["metric"] == "top1_unconditional"
               and r["verdict"] == "no_variance"
               and "floor" in (r["omnibus_note"] or "")}
    mention_won = {r["engine"] for r in rows
                   if r["metric"] == "mention_rate" and r["verdict"] == "winner"}
    stuck = sorted(floored & mention_won)
    stuck_note = (
        f"On {', '.join(stuck)} the winning creative raises MENTION but the brand is "
        f"never recommended first under ANY arm. That is an authority or corroboration "
        f"constraint, not a creative one — no creative change will move it. Route to the "
        f"Corroboration pillar, not to Visual/Multimodal." if stuck else None)

    lower_bound = [r["engine"] for r in rows_m if r.get("gap_is_lower_bound")]

    decision = ("; ".join(r["decision"] for r in clean)
                if agree and clean else
                "Engines disagree on the winning creative: "
                + ", ".join(f"{r['engine']} prefers {r['winner']}" for r in rows_m)
                + ". There is no single best creative — this belongs in an engine "
                  "overlay, not a headline recommendation.")
    if agree and magnitude_diverges:
        decision += (f" Both engines pick the same creative, but the effect SIZE differs "
                     f"by {spread:.0f}pp ("
                     + ", ".join(f"{r['engine']} {r['gap_pp']:+.0f}pp" for r in rows_m)
                     + f"), so the pooled figure describes neither. Quote the per-engine "
                       f"numbers.")
    if secondary:
        decision += (" Second validated result: "
                     + "; ".join(f"{s['engine']} {s['metric']} moved "
                                 f"{s['from_pct']:.0f}% to {s['to_pct']:.0f}% "
                                 f"({s['gap_pp']:+.0f}pp)" for s in secondary) + ".")
    if stuck_note:
        decision += " " + stuck_note

    return {
        "metric": metric,
        "verdict": ("winner" if agree and clean else
                    "engine_specific" if clean else "leader_not_separated"),
        "winner": (sorted(winners)[0] if agree and winners else None),
        "winners_by_engine": {r["engine"]: r["winner"] for r in rows_m},
        "engines_agree": agree,
        "mean_gap_pp": round(sum(gaps) / len(gaps), 1),
        "gap_spread_pp": spread,
        "magnitude_diverges": magnitude_diverges,
        "gap_is_lower_bound_on": lower_bound,
        "secondary_results": secondary,
        "mention_without_recommendation": stuck,
        "mention_without_recommendation_note": stuck_note,
        "decision": decision,
        "per_engine": rows_m,
    }
