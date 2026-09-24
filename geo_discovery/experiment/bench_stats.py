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

def analyze_bakeoff(exp_result, target_brand=None, alpha=0.05, min_gap_pp=3.0):
    """Per engine, per metric: omnibus → pairwise (FDR) → ranking → decision."""
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

            out.append({
                "engine": eng, "metric": metric, "n_arms": len(keys),
                "omnibus_p": round(p_omni, 4), "omnibus_test": test,
                "omnibus_note": onote,
                "omnibus_significant": bool(p_omni < alpha and test != "degenerate"),
                "ranking": ranking, "pairwise": pairs,
                "winner": win["arm"], "winner_rate": win["rate"],
                "runner_up": runner["arm"], "gap_pp": gap,
                "gap_ci_pp": [head["ci_lo_pp"], head["ci_hi_pp"]] if head else None,
                "winner_distinguishable": distinguishable,
                "verdict": verdict, "decision": decision,
                "eliminated": [r["arm"] for r in ranking
                               if any(pr.get("significant") and
                                      pr["arm_b"] == r["arm"] and
                                      pr["arm_a"] == win["arm"] for pr in pairs)],
                "mde_pp": round(mde_nway(win["rate"], win["n"], len(keys)) * 100, 1),
            })
    return out

def bakeoff_summary(rows, prefer=None):
    """One decision across engines. Cross-engine disagreement is itself a finding."""
    if not rows:
        return {}
    pool = [r for r in rows if r["metric"] == prefer] if prefer else rows
    usable = [r for r in pool if r["verdict"] in ("winner", "leader_not_separated")]
    if not usable:
        pool2 = pool or rows
        return {"verdict": "no_decision", "metric": (pool2[0]["metric"] if pool2 else None),
                "decision": "; ".join(sorted({r["decision"] for r in pool2}))[:600],
                "per_engine": pool2}
    by_metric = {}
    for r in usable:
        by_metric.setdefault(r["metric"], []).append(r)
    metric = max(by_metric, key=lambda m: sum(r["verdict"] == "winner"
                                              for r in by_metric[m]))
    rows_m = by_metric[metric]
    winners = {r["winner"] for r in rows_m}
    agree = len(winners) == 1
    clean = [r for r in rows_m if r["verdict"] == "winner"]
    return {
        "metric": metric,
        "verdict": ("winner" if agree and clean else
                    "engine_specific" if clean else "leader_not_separated"),
        "winner": (sorted(winners)[0] if agree else None),
        "winners_by_engine": {r["engine"]: r["winner"] for r in rows_m},
        "engines_agree": agree,
        "mean_gap_pp": round(sum(r["gap_pp"] for r in rows_m) / len(rows_m), 1),
        "decision": ("; ".join(r["decision"] for r in clean)
                     if agree and clean else
                     "Engines disagree on the winning creative: "
                     + ", ".join(f"{r['engine']} prefers {r['winner']}"
                                 for r in rows_m)
                     + ". There is no single best creative — this belongs in an engine "
                       "overlay, not a headline recommendation."),
        "per_engine": rows_m,
    }
