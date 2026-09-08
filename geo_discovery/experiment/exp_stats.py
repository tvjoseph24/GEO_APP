"""Target-brand metrics: measure HOW THE BRAND FARED (mention + first-rec),
not whether the leader changed. Both A/B and progressive."""
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.stats.multitest import multipletests
from scipy.stats import fisher_exact

def _test_pair(a_hits, a_n, b_hits, b_n):
    """A vs B significance, guarding identical/extreme cases."""
    a_rate, b_rate = a_hits/max(a_n,1), b_hits/max(b_n,1)
    try:
        if a_rate == b_rate:
            p = 1.0
        elif (a_hits in (0, a_n)) or (b_hits in (0, b_n)):
            _, p = fisher_exact([[b_hits, b_n-b_hits], [a_hits, a_n-a_hits]])
        else:
            _, p = proportions_ztest([b_hits, a_hits], [b_n, a_n])
    except Exception:
        p = 1.0
    return a_rate, b_rate, p

def analyze_experiment(exp_result, target_brand):
    """A/B — measure the TARGET BRAND on BOTH metrics (mentioned + is_first)."""
    rows, pvals = [], []
    for eng, ev in exp_result["engines"].items():
        for metric, label in [("mentioned", "mention_rate"),
                              ("is_first", "first_rec_rate")]:
            a = ev["variant_A"][metric]; b = ev["variant_B"][metric]
            na, nb = ev["variant_A"]["n"], ev["variant_B"]["n"]
            a_rate, b_rate, p = _test_pair(a, na, b, nb)
            alo, ahi = proportion_confint(a, na, method="wilson")
            blo, bhi = proportion_confint(b, nb, method="wilson")
            rows.append({"engine": eng, "metric": label,
                         "A_rate": round(a_rate,3), "B_rate": round(b_rate,3),
                         "lift_pp": round((b_rate-a_rate)*100,1),
                         "A_ci": [round(alo,3), round(ahi,3)],
                         "B_ci": [round(blo,3), round(bhi,3)],
                         "p_value": p})
            pvals.append(p)
    if pvals:
        fdr = multipletests(pvals, method="fdr_bh")[1]
        for r, pf in zip(rows, fdr):
            r["p_fdr"] = round(pf, 4); r["significant"] = pf < 0.05
    return rows

def analyze_progressive(exp_result, target_brand=None):
    """Progressive — target brand across arms, BOTH metrics, marginal per element."""
    arms = exp_result["arms"]
    rows, pvals = [], []
    for eng, ev in exp_result["engines"].items():
        for metric, label in [("mentioned", "mention_rate"),
                              ("is_first", "first_rec_rate")]:
            base_hits, base_n = ev["A"][metric], ev["A"]["n"]
            base_rate = base_hits/max(base_n,1); prev_rate = base_rate
            for arm in arms:
                h, n = ev[arm][metric], ev[arm]["n"]
                rate = h/max(n,1)
                lo, hi = proportion_confint(h, n, method="wilson")
                if arm != "A":
                    _, _, p = _test_pair(base_hits, base_n, h, n)
                else:
                    p = 1.0
                rows.append({"engine": eng, "arm": arm, "metric": label,
                             "rate": round(rate,3), "ci": [round(lo,3), round(hi,3)],
                             "vs_A_pp": round((rate-base_rate)*100,1),
                             "marginal_pp": round((rate-prev_rate)*100,1),
                             "p_value": p})
                pvals.append(p); prev_rate = rate
    if pvals:
        fdr = multipletests(pvals, method="fdr_bh")[1]
        for r, pf in zip(rows, fdr):
            r["p_fdr"] = round(pf,4); r["significant"] = pf < 0.05
    return rows

def element_attribution(prog_rows, metric="mention_rate"):
    """Which element drove the most marginal lift — for a chosen metric."""
    import pandas as pd
    df = pd.DataFrame([r for r in prog_rows if r["metric"] == metric])
    if df.empty: return {}
    marg = (df[df.arm != "A"].groupby("arm")["marginal_pp"].mean()
              .round(1).sort_values(ascending=False))
    return marg.to_dict()

def verdict(analysis_rows, min_lift=3.0):
    """Verdict based on TARGET BRAND lift — considers BOTH metrics.
    Reports the best signal (mention OR first-rec)."""
    mention_rows = [r for r in analysis_rows if r["metric"] == "mention_rate"]
    first_rows   = [r for r in analysis_rows if r["metric"] == "first_rec_rate"]

    def _sig(rows):
        return [r for r in rows if r.get("significant") and r["lift_pp"] >= min_lift]

    m_sig = _sig(mention_rows); f_sig = _sig(first_rows)
    best_metric = None; verdict_str = "rejected"

    if f_sig:                          # first-rec lift = strongest win
        verdict_str = "validated" if len(f_sig)==len(first_rows) else "engine-specific"
        best_metric = "first_rec_rate"
    elif m_sig:                        # mention lift = got INTO the answer (S001!)
        verdict_str = "validated_mention" if len(m_sig)==len(mention_rows) else "engine-specific_mention"
        best_metric = "mention_rate"
    else:
        # distinguish ceiling from floor using mention baseline
        avg_a_mention = (sum(r["A_rate"] for r in mention_rows)/len(mention_rows)
                         if mention_rows else 0)
        if avg_a_mention >= 0.9:   verdict_str = "already_winning"   # ceiling
        elif avg_a_mention <= 0.1: verdict_str = "lost"             # floor
        else:                       verdict_str = "no_lift"
    return {"verdict": verdict_str, "best_metric": best_metric}

def variance_check(exp_result, metric="is_first"):
    warnings = []
    for eng, ev in exp_result["engines"].items():
        a, na = ev["variant_A"][metric], ev["variant_A"]["n"]
        b, nb = ev["variant_B"][metric], ev["variant_B"]["n"]
        if a == na and b == nb:
            warnings.append(f"{eng}: both at CEILING (100% {metric})")
        elif a == 0 and b == 0:
            warnings.append(f"{eng}: both at FLOOR (0% {metric})")
    return warnings