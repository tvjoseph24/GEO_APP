"""Target-brand metrics: measure HOW THE BRAND FARED (mention + first-rec),
not whether the leader changed. Both A/B and progressive."""
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.stats.multitest import multipletests
from scipy.stats import fisher_exact

def arm_keys(ev):
    """Arm keys for one engine, control first. Handles both naming schemes and detects
    the case where two runs were merged (prefixed AND bare keys present)."""
    keys = [k for k, v in (ev or {}).items() if isinstance(v, dict) and "n" in v]
    pre = [k for k in keys if str(k).startswith("variant_")]
    bare = [k for k in keys if not str(k).startswith("variant_")]
    if pre and bare:
        n_pre = sum((ev[k].get("n") or 0) for k in pre)
        n_bare = sum((ev[k].get("n") or 0) for k in bare)
        keys = pre if n_pre >= n_bare else bare      # keep the larger, coherent run

    def _rank(k):
        s = str(k).replace("variant_", "")
        if not s:
            return 100
        if s == "N":
            return 99
        return ord(s[0].upper()) - 64
    return sorted(keys, key=_rank)

def arm_contamination(exp_result):
    """True if any engine has both naming schemes — the result is not trustworthy."""
    bad = []
    for eng, ev in (exp_result.get("engines") or {}).items():
        ks = [k for k, v in (ev or {}).items() if isinstance(v, dict) and "n" in v]
        if any(str(k).startswith("variant_") for k in ks) and \
                any(not str(k).startswith("variant_") for k in ks):
            bad.append(eng)
    return bad

def control_arm(ev):
    ks = arm_keys(ev)
    return ks[0] if ks else None

def _rate(a, metric):
    return (a.get(metric, 0) or 0) / max(a.get("n", 0) or 1, 1)

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
    """Every non-control arm vs the control, both metrics, FDR-corrected."""
    rows, pvals = [], []
    for eng, ev in (exp_result.get("engines") or {}).items():
        ks = arm_keys(ev)
        if len(ks) < 2:
            continue
        ctrl = ks[0]
        for metric, label in (("mentioned", "mention_rate"),
                              ("is_first", "first_rec_rate")):
            a, na = ev[ctrl].get(metric, 0), max(ev[ctrl].get("n", 0), 1)
            for arm in ks[1:]:
                b, nb = ev[arm].get(metric, 0), max(ev[arm].get("n", 0), 1)
                a_rate, b_rate, pv = _test_pair(a, na, b, nb)
                alo, ahi = proportion_confint(a, na, method="wilson")
                blo, bhi = proportion_confint(b, nb, method="wilson")
                rows.append({"engine": eng, "arm": str(arm).replace("variant_", ""),
                             "metric": label,
                             "A_rate": round(a_rate, 3), "B_rate": round(b_rate, 3),
                             "lift_pp": round((b_rate - a_rate) * 100, 1),
                             "A_ci": [round(alo, 3), round(ahi, 3)],
                             "B_ci": [round(blo, 3), round(bhi, 3)],
                             "n_A": na, "n_B": nb, "p_value": pv,
                             "is_negative_control": str(arm).endswith("N")})
                pvals.append(pv)
    if pvals:
        fdr = multipletests(pvals, method="fdr_bh")[1]
        for r, pf in zip(rows, fdr):
            r["p_fdr"] = round(pf, 4)
            r["significant"] = pf < 0.05
    return rows

def analyze_progressive(exp_result, target_brand=None):
    rows, pvals = [], []
    for eng, ev in (exp_result.get("engines") or {}).items():
        ks = arm_keys(ev)
        if len(ks) < 2:
            continue
        base = ks[0]
        for metric, label in (("mentioned", "mention_rate"),
                              ("is_first", "first_rec_rate")):
            bh, bn = ev[base].get(metric, 0), max(ev[base].get("n", 0), 1)
            base_rate = bh / bn
            prev = base_rate
            for arm in ks:
                h, n = ev[arm].get(metric, 0), max(ev[arm].get("n", 0), 1)
                rate = h / n
                lo, hi = proportion_confint(h, n, method="wilson")
                pv = 1.0 if arm == base else _test_pair(bh, bn, h, n)[2]
                rows.append({"engine": eng, "arm": str(arm).replace("variant_", ""),
                             "metric": label, "rate": round(rate, 3),
                             "ci": [round(lo, 3), round(hi, 3)], "n": n,
                             "vs_A_pp": round((rate - base_rate) * 100, 1),
                             "marginal_pp": round((rate - prev) * 100, 1),
                             "p_value": pv})
                pvals.append(pv)
                prev = rate
    if pvals:
        fdr = multipletests(pvals, method="fdr_bh")[1]
        for r, pf in zip(rows, fdr):
            r["p_fdr"] = round(pf, 4)
            r["significant"] = pf < 0.05
    return rows

def element_attribution(prog_rows, metric="mention_rate"):
    import pandas as pd
    df = pd.DataFrame([r for r in prog_rows if r.get("metric") == metric])
    if df.empty or "arm" not in df:
        return {}
    marg = (df[df.arm != "A"].groupby("arm")["marginal_pp"].mean()
            .round(1).sort_values(ascending=False))
    return marg.to_dict()

def analyze_rank(raw_path, design_id, engines=None, mode=None, min_present=5,
                 alpha=0.05):
    """Rank analysis, reported BOTH conditionally and unconditionally.

    W-24, four defects:

    1. CONDITIONING. Rank exists only where the brand appeared, so comparing mean rank
       across arms with different mention rates compares a SELECTED SUBSET against a
       full population. Observed: n_A=57, n_B=100, improvement reported as 0 while
       `A_top1_pct` read 100 — a conditional share. The brand was actually top-1 in 57
       of 100 runs, so the true movement was 57% -> 100%, +43pp, reported as zero.
       `top1_uncond_*` is now the headline and `top1_cond_*` the diagnostic.

    2. NO FDR CORRECTION. A verdict was promoted to validated_rank on one uncorrected
       p=0.0442 out of 8 comparisons, 6 of which showed the treatment performing worse.

    3. SILENT ENGINE EXCLUSION. Any engine with fewer than `min_present` present-runs was
       dropped with no notice — removing the rescue metric exactly when arm A is weakest
       and headroom is largest.

    4. BARE EXCEPT. A failed test set p=1.0, indistinguishable from a computed result.
       Fully-tied constant arrays produce a degenerate zero-variance test.

    Returns a list of comparison rows. Row 0 carries `_meta` with exclusions.
    """
    import json as _j
    import pandas as pd
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    rows = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = _j.loads(line)
            except Exception:
                continue
            if r.get("design_id") != design_id:
                continue
            if mode and r.get("mode") and r.get("mode") != mode:
                continue
            if r.get("error"):
                continue                      # errored runs are not observations
            r["arm"] = str(r.get("arm", "")).replace("variant_", "")
            rows.append(r)
    if not rows:
        return []

    df = pd.DataFrame(rows)
    out, pvals, excluded = [], [], []

    for eng, g in df.groupby("model"):
        arms = sorted(g["arm"].dropna().unique(),
                      key=lambda a: (99 if str(a).endswith("N") else ord(str(a)[0])))
        if len(arms) < 2:
            excluded.append({"engine": eng, "why": f"only {len(arms)} arm(s) recorded"})
            continue
        ctrl = arms[0]

        def _arm(a):
            s = g[g.arm == a]
            rk = s["rank"].dropna().astype(float)
            return {"n_total": int(len(s)), "n_present": int(len(rk)), "ranks": rk}

        A = _arm(ctrl)
        if A["n_present"] < min_present:
            excluded.append({
                "engine": eng, "arm": ctrl,
                "why": f"control has only {A['n_present']} run(s) where the brand was "
                       f"present (min {min_present}) — rank is undefined for the rest",
                "n_total": A["n_total"],
                "note": "The brand's near-absence in the control IS the finding; read "
                        "the binary metrics instead."})
            continue

        for arm in arms[1:]:
            B = _arm(arm)
            if B["n_present"] < min_present:
                excluded.append({"engine": eng, "arm": arm,
                                 "why": f"only {B['n_present']} present-run(s)",
                                 "n_total": B["n_total"]})
                continue

            test_ok, test_err, pv = True, None, 1.0
            if A["ranks"].nunique() == 1 and B["ranks"].nunique() == 1 \
                    and A["ranks"].iloc[0] == B["ranks"].iloc[0]:
                test_ok, test_err = False, ("all ranks identical in both arms — "
                                            "zero variance, no ordering information")
            else:
                try:
                    _, pv = mannwhitneyu(B["ranks"], A["ranks"], alternative="less")
                except Exception as e:
                    test_ok, test_err = False, f"{type(e).__name__}: {str(e)[:60]}"

            a_top1, b_top1 = int((A["ranks"] == 1).sum()), int((B["ranks"] == 1).sum())
            row = {
                "engine": eng, "arm": str(arm), "metric": "mean_rank",
                # ---- conditional: given the brand appeared ----
                "A_mean_rank": round(float(A["ranks"].mean()), 2),
                "B_mean_rank": round(float(B["ranks"].mean()), 2),
                "improvement": round(float(A["ranks"].mean() - B["ranks"].mean()), 2),
                "A_top1_cond_pct": round(100 * a_top1 / max(A["n_present"], 1), 1),
                "B_top1_cond_pct": round(100 * b_top1 / max(B["n_present"], 1), 1),
                # ---- UNCONDITIONAL: over all runs. This is the client-facing number ----
                "A_top1_uncond_pct": round(100 * a_top1 / max(A["n_total"], 1), 1),
                "B_top1_uncond_pct": round(100 * b_top1 / max(B["n_total"], 1), 1),
                "top1_uncond_lift_pp": round(
                    100 * (b_top1 / max(B["n_total"], 1) - a_top1 / max(A["n_total"], 1)),
                    1),
                "n_A_present": A["n_present"], "n_B_present": B["n_present"],
                "n_A_total": A["n_total"], "n_B_total": B["n_total"],
                "presence_rate_A": round(A["n_present"] / max(A["n_total"], 1), 3),
                "presence_rate_B": round(B["n_present"] / max(B["n_total"], 1), 3),
                "p_value": round(float(pv), 4),
                "test_ok": test_ok, "test_error": test_err,
            }
            # comparing conditional means across different presence rates is not valid
            if abs(row["presence_rate_A"] - row["presence_rate_B"]) > 0.10:
                row["conditioning_warning"] = (
                    f"Presence differs between arms ({row['presence_rate_A']:.0%} vs "
                    f"{row['presence_rate_B']:.0%}), so mean_rank compares different "
                    f"populations. Use top1_uncond_lift_pp "
                    f"({row['top1_uncond_lift_pp']:+.1f}pp) as the effect.")
            out.append(row)
            if test_ok:
                pvals.append((len(out) - 1, pv))

    # ---- FDR across ALL rank comparisons for this design ----
    if pvals:
        idxs, ps = zip(*pvals)
        fdr = multipletests(list(ps), alpha=alpha, method="fdr_bh")[1]
        for i, pf in zip(idxs, fdr):
            out[i]["p_fdr"] = round(float(pf), 4)
            out[i]["significant"] = bool(pf < alpha)
    for r in out:
        r.setdefault("p_fdr", None)
        r.setdefault("significant", False)

    if out:
        out[0]["_meta"] = {"design_id": design_id, "mode": mode,
                           "n_comparisons": len(out), "fdr_applied": bool(pvals),
                           "engines_excluded": excluded,
                           "min_present_threshold": min_present}
    elif excluded:
        return [{"engine": None, "metric": "mean_rank", "significant": False,
                 "_meta": {"design_id": design_id, "n_comparisons": 0,
                           "engines_excluded": excluded,
                           "note": "no engine had enough present-runs in both arms"}}]
    return out

def verdict(analysis_rows, min_lift=3.0, primary_metric=None):
    """Verdict based on TARGET BRAND lift across BOTH metrics.

    W-29: the metric was picked post-hoc — whichever reached significance first — so
    an analyst could not choose it, despite the power note advising exactly that.
    """
    if primary_metric and primary_metric in ("mention_rate", "first_rec_rate"):
        rows = [r for r in analysis_rows if r.get("metric") == primary_metric]
        sig = [r for r in rows if r.get("significant") and r["lift_pp"] >= min_lift]
        if sig:
            v = "validated" if len(sig) == len(rows) else "engine-specific"
            if primary_metric == "mention_rate":
                v += "_mention"
            return {"verdict": v, "best_metric": primary_metric,
                    "metric_was_selected": True}
        avg_a = sum(r["A_rate"] for r in rows) / max(len(rows), 1) if rows else 0
        return {"verdict": ("already_winning" if avg_a >= 0.9 else
                            "lost" if avg_a <= 0.1 else "no_lift"),
                "best_metric": primary_metric, "metric_was_selected": True}
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
    """Ceiling / floor / low-headroom detection. Arm-agnostic."""
    out = []
    contaminated = arm_contamination(exp_result)
    if contaminated:
        out.append(f"⛔ ARM CONTAMINATION on {contaminated}: this design was run under "
                   f"two modes and the results were merged. Purge and re-run — the "
                   f"comparisons below are not valid.")
    for eng, ev in (exp_result.get("engines") or {}).items():
        ks = arm_keys(ev)
        if len(ks) < 2:
            out.append(f"{eng}: only {len(ks)} arm(s) recorded — cannot compare")
            continue
        ctrl = ks[0]
        rates = {k: _rate(ev[k], metric) for k in ks}
        if all(r >= 0.99 for r in rates.values()):
            out.append(f"{eng}: ALL arms at ceiling (100% {metric}) — no headroom")
        elif all(r <= 0.01 for r in rates.values()):
            out.append(f"{eng}: ALL arms at floor (0% {metric})")
        elif rates[ctrl] >= 0.90:
            out.append(f"{eng}: control at {rates[ctrl]:.0%} {metric} — only "
                       f"{(1 - rates[ctrl]) * 100:.0f}pp headroom; switch metric")
        elif rates[ctrl] <= 0.02:
            out.append(f"{eng}: control at {rates[ctrl]:.0%} {metric} — near floor")
        for k in ks:
            n = ev[k].get("n", 0)
            if n and n < 20:
                out.append(f"{eng}/{k}: only {n} valid run(s)")
            if ev[k].get("n_errors", 0) > 0.1 * max(n, 1):
                out.append(f"{eng}/{k}: {ev[k]['n_errors']} error(s) excluded")
    return out