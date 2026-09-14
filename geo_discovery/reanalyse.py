"""Re-analyse EVERY (design, mode) in the raw experiment log. No LLM calls, no spend.

Recovers findings for designs whose runs completed but whose aggregation never ran
(crash, budget stop, or a session that was never emitted).

    python reanalyse.py danone aptamil            # report
    python reanalyse.py danone aptamil --save     # also write validated_findings
"""
import collections
import json
import sys

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from experiment.exp_stats import (analyze_experiment, analyze_progressive,
                                  analyze_rank, arm_keys, element_attribution,
                                  variance_check, verdict)
from experiment.results import classify_evidence
from store import load_artifact, project_dir, save_artifact

P = sys.argv[1] if len(sys.argv) > 1 else "danone"
T = sys.argv[2] if len(sys.argv) > 2 else "aptamil"
SAVE = "--save" in sys.argv
RAW = str(project_dir(P, T) / "experiment_raw.jsonl")
MIN_PER_CELL = 30            # below this, treat the estimate as indicative only

def load_rows():
    rows = []
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            r["arm"] = str(r.get("arm", "")).replace("variant_", "")
            r["mode"] = r.get("mode") or ("progressive"
                                          if r["arm"] in ("C", "D", "E") else "ab")
            rows.append(r)
    return pd.DataFrame(rows)

def build_result(df, design_id, mode):
    g = df[(df.design_id == design_id) & (df["mode"] == mode)]
    if g.empty:
        return None
    engines = {}
    for (model, arm), cell in g.groupby(["model", "arm"]):
        valid = cell[cell["error"].isna()] if "error" in cell else cell
        engines.setdefault(model, {})[arm] = {
            "n": int(len(valid)), "n_attempted": int(len(cell)),
            "n_errors": int(len(cell) - len(valid)),
            "mentioned": int(valid["mentioned"].fillna(False).sum()),
            "is_first": int(valid["is_first"].fillna(False).sum()),
            "refusals": int(valid.get("refusal", pd.Series(dtype=bool))
                            .fillna(False).sum())}
    return {"design_id": design_id, "mode": mode, "engines": engines,
            "question": (g["question"].dropna().iloc[0]
                         if g["question"].notna().any() else None),
            "n_rows": int(len(g))}

def rate_table(er):
    """Per-arm rates with Wilson CIs — makes low-n instability visible."""
    out = []
    for eng, ev in er["engines"].items():
        for arm in arm_keys(ev):
            a = ev[arm]
            n = max(a.get("n", 0), 1)
            for metric, label in (("mentioned", "mention"), ("is_first", "first_rec")):
                h = a.get(metric, 0)
                lo, hi = proportion_confint(h, n, method="wilson")
                out.append({"engine": eng, "arm": arm, "metric": label,
                            "n": a.get("n"), "rate": round(h / n, 3),
                            "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                            "ci_width_pp": round((hi - lo) * 100, 1)})
    return out

def main():
    df = load_rows()
    print(f"\n{len(df)} rows · {df.design_id.nunique()} design(s)\n")
    pairs = sorted(set(zip(df.design_id, df["mode"])))
    findings, report = [], []

    for design_id, mode in pairs:
        er = build_result(df, design_id, mode)
        if not er:
            continue
        cells = sum(len(v) for v in er["engines"].values())
        per_cell = er["n_rows"] / max(cells, 1)
        tpl = "E1" if mode == "progressive" else "E0"
        try:
            _v = load_artifact(P, T, "experiment_variants") or {}
            _r = next((x for x in (_v.get("experiments") or [])
                       if (x.get("design_id") or x.get("gap_id")) == design_id), {})
            tpl = _r.get("experiment_id") or tpl
        except Exception:
            pass

        if mode == "progressive":
            rows = analyze_progressive(er, T)
            attr = element_attribution(rows)
            best = max(attr, key=attr.get) if attr else None
            sig = [r for r in rows if r.get("significant")]
            f = {"design_id": design_id, "experiment_id": "E1", "mode": mode,
                 "verdict": "validated" if sig else "no_lift",
                 "best_metric": "mention_rate",
                 "pooled_lift_pp": round(max(attr.values()), 1) if attr else 0.0,
                 "element_attribution": attr, "best_element": best,
                 "per_engine": rows}
        else:
            rows = analyze_experiment(er, T)
            v = verdict(rows, min_lift=5.0)
            best = v.get("best_metric") or "mention_rate"
            br = [r for r in rows if r.get("metric") == best]
            f = {"design_id": design_id, "experiment_id": tpl, "mode": mode,
                 "verdict": v["verdict"], "best_metric": best,
                 "pooled_lift_pp": round(sum(r["lift_pp"] for r in br) /
                                         max(len(br), 1), 1),
                 "per_engine": rows}

        f["variance_warnings"] = variance_check(er)
        f["rank_analysis"] = analyze_rank(RAW, design_id, mode=mode)
        f["rate_table"] = rate_table(er)
        f["n_rows"] = er["n_rows"]
        f["runs_per_cell"] = round(per_cell, 1)
        f["question"] = er["question"]
        f["underpowered"] = per_cell < MIN_PER_CELL
        if f["underpowered"]:
            f.setdefault("interpretation_limits", []).append(
                f"Only ~{per_cell:.0f} runs per arm per engine (target ≥{MIN_PER_CELL}). "
                f"The direction may be real but the effect SIZE is unstable — check the "
                f"Wilson CI widths in rate_table before quoting a number.")
        # ---- promote rank when the binary metrics are saturated ----
        rt = pd.DataFrame(f["rate_table"])
        ctrl_rates = rt[(rt.arm == "A") & (rt.metric == "mention")]["rate"]
        at_ceiling = (not ctrl_rates.empty) and (ctrl_rates.min() >= 0.95)
        sig_rank = [r for r in f.get("rank_analysis", [])
                    if r.get("significant") and r.get("improvement", 0) > 0]
        f["binary_at_ceiling"] = bool(at_ceiling)
        if sig_rank and f["verdict"] in ("already_winning", "no_lift", "rejected"):
            f["verdict"] = "validated_rank"
            f["best_metric"] = "mean_rank"
            f["pooled_lift_pp"] = round(
                sum(r["improvement"] for r in sig_rank) / len(sig_rank), 2)
            f["rank_engines"] = sorted({r["engine"] for r in sig_rank})
            # ---------- INSERT HERE ----------
            # mean-rank movement reads as trivial ("0.12"); the same result expressed as
            # top-1 recommendation share is the number a reader can act on.
            f["lift_unit"] = "rank_positions"
            f["rank_top1_lift_pp"] = round(sum(
                r["B_top1_pct"] - r["A_top1_pct"] for r in sig_rank) / len(sig_rank), 1)
            f["headline_effect"] = {
                "value": f["rank_top1_lift_pp"], "unit": "pp",
                "metric": "top-1 recommendation rate",
                "detail": f"top-1 recommendation share rose "
                          f"{f['rank_top1_lift_pp']:+.0f}pp; mean rank improved "
                          f"{f['pooled_lift_pp']} positions",
                "engines": f["rank_engines"]}
            # ---------- END INSERT ----------
            f.setdefault("interpretation_limits", []).append(
                "Mention rate was at ceiling (the brand's own document is in context, so "
                "it is always named). The effect is measured as improved ordinal RANK "
                f"among brands present, significant on {f['rank_engines']}.")
        elif at_ceiling and not sig_rank:
            f.setdefault("interpretation_limits", []).append(
                "Mention rate was at ceiling and rank did not move. This is an "
                "INFORMATIVE NULL: with the document present in any form, this variable "
                "has no measurable effect. Content form is not the constraint.")

        # ---- substantiveness of the control arm: the threshold hypothesis ----
        try:
            from experiment.variant_gen import QUANT_RX
            import re as _re
            v_art = load_artifact(P, T, "experiment_variants") or {}
            _recs = v_art.get("experiments") if isinstance(v_art, dict) else v_art
            rec = next((r for r in (_recs or [])
                        if r.get("design_id") == design_id), {})
            ctrl = rec.get("variant_A") or (rec.get("arms") or {}).get("A") or ""
            ctrl = ctrl if isinstance(ctrl, str) else (ctrl or {}).get("text", "")
            if ctrl:
                f["control_profile"] = {
                    "words": len(ctrl.split()),
                    "quantified": len(QUANT_RX.findall(ctrl)),
                    "proper_nouns": len(_re.findall(r"\b[A-Z][a-z]+\s?\d*\b", ctrl)),
                    "generic_phrases": sum(
                        1 for ph in ("committed to", "dedicated to", "high-quality",
                                     "comprehensive", "designed to support",
                                     "focusing on", "understands the importance")
                        if ph in ctrl.lower())}
                cp = f["control_profile"]
                if cp["generic_phrases"] >= 3 and cp["quantified"] == 0:
                    f.setdefault("interpretation_limits", []).append(
                        f"The CONTROL arm is generic brand copy "
                        f"({cp['generic_phrases']} generic phrases, 0 quantified "
                        f"values). Any lift measures crossing the SUBSTANTIVENESS "
                        f"threshold, not the nominal isolated variable.")
        except Exception:
            pass

        f.update(classify_evidence(f, {"severity": "caveated"} if f["underpowered"]
                                   else {"severity": "clean"}))
        findings.append(f)

        sig_n = sum(1 for r in f["per_engine"] if r.get("significant"))
        report.append({"design": design_id, "mode": mode, "rows": er["n_rows"],
                       "runs/cell": round(per_cell, 1),
                       "verdict": f["verdict"], "lift_pp": f["pooled_lift_pp"],
                       "sig_rows": sig_n,
                       "powered": "" if f["underpowered"] else "✓",
                       "best_element": f.get("best_element") or ""})

    print(pd.DataFrame(report).to_string(index=False))

    print("\n--- detail per design ---")
    for f in findings:
        print(f"\n### {f['design_id']} ({f['mode']}) — {f['verdict']} · "
              f"{f['pooled_lift_pp']}pp · {f['runs_per_cell']} runs/cell"
              f"{'  ⚠ UNDERPOWERED' if f['underpowered'] else ''}")
        if f.get("question"):
            print(f"    q: {f['question']}")
        if f.get("element_attribution"):
            print(f"    elements: {f['element_attribution']}  best={f.get('best_element')}")
        rt = pd.DataFrame(f["rate_table"])
        if not rt.empty:
            print(rt.pivot_table(index=["engine", "arm"], columns="metric",
                                 values=["rate", "n", "ci_width_pp"]).to_string())
        for w in f.get("variance_warnings", []):
            print(f"    ⚠ {w}")
        rk = [r for r in f.get("rank_analysis", []) if r.get("significant")]
        for r in rk:
            print(f"    🎯 rank {r['engine']}/{r['arm']}: "
                  f"{r['A_mean_rank']}→{r['B_mean_rank']} "
                  f"(top1 {r['A_top1_pct']}%→{r['B_top1_pct']}%) p={r['p_value']}")

    if SAVE:
        save_artifact(P, T, "validated_findings", findings,
                      {"source": "reanalyse.py", "note": "rebuilt from raw log"})
        act = {"validated", "validated_mention", "engine-specific",
               "engine-specific_mention", "validated_rank"}
        save_artifact(P, T, "validated_findings_only",
                      [f for f in findings if f["verdict"] in act],
                      {"source": "reanalyse.py"})
        print(f"\nSaved {len(findings)} finding(s) → validated_findings")

if __name__ == "__main__":
    main()