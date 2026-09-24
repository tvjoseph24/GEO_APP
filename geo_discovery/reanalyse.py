"""Re-analyse EVERY (design, mode) in the raw experiment log. No LLM calls, no spend.

Recovers findings for designs whose runs completed but whose aggregation never ran
(crash, budget stop, or a session that was never emitted).

    python reanalyse.py PARENT TARGET            # report
    python reanalyse.py PARENT TARGET --save     # also write validated_findings

Delegates to experiment.results.assemble_findings. It previously reimplemented that
function, and the copy drifted: it read rank keys renamed two phases earlier, and it had
no branch for a creative bake-off — so re-analysing a bench design silently replaced the
finding with an A/B reading and dropped the bake-off data.
"""
import json
import sys
from collections import Counter

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from config import MODEL_REGISTRY
from experiment.exp_stats import arm_keys
from experiment.results import assemble_findings
from store import load_artifact, project_dir, save_artifact

P = sys.argv[1] if len(sys.argv) > 1 else "danone"
T = sys.argv[2] if len(sys.argv) > 2 else "aptamil"
SAVE = "--save" in sys.argv
KEEP_PILOTS = "--include-pilots" in sys.argv
RAW = str(project_dir(P, T) / "experiment_raw.jsonl")

MIN_PER_CELL = 30
PILOT_PREFIX = "_pilot_"

# ---------------------------------------------------------------- load
def load_rows():
    """Every run row. Pilots are EXCLUDED: a pilot is a single-arm baseline, not an
    experiment. Analysed as one it yields verdict 'lost' with no comparison, which
    classify_evidence then reports as a null — and --save writes it into
    validated_findings, where Optimise reads it as evidence."""
    rows, pilots = [], 0
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            r["arm"] = str(r.get("arm", "")).replace("variant_", "")
            r["mode"] = r.get("mode") or ("progressive"
                                          if r["arm"] in ("C", "D", "E") else "ab")
            is_pilot = (r["mode"] == "pilot"
                        or str(r.get("design_id", "")).startswith(PILOT_PREFIX))
            if is_pilot and not KEEP_PILOTS:
                pilots += 1
                continue
            rows.append(r)
    return pd.DataFrame(rows), pilots

# ---------------------------------------------------------------- reconstruct
def _visual_block(cell, model):
    """Rebuild the per-arm visual block from raw rows. The runner records
    `image_attached` and `image_tokens` per run, so payload delivery is recoverable
    without re-running anything."""
    if "image_attached" not in cell:
        return None
    att = cell[cell["image_attached"].fillna(False).astype(bool)]
    if att.empty:
        return None
    tok = att["image_tokens"].fillna(0) if "image_tokens" in att else pd.Series([0])
    prov = MODEL_REGISTRY.get(model, {}).get("provider", "litellm")
    return {"provider": prov, "model": model,
            # OpenAI folds image tokens into prompt_tokens with no breakdown, so a
            # LiteLLM engine reports zero on every vision call regardless of delivery.
            "reports_media_tokens": prov == "vertex",
            "runs_with_image": int(len(att)),
            "runs_with_image_tokens": int((tok > 0).sum()),
            "image_tokens_total": int(tok.sum()),
            "payload_delivered": bool((tok > 0).any()),
            "delivery_rate": round(float((tok > 0).mean()), 3),
            "video_runs_skipped": 0}

def build_result(df, design_id, mode, bench_meta):
    g = df[(df.design_id == design_id) & (df["mode"] == mode)]
    if g.empty:
        return None
    engines = {}
    for (model, arm), cell in g.groupby(["model", "arm"]):
        valid = cell[cell["error"].isna()] if "error" in cell else cell
        stats = {
            "n": int(len(valid)), "n_attempted": int(len(cell)),
            "n_errors": int(len(cell) - len(valid)),
            "mentioned": int(valid["mentioned"].fillna(False).sum()),
            "is_first": int(valid["is_first"].fillna(False).sum()),
            "refusals": int(valid.get("refusal", pd.Series(dtype=bool))
                            .fillna(False).sum())}
        vb = _visual_block(cell, model)
        if vb:
            stats["visual"] = vb
        engines.setdefault(model, {})[arm] = stats

    er = {"design_id": design_id, "mode": mode, "engines": engines,
          "question": (g["question"].dropna().iloc[0]
                       if g["question"].notna().any() else None),
          "n_rows": int(len(g)),
          "comp_mode": (g["comp_mode"].dropna().iloc[0]
                        if "comp_mode" in g and g["comp_mode"].notna().any() else None),
          "position_mode": (g["position"].dropna().iloc[0]
                            if "position" in g and g["position"].notna().any() else None)}

    # ---- experiment_id, and bench metadata the raw log does not carry ----
    er["experiment_id"] = ("E1" if mode == "progressive" else
                           "VBENCH" if mode == "creative" else "E0")
    try:
        v = load_artifact(P, T, "experiment_variants") or {}
        rec = next((x for x in (v.get("experiments") or [])
                    if (x.get("design_id") or x.get("gap_id")) == design_id), {})
        if rec.get("experiment_id"):
            er["experiment_id"] = rec["experiment_id"]
        if rec.get("question") and not er["question"]:
            er["question"] = rec["question"]
    except Exception:
        pass
    # text_presence and attribution are NOT in the raw log, and the creative branch needs
    # `attribution` for its behavioural delivery proof.
    er.update({k: v for k, v in (bench_meta.get(design_id) or {}).items()
               if v is not None})
    return er

def load_bench_meta():
    out = {}
    for d in (load_artifact(P, T, "bench_designs") or []):
        did = d.get("design_id")
        if did:
            out[did] = {k: d.get(k) for k in
                        ("text_presence", "attribution", "n_visual_arms", "source",
                         "isolates", "hypothesis", "pillar", "mechanism", "horizon",
                         "prediction", "primary_metric")}
    return out

# ---------------------------------------------------------------- enrichment
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

def control_profile(design_id):
    """Was the CONTROL arm generic brand copy? If so, any lift measures crossing a
    substantiveness threshold rather than the nominal isolated variable."""
    try:
        import re as _re
        from experiment.variant_gen import QUANT_RX
        v = load_artifact(P, T, "experiment_variants") or {}
        recs = v.get("experiments") if isinstance(v, dict) else v
        rec = next((r for r in (recs or []) if r.get("design_id") == design_id), {})
        ctrl = rec.get("variant_A") or (rec.get("arms") or {}).get("A") or ""
        ctrl = ctrl if isinstance(ctrl, str) else (ctrl or {}).get("text", "")
        if not ctrl:
            return None
        return {"words": len(ctrl.split()),
                "quantified": len(QUANT_RX.findall(ctrl)),
                "proper_nouns": len(_re.findall(r"\b[A-Z][a-z]+\s?\d*\b", ctrl)),
                "generic_phrases": sum(
                    1 for ph in ("committed to", "dedicated to", "high-quality",
                                 "comprehensive", "designed to support",
                                 "focusing on", "understands the importance")
                    if ph in ctrl.lower())}
    except Exception:
        return None

# ---------------------------------------------------------------- report
def main():
    df, n_pilots = load_rows()
    if df.empty:
        print(f"No analysable runs in {RAW}"
              + (f" ({n_pilots} pilot row(s) excluded)" if n_pilots else ""))
        return
    print(f"\n{len(df)} rows · {df.design_id.nunique()} design(s)"
          + (f" · {n_pilots} pilot row(s) excluded (single-arm baselines, not experiments)"
             if n_pilots else ""))

    bench_meta = load_bench_meta()
    pairs = sorted(set(zip(df.design_id, df["mode"])))
    exp_results = []
    for design_id, mode in pairs:
        er = build_result(df, design_id, mode, bench_meta)
        if er:
            exp_results.append(er)

    # THE aggregator. Not a copy of it.
    findings = assemble_findings(exp_results, T, min_lift=5.0, raw_path=RAW)

    by_id = {(er["design_id"], er["mode"]): er for er in exp_results}
    report = []
    for f in findings:
        er = by_id.get((f["design_id"], f.get("mode")))
        if er:
            f["rate_table"] = rate_table(er)
            f["n_rows"] = er["n_rows"]
            f["question"] = f.get("question") or er.get("question")
            cp = control_profile(f["design_id"])
            if cp:
                f["control_profile"] = cp
                if cp["generic_phrases"] >= 3 and cp["quantified"] == 0:
                    f.setdefault("interpretation_limits", []).append(
                        f"The CONTROL arm is generic brand copy ({cp['generic_phrases']} "
                        f"generic phrases, 0 quantified values). Any lift measures "
                        f"crossing the SUBSTANTIVENESS threshold, not the nominal "
                        f"isolated variable.")
        sig = sum(1 for r in (f.get("per_engine") or []) if r.get("significant"))
        report.append({
            "design": f["design_id"], "mode": f.get("mode"),
            "template": f.get("experiment_id"),
            "rows": f.get("n_rows"), "runs/cell": f.get("runs_per_cell"),
            "cells": f.get("n_cells"),
            "verdict": f["verdict"], "evidence": f.get("evidence_class"),
            "lift": f.get("pooled_lift_pp"), "unit": f.get("lift_unit", "pp"),
            "sig": sig,
            "best_element": f.get("best_element") or f.get("winning_arm") or ""})
    print(pd.DataFrame(report).to_string(index=False))
    print("\n  " + " · ".join(f"{k}: {v}" for k, v in
                              sorted(Counter(f.get("evidence_class")
                                             for f in findings).items())))

    print("\n--- detail per design ---")
    for f in findings:
        print(f"\n### {f['design_id']} ({f.get('mode')}) — {f['verdict']} · "
              f"{f.get('evidence_class')} · {f.get('pooled_lift_pp')}"
              f"{f.get('lift_unit', 'pp')} · {f.get('runs_per_cell')} runs/cell "
              f"across {f.get('n_cells')} cell(s)")
        if f.get("question"):
            print(f"    q: {f['question']}")

        if f.get("bakeoff_summary"):
            bs = f["bakeoff_summary"]
            print(f"    bake-off [{bs.get('verdict')}] winner={bs.get('winner')} "
                  f"spread={bs.get('gap_spread_pp')}pp")
            print(f"    {bs.get('decision')}")
            for r in (f.get("bakeoff") or []):
                mde = (f"≥{r['mde_pp']}pp" if r.get("mde_pp") is not None
                       else r.get("mde_note", "undefined"))
                print(f"      {r['engine']} · {r['metric']} · {r['verdict']} · "
                      f"omnibus {r['omnibus_test']} p={r['omnibus_p']} · MDE {mde}")
                for x in r["ranking"]:
                    print(f"        {x['rank']}. {x['arm']} {x['rate']:.1%} "
                          f"[{x['ci_lo']:.1%}–{x['ci_hi']:.1%}] n={x['n']}")

        if f.get("element_attribution"):
            print(f"    elements: {f['element_attribution']}  "
                  f"best={f.get('best_element')}")

        rt = pd.DataFrame(f.get("rate_table") or [])
        if not rt.empty and not f.get("bakeoff"):
            print(rt.pivot_table(index=["engine", "arm"], columns="metric",
                                 values=["rate", "n", "ci_width_pp"]).to_string())

        vv = f.get("visual_validity")
        if vv:
            print(f"    🖼  payload {vv['verdict']}: {vv['note']}")

        for w in (f.get("variance_warnings") or []):
            print(f"    ⚠ {w}")
        for lim in (f.get("interpretation_limits") or []):
            print(f"    📌 {lim}")

        for r in (f.get("rank_analysis") or []):
            if not r.get("significant"):
                continue
            print(f"    🎯 rank {r['engine']}/{r['arm']}: "
                  f"{r.get('A_mean_rank')}→{r.get('B_mean_rank')} "
                  f"(top-1 all runs {r.get('A_top1_uncond_pct')}%→"
                  f"{r.get('B_top1_uncond_pct')}%) p_fdr={r.get('p_fdr')}")
            if r.get("conditioning_warning"):
                print(f"        ⚠ {r['conditioning_warning']}")

    if SAVE:
        # Refuse to overwrite a richer saved set with a poorer one. A bench finding
        # carries `bakeoff`; losing it is silent and unrecoverable without a re-run.
        prior = load_artifact(P, T, "validated_findings") or []
        lost = [x["design_id"] for x in prior
                if x.get("bakeoff") and not any(
                    y["design_id"] == x["design_id"] and y.get("bakeoff")
                    for y in findings)]
        if lost:
            print(f"\n✗ REFUSING TO SAVE: {lost} currently hold bake-off data that this "
                  f"re-analysis did not reproduce. Saving would delete it. Check that "
                  f"bench_designs exists and carries `attribution`.")
            return
        save_artifact(P, T, "validated_findings", findings,
                      {"source": "reanalyse.py", "note": "rebuilt from raw log",
                       "pilots_excluded": n_pilots})
        act = {"validated", "validated_mention", "engine-specific",
               "engine-specific_mention", "validated_rank"}
        save_artifact(P, T, "validated_findings_only",
                      [f for f in findings if f["verdict"] in act],
                      {"source": "reanalyse.py"})
        print(f"\nSaved {len(findings)} finding(s) → validated_findings")

if __name__ == "__main__":
    main()