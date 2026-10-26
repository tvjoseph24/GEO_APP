"""reconcile.py — rebuild findings from the append-only raw log. No LLM calls, no spend.

The raw log is the AUTHORITY; `validated_findings` is a cache. Emit writes the session's
findings, so a second batch previously replaced the first and every phase downstream read
a fraction of the evidence. This reconstructs the complete set from disk.

Wraps the same logic as reanalyse.py, which is terminal-only. The raw log being the
authority is useless if the only way to consult it is a shell.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import MODEL_REGISTRY
from experiment.results import assemble_findings
from store import load_artifact, project_dir

PILOT_PREFIX = "_pilot_"

def raw_path(parent, target):
    return project_dir(parent, target) / "experiment_raw.jsonl"

def load_rows(parent, target, keep_pilots=False):
    """Pilots excluded: a pilot is a single-arm baseline, not an experiment. Analysed as
    one it yields verdict 'lost' with no comparison, which then reads as a null."""
    p = raw_path(parent, target)
    if not p.exists():
        return pd.DataFrame(), 0
    rows, pilots = [], 0
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            r["arm"] = str(r.get("arm", "")).replace("variant_", "")
            r["mode"] = r.get("mode") or ("progressive"
                                          if r["arm"] in ("C", "D", "E") else "ab")
            if (r["mode"] == "pilot"
                    or str(r.get("design_id", "")).startswith(PILOT_PREFIX)):
                pilots += 1
                if not keep_pilots:
                    continue
            rows.append(r)
    return pd.DataFrame(rows), pilots

def coverage(parent, target):
    """What is in the raw log that is NOT in the saved artifact. The drift check."""
    df, pilots = load_rows(parent, target)
    if df.empty:
        return {"raw_exists": raw_path(parent, target).exists(),
                "in_raw": [], "in_artifact": [], "missing": [], "orphaned": [],
                "pilot_rows": pilots, "n_raw_rows": 0}
    in_raw = sorted({(str(d), str(m)) for d, m in
                     zip(df["design_id"], df["mode"]) if d})
    saved = load_artifact(parent, target, "validated_findings") or []
    in_art = sorted({(str(f.get("design_id")), str(f.get("mode") or "ab"))
                     for f in saved})
    return {"raw_exists": True, "n_raw_rows": int(len(df)), "pilot_rows": pilots,
            "in_raw": in_raw, "in_artifact": in_art,
            "missing": [k for k in in_raw if k not in in_art],
            "orphaned": [k for k in in_art if k not in in_raw]}

def _visual_block(cell, model):
    """Rebuild the per-arm media block. `image_attached` and `image_tokens` are recorded
    per run, so payload delivery is recoverable without re-running anything."""
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

def _bench_meta(parent, target):
    out = {}
    for d in (load_artifact(parent, target, "bench_designs") or []):
        if d.get("design_id"):
            out[d["design_id"]] = {k: d.get(k) for k in
                                  ("text_presence", "attribution", "n_visual_arms",
                                   "source", "isolates", "hypothesis", "pillar",
                                   "mechanism", "horizon", "prediction",
                                   "primary_metric")}
    return out

def _first(g, col):
    """First non-null value of an optional column. The raw log is append-only and has
    gained fields over time, so an older row may simply lack one — and a bare g[col]
    raises KeyError on the whole reconciliation, which is the recovery path."""
    if col not in g.columns:
        return None
    s = g[col].dropna()
    return s.iloc[0] if len(s) else None

def build_results(parent, target, df, bench_meta):
    out = []
    for design_id, mode in sorted(set(zip(df.design_id, df["mode"]))):
        g = df[(df.design_id == design_id) & (df["mode"] == mode)]
        if g.empty:
            continue
        engines = {}
        for (model, arm), cell in g.groupby(["model", "arm"]):
            valid = cell[cell["error"].isna()] if "error" in cell else cell
            stats = {"n": int(len(valid)), "n_attempted": int(len(cell)),
                     "n_errors": int(len(cell) - len(valid)),
                     "mentioned": int(valid.get("mentioned", pd.Series(dtype=bool))
                                      .fillna(False).sum()),
                     "is_first": int(valid.get("is_first", pd.Series(dtype=bool))
                                     .fillna(False).sum()),
                     "refusals": int(valid.get("refusal", pd.Series(dtype=bool))
                                     .fillna(False).sum())}
            vb = _visual_block(cell, model)
            if vb:
                stats["visual"] = vb
            engines.setdefault(model, {})[arm] = stats

        er = {"design_id": design_id, "mode": mode, "engines": engines,
              "n_rows": int(len(g)),
              "question": _first(g, "question"),
              "comp_mode": _first(g, "comp_mode"),
              "position_mode": _first(g, "position"),
              "experiment_id": ("E1" if mode == "progressive" else
                                "VBENCH" if mode == "creative" else "E0")}
        try:
            v = load_artifact(parent, target, "experiment_variants") or {}
            rec = next((x for x in (v.get("experiments") or [])
                        if (x.get("design_id") or x.get("gap_id")) == design_id), {})
            if rec.get("experiment_id"):
                er["experiment_id"] = rec["experiment_id"]
            if rec.get("question") and not er["question"]:
                er["question"] = rec["question"]
        except Exception:
            pass
        # text_presence and attribution are not in the raw log, and the creative branch
        # needs `attribution` for its behavioural delivery proof.
        er.update({k: v for k, v in (bench_meta.get(design_id) or {}).items()
                   if v is not None})
        out.append(er)
    return out

def reconcile(parent, target, save=True, min_lift=5.0):
    """Rebuild the complete finding set from the raw log. Returns (findings, report)."""
    df, pilots = load_rows(parent, target)
    if df.empty:
        return [], {"error": "no analysable runs in the raw log",
                    "pilot_rows": pilots}
    ers = build_results(parent, target, df, _bench_meta(parent, target))
    findings = assemble_findings(ers, target, min_lift=min_lift,
                                 raw_path=str(raw_path(parent, target)))

    prior = load_artifact(parent, target, "validated_findings") or []
    # Refuse to drop bake-off data this pass could not reproduce — losing it is silent
    # and unrecoverable without a re-run.
    lost = [f["design_id"] for f in prior
            if f.get("bakeoff") and not any(
                g.get("design_id") == f["design_id"] and g.get("bakeoff")
                for g in findings)]
    report = {"n_designs": len(ers), "n_findings": len(findings),
              "n_raw_rows": int(len(df)), "pilot_rows": pilots,
              "prior_findings": len(prior), "bakeoff_at_risk": lost,
              "by_evidence": {}, "saved": False}
    from collections import Counter
    report["by_evidence"] = dict(Counter(f.get("evidence_class") for f in findings))

    if lost:
        report["error"] = (
            f"Refusing to save: {lost} currently hold bake-off data this reconstruction "
            f"did not reproduce. Saving would delete it. Check that `bench_designs` "
            f"exists and carries `attribution`.")
        return findings, report

    if save:
        from store import save_artifact
        save_artifact(parent, target, "experiment_results", ers,
                      {"source": "reconcile", "note": "rebuilt from raw log"})
        save_artifact(parent, target, "validated_findings", findings,
                      {"source": "reconcile", "note": "rebuilt from raw log",
                       "pilots_excluded": pilots})
        from experiment.results import ACTIONABLE_VERDICTS
        save_artifact(parent, target, "validated_findings_only",
                      [f for f in findings if f["verdict"] in ACTIONABLE_VERDICTS],
                      {"source": "reconcile"})
        report["saved"] = True
    return findings, report