"""Frozen baseline + minimal Tier-A snapshot."""
import json
from datetime import datetime, timezone
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint
import pandas as pd

def compute_snapshot(raw_path, target_brand):
    """Minimal Tier-A metrics for the baseline (full analysis = Module 3)."""
    rows = [json.loads(l) for l in open(raw_path, encoding="utf-8")]
    df = pd.DataFrame([r for r in rows if r.get("success")])
    if df.empty: return {"error": "no successful runs"}

    df["mentioned"] = df["brand_mentions"].apply(
        lambda b: b.get(target_brand, {}).get("mentioned", False))
    df["refusal"] = df["outcome"] == "refusal"

    snap = {}
    for (eng,), g in df.groupby(["engine"]):
        h, n = int(g["mentioned"].sum()), len(g)
        lo, hi = proportion_confint(h, n, alpha=0.05, method="wilson")
        snap[eng] = {"runs": n,
                     "mention_rate": round(h/n, 3),
                     "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                     "refusal_rate": round(g["refusal"].mean(), 3),
                     "avg_sources": round(g["num_sources"].mean(), 2)}
    return snap

def freeze_baseline(parent_brand, target_brand, raw_path, brand_for_metrics,
                    engines, config, meter):
    """Create IMMUTABLE, timestamped baseline in the portfolio path."""
    from store import project_dir      # portfolio-aware path
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    baseline = {
        "parent_brand": parent_brand, "target_brand": target_brand,
        "captured_utc": ts, "LOCKED": True,
        "engines": [{"name": e["name"], "model": e["model"]} for e in engines],
        "config": config,
        "cost_usd": round(meter.total_cost(), 4),
        "snapshot": compute_snapshot(raw_path, brand_for_metrics),
        "raw_path": raw_path,
    }
    out = project_dir(parent_brand, target_brand) / "baselines"   # ← portfolio path
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"baseline_{ts}_LOCKED.json"
    path.write_text(json.dumps(baseline, indent=2))
    return str(path), baseline