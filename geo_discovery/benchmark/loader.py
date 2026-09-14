"""Load audit JSONL → tidy per-run DataFrame (portfolio-aware)."""
import json
import pandas as pd
from pathlib import Path
from store import project_dir, load_artifact

def load_audit(parent_brand, target_brand):
    raw = project_dir(parent_brand, target_brand) / "audit_raw.jsonl"
    if not raw.exists():
        raise FileNotFoundError(f"No audit at {raw}")
    rows = [json.loads(l) for l in open(raw, encoding="utf-8")]
    df = pd.DataFrame([r for r in rows if r.get("success")])
    if df.empty:
        raise ValueError("No successful audit runs")
    '''
    # flatten key fields
    df["mentioned"] = df["brand_mentions"].apply(
        lambda b: b.get(target_brand, {}).get("mentioned", False))
    df["first_index"] = df["brand_mentions"].apply(
        lambda b: b.get(target_brand, {}).get("first_index", -1))
    df["is_first"] = df.apply(lambda r: _is_first(r, target_brand), axis=1)
    df["refusal"] = df["outcome"] == "refusal"
    df["sentiment"] = df["classification"].apply(
        lambda c: (c or {}).get("sentiment_toward_target") if isinstance(c, dict) else None)
    '''
    # prefer flat metrics_spec fields (written by audit_runner v2); fall back for
    # legacy JSONL captured before Batch 1
    if "mentioned" not in df.columns:
        df["mentioned"] = df["brand_mentions"].apply(
            lambda b: (b or {}).get(target_brand, {}).get("mentioned", False))
    if "first_index" not in df.columns:
        df["first_index"] = df["brand_mentions"].apply(
            lambda b: (b or {}).get(target_brand, {}).get("first_index", -1))
    if "is_first" not in df.columns:
        df["is_first"] = df.apply(lambda r: _is_first(r, target_brand), axis=1)
    if "refusal" not in df.columns:
        df["refusal"] = df["outcome"] == "refusal"
    for col, default in (("polarity", None), ("advisory_referral", False),
                         ("rank", None), ("refusal_kind", "none")):
        if col not in df.columns:
            df[col] = default
    df["sentiment"] = df["classification"].apply(
        lambda c: (c or {}).get("sentiment_toward_target")
        if isinstance(c, dict) else None)
    return df


def _is_first(row, target):
    bm = row.get("brand_mentions", {}) or {}
    present = [(b, v.get("first_index", -1)) for b, v in bm.items()
               if v.get("first_index", -1) >= 0]
    if not present: return False
    return min(present, key=lambda x: x[1])[0] == target

def load_context(parent_brand, target_brand):
    """Load Ground-Truth artifacts for benchmarking."""
    return {
        "claims": load_artifact(parent_brand, target_brand, "claims_register") or [],
        "query_master": load_artifact(parent_brand, target_brand, "query_master") or [],
        "competitors": load_artifact(parent_brand, target_brand, "competitive_set") or [],
        "kpis": load_artifact(parent_brand, target_brand, "kpi_targets") or [],
    }