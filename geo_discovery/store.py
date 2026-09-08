"""Versioned, timestamped artifact persistence — now portfolio-structured:
   geo_projects/{parent}/{target_brand}/artifacts/..."""
import json
from datetime import datetime, timezone
from pathlib import Path
from config import STORE_ROOT
import numpy as np

import unicodedata

def _slug(s):
    """Consistent, accent-insensitive folder name.
    'L'Oréal', 'LOreal', 'L'Oreal' → all become 'loreal'."""
    s = str(s)
    # strip accents: é→e, ñ→n, etc.
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    # lowercase, keep only alnum
    s = "".join(c.lower() if c.isalnum() else "_" for c in s)
    # collapse repeats, trim
    while "__" in s: s = s.replace("__", "_")
    return s.strip("_")

def _clean(obj):
    """Recursively convert numpy/pandas types → native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if hasattr(obj, "item"):        # catch-all for numpy scalars
        try: return obj.item()
        except Exception: return obj
    return obj

def project_dir(parent_brand, target_brand):
    """Portfolio path: parent → target brand → artifacts."""
    d = STORE_ROOT / _slug(parent_brand) / _slug(target_brand)
    (d / "artifacts").mkdir(parents=True, exist_ok=True)
    (d / "baselines").mkdir(parents=True, exist_ok=True)
    return d

def save_artifact(parent_brand, target_brand, name, obj, models_used=None,
                  scope=None):
    d = project_dir(parent_brand, target_brand) / "artifacts"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    payload = {"artifact": name, "parent_brand": parent_brand,
               "target_brand": target_brand, "scope": scope or "product",
               "timestamp_utc": ts, "models_used": models_used or {}, "data": obj}
    path = d / f"{name}_{ts}.json"
    path.write_text(json.dumps(_clean(payload), ensure_ascii=False, indent=2))
    (d / f"{name}_LATEST.json").write_text(json.dumps(_clean(payload), ensure_ascii=False, indent=2))
    return str(path)

def save_audit_log(parent_brand, target_brand, records):
    d = project_dir(parent_brand, target_brand)
    (d / "audit_log.json").write_text(json.dumps(records, ensure_ascii=False, indent=2))

def load_artifact(parent_brand, target_brand, name):
    p = project_dir(parent_brand, target_brand) / "artifacts" / f"{name}_LATEST.json"
    return json.loads(p.read_text())["data"] if p.exists() else None

def list_brands(parent_brand):
    """List target brands already under a parent (seeds Phase-II portfolio view)."""
    pd = STORE_ROOT / _slug(parent_brand)
    if not pd.exists():
        return []
    return [d.name for d in pd.iterdir() if d.is_dir()]

def list_parents():
    """List all parent/company folders in the store."""
    if not STORE_ROOT.exists():
        return []
    return sorted([d.name for d in STORE_ROOT.iterdir()
                   if d.is_dir() and not d.name.startswith(".")])

def load_all_artifacts(parent_brand, target_brand):
    """Load every saved artifact for a brand → dict. For display recovery (read-only)."""
    names = ["positioning_layers", "claims_register", "query_master",
             "competitive_set", "kpi_targets",                       # Discover
             "delta_scorecard", "gap_register_actionable", "source_hitlist",  # Benchmark
             "validated_findings_only", "experiment_variants",       # Experiment
             "geo_strategy", "roadmap", "content_briefs","ingested_sources",
             "benchmark_metrics","competitor_context"]            # Optimise
    loaded = {}
    for n in names:
        data = load_artifact(parent_brand, target_brand, n)
        if data is not None:
            loaded[n] = data
    return loaded

def save_cost_log(parent_brand, target_brand, meter):
    """Persist meter events so cost accumulates across sessions."""
    d = project_dir(parent_brand, target_brand)
    (d / "cost_log.json").write_text(json.dumps({
        "events": meter.events,
        "total": meter.total_cost(),
        "calls": meter.calls}, ensure_ascii=False, indent=2))

def load_cost_log(parent_brand, target_brand):
    """Load prior cost events (list) for restoration."""
    p = project_dir(parent_brand, target_brand) / "cost_log.json"
    if p.exists():
        return json.loads(p.read_text()).get("events", [])
    return []

def show_audit_samples(raw_path, n=3):
    import json, streamlit as st
    from pathlib import Path
    if not Path(raw_path).exists(): return
    rows = []
    for l in open(raw_path, encoding="utf-8"):
        try:
            r = json.loads(l)
            if r.get("success"): rows.append(r)
            if len(rows) >= n: break
        except Exception: pass
    if not rows: return
    st.subheader(f"🔍 Sample audit responses ({len(rows)})")
    for r in rows:
        with st.expander(f"[{r.get('engine')}] {r.get('query','')[:70]}"):
            st.write(r.get("response_text","")[:600] + "…")
            st.caption(f"Outcome: {r.get('outcome')} · "
                       f"Sources: {r.get('num_sources',0)} · "
                       f"Web: {r.get('web_on')}")
            if r.get("sources"):
                st.caption("Top sources: " +
                    ", ".join(s.get("domain","?") for s in r["sources"][:4]))