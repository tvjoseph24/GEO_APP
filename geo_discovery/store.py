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

def _row_mentioned(r, target_brand=None):
    """Legacy-safe mention lookup: pre-Batch-1 JSONL has no top-level `mentioned`."""
    if r.get("mentioned") is not None:
        return bool(r["mentioned"])
    bm = r.get("brand_mentions") or {}
    if target_brand and target_brand in bm:
        return bool(bm[target_brand].get("mentioned"))
    if target_brand:                       # case/accent-insensitive fallback
        try:
            from metrics_spec import fold
            tf = fold(target_brand)
            for k, v in bm.items():
                if fold(k) == tf:
                    return bool(v.get("mentioned"))
        except Exception:
            pass
    return None                            # unknown — don't claim absence

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
             "competitive_set", "kpi_targets",
             "delta_scorecard", "gap_register_actionable", "source_hitlist",
             "validated_findings_only", "validated_findings", "experiment_variants",
             "geo_strategy", "roadmap", "content_briefs", "ingested_sources",
             "benchmark_metrics", "competitor_context", "audit_manifest",
             "extraction_failures","gap_hypotheses", "gap_clusters","geo_strategy", "roadmap", "content_briefs", "ingested_sources",
             "benchmark_metrics", "competitor_context", "audit_manifest",
             "extraction_failures", "gap_hypotheses", "gap_clusters",
             "substantiation_backlog", "action_register","report_edits"]           # Optimise
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

def show_audit_samples(raw_path, n=3, key="samp", stratify=("engine", "intent"),
                       only_absent=False, seed=None, target_brand=None,
                       nested=False):
    """Stratified RANDOM sampling (reservoir, one pass, O(n) memory).

    nested=True renders rows as bordered containers instead of expanders, so this can
    be called from INSIDE an st.expander without tripping Streamlit's nesting rule.
    """
    import json as _json
    import random
    from pathlib import Path as _P
    import streamlit as st
    if not _P(raw_path).exists():
        return
    ss = st.session_state
    if seed is None:
        seed = ss.setdefault(f"{key}_seed", random.randrange(10 ** 6))
    rng = random.Random(seed)

    buckets, counts, total, unknown_mention = {}, {}, 0, 0
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = _json.loads(line)
            except Exception:
                continue
            if not r.get("success"):
                continue
            m = _row_mentioned(r, target_brand)
            if m is None:
                unknown_mention += 1
            if only_absent and m is not False:
                continue                       # keep ONLY known-absent rows
            total += 1
            k = tuple(r.get(s) for s in stratify)
            counts[k] = counts.get(k, 0) + 1
            b = buckets.setdefault(k, [])
            if len(b) < n:
                b.append(r)
            else:
                j = rng.randrange(counts[k])
                if j < n:
                    b[j] = r

    if not buckets:
        if only_absent and unknown_mention:
            st.caption("Cannot identify absent runs in this legacy audit file "
                       "(no `mentioned` field). Re-run the audit to populate it.")
        else:
            st.caption("No matching audit runs to sample.")
        return

    rows = [rng.choice(b) for b in buckets.values()]
    rng.shuffle(rows)
    rows = rows[: max(n, 1) * 3]

    c1, c2 = st.columns([4, 1])
    label = "absent-brand " if only_absent else ""
    c1.caption(f"🔍 {len(rows)} sampled {label}response(s) of {total} "
               f"across {len(buckets)} strata")
    if c2.button("🎲 Reshuffle", key=f"{key}_re"):
        ss[f"{key}_seed"] = random.randrange(10 ** 6)
        st.rerun()

    for i, r in enumerate(rows):
        m = _row_mentioned(r, target_brand)
        flag = "✅" if m else ("❌" if m is False else "❔")
        pol = r.get("polarity") or ""
        badge = {"neg": "⚠️", "pos": "👍", "neu": ""}.get(pol, "")
        title = (f"{flag}{badge} [{r.get('engine')}] [{r.get('intent')}] "
                 f"{(r.get('query') or '')[:60]}")

        if nested:                              # container — safe inside an expander
            box = st.container(border=True)
            box.markdown(f"**{title}**")
        else:
            box = st.expander(title)

        with box:
            st.write((r.get("response_text", "") or "")[:1200] + "…")
            st.caption(
                f"run={r.get('run_index')} · outcome={r.get('outcome')} · "
                f"rank={r.get('rank')} · web={r.get('web_on')} · "
                f"refusal={r.get('refusal')} · advisory={r.get('advisory_referral')} · "
                f"sources={r.get('num_sources', 0)} · imgs={r.get('num_images', 0)}")
            if r.get("sources"):
                st.caption("Sources: " + ", ".join(
                    f"{s.get('domain', '?')}({s.get('source_type', '?')})"
                    for s in r["sources"][:5]))

def save_blob(parent_brand, target_brand, relpath, text):
    """Persist FULL raw text (e.g. an unparseable LLM completion) outside the JSON
    artifact. Artifacts keep a pointer + short preview; nothing is lost."""
    d = project_dir(parent_brand, target_brand) / "blobs"
    p = (d / relpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(text or ""), encoding="utf-8")
    return str(p.relative_to(project_dir(parent_brand, target_brand)))

def load_blob(parent_brand, target_brand, relpath):
    p = project_dir(parent_brand, target_brand) / relpath
    return p.read_text(encoding="utf-8") if p.exists() else None

def save_cache(parent_brand, target_brand, name, obj):
    import json
    d = project_dir(parent_brand, target_brand) / "cache"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(_clean(obj), ensure_ascii=False))

def load_cache(parent_brand, target_brand, name):
    import json
    p = project_dir(parent_brand, target_brand) / "cache" / f"{name}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}