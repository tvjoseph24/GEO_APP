"""diagnostics.py — persist what Benchmark actually computed. W-30, W-31, W-32.

Benchmark produces far more than the Delta Scorecard carries. Span-level judgements with
verbatim quotes are the evidence trail behind every claim verdict; per-query winnability
is the basis of every visibility gap; per-span richness is what the specificity
experiments are sized against. All of it existed in session state only, so a refresh
discarded it and a re-run cost a full judge cycle to rebuild.

Three artifacts rather than one, because they have different sizes and different readers:

  benchmark_diagnostics   the analytical trail. Large, read by QA and the report appendix.
  prepared_claims         the register veracity ACTUALLY ran against. Small, and the
                          correct denominator for every coverage figure.
  benchmark_manifest      hashes and versions. Tiny, read on every load to detect drift.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone

DIAGNOSTICS = "benchmark_diagnostics"
PREPARED_CLAIMS = "prepared_claims"
MANIFEST = "benchmark_manifest"

# Bumped when the shape of what we persist changes, so a stale artifact is detectable
# rather than silently mixed with a fresh one.
DIAGNOSTICS_VERSION = 1

def _df(x, limit=None):
    """DataFrame, list or None → list of dicts. Benchmark mixes all three."""
    if x is None:
        return []
    if hasattr(x, "to_dict"):
        rows = x.reset_index().to_dict("records") if x.index.name else x.to_dict("records")
    else:
        rows = list(x)
    return rows[:limit] if limit else rows

def content_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]

# ---------------------------------------------------------------- prepared claims
def save_prepared_claims(parent, target, kept, prep_report, raw_register):
    """The register veracity ran against, with the full audit of how it was derived.

    THE denominator problem: prepare_claims splits and merges, then the result was lost.
    coverage_declaration computed claims_assessed_pct against the RAW register, so with
    splitting the report could print more than 100% assessed — and with heavy dropping it
    reported a coverage figure for claims that were never in scope. On one engagement 59
    of 112 claims were discarded before veracity ran and nothing downstream knew.
    """
    from store import save_artifact
    payload = {
        "version": DIAGNOSTICS_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "claims": kept,
        "n_prepared": len(kept),
        "n_raw": len(raw_register or []),
        "preparation": prep_report,
        "raw_hash": content_hash([c.get("claim") for c in (raw_register or [])]),
        "prepared_hash": content_hash([c.get("claim") for c in kept]),
        # The correct denominator for every coverage figure downstream.
        "denominator_note": (
            f"Veracity, gap detection and hypothesis generation all run against these "
            f"{len(kept)} prepared claim(s), not the {len(raw_register or [])} raw "
            f"register entries. Coverage must be reported against the prepared count: "
            f"{prep_report.get('n_split', 0)} entry(ies) were split into separate claims "
            f"and {prep_report.get('n_dropped', 0)} were excluded as unjudgeable, so the "
            f"raw count is neither an upper nor a lower bound."),
    }
    save_artifact(parent, target, PREPARED_CLAIMS, payload,
                  {"n_prepared": len(kept), "n_raw": len(raw_register or [])})
    return payload

def load_prepared_claims(parent, target):
    from store import load_artifact
    return load_artifact(parent, target, PREPARED_CLAIMS)

def claims_denominator(parent, target, raw_register=None):
    """The number every coverage percentage must divide by. Falls back to the raw count
    with the fallback declared, rather than silently using the wrong one."""
    pc = load_prepared_claims(parent, target)
    if pc and pc.get("n_prepared"):
        return {"denominator": pc["n_prepared"], "basis": "prepared",
                "n_raw": pc.get("n_raw"),
                "note": pc.get("denominator_note")}
    n = len(raw_register or [])
    return {"denominator": n, "basis": "raw_register_fallback", "n_raw": n,
            "note": ("The prepared register is not persisted for this engagement, so "
                     "coverage is reported against the RAW register. Claim preparation "
                     "splits and merges entries, so this figure may be wrong in either "
                     "direction. Re-run Benchmark to persist it.")}

# ---------------------------------------------------------------- diagnostics
def build_diagnostics(*, veracity=None, winnability=None, richness_detail=None,
                      richness_agg=None, reason_gap=None, lexicon=None,
                      cross_engine=None, consistency=None, social=None,
                      hitlist=None, qa=None, claim_prep=None, judge_model=None,
                      span_limit=4000, winnability_limit=400):
    """The analytical trail. Capped, because span judgements can run to tens of thousands
    of rows and an artifact nobody can open is not persistence."""
    v = veracity or {}
    judgements = _df(v.get("judgements"), span_limit)
    halluc = _df(v.get("hallucinations"), span_limit)
    return {
        "version": DIAGNOSTICS_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "judge_model": judge_model,

        # ---- the evidence trail. Every claim verdict traces to a verbatim quote ----
        "span_judgements": judgements,
        "span_judgements_truncated": len(_df(v.get("judgements"))) > len(judgements),
        "n_span_judgements": len(_df(v.get("judgements"))),
        "reverse_scan": halluc,
        "n_reverse_scan": len(_df(v.get("hallucinations"))),
        "per_claim": _df(v.get("per_claim")),

        # ---- judge reliability, which could not be audited after the session ----
        "judge": {k: v.get(k) for k in
                  ("method", "span_count", "spans_by_engine", "spans_by_intent",
                   "claims_considered", "claims_with_evidence", "claims_absent",
                   "claims_judged", "judge_yield", "judge_failure_count",
                   "judge_failures", "quote_verification_failures",
                   "quote_match_breakdown", "rescued_retrievals",
                   "retrieval_score_median", "calibration", "engine_scores",
                   "halluc_counts")},

        # ---- the basis of every visibility gap ----
        "winnability": _df(winnability, winnability_limit),

        # ---- what the specificity experiments are sized against ----
        "richness_per_span": _df(richness_detail, span_limit),
        "richness_aggregate": _df(richness_agg),
        "reason_gap": _df(reason_gap),
        "category_lexicon": lexicon or {},

        "cross_engine": cross_engine or {},
        "consistency": consistency or {},
        "social": social or {},
        "hitlist": _df(hitlist),
        "qa": qa or [],
        "claim_preparation": claim_prep or {},
    }

def save_diagnostics(parent, target, diagnostics):
    from store import save_artifact
    save_artifact(parent, target, DIAGNOSTICS, diagnostics,
                  {"version": DIAGNOSTICS_VERSION,
                   "n_span_judgements": diagnostics.get("n_span_judgements"),
                   "judge_model": diagnostics.get("judge_model")})
    return diagnostics

def load_diagnostics(parent, target):
    from store import load_artifact
    return load_artifact(parent, target, DIAGNOSTICS)

def diagnostics_summary(d):
    """One screen. For the QA gate and the report appendix."""
    if not d:
        return {"persisted": False,
                "note": "Benchmark diagnostics were not persisted for this engagement. "
                        "Span-level evidence, winnability and richness detail exist only "
                        "in the session that produced them."}
    j = d.get("judge") or {}
    return {
        "persisted": True, "version": d.get("version"),
        "judge_model": d.get("judge_model"),
        "span_judgements": d.get("n_span_judgements"),
        "truncated": d.get("span_judgements_truncated"),
        "reverse_scan_assertions": d.get("n_reverse_scan"),
        "claims_judged": j.get("claims_judged"),
        "judge_yield": j.get("judge_yield"),
        "quote_failures": j.get("quote_verification_failures"),
        "calibration": (j.get("calibration") or {}).get("agreement"),
        "winnability_rows": len(d.get("winnability") or []),
        "richness_spans": len(d.get("richness_per_span") or []),
    }

# ---------------------------------------------------------------- manifest
def save_manifest(parent, target, *, df_rows, engines, intents, judge_model,
                  prepared_claims, source_pack=None, config=None):
    """Hashes and versions, read on every load to detect drift.

    The specific failure: an audit backfilled after Benchmark ran leaves the scorecard
    describing a corpus that no longer exists. Nothing currently notices.
    """
    from store import save_artifact
    from metrics_spec import SCHEMA_VERSION
    payload = {
        "version": DIAGNOSTICS_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_spec_version": SCHEMA_VERSION,
        "n_audit_rows": int(df_rows),
        "engines": sorted(engines or []),
        "intents": sorted(intents or []),
        "judge_model": judge_model,
        "prepared_claims_hash": content_hash(
            [c.get("claim") for c in (prepared_claims or [])]),
        "n_prepared_claims": len(prepared_claims or []),
        "source_pack_version": (source_pack or {}).get("pack_version"),
        "source_pack_domains": (source_pack or {}).get("n_domains"),
        "config": config or {},
    }
    save_artifact(parent, target, MANIFEST, payload, {"version": DIAGNOSTICS_VERSION})
    return payload

def load_manifest(parent, target):
    from store import load_artifact
    return load_artifact(parent, target, MANIFEST)

def drift_check(parent, target, current_rows=None, current_pack=None):
    """Has the corpus moved beneath the saved Benchmark?"""
    m = load_manifest(parent, target)
    if not m:
        return {"has_manifest": False,
                "note": "No benchmark manifest, so drift cannot be detected. The saved "
                        "scorecard may describe a corpus that has since been backfilled."}
    out = {"has_manifest": True, "benchmarked_rows": m.get("n_audit_rows"),
            "benchmarked_utc": m.get("generated_utc"),
            "source_pack_version": m.get("source_pack_version"), "drift": []}
    if current_rows is not None and m.get("n_audit_rows") != current_rows:
        out["drift"].append(
            f"The audit log held {m['n_audit_rows']:,} successful run(s) when Benchmark "
            f"ran and holds {current_rows:,} now. The saved scorecard describes a "
            f"different corpus — re-run Benchmark.")
    _cp = (current_pack or {}).get("pack_version")
    if _cp != m.get("source_pack_version"):
        out["drift"].append(
            f"The source pack has changed since Benchmark ran (v{m.get('source_pack_version')} "
            f"→ v{_cp}). Citation provenance in the saved scorecard predates the current "
            f"classification — backfill, then re-run Benchmark.")
    from metrics_spec import SCHEMA_VERSION
    if m.get("metrics_spec_version") != SCHEMA_VERSION:
        out["drift"].append(
            f"Measurement schema v{m.get('metrics_spec_version')} when Benchmark ran, "
            f"v{SCHEMA_VERSION} now. Metric definitions have changed — re-run Benchmark "
            f"rather than comparing across versions.")
    out["has_drift"] = bool(out["drift"])
    return out