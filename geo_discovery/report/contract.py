# report/contract.py
"""Declares every field the report requires, and audits the artifacts against it.

Run BEFORE building the renderer. Any REQUIRED field that is missing must either be
emitted by an upstream phase, or the section it feeds must be declared out-of-scope.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from store import load_artifact
from collections import Counter

@dataclass
class Req:
    section: str
    label: str
    artifact: str
    path: str                  # dotted, [] for list-any
    required: bool = True
    note: str = ""

CONTRACT: list[Req] = [
    # ---- Executive summary ----
    Req("exec", "Headline narrative", "delta_scorecard", "headline"),
    Req("exec", "Retrieval gap", "benchmark_metrics", "mention_by_intent",
        note="Derived: 1 - need mention vs in-context ≈100%"),
    Req("exec", "Strategy narrative", "geo_strategy", "overall_strategy.narrative"),
    Req("exec", "Big bets", "geo_strategy", "overall_strategy.big_bets"),
    Req("exec", "Not doing", "geo_strategy", "overall_strategy.not_doing",
        note="Evidence-based exclusions — required for credibility"),

    # ---- Discovery ----
    Req("discover", "Positioning layers", "positioning_layers", ""),
    Req("discover", "Claims register", "claims_register", ""),
    Req("discover", "Competitive set", "competitive_set", ""),
    Req("discover", "Query master", "query_master", ""),
    Req("discover", "Ingested sources", "ingested_sources", "", required=False),

    # ---- Audit ----
    Req("audit", "Runs / engines", "benchmark_metrics", "n_runs"),
    Req("audit", "Mention by intent", "benchmark_metrics", "mention_by_intent"),
    Req("audit", "First mention by intent", "benchmark_metrics",
        "first_mention_by_intent"),
    Req("audit", "Mention by engine × intent", "benchmark_metrics",
        "mention_by_engine_intent"),
    Req("audit", "Rank when present", "benchmark_metrics",
        "rank_when_present_by_intent"),
    Req("audit", "Refusal by intent", "benchmark_metrics", "refusal_by_intent"),
    Req("audit", "Advisory deferral", "benchmark_metrics", "advisory_rate_by_intent",
        required=False, note="Regulated categories only"),
    Req("audit", "Source-type mix", "benchmark_metrics", "source_mix"),
    Req("audit", "Audit manifest / coverage", "audit_manifest", "", required=False),
    Req("audit", "Top-3 rate", "benchmark_metrics", "top3_by_intent", required=False,
        note="NOT YET COMPUTED — add to metrics_bundle"),
    Req("audit", "Recommendation rate", "benchmark_metrics", "recommendation_by_intent",
        required=False, note="NOT YET COMPUTED — add to metrics_bundle"),

    # ---- Benchmark / delta ----
    Req("delta", "Claim deltas", "delta_scorecard", "claim_deltas"),
    Req("delta", "Risk register", "delta_scorecard", "risk_register"),
    Req("delta", "Engine veracity", "delta_scorecard", "engine_veracity"),
    Req("delta", "Provenance", "delta_scorecard", "provenance"),
    Req("delta", "Framing", "delta_scorecard", "framing"),
    Req("delta", "Reason richness", "delta_scorecard", "reason_richness"),
    Req("delta", "Reason gap", "delta_scorecard", "reason_gap"),
    Req("delta", "Flatness", "delta_scorecard", "flatness"),
    Req("delta", "Leaderboard", "delta_scorecard", "leaderboard"),
    Req("delta", "Winnability", "delta_scorecard", "winnability_summary"),
    Req("delta", "Consistency", "delta_scorecard", "consistency"),
    Req("delta", "Judge calibration", "delta_scorecard", "calibration", required=False),
    Req("delta", "Source hit-list", "source_hitlist", ""),
    Req("delta", "Gap register", "gap_register_actionable", ""),

    # ---- Experiment ----
    Req("experiment", "All findings incl. nulls", "validated_findings", ""),
    Req("experiment", "Tested variants", "experiment_variants", ""),
    Req("experiment", "Substantiation backlog", "substantiation_backlog", "",
        note="Pillar-6 / compliance section"),
    Req("experiment", "Hypotheses", "gap_hypotheses", "", required=False),

    # ---- Strategy ----
    Req("strategy", "Pillars", "geo_strategy", "pillars"),
    Req("strategy", "Engine overlays", "geo_strategy", "engine_overlays"),
    Req("strategy", "Actions", "geo_strategy", "actions"),
    Req("strategy", "Structural truths", "geo_strategy",
        "overall_strategy.structural_truths"),
    Req("strategy", "Critic review", "geo_strategy", "critic_review", required=False),
    Req("strategy", "Content briefs", "content_briefs", ""),

    # ---- Roadmap ----
    Req("roadmap", "Waves", "roadmap", "waves"),
    Req("roadmap", "Measurement contracts", "roadmap", "waves",
        note="Each wave must carry measurement_contract"),
    Req("roadmap", "Projection", "roadmap", "projection", required=False),
    Req("roadmap", "Dependency DAG", "roadmap", "dependency_dag", required=False),
    Req("roadmap", "Action register", "action_register", ""),

    # ---- Methodology ----
    Req("method", "KPI targets", "kpi_targets", ""),
    Req("method", "QA gate", "delta_scorecard", "qa", required=False),
    Req("method", "Frozen baseline", "audit_manifest", "config", required=False),
]

def _dig(obj, path):
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            if not cur:
                return None
            cur = cur[0].get(part) if isinstance(cur[0], dict) else None
        else:
            return None
    return cur

def audit_contract(parent, target):
    cache, rows = {}, []
    for r in CONTRACT:
        if r.artifact not in cache:
            cache[r.artifact] = load_artifact(parent, target, r.artifact)
        art = cache[r.artifact]
        if art is None:
            state, detail = "MISSING_ARTIFACT", "-"
        else:
            val = _dig(art, r.path)
            if val is None:
                state, detail = "MISSING_FIELD", "-"
            elif isinstance(val, (list, dict)) and len(val) == 0:
                state, detail = "EMPTY", "len=0"
            else:
                state = "OK"
                detail = (f"len={len(val)}" if isinstance(val, (list, dict))
                          else str(val)[:40])
        rows.append({"section": r.section, "label": r.label,
                     "artifact": r.artifact, "path": r.path or "(root)",
                     "required": r.required, "state": state, "detail": detail,
                     "note": r.note})
    return rows

def coverage_declaration(parent, target):
    """What was and was not assessed. Renders in report front-matter — a section with no
    data must be declared, never left blank."""
    man = load_artifact(parent, target, "audit_manifest") or {}
    m = load_artifact(parent, target, "benchmark_metrics") or {}
    qm = load_artifact(parent, target, "query_master") or []
    findings = load_artifact(parent, target, "validated_findings") or []

    planned = sorted({q.get("intent") for q in qm if q.get("intent")})
    audited = sorted({r.get("intent") for r in m.get("mention_by_intent", [])})
    cfg = man.get("config") or {}
    if not cfg.get("web_modes"):
        # audit predates the manifest — derive from the raw log
        import json as _j
        from pathlib import Path
        from store import project_dir
        raw = project_dir(parent, target) / "audit_raw.jsonl"
        modes, models = set(), set()
        if raw.exists():
            with open(raw, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = _j.loads(line)
                    except Exception:
                        continue
                    if r.get("web_on") is not None:
                        modes.add("on" if r["web_on"] else "off")
                    if r.get("model"):
                        models.add(r["model"])
        cfg = {"web_modes": sorted(modes) or ["unknown"], "models": sorted(models),
               "_derived_from_raw": True}
    vs = man.get("vision_selftest") or {}
    return {
        "engines": m.get("engines") or [],
        "n_runs": m.get("n_runs"),
        "intents_planned": planned,
        "intents_audited": audited,
        "intents_not_audited": [i for i in planned if i not in audited],
        "web_modes": cfg.get("web_modes") or ["unknown"],
        "memory_vs_retrieval_separated": len(cfg.get("web_modes") or []) > 1,
        "modalities_assessed": ["text"],
        "modalities_not_assessed": ["visual-only", "visual+text", "conversational"],
        "vision_verified": vs.get("vision_live"),
        "n_experiments": len(findings),
        "evidence_classes": dict(Counter(f.get("evidence_class") for f in findings)),
        "n_proven": sum(1 for f in findings
                        if f.get("evidence_class") in
                        ("PROVEN", "PROVEN_ENGINE_SPECIFIC")),
        "n_proven_caveated": sum(1 for f in findings
                                 if f.get("evidence_class") == "PROVEN_WITH_CAVEAT"),
        "n_disproven": sum(1 for f in findings
                           if f.get("evidence_class") == "DISPROVEN"),
        "n_inconclusive": sum(1 for f in findings
                              if f.get("evidence_class") == "INCONCLUSIVE"),
        "n_wellpowered_nulls": sum(1 for f in findings
                                   if f.get("verdict") in
                                   ("no_lift", "already_winning", "rejected")
                                   and not f.get("underpowered")),
        "query_volume_data": False,
    }

def input_hashes(parent, target):
    out = {}
    for name in sorted({r.artifact for r in CONTRACT}):
        art = load_artifact(parent, target, name)
        if art is None:
            continue
        blob = json.dumps(art, sort_keys=True, default=str).encode()
        out[name] = {"hash": hashlib.sha256(blob).hexdigest()[:16],
                     "size": len(blob)}
    return out