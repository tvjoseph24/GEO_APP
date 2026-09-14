"""Dump experiment results for review. Run: python dump_results.py danone aptamil"""
import json
import sys

from store import load_artifact

P = sys.argv[1] if len(sys.argv) > 1 else "danone"
T = sys.argv[2] if len(sys.argv) > 2 else "aptamil"

def A(name):
    return load_artifact(P, T, name)

def pick(d, keys):
    d = d or {}
    return {k: d.get(k) for k in keys}

findings = A("validated_findings") or A("validated_findings_only") or []
variants = A("experiment_variants") or {}
backlog = A("substantiation_backlog") or []

if isinstance(variants, dict):
    recs = variants.get("experiments") or []
    comp_text = variants.get("competitor_context_text") or ""
else:
    recs = variants or []
    comp_text = ""

vmap = {}
for r in recs:
    key = r.get("design_id") or r.get("gap_id")
    if key:
        vmap[key] = r

BIN_KEYS = ["engine", "arm", "metric", "A_rate", "B_rate", "lift_pp",
            "n_A", "n_B", "p_fdr", "significant", "is_negative_control"]
PILOT_KEYS = ["p0_mention", "p0_first", "ceiling_risk", "floor_risk", "n_total"]
BACK_KEYS = ["cluster_id", "severity", "assertion_to_substantiate"]

out = {
    "brand": T,
    "n_findings": len(findings),
    "n_substantiation_backlog": len(backlog),
    "competitor_field_words": len(comp_text.split()),
    "verdicts": {},
    "evidence_classes": {},
    "findings": [],
}

for f in findings:
    v = f.get("verdict")
    ec = f.get("evidence_class")
    out["verdicts"][v] = out["verdicts"].get(v, 0) + 1
    out["evidence_classes"][ec] = out["evidence_classes"].get(ec, 0) + 1

    d = vmap.get(f.get("design_id")) or {}
    rows = []
    for r in (f.get("per_engine") or [])[:24]:
        rows.append(pick(r, BIN_KEYS))

    out["findings"].append({
        "design_id": f.get("design_id"),
        "template": f.get("experiment_id"),
        "mode": f.get("mode"),
        "verdict": v,
        "evidence_class": ec,
        "best_metric": f.get("best_metric"),
        "pooled_lift_pp": f.get("pooled_lift_pp"),
        "mechanism": f.get("mechanism") or d.get("mechanism"),
        "pillar": f.get("pillar") or d.get("pillar"),
        "isolated_variable": f.get("isolated_variable") or d.get("isolates"),
        "hypothesis": (f.get("hypothesis") or d.get("hypothesis") or "")[:220],
        "prediction": f.get("prediction"),
        "question": d.get("question"),
        "binary_rows": rows,
        "rank_analysis": f.get("rank_analysis") or [],
        "element_attribution": f.get("element_attribution"),
        "best_element": f.get("best_element"),
        "variance_warnings": f.get("variance_warnings") or [],
        "caveats": f.get("caveats") or [],
        "interpretation_limits": f.get("interpretation_limits") or [],
        "pilot": pick(f.get("pilot"), PILOT_KEYS),
    })

sample = []
for b in backlog[:8]:
    sample.append(pick(b, BACK_KEYS))
out["backlog_sample"] = sample

print(json.dumps(out, indent=2, default=str))