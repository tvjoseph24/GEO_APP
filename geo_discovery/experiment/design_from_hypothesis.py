"""Turn validated Benchmark hypotheses into experiment designs.

Before this, experiment_tab only knew about signal_designer's heuristic designs, so the
LLM-reasoned hypotheses (with mechanism, prediction, falsification, isolated variable,
approved claim list and a brand-free target query) never reached the runner.
"""
from __future__ import annotations

from benchmark.hypothesis import TEMPLATES
from metrics_spec import fold

def designs_from_hypotheses(hyps, *, target_brand, query_master=None,
                            only_ready=True, include_by_default=False):
    """Returns (designs, skipped, not_testable).
    not_testable = substantiation gaps → Optimise Pillar 6, never the experiment queue."""
    bf = {fold(q["query"]) for q in (query_master or [])
          if q.get("intent") in ("need", "attribute", "persona", "longtail",
                                 "negative", "freshness")}
    out, skipped, not_testable = [], [], []
    for h in (hyps or []):
        if h.get("status") == "not_testable" or \
                h.get("mechanism") == "substantiation_deficit":
            not_testable.append({
                "cluster_id": h.get("cluster_id"), "title": h.get("cluster_title"),
                "pillar": "Claims-Substantiation", "severity": h.get("severity"),
                "gap_ids": h.get("gap_ids"),
                "assertion_to_substantiate": h.get("hypothesis"),
                "reason": h.get("not_testable_reason",
                                "claim absent from the substantiated register"),
                "engines_asserting": h.get("engines"),
                "next_step": "Substantiate, publish, then test with E4."})
            continue
        errs = h.get("validation_errors") or []
        if only_ready and errs:
            skipped.append({"cluster_id": h.get("cluster_id"),
                            "title": h.get("cluster_title"),
                            "template": (h.get("experiment") or {}).get("template"),
                            "pillar": h.get("pillar"), "severity": h.get("severity"),
                            "blockers": "; ".join(errs)[:200],
                            "warnings": "; ".join(h.get("validation_warnings") or [])[:160]})
            continue
        exp = h.get("experiment") or {}
        tpl = exp.get("template") if exp.get("template") in TEMPLATES else "E0"
        q = (exp.get("target_query") or "").strip()
        if not q or fold(target_brand) in fold(q):
            skipped.append({"cluster_id": h.get("cluster_id"),
                            "title": h.get("cluster_title"),
                            "blockers": "target query missing or names the brand"})
            continue
        out.append({
            "design_id": h.get("cluster_id"), "source": "hypothesis",
            "signal": h.get("family"),
            "priority_signal": ("high" if h.get("severity") in ("critical", "high")
                                else "medium"),
            "experiment_id": tpl, "experiment_name": TEMPLATES[tpl]["name"],
            "isolates": exp.get("isolated_variable") or TEMPLATES[tpl]["isolates"],
            "hypothesis": h.get("hypothesis"), "question": q,
            "question_in_audit": fold(q) in bf,
            "gap": h.get("cluster_title"), "gap_ids": h.get("gap_ids"),
            "gap_claim": (h.get("claims_used") or [None])[0],
            "claims_allowed": h.get("claims_used") or [],
            "mechanism": h.get("mechanism"), "pillar": h.get("pillar"),
            "horizon": h.get("horizon"), "prediction": h.get("prediction") or {},
            "falsification": h.get("falsification"),
            "held_constant": exp.get("held_constant") or [],
            "mde_pp_audit": h.get("mde_pp"),
            "mode": "Progressive (E1)" if tpl == "E1" else "A/B (fast)",
            "include": bool(include_by_default),
            "needs_visual": tpl in ("E7", "E8"),
            "validation_errors": errs,
            "validation_warnings": h.get("validation_warnings") or [],
            "forced": bool(errs),
            "_hypothesis": h})
    out.sort(key=lambda d: (0 if d["priority_signal"] == "high" else 1, d["design_id"]))
    return out, skipped, not_testable