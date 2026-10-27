"""Constrained roadmap: dependency DAG → capacity-packed waves → measurement contracts
→ overlap-discounted projection. A sorted list is not a roadmap."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from optimise.strategy_llm import EVIDENCE_WEIGHT

WAVES = ["Wave 0 (0-2 weeks)", "Wave 1 (2-8 weeks)", "Wave 2 (2-4 months)",
         "Wave 3 (2-4 quarters)"]
WAVE_WEEKS = {WAVES[0]: 2, WAVES[1]: 8, WAVES[2]: 16, WAVES[3]: 52}
EFFORT_DAYS = {"low": 2, "medium": 5, "high": 12}

DEFAULT_CAPACITY = {"content": 10, "dev": 8, "PR": 5, "design": 4,
                    "legal": 3, "community": 4}          # person-days per wave

CAVEATED_CONTRIB_CAP_PP = 8.0
DIAGNOSED_CONTRIB_CAP_PP = 5.0    # observational evidence cannot carry a large projection

# A must precede B. Real production dependencies, not preferences.
DEP_RULES = [
    (lambda a: a["pillar"] == "Claims-Substantiation",
     lambda b: b["pillar"] in ("Content", "Corroboration"),
     "a claim cannot be published or pitched before it is substantiated"),
    (lambda a: a["pillar"] == "Visual/Multimodal",
     lambda b: "visual" in (b.get("title", "") + b.get("what", "")).lower(),
     "the on-pack claim must exist before vision can read it"),
    (lambda a: a["pillar"] in ("Content", "Technical/Structured-Data"),
     lambda b: b["pillar"] == "Corroboration",
     "third parties need published substance to cite"),
    (lambda a: a["pillar"] == "Content",
     lambda b: b["pillar"] == "Social/Community",
     "community seeding is not credible without published substance"),
]

def dedupe_action_ids(actions):
    """One action_id, one action. Two actions sharing an ID produce a self-loop in the
    dependency graph — observed as `VM-3 → VM-4 → VM-4` — and `idx` keeps only one of
    them, so the other is invisible to the roadmap, the projection and the briefs.

    Renames rather than dropping: the action is real, only its label collided.
    """
    seen, renamed = {}, []
    for a in actions:
        aid = a.get("action_id")
        if not aid:
            continue
        if aid not in seen:
            seen[aid] = a
            continue
        base, n = str(aid), 2
        while f"{base}.{n}" in seen:
            n += 1
        new = f"{base}.{n}"
        renamed.append({"was": base, "now": new, "pillar": a.get("pillar"),
                        "title": (a.get("title") or "")[:70]})
        a["action_id"] = new
        a["id_collision"] = (f"The model issued `{base}` twice. Renamed to `{new}`; "
                             f"dependency references to `{base}` may be ambiguous.")
        seen[new] = a
    return renamed


def build_dag(actions):
    idx = {a["action_id"]: a for a in actions if a.get("action_id")}
    edges = defaultdict(set)
    for a in actions:
        for dep in (a.get("dependencies") or []):
            if dep in idx and dep != a.get("action_id"):
                edges[a["action_id"]].add(dep)
    for b in actions:
        for a in actions:
            if a is b or not a.get("action_id") or not b.get("action_id"):
                continue
            for pre, post, _why in DEP_RULES:
                try:
                    if pre(a) and post(b):
                        edges[b["action_id"]].add(a["action_id"])
                except Exception:
                    pass
    return {k: sorted(v) for k, v in edges.items()}

def topo_sort(actions, dag):
    ids = [a["action_id"] for a in actions if a.get("action_id")]
    seen, order, cycles = set(), [], []

    def visit(n, path):
        if n in seen:
            return
        if n in path:
            cycles.append(" → ".join(list(path) + [n]))
            return
        for d in dag.get(n, []):
            if d in ids:
                visit(d, path | {n})
        seen.add(n)
        order.append(n)

    for i in ids:
        visit(i, set())
    return order, cycles

def priority(a):
    ew = EVIDENCE_WEIGHT.get(a.get("evidence_class", "HYPOTHESIS"), 0.2)
    eff = {"low": 1, "medium": 2, "high": 3}.get(a.get("effort", "medium"), 2)
    val = a.get("expected_effect") or {}
    try:
        size = abs(float(val.get("value") or 0))
    except Exception:
        size = 0.0
    reach = max(len(a.get("target_queries") or []), 1)
    return round(ew * (1 + size) * (reach ** 0.3) / eff, 3)

def pack_waves(actions, dag, capacity=None):
    capacity = {**DEFAULT_CAPACITY, **(capacity or {})}
    order, cycles = topo_sort(actions, dag)
    idx = {a["action_id"]: a for a in actions if a.get("action_id")}
    placed, waves = {}, {w: {"actions": [], "load": defaultdict(float)} for w in WAVES}

    for aid in sorted(order, key=lambda x: -priority(idx.get(x, {}))):
        a = idx.get(aid)
        if not a or a.get("blocked"):
            continue
        earliest = WAVES.index(a.get("horizon") or WAVES[1])
        for dep in dag.get(aid, []):
            if dep in placed:
                earliest = max(earliest, WAVES.index(placed[dep]))
        owner = a.get("owner", "content")
        cost = EFFORT_DAYS.get(a.get("effort", "medium"), 5)
        for wi in range(earliest, len(WAVES)):
            w = WAVES[wi]
            if waves[w]["load"][owner] + cost <= capacity.get(owner, 8) or wi == len(WAVES) - 1:
                waves[w]["actions"].append(aid)
                waves[w]["load"][owner] += cost
                placed[aid] = w
                break

    for w in WAVES:
        waves[w]["utilisation"] = {
            o: f"{waves[w]['load'][o]:.0f}/{capacity.get(o, 8)}d"
            for o in sorted(waves[w]["load"]) if waves[w]["load"][o]}
        waves[w]["load"] = dict(waves[w]["load"])
    return waves, placed, cycles

# ---- overlap-discounted projection ----
OVERLAP_GROUPS = {
    "structure": {"E0", "E1", "E5"}, "specificity": {"E2", "E4"},
    "corroboration": {"E3"},
    # VBENCH belongs here. Without it an E7 action groups as "visual" and a bench action
    # falls back to its pillar string — different groups, so their effects never decay
    # against each other and the projection double-counts the same mechanism.
    "visual": {"E7", "E8", "VBENCH"},
    "objection": {"E9"},
}

# A creative-bench effect is measured with the asset IN CONTEXT, on a multimodal read.
# The projection below is built on need-query mention rate from a TEXT audit, where the
# creative acts only insofar as the pack copy is transcribed. Discounted rather than
# excluded, because the work is real — and the full effect is reported separately so it
# is not lost.
VISUAL_SURFACE_FACTOR = 0.25
MAX_SINGLE_CONTRIBUTION_PP = 20.0

def _group(a, design_to_template=None):
    """OVERLAP_GROUPS holds TEMPLATE ids; evidence_refs hold DESIGN ids. Without the
    mapping every action falls back to its pillar and overlap decay never applies —
    two actions citing the same experiment then double-count."""
    refs = set(a.get("evidence_refs") or [])
    tpls = {(design_to_template or {}).get(r, r) for r in refs}
    for g, t in OVERLAP_GROUPS.items():
        if tpls & t:
            return g
    return a.get("pillar", "other")


def project(waves, idx, baseline_pct, decay=0.6, cap_pct=95.0,
            design_to_template=None):
    """Illustrative projection of need-query mention rate.

    Three things a naive sum gets wrong.

    Contributions from the same mechanism are not additive, so they decay geometrically
    within an overlap group — AND ACROSS WAVES. Decay was applied per wave only, so three
    structure actions in three waves each paid out near-full strength and one visual
    finding cited by two actions paid twice. Observed: a 35.4% baseline reaching 88.8%.

    A creative-bench effect was measured with the asset in context on a multimodal read,
    so it is discounted on this text-answer axis and reported separately.

    And a single finding cannot pay out twice however many actions cite it.
    """
    d2t = design_to_template or {}
    bench_ids = {d for d, t in d2t.items() if str(t).upper() == "VBENCH"}
    cum, traj = float(baseline_pct), []
    # Carried ACROSS waves: how many contributions each mechanism group has already made,
    # and which experiment designs have already been counted.
    group_count = defaultdict(int)
    designs_counted = {}

    for w in WAVES:
        groups, sources, multimodal = defaultdict(list), [], []
        for aid in waves[w]["actions"]:
            a = idx.get(aid) or {}
            eff = a.get("expected_effect") or {}
            if eff.get("basis") not in ("measured", "inferred"):
                continue
            try:
                v = float(eff.get("value") or 0)
            except Exception:
                v = 0.0
            if v <= 0 or eff.get("unit") not in (None, "pp"):
                continue
            ec = a.get("evidence_class", "HYPOTHESIS")
            refs = set(a.get("evidence_refs") or [])

            if ec in ("DIAGNOSED", "HYPOTHESIS") and refs and \
                    eff.get("basis") == "measured":
                sources.append({"action": aid, "evidence": ec, "group": "excluded",
                                "quoted_pp": round(v, 1), "contribution_pp": 0.0,
                                "excluded_reason": "class/effect mismatch — a DIAGNOSED "
                                                   "action cannot claim a measured lift"})
                continue

            # ---- a finding pays out ONCE, however many actions cite it ----
            _dupe = sorted(r for r in refs if r in designs_counted)
            if _dupe:
                sources.append({
                    "action": aid, "evidence": ec, "group": "excluded",
                    "quoted_pp": round(v, 1), "contribution_pp": 0.0,
                    "excluded_reason": (
                        f"the effect from {_dupe} was already counted in "
                        f"{designs_counted[_dupe[0]]}. The work may be real and worth "
                        f"doing; the measured lift cannot be banked twice.")})
                continue

            raw = v
            v *= EVIDENCE_WEIGHT.get(ec, 0.2)
            if ec == "PROVEN_WITH_CAVEAT":
                v = min(v * 0.5, CAVEATED_CONTRIB_CAP_PP)
            elif ec in ("DIAGNOSED", "HYPOTHESIS"):
                v = min(v, DIAGNOSED_CONTRIB_CAP_PP)

            is_bench = bool(refs & bench_ids)
            if is_bench:
                full = v
                v *= VISUAL_SURFACE_FACTOR
                multimodal.append({
                    "action": aid, "designs": sorted(refs & bench_ids),
                    "multimodal_effect_pp": round(full, 1),
                    "counted_toward_text_projection_pp": round(v, 1),
                    "note": "Full effect applies to multimodal surfaces, which this "
                            "text-answer projection does not measure."})

            v = min(v, MAX_SINGLE_CONTRIBUTION_PP)
            g = _group(a, d2t)
            groups[g].append(v)
            sources.append({"action": aid, "evidence": ec, "group": g,
                            "quoted_pp": round(raw, 1),
                            "contribution_pp_before_decay": round(v, 1),
                            "modality_discounted": is_bench})
            for r in refs:
                designs_counted[r] = w

        # ---- decay, with the group's position carried forward from earlier waves ----
        add = 0.0
        for g, vals in groups.items():
            for v in sorted(vals, reverse=True):
                factor = decay ** group_count[g]
                add += v * factor
                group_count[g] += 1
                for s in sources:
                    if (s.get("group") == g
                            and s.get("contribution_pp_before_decay") == round(v, 1)
                            and "contribution_pp" not in s):
                        s["contribution_pp"] = round(v * factor, 1)
                        s["decay_factor"] = round(factor, 2)
                        s["nth_in_group"] = group_count[g]
                        break
        for s in sources:
            s.setdefault("contribution_pp", 0.0)

        cum = min(cum + add, cap_pct)
        traj.append({"wave": w, "added_pp": round(add, 1),
                     "projected_pct": round(cum, 1),
                     "low_pct": round(max(baseline_pct, cum - add * 0.7), 1),
                     "high_pct": round(min(cum + add * 0.3, cap_pct), 1),
                     "contributors": sources,
                     "multimodal_effects": multimodal,
                     "double_counted_excluded": [
                         s["action"] for s in sources
                         if "already counted" in str(s.get("excluded_reason", ""))],
                     "dominant_evidence": (
                         max((s for s in sources if s["contribution_pp"] > 0),
                             key=lambda s: s["contribution_pp"])["evidence"]
                         if any(s["contribution_pp"] > 0 for s in sources) else None)})
    return traj

ROADMAP_PROMPT = """Turn this wave plan into an executable GEO roadmap for "{brand}".

For EACH wave produce a MEASUREMENT CONTRACT so the Measure phase can verify it: which
queries to re-audit, N, which engines, the expected delta with its basis, the re-audit
date, and the kill criterion.

Be honest about uncertainty. Where an expected effect comes from a single-engine or
underpowered experiment, say so. Do NOT sum overlapping effects as if additive.

WAVES (dependency-ordered, capacity-packed): {waves}
ACTION DETAIL: {actions}
STRATEGIC NARRATIVE: {narrative}
CURRENT BASELINE (audit): {baseline}
KPI TARGETS: {kpis}
TEAM CAPACITY (person-days per wave): {capacity}
DEPENDENCY EDGES: {dag}
NOT DOING (evidence-based exclusions): {not_doing}

Return STRICT JSON:
{{
 "waves": [{{
   "wave":"", "window":"", "theme":"one line",
   "objective":"what this wave proves or delivers",
   "actions":[{{"action_id":"","title":"","owner":"","effort":"","evidence_class":"","depends_on":[]}}],
   "measurement_contract":{{"re_audit_queries":[],"n_runs":0,"engines":[],
                            "primary_metric":"","expected_delta":"","basis":"",
                            "re_audit_after_weeks":0,"kill_criterion":""}},
   "exit_criteria":"what must be true to proceed",
   "risks":[{{"risk":"","mitigation":"","rollback":""}}]
 }}],
 "critical_path":["action_id"],
 "quick_wins":["action_id"],
 "sequencing_rationale":"why this order and not another",
 "assumptions":["every assumption behind the plan"]
}}"""


def build_roadmap(gateway, strategy, model, capacity=None, module="optimise"):
    actions = [a for a in strategy.get("actions", []) if not a.get("blocked")]
    # Must precede idx and the DAG: a duplicate ID makes idx silently lose an action and
    # produces a self-loop that reads as a dependency cycle.
    renamed = dedupe_action_ids(actions)
    idx = {a["action_id"]: a for a in actions if a.get("action_id")}
    dag = build_dag(actions)
    waves, placed, cycles = pack_waves(actions, dag, capacity)

    si_metrics = (strategy.get("_si") or {}).get("metrics", {})
    mbi = {r["intent"]: r["mention_rate"]
           for r in si_metrics.get("mention_by_intent", [])}
    baseline_pct = round(mbi.get("need", 0.05) * 100, 1)

    res = gateway.generate_json(model, ROADMAP_PROMPT.format(
        brand=strategy.get("target_brand"),
        waves=json.dumps({w: {"actions": v["actions"],
                              "utilisation": v["utilisation"]}
                          for w, v in waves.items()}, indent=1)[:3500],
        actions=json.dumps([{k: a.get(k) for k in
                             ("action_id", "title", "what", "pillar", "owner", "effort",
                              "evidence_class", "evidence_refs", "expected_effect",
                              "clock", "horizon", "target_queries", "kill_criterion",
                              "leading_indicator", "measurement_note")}
                            for a in actions], indent=1)[:9000],
        narrative=(strategy.get("overall_strategy") or {}).get("narrative", ""),
        baseline=json.dumps(si_metrics.get("mention_by_intent") or [])[:900],
        kpis=json.dumps(((strategy.get("_si") or {}).get("kpis")) or [])[:1000],
        capacity=json.dumps({**DEFAULT_CAPACITY, **(capacity or {})}),
        dag=json.dumps(dag)[:1500],
        not_doing=json.dumps((strategy.get("overall_strategy") or {})
                             .get("not_doing") or [])[:1500]),
        expect="dict", module=module, task="roadmap", temperature=0.3, max_tokens=5000)

    rm = res.get("data") or {}
    rm["waves_packed"] = waves
    rm["dependency_dag"] = dag
    rm["dependency_cycles"] = cycles
    rm["renamed_action_ids"] = renamed
    if renamed:
        rm["id_collision_note"] = (
            f"{len(renamed)} action(s) were issued a duplicate action_id and have been "
            f"renamed. A duplicate makes the action index lose one of the pair, so it "
            f"would otherwise be absent from the roadmap, the projection and the briefs.")
    rm["action_placement"] = placed
    si = strategy.get("_si") or {}
    d2t = {r["design_id"]: r.get("template")
           for r in (si.get("wins") or []) + (si.get("nulls") or [])
           if r.get("design_id")}
    rm["projection"] = project(waves, idx, baseline_pct, design_to_template=d2t)
    weak = [t for t in rm["projection"]
            if t["added_pp"] > 10 and t.get("dominant_evidence") in
            ("PROVEN_WITH_CAVEAT", "DIAGNOSED", "HYPOTHESIS")]
    rm["projection_warning"] = (
        f"⚠️ {len(weak)} wave(s) show a large projected gain driven by CAVEATED or "
        f"DIAGNOSED evidence. Treat as an illustrative upper bound, not a forecast."
        if weak else "")
    _mm = [m for t in rm["projection"] for m in (t.get("multimodal_effects") or [])]
    rm["projection_note"] = (
            "Projection, not forecast. Contributions are discounted by evidence class "
            "(caveated results halved again), by geometric overlap within a mechanism group, "
            "and capped per action. Actions with no measured basis contribute nothing. Memory "
            "plays contribute no in-quarter mention-rate lift by design."
            + (f" Creative-bench effects are discounted to {VISUAL_SURFACE_FACTOR:.0%} here: "
               f"this projection tracks need-query mention rate from a TEXT audit, while a "
               f"bake-off measures a multimodal read with the asset in context. The full "
               f"multimodal effect is reported separately in `multimodal_effects` and is not "
               f"lost — it simply does not belong on this axis."
               if _mm else ""))
    rm["multimodal_effects"] = _mm
    rm["baseline_pct"] = baseline_pct
    rm["generated_utc"] = datetime.now(timezone.utc).isoformat()
    rm["_generated_ok"] = bool(res.get("ok"))

    # calendar dates for the measurement contracts
    start = datetime.now(timezone.utc)
    for w in (rm.get("waves") or []):
        wk = WAVE_WEEKS.get(w.get("wave"), 8)
        mc = w.setdefault("measurement_contract", {})
        after = int(mc.get("re_audit_after_weeks") or wk)
        mc["re_audit_date"] = (start + timedelta(weeks=after)).strftime("%Y-%m-%d")
    return rm