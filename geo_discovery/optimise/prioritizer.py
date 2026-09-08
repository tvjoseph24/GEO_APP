"""Prioritize actions: Impact × Feasibility × Winnability → Now/Next/Later."""

def _impact_score(finding):
    """From experiment lift (pooled_lift_pp). 0-5 scale."""
    lift = abs(finding.get("pooled_lift_pp", 0))
    if lift >= 50: return 5
    if lift >= 30: return 4
    if lift >= 15: return 3
    if lift >= 5:  return 2
    return 1

def _winnability_score(finding):
    """From verdict + engine breadth."""
    v = finding.get("verdict", "")
    if v == "validated":       return 5   # works everywhere
    if v == "engine-specific": return 3   # works on some engines
    return 1

def _feasibility_default(finding):
    """Heuristic feasibility by pillar (human can override). 0-5."""
    pillar = finding.get("pillar", "")
    # own-content pillars are easier than earned/corroboration
    easy = {"Content": 5, "Technical/Structured-Data": 4, "Claims-Substantiation": 3,
            "Visual/Multimodal": 3, "Corroboration": 2, "Social/Community": 2}
    return easy.get(pillar, 3)

def score_actions(findings, feasibility_overrides=None):
    """Compute priority = Impact × Feasibility × Winnability, assign horizon."""
    fo = feasibility_overrides or {}
    for f in findings:
        imp = _impact_score(f)
        win = _winnability_score(f)
        feas = fo.get(f.get("gap_id"), _feasibility_default(f))
        f["impact"] = imp; f["winnability"] = win; f["feasibility"] = feas
        f["priority_score"] = imp * feas * win        # 1-125
        # horizon by score + time-nature
        if f["priority_score"] >= 60:   f["horizon"] = "Now"
        elif f["priority_score"] >= 25: f["horizon"] = "Next"
        else:                           f["horizon"] = "Later"
        # time-clock: retrieval (publishable now) vs memory (slow)
        f["clock"] = ("memory" if f["pillar"] == "Social/Community" else "retrieval")
    return sorted(findings, key=lambda x: -x["priority_score"])