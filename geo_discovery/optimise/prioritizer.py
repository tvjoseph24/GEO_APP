"""Evidence-weighted prioritisation. Weights are exposed so an analyst can see the
ranking move when the trade-off changes."""
from optimise.strategy_llm import EVIDENCE_WEIGHT

EFFORT = {"low": 1.0, "medium": 2.0, "high": 3.5}
TIME = {"Wave 0 (0-2 weeks)": 1.0, "Wave 1 (2-8 weeks)": 1.3,
        "Wave 2 (2-4 months)": 2.0, "Wave 3 (2-4 quarters)": 3.5}

DEFAULT_WEIGHTS = {"evidence": 1.0, "effect": 1.0, "reach": 1.0,
                   "effort": 1.0, "speed": 1.0}

def score_action(a, weights=None, engine_count=1):
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    ew = EVIDENCE_WEIGHT.get(a.get("evidence_class", "HYPOTHESIS"), 0.2)
    eff = a.get("expected_effect") or {}
    try:
        size = abs(float(eff.get("value") or 0))
    except Exception:
        size = 0.0
    # rank positions are not percentage points — normalise to a comparable scale
    if eff.get("unit") == "rank_positions":
        size *= 10.0
    breadth = max(len(eff.get("engines") or []), 1) / max(engine_count, 1)
    reach = max(len(a.get("target_queries") or []), 1)
    num = ((ew ** w["evidence"]) * ((1 + size) ** w["effect"])
           * (reach ** (0.3 * w["reach"])) * (0.6 + 0.4 * breadth))
    den = (EFFORT.get(a.get("effort", "medium"), 2.0) ** w["effort"]) * \
          (TIME.get(a.get("horizon"), 1.5) ** (0.5 * w["speed"]))
    return round(num / den, 3)

def score_actions(actions, weights=None, engine_count=1):
    for a in actions:
        a["priority_score"] = score_action(a, weights, engine_count)
        a["evidence_weight"] = EVIDENCE_WEIGHT.get(
            a.get("evidence_class", "HYPOTHESIS"), 0.2)
    return sorted(actions, key=lambda x: -x["priority_score"])

def summary(actions):
    from collections import Counter
    return {"n": len(actions),
            "by_evidence": dict(Counter(a.get("evidence_class") for a in actions)),
            "by_pillar": dict(Counter(a.get("pillar") for a in actions)),
            "by_clock": dict(Counter(a.get("clock") for a in actions)),
            "by_owner": dict(Counter(a.get("owner") for a in actions)),
            "blocked": sum(1 for a in actions if a.get("blocked")),
            "mean_priority": round(sum(a.get("priority_score", 0) for a in actions)
                                   / max(len(actions), 1), 3)}