"""Verify deterministic_validate. Run: python test_validate.py"""
from optimise.strategy_llm import deterministic_validate

si = {
    "claims": [{"claim": "Aptamil first infant milks use a palm-oil-free oil blend."}],
    "nulls": [{
        "design_id": "C20",
        "template": "E1",
        "runs_per_cell": 100,
        "isolated_variable": "cumulative structure",
        "underpowered": False,
    }],
    "wins": [{
        "design_id": "C08",
        "underpowered": True,
        "runs_per_cell": 20,
        "evidence_class": "PROVEN_WITH_CAVEAT",
    }],
}

# 1. must NOT flag — same wording, different pillar from the null
a1 = {
    "action_id": "P4-1",
    "pillar": "Corroboration",
    "title": "Earn citations with a cumulative structure of cited claims",
    "what": "pitch editorial placements",
    "evidence_class": "DIAGNOSED",
    "validating_experiment": "E3",
    "kill_criterion": "stop if no citations after 2 quarters",
    "expected_effect": {"basis": "none", "value": None},
}

# 2. SHOULD flag — same pillar as the null, all null tokens present
a2 = {
    "action_id": "P2-9",
    "pillar": "Technical/Structured-Data",
    "title": "Reformat with cumulative structure",
    "what": "apply cumulative structure across all PDPs",
    "evidence_class": "DIAGNOSED",
    "validating_experiment": "E1",
    "kill_criterion": "stop if no movement",
    "expected_effect": {"basis": "none", "value": None},
}

# 3. SHOULD flag twice — DIAGNOSED quoting a measured effect, and >25pp
a3 = {
    "action_id": "P1-1",
    "pillar": "Content",
    "title": "Publish HMO detail",
    "what": "publish specific HMO content",
    "evidence_class": "DIAGNOSED",
    "evidence_refs": ["C08"],
    "validating_experiment": "E4",
    "kill_criterion": "stop if no movement",
    "expected_effect": {"value": 62.5, "basis": "measured", "source": "C08"},
}

pillar_strategies = {"Content": {"actions": [a1, a2, a3]}}
strategy = {"not_doing": [{"item": "placeholder"}], "big_bets": []}

out = deterministic_validate(si, pillar_strategies, strategy)

print(f"{len(out)} error(s)\n")
for e in out:
    print(f"  {e['severity']:8} {str(e['target']):8} {e['error']}")
    if e.get("detail"):
        print(f"           detail: {e['detail']}")

codes = [(e["target"], e["error"]) for e in out]
checks = [
    ("P4-1 NOT flagged as contradiction (false-positive fix)",
     ("P4-1", "contradicts_well_powered_null") not in codes),
    ("P2-9 flagged as contradiction",
     ("P2-9", "contradicts_well_powered_null") in codes),
    ("P1-1 flagged for class/effect mismatch",
     ("P1-1", "diagnosed_action_quotes_measured_effect") in codes),
    ("P1-1 flagged for implausible effect",
     ("P1-1", "implausible_effect_for_unevidenced_action") in codes),
]
print()
for label, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + label)
print("\n" + ("ALL PASS" if all(k for _, k in checks) else "SOME FAILED"))