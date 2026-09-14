"""Diagnose the three Optimise defects. No LLM calls, no spend.

    python diag_optimise.py danone aptamil
"""
import sys

from metrics_spec import fold
from store import load_artifact

P = sys.argv[1] if len(sys.argv) > 1 else "danone"
T = sys.argv[2] if len(sys.argv) > 2 else "aptamil"

FLAGGED = [
    "APTAMIL GOLD 1 contains 2\u2019-FL, 3- FL, LNT, 6\u2019-SL and 3\u2019-SL.",
    "Aptamil Gold Infant Formula contains HMO & Prebiotic (GOS & FOS).",
    "Aptamil first infant milks use a palm-oil-free oil blend.",
    "Aptamil first infant milks are made with high-quality ingredients.",
]

def jac(a, b):
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / max(len(sa | sb), 1)

# ---------- D1: are the critic-flagged claims actually in the register? ----------
claims = load_artifact(P, T, "claims_register") or []
reg = {fold(c.get("claim", "")) for c in claims}
print(f"=== D1 · claims register: {len(claims)} claims ===\n")
for c in FLAGGED:
    exact = fold(c) in reg
    near = sorted(((jac(fold(c), r), r) for r in reg), reverse=True)[:1]
    tag = "IN REGISTER" if exact else "NOT EXACT"
    print(f"  [{tag:11}] {c[:66]}")
    if not exact and near and near[0][0] > 0.5:
        print(f"                near ({near[0][0]:.2f}): {near[0][1][:66]}")

# ---------- D2: evidence class vs quoted effect ----------
strat = load_artifact(P, T, "geo_strategy") or {}
findings = load_artifact(P, T, "validated_findings") or []
exp_ids = {f.get("design_id") for f in findings}
print("\n=== D2 · action evidence class vs quoted effect ===")
print(f"  {'action':9} {'class':26} {'value':>7} {'basis':10} refs")
bad = 0
for a in strat.get("actions", []):
    e = a.get("expected_effect") or {}
    refs = a.get("evidence_refs") or []
    v = e.get("value")
    mismatch = (a.get("evidence_class") in ("DIAGNOSED", "HYPOTHESIS")
                and v and set(refs) & exp_ids and e.get("basis") == "measured")
    bad += bool(mismatch)
    print(f"  {str(a.get('action_id')):9} {str(a.get('evidence_class')):26} "
          f"{str(v):>7} {str(e.get('basis')):10} {refs}"
          + ("   <-- MISMATCH" if mismatch else ""))
print(f"\n  {bad} action(s) quote a measured experiment effect while classed as "
      f"unevidenced.")

# ---------- D3: did headline_effect reach the strategy input? ----------
print("\n=== D3 · what the pillar prompts saw for each win ===")
try:
    from optimise.strategy_llm import assemble_strategy_input
    si = assemble_strategy_input(P, T, "infant formula")
    for w in si["wins"]:
        he = w.get("headline_effect")
        print(f"  {w['design_id']}: lift={w.get('lift')} {w.get('lift_unit')} "
              f"| headline_effect={'MISSING' if not he else he}")
    if not any(w.get("headline_effect") for w in si["wins"]):
        print("\n  -> headline_effect absent. Add this line to the `row` dict in "
              "assemble_strategy_input:\n"
              '       "headline_effect": f.get("headline_effect"),')
except Exception as e:
    print(f"  could not load strategy input: {type(e).__name__}: {e}")

# ---------- summary of evidence classes on findings vs actions ----------
from collections import Counter
print("\n=== evidence classes ===")
print("  findings:", dict(Counter(f.get("evidence_class") for f in findings)))
print("  actions :", dict(Counter(a.get("evidence_class")
                                  for a in strat.get("actions", []))))
prov = [f for f in findings
        if str(f.get("evidence_class", "")).startswith("PROVEN")]
if prov and not any(str(a.get("evidence_class", "")).startswith("PROVEN")
                    for a in strat.get("actions", [])):
    print(f"\n  -> {len(prov)} proven finding(s) exist "
          f"({', '.join(f['design_id'] for f in prov)}) but NO action is classed as "
          f"proven. The strategy is understating its strongest evidence.")