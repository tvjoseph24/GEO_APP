"""Map each actionable gap → the experiment type (E1-E10) that tests it."""

# Experiment catalog
EXPERIMENTS = {
    # ---- A/B experiments (gap-driven) ----
    "E4": {"name": "Gap-Closure", "isolates": "adding the missing claim",
           "arms": ["A: without claim", "B: with claim"], "type": "ab", "visual": False},
    "E2": {"name": "Specificity", "isolates": "quantified vs vague",
           "arms": ["A: vague", "B: quantified"], "type": "ab", "visual": False},
    "E3": {"name": "Corroboration", "isolates": "brand-only vs 3rd-party echo",
           "arms": ["A: brand source", "B: brand + editorial source"], "type": "ab", "visual": False},
    "E5": {"name": "Format", "isolates": "prose vs table vs bullets",
           "arms": ["A: prose", "B: table", "C: bullets"], "type": "multi", "visual": False},
    "E9": {"name": "Objection-Rebuttal", "isolates": "countering a negative",
           "arms": ["A: no rebuttal", "B: with rebuttal"], "type": "ab", "visual": False},
    "E7": {"name": "Visual-Extraction", "isolates": "can vision read the claim",
           "arms": ["A: text only", "B: text + image"], "type": "ab", "visual": True},
    "E8": {"name": "Visual+Text Align", "isolates": "aligned image+text lift",
           "arms": ["A: text only", "B: text + aligned visual"], "type": "ab", "visual": True},
    # ---- E1: foundational multi-arm (not per-gap; run once) ----
    "E1": {"name": "Structure Isolation", "isolates": "which structural element wins",
           "arms": ["A: plain", "B: +capsule", "C: +table", "D: +citations", "E: full"],
           "type": "multi", "visual": False},
}
# E6 (Position Control) = the target_pos parameter — run all 3 positions as a robustness check
# E10 (Cross-Model Transfer) = automatic when engines list has >1 model (see Clarification 2)

def map_gap_to_experiment(gap):
    """Propose the experiment type for a gap based on its delta_type + content."""
    dt = gap.get("delta_type", "")
    pillar = gap.get("pillar", "")
    gap_text = gap.get("gap", "").lower()

    if dt == "hallucination":
        return "E3"          # corroboration fixes wrong/contradicted claims
    if "negative" in gap_text or "objection" in gap_text or "tastes bad" in gap_text:
        return "E9"          # objection rebuttal
    if dt == "under-surfaced" or "winnable query" in gap_text:
        return "E4"          # gap-closure: add the relevant claim
    if pillar == "Corroboration":
        return "E3"
    if pillar == "Visual / Multimodal":
        return "E7"
    if dt == "missing":
        return "E4"          # default for missing claims: gap-closure
    return "E4"              # safe default

def propose_experiments(actionable_gaps):
    """Build the proposed experiment design for each actionable gap."""
    designs = []
    for g in actionable_gaps:
        eid = map_gap_to_experiment(g)
        exp = EXPERIMENTS[eid]
        designs.append({
            "gap_id": g["gap_id"],
            "gap": g["gap"],
            "hypothesis": g.get("hypothesis", ""),
            "experiment_id": eid,
            "experiment_name": exp["name"],
            "isolates": exp["isolates"],
            "variant_plan": exp["arms"],                 # ← "arms" not "variants"
            "experiment_type": exp.get("type", "ab"),    # ← ab / multi
            "needs_visual": exp["visual"],
            "include": True,
            "add_visual_variant": False,
        })
    return designs