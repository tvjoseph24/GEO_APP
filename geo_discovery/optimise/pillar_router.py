"""Route each validated finding / gap → one of the 6 strategy pillars."""

PILLARS = {
    "Content": "Messaging, specificity, quantified claims",
    "Technical/Structured-Data": "Schema, answer capsules, comparison tables, machine-readable PDPs",
    "Visual/Multimodal": "On-pack claims, alt-text, image legibility",
    "Corroboration": "Editorial, PR, expert quotes, third-party echo",
    "Social/Community": "Reddit & forums (query-triggered, engine-weighted)",
    "Claims-Substantiation": "Certify/verify claims so they're citable",
}

def route_finding(finding, gap=None):
    """Map a validated finding to its primary pillar based on experiment type + gap."""
    eid = finding.get("experiment_id", "")
    dt = (gap or {}).get("delta_type", "")
    pillar_hint = (gap or {}).get("pillar", "")

    # experiment-type → pillar
    if eid == "E3":              return "Corroboration"       # corroboration test won
    if eid in ("E7", "E8"):      return "Visual/Multimodal"   # visual test won
    if eid == "E1":              return "Technical/Structured-Data"  # structure isolation
    if eid == "E2":              return "Content"             # specificity
    if eid == "E9":              return "Content"             # objection rebuttal = messaging
    if dt == "hallucination":    return "Claims-Substantiation"
    # E4 gap-closure → depends on the gap's pillar
    if pillar_hint in PILLARS:   return pillar_hint
    return "Content"             # default

def route_all(findings, gap_lookup):
    """Attach pillar to each finding. gap_lookup: {gap_id: gap_dict}."""
    for f in findings:
        gap = gap_lookup.get(f.get("gap_id"))
        f["pillar"] = route_finding(f, gap)
        f["gap_detail"] = gap
    return findings

def pillar_summary(findings):
    """Count findings per pillar (for the strategy overview)."""
    from collections import Counter
    return dict(Counter(f["pillar"] for f in findings))