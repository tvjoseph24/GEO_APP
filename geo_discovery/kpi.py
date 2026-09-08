"""KPI template + optional grounded target suggestions."""
import json

DEFAULT_KPIS = [
    {"kpi": "need_based_mention_rate", "definition": "% of need-based queries where brand is mentioned",
     "baseline": None, "target": "+15pp", "measured_by": "Audit need queries", "priority": "primary"},
    {"kpi": "first_mention_share", "definition": "% of runs where brand is named first",
     "baseline": None, "target": "reach top-3", "measured_by": "Audit first-mention", "priority": "primary"},
    {"kpi": "veracity_rate", "definition": "% of stated brand claims that are correct",
     "baseline": None, "target": ">=90%", "measured_by": "Benchmark veracity", "priority": "primary"},
    {"kpi": "refusal_rate", "definition": "% of runs where engine declined to recommend",
     "baseline": None, "target": "track", "measured_by": "Audit refusals", "priority": "secondary"},
    {"kpi": "hallucination_rate", "definition": "% of runs stating a false brand claim",
     "baseline": None, "target": "0%", "measured_by": "Benchmark veracity", "priority": "primary"},
    {"kpi": "source_corroboration", "definition": "% of brand claims echoed by editorial sources",
     "baseline": None, "target": "increase", "measured_by": "Benchmark provenance", "priority": "secondary"},
]

SUGGEST_PROMPT = """For a brand in the "{category}" category, suggest realistic GEO KPI targets.
Given these KPIs, propose a sensible 'target' for each based on category norms.
Return STRICT JSON list mirroring the input with updated 'target' fields:
{kpis}"""

def default_kpis():
    import copy
    return copy.deepcopy(DEFAULT_KPIS)

def suggest_targets(gateway, category, model="gemini-2.5-flash", module="discover"):
    """Optional: grounded LLM suggests targets from category norms."""
    r = gateway.generate_grounded(
        SUGGEST_PROMPT.format(category=category, kpis=json.dumps(DEFAULT_KPIS)),
        model=model, module=module, task="kpi_suggest", json_mode=True)
    return r["parsed"] if isinstance(r["parsed"], list) else default_kpis()