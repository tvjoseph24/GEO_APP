"""Generate production-ready content briefs from winning variants."""
import json

BRIEF_PROMPT = """You are writing a production brief for "{brand}". An experiment PROVED that
a SPECIFIC content treatment increased AI recommendation by {lift}pp on {engines}.

Your job is NOT to rewrite the content — it already works. Your job is to:
1. Present the PROVEN content as-is (it's the recommendation).
2. Add placement, corroboration, and success-metric guidance around it.

PROVEN WINNING CONTENT (this exact treatment worked — publish this):
---
{winning_content}
---

Gap: {gap} | Pillar: {pillar} | Source targets: {hitlist}

Return STRICT JSON:
{{
 "title": "",
 "objective": "one line",
 "why_it_works": "briefly, what structural elements made this content win (capsule/table/etc.)",
 "placement": ["specific places to publish — use source targets"],
 "corroboration": ["3rd-party echoes needed to make claims citable"],
 "success_metric": "how to measure post-publish lift"
}}
Do NOT rewrite or replace the proven content — reference it as the deliverable."""

def generate_brief(gateway, finding, winning_variant, hitlist_domains,
                   target_brand, model="gemini-2.5-flash", module="optimise"):
    sig_engines = [e["engine"] for e in finding.get("per_engine", []) if e.get("significant")]
    best_lift = max((e["lift_pp"] for e in finding.get("per_engine",[])
                     if e.get("significant")), default=finding.get("pooled_lift_pp",0))
    r = gateway.generate(model, BRIEF_PROMPT.format(
            brand=target_brand,
            gap=finding.get("gap_detail", {}).get("gap", finding.get("gap_id")),
            lift=best_lift, engines=", ".join(sig_engines) or "tested engines",
            winning_content=winning_variant[:2000],
            pillar=finding.get("pillar"),
            hitlist=json.dumps(hitlist_domains[:8]),
        ), json_mode=True, module=module, task="brief_gen", temperature=0.3)
    brief = r["parsed"] or {}
    # ⭐ ATTACH the actual proven content — don't let LLM replace it
    brief["proven_content"] = winning_variant       # the verbatim winning variant
    brief["gap_id"] = finding.get("gap_id")
    brief["pillar"] = finding.get("pillar")
    brief["horizon"] = finding.get("horizon")
    brief["priority_score"] = finding.get("priority_score")
    brief["engines_targeted"] = ", ".join(sig_engines) or "none significant"  # Fix 3: real engines
    brief["best_engine_lift_pp"] = best_lift
    brief["_provisional"] = True
    return brief