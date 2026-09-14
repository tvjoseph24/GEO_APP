"""Production briefs. A brief is the faithful translation of a validated experiment arm —
that is the payoff of having run the experiments."""
import json

BRIEF_PROMPT = """Write a production brief for "{brand}".

{evidence_block}

PROVEN WINNING CONTENT (this exact treatment was tested — publish this pattern):
---
{winning_content}
---

ACTION: {title}
WHAT: {what}
PILLAR: {pillar} · HORIZON: {horizon} · EVIDENCE: {evidence_class}
TARGET QUERIES: {queries}
APPROVED CLAIMS (the ONLY publishable facts — never add others):
{claims}
PLACEMENT TARGETS: {domains}
KILL CRITERION: {kill}

Your job is NOT to rewrite the content. It is to make it executable: placement,
corroboration, structural requirements and measurement.

Return STRICT JSON:
{{
 "title":"", "objective":"one line",
 "why_it_works":"the mechanism, citing the evidence",
 "deliverable":"exactly what gets produced, and where it is published",
 "must_contain":["structural and factual requirements a reviewer can tick off"],
 "must_not_contain":["claims or patterns excluded by evidence or regulation"],
 "placement":["specific surfaces or domains"],
 "corroboration":["third-party echoes needed to make the claims citable"],
 "owner":"", "effort":"",
 "success_metric":"how post-publish lift is measured",
 "measurement":"which queries to re-audit, N, engines, when",
 "caveats":["limits a reader must know before quoting the effect size"]
}}

Never invent a claim. If the evidence is caveated, say so in caveats."""

def _evidence_block(action, findings_by_id, nulls):
    refs = action.get("evidence_refs") or []
    lines = []
    for r in refs:
        f = findings_by_id.get(r)
        if not f:
            continue
        lines.append(
            f"- {r} ({f.get('template')}, {f.get('runs_per_cell')} runs/arm/engine): "
            f"{f.get('verdict')} · {f.get('lift')} {f.get('lift_unit','pp')} on "
            f"{f.get('best_metric')}"
            + (f" · engines {f.get('rank_engines')}" if f.get("rank_engines") else "")
            + ("  ⚠ UNDERPOWERED" if f.get("underpowered") else ""))
        for lim in (f.get("interpretation_limits") or [])[:2]:
            lines.append(f"    limit: {lim}")
    nl = [f"- {n['design_id']}: {n.get('isolated_variable')} had NO effect "
          f"({n.get('runs_per_cell')} runs/arm)"
          for n in nulls if not n.get("underpowered")][:4]
    return ("=== EVIDENCE FOR THIS ACTION ===\n" + ("\n".join(lines) or "(diagnostic only)")
            + ("\n\n=== WHAT THE EXPERIMENTS RULED OUT — do not recommend these ===\n"
               + "\n".join(nl) if nl else ""))

def generate_brief(gateway, action, winning_content, si, model="gemini-2.5-flash",
                   module="optimise"):
    findings_by_id = {f["design_id"]: f for f in (si.get("wins") or [])
                      if f.get("design_id")}
    res = gateway.generate_json(model, BRIEF_PROMPT.format(
        brand=si.get("brand"),
        evidence_block=_evidence_block(action, findings_by_id, si.get("nulls") or []),
        winning_content=(winning_content or "(no tested variant — DIAGNOSED action)")[:2200],
        title=action.get("title"), what=action.get("what"),
        pillar=action.get("pillar"), horizon=action.get("horizon"),
        evidence_class=action.get("evidence_class"),
        queries=json.dumps(action.get("target_queries") or []),
        claims=json.dumps(action.get("claims_used") or [], indent=1)[:2000],
        domains=json.dumps(action.get("target_domains") or []),
        kill=action.get("kill_criterion") or "(not stated)"),
        expect="dict", module=module, task="brief_gen", temperature=0.3, max_tokens=2500)

    b = res.get("data") or {}
    b.update({
        "action_id": action.get("action_id"), "pillar": action.get("pillar"),
        "horizon": action.get("horizon"), "clock": action.get("clock"),
        "evidence_class": action.get("evidence_class"),
        "evidence_refs": action.get("evidence_refs"),
        "priority_score": action.get("priority_score"),
        "proven_content": winning_content or "",
        "claims_required": action.get("claims_used") or [],
        "expected_effect": action.get("expected_effect"),
        "_generated_ok": bool(res.get("ok")),
    })
    return b

def winning_content_for(action, variants):
    """The tested arm this action derives from. Prefers the strongest arm of a cited
    design; falls back to empty for DIAGNOSED actions."""
    recs = variants.get("experiments") if isinstance(variants, dict) else (variants or [])
    by_id = {}
    for r in (recs or []):
        k = r.get("design_id") or r.get("gap_id")
        if k:
            by_id[k] = r
    for ref in (action.get("evidence_refs") or []):
        r = by_id.get(ref)
        if not r:
            continue
        arms = r.get("arms")
        if isinstance(arms, dict) and arms:
            for k in ("E", "D", "C", "B"):
                v = arms.get(k)
                txt = v if isinstance(v, str) else (v or {}).get("text", "")
                if txt:
                    return txt
        b = r.get("variant_B")
        b = b if isinstance(b, str) else (b or {}).get("text", "")
        if b:
            return b
    return ""

def save_briefs(parent, target, briefs, models_used=None):
    from store import save_artifact
    save_artifact(parent, target, "content_briefs", briefs, models_used or {})