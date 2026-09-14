"""Generate A/B content variants: LLM-drafted text + user-uploaded visual."""
import json

# Base competitor docs are FROZEN (the controlled field). Only the target doc varies.
VARIANT_PROMPT = """You are creating a controlled A/B GEO content experiment for "{brand}".
Generate TWO genuinely different versions of a brand content block.

EXPERIMENT: {exp_name} — {isolates}
HYPOTHESIS: {hypothesis}
TARGET QUESTION (content must help answer THIS): "{question}"
VERIFIED CLAIMS (use ONLY these — never invent):
{claims}

Return STRICT JSON:
{{"variant_A": {{"label":"A (plain)", "text":""}},
  "variant_B": {{"label":"B (GEO-enriched)", "text":""}}}}

VARIANT A — plain baseline:
- Simple prose. No formatting, no tables, no explicit structure. One flowing paragraph.

VARIANT B — FULL GEO treatment. MUST include ALL of:
1. A **bold answer-capsule sentence** at the top directly answering the question.
2. QUANTIFIED, SPECIFIC claims (numbers, dates, proof) — not vague statements.
3. A MARKDOWN COMPARISON TABLE (| col | col |) with ≥3 attribute rows relevant to the question.
4. CITED claims (inline source, e.g. "(NESCAFÉ Plan)").
5. Clear structure (headings/bullets).

CRITICAL: A and B must be SUBSTANTIALLY different in structure and specificity.
Both use ONLY the verified claims. B adds STRUCTURE + SPECIFICITY, never new facts."""

def generate_text_variants(gateway, design, claims, target_brand,
                           model="gemini-2.5-flash", module="experiment"):
    """LLM-draft A (plain) vs B (GEO-enriched). Retries on empty/missing-table."""
    gap_words = set(design["gap"].lower().split())
    relevant = [c["claim"] for c in claims
                if gap_words & set(c.get("claim","").lower().split())]
    if len(relevant) < 4:
        relevant = list({*relevant, *[c["claim"] for c in claims]})[:8]

    def _gen():
        r = gateway.generate(model, VARIANT_PROMPT.format(
                brand=target_brand, exp_name=design["experiment_name"],
                isolates=design["isolates"], hypothesis=design["hypothesis"],
                question=design.get("question", ""), claims=json.dumps(relevant[:8])),
            json_mode=True, module=module, task="variant_gen", temperature=0.4)
        v = r["parsed"] or {}
        return (v.get("variant_A", {"label":"A","text":""}),
                v.get("variant_B", {"label":"B","text":""}))

    va, vb = _gen()
    va_t, vb_t = va.get("text","").strip(), vb.get("text","").strip()

    # RETRY once if either empty OR B lacks a table (Fix #2 + #4)
    if (not va_t or not vb_t) or ("|" not in vb_t):
        va2, vb2 = _gen()
        if va2.get("text","").strip() and vb2.get("text","").strip():
            va, vb = va2, vb2
            va_t, vb_t = va.get("text","").strip(), vb.get("text","").strip()

    # final quality flag
    if not va_t or not vb_t:
        warn = "⚠️ EMPTY variant — SKIP this experiment"
    elif "|" not in vb_t and "**" not in vb_t:
        warn = "⚠️ B lacks table AND capsule — weak treatment"
    elif "|" not in vb_t:
        warn = "⚠️ B has capsule but NO table"
    else:
        warn = ""
    return {"variant_A": va, "variant_B": vb, "claims_used": relevant[:8],
            "quality_warning": warn}

def attach_visual_variant(design, image_bytes, mime):
    """User-uploaded visual for E7/E8 (or opt-in on any experiment).
    B variant gets the image; A stays text-only."""
    design["visual_asset"] = {"bytes": image_bytes, "mime": mime}
    return design

# FROZEN competitor context (built once from competitive_set)
def build_competitor_context(competitors, claims_by_competitor=None):
    """Simple frozen competitor blocks — the controlled field."""
    blocks = []
    for c in competitors[:3]:
        name = c if isinstance(c, str) else c.get("brand", "")
        blocks.append(f"[{name.upper()}]\n{name} is a competing product in this category.")
    return "\n\n".join(blocks)

PROGRESSIVE_PROMPT = """Create PROGRESSIVE GEO content variants for "{brand}", each ADDING one
element to the previous (cumulative). Use ONLY the verified claims — never invent.

TARGET QUESTION (content must help answer): "{question}"
VERIFIED CLAIMS:
{claims}

Return STRICT JSON with these cumulative arms:
{{
 "A": {{"label":"plain", "text":"<simple prose paragraph, no formatting>"}},
 "B": {{"label":"+capsule", "text":"<A's content + a **bold answer-capsule sentence** at top>"}},
 "C": {{"label":"+table", "text":"<B's content + a markdown comparison table, ≥3 rows>"}},
 "D": {{"label":"+citations", "text":"<C's content + inline cited/quantified claims e.g. '(NESCAFÉ Plan)'>"}},
 "E": {{"label":"full", "text":"<D's content + clear headings/bullets — fully structured>"}}
}}

CRITICAL:
- Each arm is CUMULATIVE (contains everything from the prior arm + ONE new element).
- A must be plain (no bold, no table, no citations).
- The table in C+ must be real markdown (| col | col |) with attributes relevant to the question.
- All arms use the SAME facts; only STRUCTURE/SPECIFICITY increases."""

def generate_progressive_variants(gateway, design, claims, target_brand,
                                  model="gemini-2.5-flash", module="experiment"):
    """Generate 5 cumulative arms (A→E) for E1-style structure isolation."""
    gap_words = set(design["gap"].lower().split())
    relevant = [c["claim"] for c in claims
                if gap_words & set(c.get("claim","").lower().split())]
    if len(relevant) < 4:
        relevant = list({*relevant, *[c["claim"] for c in claims]})[:8]
    r = gateway.generate(model, PROGRESSIVE_PROMPT.format(
            brand=target_brand, question=design.get("question",""),
            claims=json.dumps(relevant[:8])),
        json_mode=True, module=module, task="progressive_variants", temperature=0.4)
    v = r["parsed"] or {}
    arms = {}
    for k in ["A","B","C","D","E"]:
        arms[k] = v.get(k, {"label": k, "text": ""})
    # quality guards
    warnings = []
    if "**" not in arms["B"].get("text",""): warnings.append("B missing capsule")
    if "|" not in arms["C"].get("text",""):  warnings.append("C missing table")
    return {"arms": arms, "claims_used": relevant[:8],
            "quality_warning": "; ".join(warnings)}
