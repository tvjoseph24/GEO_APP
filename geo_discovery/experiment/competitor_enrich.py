"""P2-1 — Rich competitor content for experiments.
Strategy: audit-derived (real) → grounded-search (fallback) → rich frozen mini-docs.
Fixes the 'no lift' problem: real competition vs. the target, not strawmen."""
import json
import re
from pathlib import Path
from collections import defaultdict

MIN_AUDIT_SNIPPETS = 3        # below this → trigger grounded search
MAX_COMPETITORS = 5           # cap (cost + context size)

# ======================================================================
#  1. AUDIT-DERIVED — mine what competitors actually said in audit responses
# ======================================================================
def extract_from_audit(raw_path, competitors, target_brand, query_filter=None):
    """Mine competitor claims/positioning from real audit responses.
    query_filter: if given, only mine responses to that specific query (query-relevant)."""
    if not Path(raw_path).exists():
        return {}
    comp_snippets = defaultdict(list)
    comp_names = [c["brand"] if isinstance(c, dict) else c for c in competitors]

    for line in open(raw_path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if not r.get("success"):
            continue
        if query_filter and query_filter.lower() not in r.get("query", "").lower():
            continue                                    # only the relevant query's responses
        text = r.get("response_text", "")
        # split into sentences; keep those naming a competitor
        for sent in re.split(r'(?<=[.!?])\s+', text):
            low = sent.lower()
            for cname in comp_names:
                if cname.lower() == target_brand.lower():
                    continue
                if cname.lower() in low and len(sent.strip()) > 30:
                    comp_snippets[cname].append(sent.strip())
    # dedupe + cap per competitor
    return {c: list(dict.fromkeys(snips))[:8] for c, snips in comp_snippets.items()}

# ======================================================================
#  2. GROUNDED-SEARCH FALLBACK — rich mini-brief for thin competitors
# ======================================================================
COMP_ENRICH_PROMPT = """Using live web search, write a realistic, factual product mini-brief
for the competitor brand "{comp}" in the "{category}" category. Base it on retailer listings,
reviews, and the brand's real, current positioning.

Return STRICT JSON:
{{
 "brand": "{comp}",
 "positioning": "1-2 sentences: how it positions itself in this category",
 "key_claims": ["3-5 concrete, specific claims/attributes it is actually known for"],
 "attributes": {{"attribute": "value"}}
}}
Examples of good attributes: format, price tier, key ingredient, target user, certifications.
Be factual and specific — real claims this brand makes. Do NOT invent."""

def enrich_via_grounding(gateway, comp_name, category, model="gemini-2.5-flash"):
    """Grounded web search → rich competitor mini-doc."""
    try:
        r = gateway.generate_grounded(
            COMP_ENRICH_PROMPT.format(comp=comp_name, category=category),
            model=model, module="experiment", task="comp_enrich", json_mode=True)
        return r["parsed"] if isinstance(r["parsed"], dict) else {}
    except Exception:
        return {}

# ======================================================================
#  3. BUILD RICH FROZEN MINI-DOCS  (audit-first, grounded-fallback)
# ======================================================================
def build_rich_competitor_context(gateway, competitors, category, target_brand,
                                  raw_path, model="gemini-2.5-flash",
                                  query_filter=None, progress_cb=None):
    """Return {competitor: rich_doc}. Frozen — built once, reused across variants."""
    audit_content = extract_from_audit(raw_path, competitors, target_brand, query_filter)
    comp_docs = {}
    comp_list = [c for c in competitors
                 if (c["brand"] if isinstance(c, dict) else c).lower() != target_brand.lower()]
    comp_list = comp_list[:MAX_COMPETITORS]
    n = max(len(comp_list), 1)

    for i, comp in enumerate(comp_list):
        cname = comp["brand"] if isinstance(comp, dict) else comp
        known = comp.get("known_strength", "") if isinstance(comp, dict) else ""
        snippets = audit_content.get(cname, [])

        if len(snippets) >= MIN_AUDIT_SNIPPETS:
            # rich enough from REAL audit data
            doc = {"brand": cname, "source": "audit-derived",
                   "positioning": snippets[0],
                   "key_claims": snippets[:5],
                   "attributes": {},
                   "known_strength": known}
        else:
            # thin → grounded search for rich content
            g = enrich_via_grounding(gateway, cname, category, model)
            doc = {"brand": cname, "source": "grounded-search",
                   "positioning": g.get("positioning", "") or known,
                   "key_claims": g.get("key_claims", []) or ([known] if known else []),
                   "attributes": g.get("attributes", {}),
                   "known_strength": known}
        comp_docs[cname] = doc
        if progress_cb:
            progress_cb((i + 1) / n)
    return comp_docs

# ======================================================================
#  4. RENDER → frozen text block for RAG injection (with tables)
# ======================================================================
def render_competitor_context(comp_docs):
    """Format rich competitor docs into the frozen markdown block injected into experiments."""
    if not comp_docs:
        return ""
    blocks = []
    for cname, doc in comp_docs.items():
        parts = [f"[{cname.upper()}]"]
        if doc.get("positioning"):
            parts.append(doc["positioning"])
        claims = doc.get("key_claims", [])
        if claims:
            parts.append("\n".join(f"- {c}" for c in claims))
        attrs = doc.get("attributes", {})
        if attrs:
            parts.append("| Attribute | Value |\n|---|---|\n" +
                         "\n".join(f"| {k} | {v} |" for k, v in attrs.items()))
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)

# ======================================================================
#  5. PERSIST  (frozen — reuse across runs; rebuild button in UI)
# ======================================================================
def save_competitor_context(parent_brand, target_brand, comp_docs):
    from store import save_artifact
    save_artifact(parent_brand, target_brand, "competitor_context", comp_docs,
                  {"note": "rich frozen competitor mini-docs (audit + grounded)"})

def load_competitor_context(parent_brand, target_brand):
    from store import load_artifact
    return load_artifact(parent_brand, target_brand, "competitor_context")