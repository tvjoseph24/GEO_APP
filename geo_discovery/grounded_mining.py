"""Grounded-search mining: competitors + real-demand queries.
Uses Gemini + Google Search grounding (no agent). Feeds human gates.
Brand-scoped: operates on a PRODUCT brand (target) with parent as context."""
import json, re
from datetime import datetime

# ---------------- COMPETITOR DISCOVERY ----------------
COMPETITOR_PROMPT = """You are a category analyst. Using live web search, identify the main
competitors of the PRODUCT BRAND "{target_brand}" (owned by {parent_brand}) in its
category: "{category}".

CRITICAL: Find competitors of the PRODUCT "{target_brand}" — NOT of the parent company
"{parent_brand}".
Example: for "Nescafé" → "Folgers, Starbucks, Maxwell House" (NOT "Unilever, PepsiCo").

Base this on RETAILER listings, EDITORIAL "best of" roundups, and COMMUNITY discussion
(Reddit/forums) — NOT the brand's own marketing.

Return STRICT JSON list (5-10 product-level competitors):
[{{"brand":"", "role":"leader|challenger|niche",
   "known_strength":"", "evidence":"where you saw this (site type)"}}]
Only include real, current PRODUCT-level competitors. Do not invent."""

def discover_competitors(gateway, target_brand, parent_brand, category,
                         doc_mentions=None, model="gemini-2.5-flash", module="discover"):
    r = gateway.generate_grounded(
        COMPETITOR_PROMPT.format(target_brand=target_brand, parent_brand=parent_brand,
                                 category=category),
        model=model, module=module, task="competitor_discovery", json_mode=True)
    comps = r["parsed"] if isinstance(r["parsed"], list) else []
    for c in comps:
        c.setdefault("source", "web_grounded")
    # merge any competitors named in the brand's own docs (secondary)
    for name in (doc_mentions or []):
        if not any(name.lower() == c.get("brand", "").lower() for c in comps):
            comps.append({"brand": name, "role": "unknown", "known_strength": "",
                          "evidence": "mentioned in brand docs", "source": "doc_mention"})
    return comps, r.get("sources", [])

# ---------------- REAL-DEMAND QUERY MINING ----------------
DEMAND_PROMPT = """You are a consumer-search analyst. Using live web search, find how REAL people
ask about the "{category}" category (needs the brand "{target_brand}" could serve).

CURRENT DATE: {current_date}.

CRITICAL: Capture BRAND-FREE consumer queries — how people phrase NEEDS, not brand lookups.
Do NOT return queries naming "{target_brand}" or "{parent_brand}" or sibling brands.
We measure whether the brand surfaces UNPROMPTED, so brand-free queries are essential.

Capture actual phrasing from Reddit, forums, People-Also-Ask, Q&A sites.
Prioritise: need-based, attribute-seeking, persona, negative/skeptical, and freshness
(latest as of {current_date}) — ALL without naming any brand.

Return STRICT JSON (15-25 brand-free queries):
[{{"query":"", "intent":"need|attribute|persona|longtail|negative|freshness",
   "modality":"text", "winnability_hypothesis":"winnable|commoditized|unwinnable",
   "evidence_url":""}}]"""

def mine_real_demand(gateway, target_brand, parent_brand, category,
                     model="gemini-2.5-flash", module="discover"):
    now = datetime.now()
    r = gateway.generate_grounded(
        DEMAND_PROMPT.format(target_brand=target_brand, parent_brand=parent_brand,
                             category=category, current_date=now.strftime("%B %Y")),
        model=model, module=module, task="demand_mining", json_mode=True)
    queries = r["parsed"] if isinstance(r["parsed"], list) else []
    for q in queries:
        q["source"] = "real_demand_mined"
    return queries, r.get("sources", [])

# ---------------- MERGE brand-derived + mined queries ----------------
def merge_query_masters(brand_derived, mined):
    """Combine, dedupe (case-insensitive), tag source."""
    seen, out = {}, []
    for q in (brand_derived or []):
        q.setdefault("source", "brand_derived")
        key = q.get("query", "").strip().lower()
        if key and key not in seen:
            seen[key] = q; out.append(q)
    for q in (mined or []):
        key = q.get("query", "").strip().lower()
        if key and key not in seen:
            seen[key] = q; out.append(q)
    return out