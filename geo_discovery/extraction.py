"""LLM extraction: 3-layer positioning + draft Claims Register + draft Query Master.
Vision path for image sources included (Gemini)."""
import json
from datetime import datetime

POSITIONING_PROMPT = """You are a brand strategist. From the SOURCE below, extract brand
positioning pillars. Classify EACH pillar independently into one of three layers:

- "aspirational"   = a benefit promise / brand intent (e.g., "gentle care for sensitive skin")
- "expressed"      = current in-market campaign messaging / tone (from ads/creative)
- "substantiated"  = a provable, citable fact (ingredient, certification, clinical result)

A single source often contains MULTIPLE layers — classify each pillar on its own merits,
do not force them all into one layer.

Return STRICT JSON:
{{
 "pillars": [
   {{"pillar": "", "statement": "", "layer": "aspirational|expressed|substantiated",
     "evidence_in_source": ""}}
 ],
 "source_dominant_layer": "aspirational|expressed|substantiated",
 "notes": ""
}}
Only use what the source supports. Do not invent.
IGNORE and DO NOT extract any content about: cookie policies, privacy notices,
consent banners, opt-out mechanisms, data-sharing/advertising disclosures, security
checks, or website terms of use. These are legal/technical boilerplate, NOT brand
positioning or product claims. Extract ONLY brand/product substance.

SOURCE_TYPE: {stype}
SOURCE:
{content}"""

CLAIMS_PROMPT = """You are a claims analyst for a regulated category. From the SOURCE,
extract every FACTUAL, potentially-citable claim about the product/brand.
Return STRICT JSON: a list of objects:
[{{"claim":"", "category":"ingredient|efficacy|credential|safety|feature|other",
   "proof_in_source":"", "proof_type":"regulatory|clinical|third_party|internal|none",
   "legal_ok":"yes|no|verify", "confidence":0.0}}]
Rules: extract only claims the source states. Mark legal_ok='verify' unless proof is explicit.
Do NOT fabricate claims or proofs.
IGNORE and DO NOT extract any content about: cookie policies, privacy notices,
consent banners, opt-out mechanisms, data-sharing/advertising disclosures, security
checks, or website terms of use. These are legal/technical boilerplate, NOT brand
positioning or product claims. Extract ONLY brand/product substance.

SOURCE:
{content}"""

QUERY_DRAFT_PROMPT = """You are a consumer-search analyst. Propose a Query Master for the
PRODUCT BRAND "{target_brand}" (owned by {parent_brand}) in category "{category}".

CURRENT DATE: {current_date}  — use this for freshness queries.

CRITICAL BRAND-NAMING RULES (this is essential for valid measurement):
- ONLY "branded" and "comparative" queries may contain the brand name "{target_brand}".
- ALL OTHER categories MUST NOT mention "{target_brand}" or "{parent_brand}" — because we
  measure whether the brand surfaces UNPROMPTED. Naming it would rig the result.

PER-CATEGORY DESIGN:
- branded (name brand): probe the brand's claims to test if AI describes them CORRECTLY.
  e.g. "Is {target_brand}'s [claim/product] effective?"
- comparative (name brand + a competitor): head-to-head on attributes.
  e.g. "{target_brand} vs [competitor] for [attribute]"
- need (NO brand): pure consumer need in the category.
  e.g. "best [category] for [need]"
- attribute (NO brand): built from the brand's OWN claims below — test if it's recommended
  when someone wants that attribute. e.g. "best [category] with [attribute]"
- persona (NO brand): need + user-type. e.g. "best [category] for [persona]"
- longtail (NO brand): specific niche need, no brand.
- negative (NO brand): category concern/skepticism. e.g. "which [category] products cause [problem]?"
- freshness (NO brand): newest/latest as of {current_date}. e.g. "newest [category] launched in {year}"

BRAND'S VERIFIED CLAIMS (use these to build ATTRIBUTE queries — strip the brand name):
{claims}

Return STRICT JSON (18-25 queries, spread across ALL categories):
[{{"query":"", "intent":"branded|comparative|need|attribute|persona|longtail|negative|freshness",
   "modality":"text", "winnability_hypothesis":"winnable|commoditized|unwinnable"}}]

REMINDER: only branded & comparative name the brand. Everything else is brand-free."""

def _layer_for(source_type):
    return {"doc": "aspirational", "web": "aspirational",
            "image": "expressed", "video": "expressed"}.get(source_type, "aspirational")

def extract_positioning(gateway, source, model, module="discover"):
    images = source.media if source.source_type in ("image", "video") else None
    prompt = POSITIONING_PROMPT.format(stype=source.source_type,
                                       content=(source.text or "(see image)")[:30000])
    r = gateway.generate(model, prompt, images=images, json_mode=True,
                         module=module, task="positioning", temperature=0.2)
    data = r["parsed"] or {"pillars": [], "source_dominant_layer": _layer_for(source.source_type),
                           "notes": "parse_failed", "_raw": r["text"][:500]}
    # source-type hint as fallback if the model omitted dominant layer
    data.setdefault("source_dominant_layer", _layer_for(source.source_type))
    data["_origin"] = source.origin
    data["_source_type_hint"] = _layer_for(source.source_type)   # keep the hint for audit
    return data

def extract_claims(gateway, source, model, module="discover"):
    images = source.media if source.source_type in ("image", "video") else None
    prompt = CLAIMS_PROMPT.format(content=(source.text or "(see image)")[:25000])
    r = gateway.generate(model, prompt, images=images, json_mode=True,
                         module=module, task="claims", temperature=0.1)
    claims = r["parsed"] if isinstance(r["parsed"], list) else []
    for c in claims:
        c["_origin"] = source.origin
    return claims

def draft_query_master(gateway, positioning_layers, competitors, target_brand,
                       parent_brand, category, model, claims=None, module="discover"):
    now = datetime.now()
    claims_txt = json.dumps([c.get("claim","") for c in (claims or [])][:15])[:3000]
    context = json.dumps({"positioning": positioning_layers,
                          "competitors": competitors}, indent=2)[:5000]
    r = gateway.generate(model, QUERY_DRAFT_PROMPT.format(
        target_brand=target_brand, parent_brand=parent_brand, category=category,
        current_date=now.strftime("%B %Y"), year=now.year, claims=claims_txt),
        json_mode=True, module=module, task="query_mining", temperature=0.4)
    queries = r["parsed"] if isinstance(r["parsed"], list) else []
    # append the context separately (kept short above to leave room for claims)
    return queries

def dedupe_claims(claims):
    """Merge duplicate claims across sources."""
    seen = {}
    for c in claims:
        key = (c.get("claim", "").strip().lower())
        if not key: continue
        if key in seen:
            seen[key].setdefault("_origins", []).append(c.get("_origin"))
        else:
            c["_origins"] = [c.get("_origin")]
            seen[key] = c
    return list(seen.values())

CREATIVE_PROMPT = """You are a brand-creative analyst. Interrogate this brand CREATIVE
(a static ad or video). Creative PERFORMS positioning — extract what it communicates.

Return STRICT JSON:
{{
 "expressed_message": "the core benefit/message the creative communicates",
 "visible_claims": ["exact text of any claim shown or spoken"],
 "brand_tone": "emotional/positioning tone (e.g., warm, clinical, premium)",
 "machine_readable": "what a machine can DIRECTLY read (on-screen text, logos)",
 "human_inferred": "what a human INFERS but is NOT explicit (mood, implication)",
 "positioning_pillars": [{{"pillar":"", "statement":"", "layer":"expressed"}}],
 "claims": [{{"claim":"", "category":"ingredient|efficacy|credential|safety|feature|other",
              "proof_in_source":"visible/spoken in creative", "proof_type":"internal",
              "legal_ok":"verify"}}]
}}
IGNORE cookie/legal/technical text. Extract ONLY brand/product substance.
Do not invent claims not present in the creative."""

def extract_creative(gateway, source, model, module="discover"):
    """Vision/video interrogation of an ad. source.media = [(bytes, mime)]."""
    r = gateway.generate(model, CREATIVE_PROMPT, images=source.media,
                         json_mode=True, module=module, task="creative", temperature=0.2)
    data = r["parsed"] or {"positioning_pillars": [], "claims": [],
                           "notes": "parse_failed", "_raw": r["text"][:500]}
    data["_origin"] = source.origin
    # tag origin on each extracted item so it flows into the shared gates
    for p in data.get("positioning_pillars", []):
        p["_origin"] = source.origin; p.setdefault("layer", "expressed")
    for c in data.get("claims", []):
        c["_origin"] = source.origin; c["_origins"] = [source.origin]
    return data