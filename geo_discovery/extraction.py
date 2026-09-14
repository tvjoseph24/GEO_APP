"""LLM extraction: 3-layer positioning + draft Claims Register + draft Query Master.
Vision path for image sources included (Gemini)."""
import json
from datetime import datetime
import re

MAX_TOKENS_EXTRACT = 6000     # was implicitly 1500 → arrays truncated mid-claim

def _repair_json(text):
    """Recover fenced / preambled / trailing-comma JSON without an LLM call."""
    if not text:
        return None
    t = str(text).strip()
    t = re.sub(r"^$", "", t, flags=re.S).strip()
    for opener, closer in (("[", "]"), ("{", "}")):
        i = t.find(opener)
        if i == -1:
            continue
        j = t.rfind(closer)
        if j > i:
            cand = t[i:j + 1]
            cand = re.sub(r",\s*([\]}])", r"\1", cand)      # trailing commas
            try:
                return json.loads(cand)
            except Exception:
                pass
    return None

def _parse_or_repair(gateway, r, model, schema_hint, source, module, task,
                     retry_prompt=None, repair_model="gemini-2.5-flash"):
    """Returns (data, note). Escalates: truncation-retry → local repair → LLM repair."""
    if r.get("parsed") is not None and not r.get("truncated"):
        return r["parsed"], None
    if r.get("truncated") and retry_prompt:
        try:
            r2 = gateway.generate(model, retry_prompt, json_mode=True, module=module,
                                  task=f"{task}_retry", temperature=0.1,
                                  max_tokens=12000)
            if r2.get("parsed") is not None and not r2.get("truncated"):
                return r2["parsed"], "recovered_after_truncation"
            r = r2
        except Exception:
            pass
    fixed = _repair_json(r.get("text"))
    if fixed is not None:
        return fixed, "repaired_locally"
    try:
        rr = gateway.generate(
            repair_model,
            f"Return ONLY valid JSON matching this shape:\n{schema_hint}\n\n"
            f"Fix this malformed model output:\n{(r.get('text') or '')[:12000]}",
            json_mode=True, module=module, task=f"{task}_json_repair",
            temperature=0.0, max_tokens=8000)
        if rr.get("parsed") is not None:
            return rr["parsed"], "repaired_by_llm"
    except Exception:
        pass
    source.meta.setdefault("extract_failures", []).append({
        "task": task, "origin": source.origin,
        "finish_reason": r.get("finish_reason"),
        "truncated": bool(r.get("truncated")),
        "raw_full": r.get("text") or "",          # discover_tab writes this to a blob
        "raw_preview": (r.get("text") or "")[:1500]})
    return None, "parse_failed"



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

COMPETITORS (use these EXACT names for comparative queries — never invent a competitor):
{competitors}

BRAND POSITIONING PILLARS (use for ATTRIBUTE and PERSONA queries — strip the brand name):
{pillars}

BRAND'S VERIFIED CLAIMS (use these to build ATTRIBUTE queries — strip the brand name):
{claims}

Return STRICT JSON (18-25 queries, spread across ALL categories):
[{{"query":"", "intent":"branded|comparative|need|attribute|persona|longtail|negative|freshness",
   "modality":"text", "winnability_hypothesis":"winnable|commoditized|unwinnable"}}]

REMINDER: only branded & comparative name the brand. Everything else is brand-free."""

def _layer_for(source_type):
    return {"doc": "aspirational", "web": "aspirational",
            "image": "expressed", "video": "expressed"}.get(source_type, "aspirational")

'''

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
'''
def extract_positioning(gateway, source, model, module="discover"):
    images = source.media if source.source_type in ("image", "video") else None
    prompt = POSITIONING_PROMPT.format(stype=source.source_type,
                                       content=(source.text or "(see image)")[:30000])
    r = gateway.generate(model, prompt, images=images, json_mode=True,
                         module=module, task="positioning", temperature=0.2,
                         max_tokens=MAX_TOKENS_EXTRACT)
    data, note = _parse_or_repair(gateway, r, model,
                                  '{"pillars":[{"pillar","statement","layer"}]}',
                                  source, module, "positioning", retry_prompt=prompt)
    if data is None:
        data = {"pillars": [], "source_dominant_layer": _layer_for(source.source_type),
                "notes": "parse_failed"}
    if note:
        data["_parse_note"] = note
    data.setdefault("source_dominant_layer", _layer_for(source.source_type))
    data["_origin"] = source.origin
    data["_source_type_hint"] = _layer_for(source.source_type)
    for p in data.get("pillars", []):
        p["_origin"] = source.origin
    return data

def extract_claims(gateway, source, model, module="discover"):
    images = source.media if source.source_type in ("image", "video") else None
    prompt = CLAIMS_PROMPT.format(content=(source.text or "(see image)")[:25000])
    r = gateway.generate(model, prompt, images=images, json_mode=True,
                         module=module, task="claims", temperature=0.1,
                         max_tokens=MAX_TOKENS_EXTRACT)
    data, note = _parse_or_repair(gateway, r, model,
                                  '[{"claim","category","proof_in_source",'
                                  '"proof_type","legal_ok","confidence"}]',
                                  source, module, "claims", retry_prompt=prompt)
    claims = data if isinstance(data, list) else []
    for i, c in enumerate(claims):
        c["_origin"] = source.origin
        c.setdefault("claim_id", f"CL-{abs(hash((source.origin, c.get('claim', ''))))%10**6:06d}")
        if note:
            c["_parse_note"] = note
    return claims

'''

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
'''
def draft_query_master(gateway, positioning_layers, competitors, target_brand,
                       parent_brand, category, model, claims=None, module="discover"):
    now = datetime.now()
    comp_names = [c["brand"] if isinstance(c, dict) else c
                  for c in (competitors or [])][:8]
    pillars = []
    for layer, items in (positioning_layers or {}).items():
        for pil in (items or [])[:6]:
            pillars.append(f"[{layer}] {pil.get('pillar','')}: {pil.get('statement','')}")
    claims_txt = json.dumps([c.get("claim", "") for c in (claims or [])][:15])[:3000]
    r = gateway.generate(model, QUERY_DRAFT_PROMPT.format(
        target_brand=target_brand, parent_brand=parent_brand, category=category,
        current_date=now.strftime("%B %Y"), year=now.year, claims=claims_txt,
        competitors=json.dumps(comp_names),
        pillars=json.dumps(pillars)[:2500]),
        json_mode=True, module=module, task="query_mining", temperature=0.4,
        max_tokens=4000)
    queries = r["parsed"] if isinstance(r["parsed"], list) else []
    # deterministic top-up: comparative coverage must not depend on model compliance
    have_comp = sum(1 for q in queries if q.get("intent") == "comparative")
    for comp in comp_names[: max(0, 4 - have_comp)]:
        queries.append({"query": f"{target_brand} vs {comp} for {category}".strip(),
                        "intent": "comparative", "modality": "text",
                        "winnability_hypothesis": "winnable",
                        "source": "deterministic_topup"})
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

'''
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

'''
def extract_creative(gateway, source, model, module="discover"):
    r = gateway.generate(model, CREATIVE_PROMPT, images=source.media,
                         json_mode=True, module=module, task="creative",
                         temperature=0.2, max_tokens=MAX_TOKENS_EXTRACT)
    data, note = _parse_or_repair(gateway, r, model,
                                  '{"positioning_pillars":[],"claims":[]}',
                                  source, module, "creative")
    if data is None:
        data = {"positioning_pillars": [], "claims": [], "notes": "parse_failed"}
    if note:
        data["_parse_note"] = note
    data["_origin"] = source.origin
    for p in data.get("positioning_pillars", []):
        p["_origin"] = source.origin
        p.setdefault("layer", "expressed")
    for c in data.get("claims", []):
        c["_origin"] = source.origin
        c["_origins"] = [source.origin]
        c.setdefault("claim_id", f"CL-{abs(hash((source.origin, c.get('claim', ''))))%10**6:06d}")
    return data
