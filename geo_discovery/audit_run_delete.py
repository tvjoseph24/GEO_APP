"""Audit runner: text (multi-engine, N, web on/off) + returned-image extraction
+ user-uploaded visual interrogation. Reuses Gateway/Meter. Resumable JSONL."""
import json, re, time, base64
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from classifier import classify
from url_resolver import HEADERS
import unicodedata



# ---- source-type map (reuse your canonical one) ----
SOURCE_TYPE_MAP = {
    "community_reviews": ["reddit.com","quora.com","trustpilot.com","makeupalley.com"],
    "video_social": ["youtube.com","youtube","tiktok.com","instagram.com"],
    "retailer": ["amazon.","walmart.com","target.com","cvs.com","walgreens.com","ulta.com","sephora.com"],
    "editorial_authority": ["healthline.com","webmd.com","mayoclinic.org","nih.gov","aad.org",
                            "nationaleczema.org","nytimes.com","wirecutter.com","byrdie.com"],
    "brand_owned": ["aveeno.com","cerave.com","cetaphil.com","nestle","jnj.com"],
}

_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"

def _norm(t):
    if not t: return ""
    for d in _DASHES: t = t.replace(d, "-")
    return unicodedata.normalize("NFKC", t).lower()

def _brand_hits(text, brands):
    low = _norm(text)
    out = {}
    for b in brands:
        nb = _norm(b)
        idx = low.find(nb)
        out[b] = {"mentioned": idx != -1, "first_index": idx}
    return out
def _stype(domain):
    d = (domain or "").lower()
    for st, pats in SOURCE_TYPE_MAP.items():
        if any(p in d for p in pats): return st
    return "other"

VISUAL_PAGE_PATTERNS = [r'facebook\.com/.+/(photos|videos)', r'instagram\.com',
                        r'youtube\.com', r'youtu\.be', r'/videos?/', r'pinterest\.']

# ---- IMAGE URL extraction from a response ----
def _extract_image_urls(text, sources):
    urls = set(re.findall(r'https?://\S+\.(?:png|jpe?g|webp|gif)', text, re.I))
    visual_pages = []
    for s in sources:
        u = s.get("resolved_url") or s.get("url", "")
        if re.search(r'\.(png|jpe?g|webp|gif)(\?|$)', u, re.I):
            urls.add(u)
        elif any(re.search(p, u, re.I) for p in VISUAL_PAGE_PATTERNS):
            visual_pages.append(u)          # visual content, but on a page
    return list(urls), visual_pages          # return both

# ---- Step 3: fetch + interrogate returned images (opt-in) ----
IMG_PROMPT = """Analyze this image found in an AI answer about products. Return JSON:
{"shows_product": true, "brand_visible": "", "product_type": "",
 "visible_claims": [], "image_role": "product_shot|comparison|lifestyle|logo|other"}"""

def _fetch_image(url, timeout=8, max_bytes=4_000_000):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        if r.status_code != 200: return None
        data = r.content[:max_bytes]
        mime = r.headers.get("Content-Type", "image/jpeg").split(";")[0]
        if "image" not in mime: return None
        return data, mime
    except Exception:
        return None

def interrogate_returned_images(gateway, image_urls, vision_model, max_images=3):
    """Fetch a few returned images and vision-interrogate them (opt-in, cost-adding)."""
    out = []
    for url in image_urls[:max_images]:
        fetched = _fetch_image(url)
        if not fetched:
            out.append({"url": url, "fetched": False}); continue
        data, mime = fetched
        try:
            r = gateway.generate(vision_model, IMG_PROMPT, images=[(data, mime)],
                                 json_mode=True, module="audit", task="image_interrogate")
            insight = r["parsed"] or {}
            insight.update({"url": url, "fetched": True})
            out.append(insight)
        except Exception as e:
            out.append({"url": url, "fetched": True, "error": str(e)[:100]})
    return out

# ---- Step 2: ONE audit run ----
def _one_run(gateway, engine_cfg, query, target_brand, competitors, web_on,
             run_index, *, use_llm_classify, judge_model,
             extract_images, vision_model):
    model = engine_cfg["model"]
    provider = engine_cfg["provider"]
    try:
        if provider == "vertex" and web_on:
            r = gateway.generate_grounded(query, model=model, module="audit", task="run")
            text, sources = r["text"], r["sources"]
            web_queries = r.get("web_search_queries", [])

        elif provider == "litellm" and web_on:
            # PROVEN ChatGPT web-search path: responses API + url_citation parsing
            r = gateway.generate_web_litellm(model, query, module="audit", task="run",
                                             temperature=1.0 if "gpt-5" in model else 0.3)
            text = r["text"]
            sources = r["sources"]  # REAL urls, already classified-ready
            web_queries = r["web_search_queries"]

        else:  # web OFF (memory) — either provider
            r = gateway.generate(model, query, module="audit", task="run",
                                 temperature=1.0 if "gpt-5" in model else 0.3)
            text = r["text"];
            sources = [];
            web_queries = []

        # enrich sources with type (works for both engines now)
        for s in sources:
            s["source_type"] = _stype(s.get("domain", ""))
        ...

        cls = classify(gateway, text, target_brand, competitors,
                       use_llm=use_llm_classify, judge_model=judge_model)
        img_urls = _extract_image_urls(text, sources)
        img_insights = (interrogate_returned_images(gateway, img_urls, vision_model)
                        if (extract_images and img_urls) else [])

        return {"query": query, "engine": engine_cfg["name"], "model": model,
                "web_on": web_on, "run_index": run_index,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True, "response_text": text,
                "sources": sources, "num_sources": len(sources),
                "source_type_tally": _tally(sources),
                "image_urls": img_urls, "num_images": len(img_urls),
                "web_search_queries": web_queries,
                "image_insights": img_insights,
                "brand_mentions": _brand_hits(text, [target_brand] + competitors),
                "outcome": cls.get("outcome"), "classification": cls}
    except Exception as e:
        return {"query": query, "engine": engine_cfg["name"], "web_on": web_on,
                "run_index": run_index, "success": False, "error": str(e)[:200]}

def _tally(sources):
    t = {}
    for s in sources:
        st = s.get("source_type", "other"); t[st] = t.get(st, 0) + 1
    return t

# ---- Resumable batch runner with cost governance ----
class BudgetExceeded(Exception): pass

def _already_done(out_path):
    if not Path(out_path).exists(): return set()
    done = set()
    for l in open(out_path, encoding="utf-8"):
        try:
            r = json.loads(l)
            if r.get("success"):
                done.add((r["query"], r["engine"], r["web_on"], r["run_index"]))
        except Exception: pass
    return done

def estimate_cost(n_queries, n_engines, n_runs, web_modes, avg_cost_per_call=0.002):
    calls = n_queries * n_engines * n_runs * len(web_modes)
    return calls, round(calls * avg_cost_per_call, 2)

def run_audit(gateway, queries, engines, target_brand, competitors, *,
              n_runs=10, web_modes=("on",), out_path="audit_raw.jsonl",
              use_llm_classify=False, judge_model="gemini-2.5-flash",
              extract_images=False, vision_model="gemini-2.5-flash",
              budget=None, hard_stop=False, progress_cb=None):
    """queries: [{query,intent},...]; engines: [{name,model,provider},...]"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    done = _already_done(out_path)             # resumability
    tasks = [(q, e, (w == "on"), r)
             for q in queries for e in engines
             for w in web_modes for r in range(n_runs)
             if (q["query"], e["name"], (w == "on"), r) not in done]
    total = len(tasks); completed = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for (q, e, web_on, r) in tasks:
            # cost governance
            if budget is not None and gateway.meter.total_cost() >= budget:
                if hard_stop:
                    raise BudgetExceeded(f"Budget ${budget} reached at "
                                         f"${gateway.meter.total_cost():.2f}")
                # soft: warn via callback, continue
                if progress_cb: progress_cb(completed, total, warn=True)
            rec = _one_run(gateway, e, q["query"], target_brand, competitors, web_on, r,
                           use_llm_classify=use_llm_classify, judge_model=judge_model,
                           extract_images=extract_images, vision_model=vision_model)
            rec["intent"] = q.get("intent")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            completed += 1
            if progress_cb: progress_cb(completed, total, warn=False)
    return out_path

# ---- Step 4: user-uploaded visual interrogation (opt-in) ----
VT_PROMPT = """A shopper shows you this image and asks: "{query}"
Answer as an AI assistant would. Then note: what could you READ from the image,
and did it influence your answer?"""

def visual_interrogation(gateway, images, query, engines, vision_model="gemini-2.5-flash"):
    """images: [(bytes,mime)]. Tests how engines respond to image+text."""
    results = []
    for e in engines:
        if "vision" not in e.get("supports", []):
            results.append({"engine": e["name"], "skipped": "no vision support"}); continue
        try:
            r = gateway.generate(e["model"], VT_PROMPT.format(query=query),
                                 images=images, module="audit", task="visual_interrogate")
            results.append({"engine": e["name"], "response": r["text"], "success": True})
        except Exception as ex:
            results.append({"engine": e["name"], "error": str(ex)[:150]})
    return results

############ from experiment tab to be deleted

