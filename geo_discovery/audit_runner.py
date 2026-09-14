
############# New Code ##########################
"""Audit runner: text (multi-engine, N, web on/off) + returned-image extraction
+ user-uploaded visual interrogation. Resumable JSONL. All measurement via metrics_spec."""
import base64
import hashlib
import io
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from classifier import classify
from metrics_spec import build_registry, measure, fold
from url_resolver import HEADERS

# ---- source-type map (BASE — extend at runtime with the brand's own domains) ----
SOURCE_TYPE_MAP = {
    "community_reviews": ["reddit.com", "quora.com", "trustpilot.com", "stackexchange",
                          "mumsnet.com", "netmums.com", "babycenter.com", "babycentre.co.uk",
                          "whattoexpect.com", "thebump.com", "madeformums.com",
                          "productreview.com.au", "influenster.com", "peanut-app.io"],

    "video_social": ["youtube.com", "youtu.be", "tiktok.com", "instagram.com",
                     "facebook.com", "pinterest.", "x.com", "twitter.com"],

    "editorial_authority": [
        # clinical / regulatory / academic
        "healthychildren.org", "aap.org", "nih.gov", "ncbi.nlm.nih.gov", "pubmed",
        "cochrane.org", "bmj.com", "thelancet.com", "cdc.gov", "fda.gov", "who.int",
        "efsa.europa.eu", "nhs.uk", "nice.org.uk", "mayoclinic.org",
        "kidswithfoodallergies.org", "healthcare.utah.edu", ".edu/", "hopkinsmedicine",
        "clevelandclinic.org", "chop.edu",
        # editorial / consumer press
        "healthline.com", "webmd.com", "medicalnewstoday.com", "verywellfamily.com",
        "verywellhealth.com", "parents.com", "forbes.com", "nytimes.com",
        "wirecutter.com", "goodhousekeeping", "which.co.uk", "consumerreports.org",
        "solidstarts.com",
    ],

    "trade_press": ["nutraingredients", "foodnavigator", "just-food", "fooddive",
                    "dairyreporter", "beveragedaily"],

    "retailer": ["amazon.", "walmart.com", "target.com", "cvs.com", "walgreens.com",
                 "kroger.com", "costco.com", "samsclub.com", "boots.com",
                 "chemistwarehouse", "tesco.com", "sainsburys", "asda.com", "ocado.com",
                 "woolworths.com", "coles.com", "iherb.com", "babylist.com",
                 "buybuybaby", "firstcry", "pharmeasy", "netmeds"],

    # ── Parallel-import / grey-market resellers. Broken out DELIBERATELY: they carry no
    # ── compliance obligation, publish uncontrolled claims, and are a distinct
    # ── strategic problem from authorised retail. Quantify, don't bury in `retailer`.
    "grey_market_retailer": [
        "organiclifestart", "euromallusa", "organicsbestshop", "formuland",
        "groworganicbaby", "organicformulashop", "babybest.store", "babysvariety",
        "bonpek", "childrensland", "kidscity.in", "blwstore", "formulaatlas",
        "myorganiccompany", "littlebundle", "organicbabyfood",
    ],

    # ── SEO/affiliate content: reads like editorial, has no editorial standard.
    "content_affiliate": ["teachtoddler.com", "pickyeaterblog", "stuckathomemom",
                          "thegoodtrade.com", "mamaheart.org", "sanupostpartum",
                          "medium.com", "substack.com", "blogspot.", "wordpress.com"],

    "reference": ["wikipedia.org", "wikidata.org", "britannica.com"],

    # ── Non-content: CDN/asset hosts. Excluded from provenance maths.
    "cdn_asset": ["squarespace.com", "cloudfront.net", "amazonaws.com", "shopify",
                  "wp.com", "imgix.net", "cdn."],

    # ⚠ NEVER hardcode brand names in these two. Runtime-injected only — a hardcoded
    # ⚠ competitor domain would be counted as YOUR self-reference.
    "brand_owned": [],
    "competitor_owned": [],
}

def build_source_classifier(brand_domains=(), competitor_domains=()):
    """Return a domain→source_type function. Brand-agnostic: domains are injected."""
    m = {k: list(v) for k, v in SOURCE_TYPE_MAP.items()}
    m["brand_owned"] += [d.lower() for d in brand_domains if d]
    m["competitor_owned"] = [d.lower() for d in competitor_domains if d]

    def _st(domain):
        d = (domain or "").lower()
        if not d:
            return "other"
        # most-specific buckets first so brand/competitor beat generic retailer matches
        for st in ("cdn_asset", "brand_owned", "competitor_owned",
                   "editorial_authority", "trade_press", "reference",
                   "community_reviews", "grey_market_retailer", "content_affiliate",
                   "retailer", "video_social"):
            if any(p and p in d for p in m.get(st, [])):
                return st
        return "other"

    return _st

_DEFAULT_STYPE = build_source_classifier()

def report_unclassified_domains(raw_path, brand_domains=(), competitor_domains=(),
                               top=40):
    """Which domains are falling into 'other'? Feed the winners back into the map."""
    import json as _j
    from collections import Counter
    from pathlib import Path as _P
    if not _P(raw_path).exists():
        return []
    stype = build_source_classifier(brand_domains, competitor_domains)
    c, tot = Counter(), Counter()
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = _j.loads(line)
            except Exception:
                continue
            if not r.get("success"):
                continue
            for s in (r.get("sources") or []):
                d = (s.get("domain") or "").lower()
                if not d:
                    continue
                tot[d] += 1
                if stype(d) == "other":
                    c[d] += 1
    return [{"domain": d, "citations": n, "share_of_all": round(n / max(sum(tot.values()), 1), 4)}
            for d, n in c.most_common(top)]

def domains_from_sources(ingested_sources):
    """Derive the brand's own domains from Discover's ingested_sources artifact."""
    from urllib.parse import urlparse
    out = set()
    for s in (ingested_sources or []):
        origin = s.get("origin") if isinstance(s, dict) else getattr(s, "origin", "")
        if origin and str(origin).startswith("http"):
            netloc = urlparse(str(origin)).netloc.replace("www.", "").lower()
            if netloc:
                out.add(netloc)
    return sorted(out)

# ---------------------------------------------------------------- images
VISUAL_PAGE_PATTERNS = [r'facebook\.com/.+/(photos|videos)', r'instagram\.com',
                        r'youtube\.com', r'youtu\.be', r'/videos?/', r'pinterest\.']

IMG_PROMPT = """Analyze this image found in an AI answer about products. Return JSON:
{"shows_product": true, "brand_visible": "", "product_type": "",
 "visible_claims": [], "legible_text": [],
 "image_role": "product_shot|comparison|lifestyle|logo|chart|other"}"""

def _extract_image_urls(text, sources):
    """Returns (direct_image_urls, visual_page_urls). BOTH must be unpacked by callers."""
    urls = set(re.findall(r'https?://\S+?\.(?:png|jpe?g|webp|gif)', text or "", re.I))
    visual_pages = []
    for s in (sources or []):
        u = s.get("resolved_url") or s.get("url", "")
        if re.search(r'\.(png|jpe?g|webp|gif)(\?|$)', u, re.I):
            urls.add(u)
        elif any(re.search(p, u, re.I) for p in VISUAL_PAGE_PATTERNS):
            visual_pages.append(u)
    return sorted(urls), visual_pages

def _fetch_image(url, timeout=8, max_bytes=4_000_000):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        if r.status_code != 200:
            return None
        data = r.content[:max_bytes]
        mime = r.headers.get("Content-Type", "image/jpeg").split(";")[0]
        if "image" not in mime or len(data) < 512:
            return None
        return data, mime
    except Exception:
        return None

def interrogate_returned_images(gateway, image_urls, vision_model, max_images=3,
                                cache=None, budget_ok=None):
    """Vision-interrogate returned images. `cache` dedupes across the WHOLE audit —
    engines return the same product shots repeatedly, so this is ~10x cheaper."""
    out = []
    cache = cache if cache is not None else {}
    for url in list(image_urls)[:max_images]:
        if url in cache:
            out.append({**cache[url], "cache_hit": True})
            continue
        if budget_ok is not None and not budget_ok():
            out.append({"url": url, "fetched": False, "skipped": "budget"})
            continue
        fetched = _fetch_image(url)
        if not fetched:
            rec = {"url": url, "fetched": False, "error": "fetch_failed"}
            cache[url] = rec
            out.append(rec)
            continue
        data, mime = fetched
        try:
            r = gateway.generate(vision_model, IMG_PROMPT, images=[(data, mime)],
                                 json_mode=True, module="audit", task="image_interrogate")
            rec = dict(r["parsed"] or {})
            rec.update({"url": url, "fetched": True,
                        "bytes": len(data), "mime": mime,
                        "image_tokens": r.get("image_tokens"),
                        "vision_warning": r.get("vision_warning")})
        except Exception as e:
            rec = {"url": url, "fetched": True, "error": str(e)[:150]}
        cache[url] = rec
        out.append(rec)
    return out

# ---------------------------------------------------------------- vision canary
CANARY_NONCE = "ZQ7-KESTREL-4419"

def _make_canary_png(nonce=CANARY_NONCE, blank=False):
    """Generate the canary in-memory (no fixture files to ship)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    img = Image.new("RGB", (640, 240), "white")
    if not blank:
        d = ImageDraw.Draw(img)
        d.rectangle([20, 20, 620, 220], outline="black", width=4)
        d.text((60, 100), f"BATCH CODE {nonce}", fill="black")
        d.text((60, 140), "NET 800 g", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def vision_selftest(gateway, vision_model):
    """Prove vision is actually live. Also detects a CONFABULATING model via a
    blank-image negative control. Returns a dict for the audit manifest."""
    canary = _make_canary_png()
    blank = _make_canary_png(blank=True)
    if canary is None:
        return {"vision_live": None, "note": "Pillow not installed — cannot self-test"}
    prompt = ("List every alphanumeric code or batch number printed in this image. "
              "If there is none, reply exactly NONE.")

    def _ask(img):
        try:
            r = gateway.generate(vision_model, prompt, images=[(img, "image/png")],
                                 module="audit", task="vision_selftest", temperature=0.0,
                                 max_tokens=200)
            return r["text"] or "", r.get("image_tokens"), r.get("vision_warning")
        except Exception as e:
            return f"__ERROR__ {e}", None, str(e)[:120]

    pos, pos_tok, pos_warn = _ask(canary)
    neg, _, _ = _ask(blank)
    live = CANARY_NONCE.lower() in fold(pos)
    confab = (CANARY_NONCE.lower() in fold(neg)
              or bool(re.search(r"[a-z]{2,}-\d{3,}", fold(neg))))
    return {"vision_live": live, "confabulates": confab,
            "image_tokens_reported": pos_tok, "warning": pos_warn,
            "positive_response": pos[:220], "negative_response": neg[:220],
            "model": vision_model,
            "checked_utc": datetime.now(timezone.utc).isoformat()}

# ---------------------------------------------------------------- one run
def _tally(sources):
    t = {}
    for s in (sources or []):
        st = s.get("source_type", "other")
        t[st] = t.get(st, 0) + 1
    return t

def _one_run(gateway, engine_cfg, query, registry, web_on, run_index, *,
             use_llm_classify, judge_model, extract_images, vision_model,
             stype_fn, image_cache, budget_ok):
    model = engine_cfg["model"]
    provider = engine_cfg["provider"]
    target_brand = registry.target.canonical
    competitors = [c.canonical for c in registry.competitors]
    try:
        if provider == "vertex" and web_on:
            r = gateway.generate_grounded(query, model=model, module="audit", task="run")
            text, sources = r["text"], r["sources"]
            web_queries = r.get("web_search_queries", [])
        elif provider == "litellm" and web_on:
            r = gateway.generate_web_litellm(model, query, module="audit", task="run",
                                            temperature=1.0 if "gpt-5" in model else 0.3)
            text, sources, web_queries = r["text"], r["sources"], r["web_search_queries"]
        else:
            r = gateway.generate(model, query, module="audit", task="run",
                                 temperature=1.0 if "gpt-5" in model else 0.3)
            text, sources, web_queries = r["text"], [], []

        for s in sources:
            s["source_type"] = stype_fn(s.get("domain", ""))

        # ---- canonical measurement (metrics_spec is the ONLY implementation) ----
        m = measure(text, registry)

        cls = classify(gateway, text, target_brand, competitors,
                       use_llm=use_llm_classify, judge_model=judge_model,
                       precomputed=m)

        # ---- B2 FIX: unpack the tuple. Previously img_urls WAS the tuple, so every
        # ---- fetch received a list and silently failed. No image was ever read.
        img_urls, visual_pages = _extract_image_urls(text, sources)
        img_insights = (interrogate_returned_images(
            gateway, img_urls, vision_model, cache=image_cache, budget_ok=budget_ok)
            if (extract_images and img_urls) else [])

        return {"query": query, "engine": engine_cfg["name"], "model": model,
                "web_on": web_on, "run_index": run_index,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "success": True, "response_text": text,
                "response_chars": len(text or ""),
                "sources": sources, "num_sources": len(sources),
                "source_type_tally": _tally(sources),
                "image_urls": img_urls, "num_images": len(img_urls),
                "visual_pages": visual_pages, "num_visual_pages": len(visual_pages),
                "web_search_queries": web_queries,
                "image_insights": img_insights,
                # canonical metrics (flat, so loaders need no recomputation)
                "brand_mentions": m["brand_mentions"],
                "mentioned": m["mentioned"], "first_index": m["first_index"],
                "rank": m["rank"], "is_first": m["is_first"],
                "first_brand": m["first_brand"],
                "positive_first_brand": m["positive_first_brand"],
                "polarity": m["polarity"], "hedged": m["hedged"],
                "refusal": m["refusal"], "refusal_kind": m["refusal_kind"],
                "advisory_referral": m["advisory_referral"],
                "n_brands_present": m["n_brands_present"],
                "outcome": cls.get("outcome") or m["outcome"],
                "classification": cls,
                "metrics_spec_version": 1}
    except Exception as e:
        return {"query": query, "engine": engine_cfg["name"], "model": model,
                "web_on": web_on, "run_index": run_index, "success": False,
                "error": str(e)[:300],
                "timestamp": datetime.now(timezone.utc).isoformat()}

# ---------------------------------------------------------------- batch
class BudgetExceeded(Exception):
    pass

def _already_done(out_path):
    """Resume key now includes MODEL — swapping models under the same engine name
    previously made stale runs look complete."""
    if not Path(out_path).exists():
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("success"):
                    done.add((r["query"], r["engine"], r.get("model", ""),
                              r["web_on"], r["run_index"]))
            except Exception:
                pass
    return done

def audit_progress(out_path, queries, engines, n_runs, web_modes):
    done = _already_done(out_path)
    planned = len(queries) * len(engines) * n_runs * len(web_modes)
    return len(done), planned

def estimate_cost(n_queries, n_engines, n_runs, web_modes, avg_cost_per_call=0.002):
    calls = n_queries * n_engines * n_runs * len(web_modes)
    return calls, round(calls * avg_cost_per_call, 2)

def run_audit(gateway, queries, engines, target_brand, competitors, *,
              n_runs=10, web_modes=("on",), out_path="audit_raw.jsonl",
              use_llm_classify=False, judge_model="gemini-2.5-flash",
              extract_images=False, vision_model="gemini-2.5-flash",
              budget=None, hard_stop=False, progress_cb=None,
              brand_domains=(), competitor_domains=(), alias_overrides=None,
              manifest_cb=None):
    """queries: [{query,intent}]; engines: [{name,model,provider,supports}]"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    registry = build_registry(target_brand, competitors, alias_overrides)
    stype_fn = build_source_classifier(brand_domains, competitor_domains)
    image_cache: dict = {}

    def budget_ok():
        return budget is None or gateway.meter.total_cost() < budget

    done = _already_done(out_path)
    tasks = [(q, e, (w == "on"), r)
             for q in queries for e in engines
             for w in web_modes for r in range(n_runs)
             if (q["query"], e["name"], e["model"], (w == "on"), r) not in done]
    total, completed, errors = len(tasks), 0, 0

    with open(out_path, "a", encoding="utf-8") as f:
        for (q, e, web_on, r) in tasks:
            if budget is not None and gateway.meter.total_cost() >= budget:
                if hard_stop:
                    raise BudgetExceeded(
                        f"Budget ${budget} reached at ${gateway.meter.total_cost():.2f} "
                        f"({completed}/{total} done — re-run to resume)")
                if progress_cb:
                    progress_cb(completed, total, warn=True)
            rec = _one_run(gateway, e, q["query"], registry, web_on, r,
                           use_llm_classify=use_llm_classify, judge_model=judge_model,
                           extract_images=extract_images, vision_model=vision_model,
                           stype_fn=stype_fn, image_cache=image_cache,
                           budget_ok=budget_ok)
            rec["intent"] = q.get("intent")
            rec["modality"] = q.get("modality", "text")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            completed += 1
            if not rec.get("success"):
                errors += 1
            if progress_cb:
                progress_cb(completed, total, warn=False)

    if manifest_cb:
        manifest_cb({"registry_aliases": {b.canonical: list(b.aliases)
                                          for b in registry.all},
                     "brand_domains": list(brand_domains),
                     "competitor_domains": list(competitor_domains),
                     "unique_images_interrogated": len(image_cache),
                     "errors": errors, "runs_written": completed})
    return out_path

# ---------------------------------------------------------------- V/VT
VT_PROMPT = """A shopper shows you this image and asks: "{query}"
Answer as an AI assistant would. Then note: what could you READ from the image,
and did it influence your answer?"""

def visual_interrogation(gateway, images, query, engines, vision_model="gemini-2.5-flash",
                         target_brand=None, competitors=()):
    results = []
    registry = (build_registry(target_brand, competitors) if target_brand else None)
    for e in engines:
        if "vision" not in e.get("supports", []):
            results.append({"engine": e["name"], "skipped": "no vision support"})
            continue
        try:
            r = gateway.generate(e["model"], VT_PROMPT.format(query=query),
                                 images=images, module="audit", task="visual_interrogate")
            row = {"engine": e["name"], "response": r["text"], "success": True,
                   "image_tokens": r.get("image_tokens"),
                   "vision_warning": r.get("vision_warning")}
            if registry:
                row.update({k: v for k, v in measure(r["text"], registry).items()
                            if k != "brand_mentions"})
            results.append(row)
        except Exception as ex:
            results.append({"engine": e["name"], "error": str(ex)[:200]})
    return results