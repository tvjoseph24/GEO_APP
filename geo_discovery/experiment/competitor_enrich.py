"""Rich, contamination-free, length-matched competitor documents.

Three problems in v1 that invalidated E1-E5:
  1. `positioning = snippets[0]` — a raw AI sentence, often naming OTHER brands, and
     often actually about the TARGET. Any leaked "unlike Brand X" biases every arm.
  2. `attributes = {}` on the audit path — so competitor docs had no table while the
     target's Variant B did. The experiment measured document quality, not the variable.
  3. No length parity. Target ~1200 words vs competitor ~200 words is a confound.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from metrics_spec import build_registry, fold

MIN_AUDIT_SPANS = 3
MAX_COMPETITORS = 5
SENT = re.compile(r"(?<=[.!?])\s+|\n+")

# ------------------------------------------------------------------ 1. harvest
def harvest_evidence(raw_path, competitors, target_brand, alias_overrides=None,
                     query_filter=None, max_spans=40):
    """Mine ALL audit evidence per competitor: spans, cited domains, queries won."""
    if not Path(raw_path).exists():
        return {}
    names = [c["brand"] if isinstance(c, dict) else c for c in competitors]
    names = [n for n in names if n and fold(n) != fold(target_brand)]
    reg = build_registry(target_brand, names, alias_overrides)
    pats = {c.canonical: c.pattern for c in reg.competitors}
    all_names = [target_brand] + names

    ev = defaultdict(lambda: {"spans": [], "domains": Counter(), "won": Counter(),
                              "mentions": 0, "first": 0, "types": Counter()})
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("success"):
                continue
            if query_filter and fold(query_filter) not in fold(r.get("query", "")):
                continue
            text = r.get("response_text") or ""
            bm = r.get("brand_mentions") or {}
            fb = r.get("first_brand")
            for cname, pat in pats.items():
                e = ev[cname]
                for s in SENT.split(text):
                    s = s.strip()
                    if len(s) > 40 and pat.search(fold(s)):
                        e["spans"].append(s)
                if (bm.get(cname) or {}).get("mentioned"):
                    e["mentions"] += 1
                    e["won"][r.get("query")] += 1
                    for src in (r.get("sources") or []):
                        if src.get("domain"):
                            e["domains"][src["domain"]] += 1
                            e["types"][src.get("source_type", "other")] += 1
                if fb == cname:
                    e["first"] += 1
    for c in ev:
        ev[c]["spans"] = _dedupe(ev[c]["spans"], max_spans)
    return {k: v for k, v in ev.items()}

def _dedupe(sents, limit, thresh=0.85):
    keep = []
    for s in sents:
        t = set(fold(s).split())
        if not t:
            continue
        if not any(len(t & set(fold(k).split())) / len(t | set(fold(k).split())) > thresh
                   for k in keep):
            keep.append(s)
        if len(keep) >= limit:
            break
    return keep

# ------------------------------------------------------------------ 2. scrub
def scrub(sentences, keep_brand, forbidden, alias_overrides=None):
    """Drop any sentence naming a brand other than keep_brand."""
    reg = build_registry(keep_brand, [b for b in forbidden if fold(b) != fold(keep_brand)],
                         alias_overrides)
    pats = [(c.canonical, c.pattern) for c in reg.competitors]
    kept, removed = [], []
    for s in sentences:
        low = fold(s)
        leaks = [b for b, p in pats if p.search(low)]
        (removed if leaks else kept).append(s)
    return kept, removed

def assert_clean(text, keep_brand, forbidden, alias_overrides=None):
    reg = build_registry(keep_brand, [b for b in forbidden if fold(b) != fold(keep_brand)],
                         alias_overrides)
    low = fold(text or "")
    leaks = [c.canonical for c in reg.competitors if c.pattern.search(low)]
    return leaks

# ------------------------------------------------------------------ 3. structure
STRUCTURE_PROMPT = """You are building a controlled competitive document for a GEO experiment.

Below is real evidence about "{comp}" in the {category} category, mined from AI assistant
answers and the pages those answers cited.

Write a factual product document for "{comp}" ONLY.

HARD RULES
1. Mention ONLY "{comp}". NEVER name any other brand. If the evidence compares brands,
   restate only the part about {comp}.
2. Use ONLY facts present in the evidence. Do NOT invent ingredients, numbers, awards or
   certifications.
3. Include quantified specifics wherever the evidence supplies them.
4. Target length: {target_words} words (±10%). This is a controlled experiment — length
   parity between documents matters more than completeness.
5. Produce the SAME sections as every other document in this experiment: answer capsule,
   claims, attribute table, citations.

Return STRICT JSON:
{{"brand":"{comp}",
  "answer_capsule":"one sentence stating what {comp} is best for",
  "positioning":"2-3 sentences",
  "claims":[{{"claim":"","quantified":false,"source_domain":"","source_type":"editorial|retailer|community|brand"}}],
  "attributes":{{"attribute":"value"}},
  "citations":[{{"text":"","domain":""}}],
  "certifications":[],
  "review_summary":"what reviewers consistently say — only if the evidence supports it"}}

EVIDENCE — sentences from AI answers ({n_spans} available, contamination removed):
{spans}

EVIDENCE — domains that cited this brand ({n_domains} distinct):
{domains}

EVIDENCE — queries where this brand was recommended:
{won}

Known strength from the competitive set: {known}"""

def structure_doc(gateway, cname, category, evidence, forbidden, target_words, model,
                  known="", alias_overrides=None, module="experiment"):
    clean, removed = scrub(evidence["spans"], cname, forbidden, alias_overrides)
    res = gateway.generate_json(model, STRUCTURE_PROMPT.format(
        comp=cname, category=category or "this category", target_words=target_words,
        spans=json.dumps(clean[:25], indent=1)[:9000], n_spans=len(clean),
        domains=json.dumps([d for d, _ in evidence["domains"].most_common(12)]),
        n_domains=len(evidence["domains"]),
        won=json.dumps([q for q, _ in evidence["won"].most_common(8)])[:1200],
        known=known or "(none recorded)"),
        expect="dict", module=module, task="comp_structure",
        temperature=0.2, max_tokens=3000)
    doc = res.get("data") or {}
    doc.setdefault("brand", cname)
    rendered = json.dumps(doc)
    leaks = assert_clean(rendered, cname, forbidden, alias_overrides)
    doc["_meta"] = {"source": "audit+llm", "ok": bool(res.get("ok")),
                    "failure": res.get("failure"),
                    "spans_used": len(clean), "spans_removed_contaminated": len(removed),
                    "domains": len(evidence["domains"]),
                    "audit_mentions": evidence["mentions"],
                    "audit_first": evidence["first"],
                    "source_mix": dict(evidence["types"]),
                    "contamination_leaks": leaks}
    return doc

# ------------------------------------------------------------------ 4. build
def build_rich_competitor_context(gateway, competitors, category, target_brand, raw_path,
                                  model="gemini-2.5-flash", query_filter=None,
                                  target_words=180, alias_overrides=None,
                                  progress_cb=None, max_competitors=MAX_COMPETITORS):
    ev = harvest_evidence(raw_path, competitors, target_brand, alias_overrides,
                          query_filter)
    names = [c["brand"] if isinstance(c, dict) else c for c in competitors]
    names = [n for n in names if n and fold(n) != fold(target_brand)]
    known_map = {(c.get("brand") if isinstance(c, dict) else c):
                 (c.get("known_strength", "") if isinstance(c, dict) else "")
                 for c in competitors}
    # prioritise competitors the engines ACTUALLY recommend
    ranked = sorted(names, key=lambda n: -(ev.get(n, {}).get("mentions", 0)))
    ranked = ranked[:max_competitors]

    docs = {}
    for i, cname in enumerate(ranked):
        e = ev.get(cname) or {"spans": [], "domains": Counter(), "won": Counter(),
                              "mentions": 0, "first": 0, "types": Counter()}
        docs[cname] = structure_doc(gateway, cname, category, e, names + [target_brand],
                                    target_words, model, known_map.get(cname, ""),
                                    alias_overrides)
        if progress_cb:
            progress_cb((i + 1) / max(len(ranked), 1))
    return docs

# ------------------------------------------------------------------ 5. render
def render_competitor_blocks(comp_docs):
    """list[str] — one block per competitor. NEVER join for injection (see rag_runner B6)."""
    blocks = []
    for cname, d in (comp_docs or {}).items():
        p = [f"[{cname.upper()}]"]
        if d.get("answer_capsule"):
            p.append(f"**{d['answer_capsule']}**")
        if d.get("positioning"):
            p.append(d["positioning"])
        cl = d.get("claims") or []
        if cl:
            p.append("\n".join(
                f"- {c.get('claim')}" + (f" ({c.get('source_domain')})"
                                         if c.get("source_domain") else "")
                for c in cl[:6] if c.get("claim")))
        a = d.get("attributes") or {}
        if a:
            p.append("| Attribute | Value |\n|---|---|\n" +
                     "\n".join(f"| {k} | {v} |" for k, v in list(a.items())[:6]))
        cits = [c.get("domain") for c in (d.get("citations") or []) if c.get("domain")]
        if cits:
            p.append("Sources: " + ", ".join(cits[:4]))
        blocks.append("\n".join(p))
    return blocks

def render_competitor_context(comp_docs):
    return "\n\n".join(render_competitor_blocks(comp_docs))

# ------------------------------------------------------------------ 6. parity
def parity_report(target_text, comp_blocks, exp_id="E0", control_text=None):
    """Parity is template-dependent: for E0/E1/E5 the arms deliberately differ in
    structure, so only the TREATMENT arm is compared to the competitor field."""
    def prof(name, tx):
        tx = tx or ""
        return {"doc": name, "words": len(tx.split()),
                "has_table": len(re.findall(r"^\s*\|.+\|\s*$", tx, re.M)) >= 2,
                "has_capsule": bool(re.search(r"\*\*[^*]{15,}\*\*", tx)),
                "n_citations": len(re.findall(r"\(([a-z0-9\-]+\.[a-z]{2,})", fold(tx))),
                "n_bullets": len(re.findall(r"^\s*[-*•]\s+\S", tx, re.M))}

    rows = [prof("TREATMENT", target_text)]
    if control_text is not None:
        rows.append(prof("CONTROL", control_text))
    rows += [prof(b.split("\n")[0].strip("[]")[:24], b) for b in comp_blocks]
    df = pd.DataFrame(rows)

    tw = max(int(df.loc[df.doc == "TREATMENT", "words"].iloc[0]), 1)
    comp = df[~df.doc.isin(("TREATMENT", "CONTROL"))]
    warns = []
    if len(comp):
        ratio = float(comp["words"].mean()) / tw
        df.attrs["length_parity_ratio"] = round(ratio, 2)
        if not (0.6 <= ratio <= 1.7):
            warns.append(f"treatment arm is {tw} words vs competitor average "
                         f"{comp['words'].mean():.0f} ({ratio:.2f}×) — length asymmetry.")
        tgt = df[df.doc == "TREATMENT"].iloc[0]
        for feat, label in (("has_table", "table"), ("has_capsule", "capsule")):
            share = float(comp[feat].mean())
            if share >= 0.5 and not bool(tgt[feat]):
                warns.append(f"{share:.0%} of competitor docs have a {label}; the "
                             f"treatment arm has none — structurally handicapped.")
        if float(comp["n_citations"].mean()) >= 1 and int(tgt["n_citations"]) == 0:
            warns.append("competitor docs carry citations, the treatment arm none.")
    if exp_id in ("E0", "E1", "E5") and control_text is not None:
        df.attrs["note"] = (f"{exp_id}: the control arm is deliberately below parity — "
                            f"structure is the variable. Only the treatment arm is "
                            f"compared to the competitor field.")
    df.attrs["warnings"] = warns
    return df

# ------------------------------------------------------------------ 7. freeze
def freeze_manifest(comp_docs, context_text):
    return {"content_hash": hashlib.sha256(
                (context_text or "").encode()).hexdigest()[:16],
            "n_competitors": len(comp_docs or {}),
            "competitors": sorted(comp_docs or {}),
            "words": len((context_text or "").split()),
            "contamination_leaks": {k: (v.get("_meta") or {}).get("contamination_leaks")
                                    for k, v in (comp_docs or {}).items()
                                    if (v.get("_meta") or {}).get("contamination_leaks")},
            "llm_failures": {k: (v.get("_meta") or {}).get("failure")
                             for k, v in (comp_docs or {}).items()
                             if not (v.get("_meta") or {}).get("ok")}}

def save_competitor_context(parent_brand, target_brand, comp_docs, manifest=None):
    from store import save_artifact
    save_artifact(parent_brand, target_brand, "competitor_context", comp_docs,
                  manifest or {"note": "frozen competitor docs"})

def load_competitor_context(parent_brand, target_brand):
    from store import load_artifact
    return load_artifact(parent_brand, target_brand, "competitor_context")

def split_competitor_blocks(comp_context):
    from experiment.rag_runner import split_competitor_blocks as _s
    return _s(comp_context)

# ------------------------------------------------------------------ 8. visual grounding
COMPETITOR_VISUAL_PROMPT = """You are reading a COMPETITOR's product image, as an AI
assistant would when it surfaces in an answer.

Report ONLY what is directly legible or visible. Do not infer, and do not use prior
knowledge of the brand.

Return STRICT JSON:
{"brand_read":"", "product_type_read":"",
 "visible_claims":["claims a shopper could read off this pack"],
 "numbers_read":[], "certifications_read":[],
 "text_legibility":"clear|partial|illegible",
 "pack_description":"one factual sentence on what the pack shows"}"""

def harvest_competitor_images(raw_path, competitors, target_brand,
                              alias_overrides=None, per_competitor=2):
    """Direct image URLs the audit captured in answers naming each competitor.

    Deliberately narrow. Engines return social and video PAGE urls far more often than
    direct image urls, and those pages will refuse a plain fetch. Where nothing is
    obtainable we record that rather than substituting a guess.
    """
    if not Path(raw_path).exists():
        return {}
    names = [c["brand"] if isinstance(c, dict) else c for c in (competitors or [])]
    names = [n for n in names if n and fold(n) != fold(target_brand)]
    reg = build_registry(target_brand, names, alias_overrides)
    pats = {c.canonical: c.pattern for c in reg.competitors}

    found = {c: [] for c in pats}
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("success"):
                continue
            urls = r.get("image_urls") or []
            if not urls:
                continue
            low = fold(r.get("response_text") or "")
            named = [c for c, p in pats.items() if p.search(low)]
            if len(named) != 1:
                continue                   # ambiguous attribution — skip
            c = named[0]
            for u in urls:
                if u not in found[c] and len(found[c]) < per_competitor:
                    found[c].append(u)
    return {k: v for k, v in found.items() if v}

def interrogate_competitor_visuals(gateway, images_by_comp, vision_model,
                                   module="experiment", budget_images=8):
    """Vision-read competitor packs so the competitor block can state what their pack
    VISIBLY shows — sourced fact, not an assumption."""
    from audit_runner import _fetch_image
    out, spent = {}, 0
    for comp, urls in (images_by_comp or {}).items():
        for u in urls:
            if spent >= budget_images:
                break
            got = _fetch_image(u)
            if not got:
                out.setdefault(comp, []).append({"url": u, "fetched": False,
                                                 "error": "fetch_failed"})
                continue
            data, mime = got
            spent += 1
            try:
                r = gateway.generate(vision_model, COMPETITOR_VISUAL_PROMPT,
                                     images=[(data, mime)], json_mode=True,
                                     module=module, task="competitor_visual",
                                     temperature=0.0, max_tokens=900)
                d = dict(r.get("parsed") or {})
                d.update({"url": u, "fetched": True,
                          "image_tokens": r.get("image_tokens")})
                out.setdefault(comp, []).append(d)
            except Exception as e:
                out.setdefault(comp, []).append({"url": u, "fetched": True,
                                                 "error": str(e)[:150]})
    return out

def fold_visuals_into_docs(comp_docs, visuals):
    """Add a `pack_visible` layer to each competitor doc. Facts only, sourced from a
    fetched image, so the competitor field is visually grounded without injecting a
    competitor IMAGE into context — which would make the model attribute claims across
    products and turn the test into an attribution test."""
    n_folded = 0
    for comp, reads in (visuals or {}).items():
        good = [r for r in reads if r.get("fetched") and not r.get("error")
                and (r.get("visible_claims") or r.get("pack_description"))]
        if not good or comp not in (comp_docs or {}):
            continue
        claims, descs, certs = [], [], []
        for r in good:
            claims += [c for c in (r.get("visible_claims") or []) if c]
            certs += [c for c in (r.get("certifications_read") or []) if c]
            if r.get("pack_description"):
                descs.append(r["pack_description"])
        comp_docs[comp]["pack_visible"] = {
            "description": descs[0] if descs else "",
            "visible_claims": list(dict.fromkeys(claims))[:6],
            "certifications": list(dict.fromkeys(certs))[:4],
            "n_images_read": len(good),
            "source": "vision-read from an image the audit captured"}
        n_folded += 1
    return comp_docs, n_folded

def render_competitor_blocks_visual(comp_docs):
    """Blocks including the pack_visible layer. Used by the bench, where what a
    competitor's pack VISIBLY shows is the relevant competitive pressure."""
    blocks = []
    for cname, d in (comp_docs or {}).items():
        p = [f"[{cname.upper()}]"]
        if d.get("answer_capsule"):
            p.append(f"**{d['answer_capsule']}**")
        if d.get("positioning"):
            p.append(d["positioning"])
        pv = d.get("pack_visible") or {}
        if pv.get("description") or pv.get("visible_claims"):
            line = "On pack: " + (pv.get("description") or "")
            if pv.get("visible_claims"):
                line += " Visible claims: " + "; ".join(pv["visible_claims"][:4]) + "."
            if pv.get("certifications"):
                line += " Marks shown: " + ", ".join(pv["certifications"][:3]) + "."
            p.append(line.strip())
        cl = [c.get("claim") for c in (d.get("claims") or []) if c.get("claim")]
        if cl:
            p.append("\n".join(f"- {c}" for c in cl[:5]))
        blocks.append("\n".join(p))
    return blocks

def visual_grounding_manifest(comp_docs, visuals, n_folded):
    attempted = sum(len(v) for v in (visuals or {}).values())
    fetched = sum(1 for v in (visuals or {}).values() for r in v if r.get("fetched"))
    return {"competitors_with_pack_read": n_folded,
            "images_attempted": attempted, "images_fetched": fetched,
            "competitors_without_pack_read": sorted(
                set(comp_docs or {}) - {c for c, d in (comp_docs or {}).items()
                                        if d.get("pack_visible")}),
            "note": ("Competitor packs are described from images the audit captured and a "
                     "vision model read. Where no image was obtainable the competitor "
                     "block carries text evidence only — nothing is invented. Engines "
                     "return social and video PAGE urls far more often than direct image "
                     "urls, so partial coverage is expected."
                     if attempted else
                     "No competitor image URLs were available in the audit, so the "
                     "competitor field is text-only. This does not bias an A-vs-B "
                     "comparison — the field is constant across arms — but it does mean "
                     "the competitive pressure is described rather than shown.")}
