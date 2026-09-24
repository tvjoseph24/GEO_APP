"""Creative bench: evaluate a brand's own visual assets against a frozen competitive field.

Distinct from E7/E8, and the difference matters.

  E7/E8 are DIAGNOSTIC. They start from a gap ("this claim is absent from machine
  reality") and ask whether printing it on-pack makes it readable. The creative exists to
  answer the question.

  The bench is EVALUATIVE. The creative already exists and the question is which version
  performs. That inverts the pipeline — asset first, question second — which is why it
  cannot be reached through clustering and hypothesis generation.

Design contract
---------------
* TEXT MODE is declared, never inferred:
    isolation — one text block, generated once, copied byte-identical. The asset is the
                only variable, so a lift is attributable to the creative.
    bundle    — per-arm text generated from that arm's asset. Tests the shipping package
                and can make NO claim about the creative alone.
* TEXT PRESENCE has three states (none / minimal / full), each answering a different
  question and each with a different metric ceiling.
* No claim may appear in the text that the IMAGE is supposed to carry. If it does, the
  model answers without looking at the image and the experiment measures nothing.
* A LEGIBILITY PRE-CHECK runs before any spend. If two assets read identically to the
  model, the null is known in advance for four vision calls.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from metrics_spec import build_registry, fold

BENCH_TEMPLATE = "VBENCH"
BENCH_MODE = "creative"          # never merges with 'ab' or 'progressive' in the raw log

TEXT_PRESENCE = {
    "none": {
        "label": "No text — image only",
        "question": "Can the engine identify the brand from the creative alone?",
        "metric_note": "Mention rate floors near zero unless vision reads the brand off "
                       "the pack. A floor IS the finding: it means the creative is not "
                       "machine-legible.",
        "words": 0},
    "minimal": {
        "label": "Minimal — brand and category only",
        "question": "Can the engine read the CLAIMS off the creative?",
        "metric_note": "The brand is named, so mention has a floor above zero and the "
                       "creative is tested on whether its claims are extractable. The "
                       "recommended default.",
        "words": 20},
    "full": {
        "label": "Full — claims stated in text",
        "question": "Does the creative reinforce claims the copy already makes?",
        "metric_note": "Mention saturates near 100%, so the finding must come from RANK "
                       "and unconditional top-1, not from mention rate.",
        "words": 60},
}

ATTRIBUTION = {
    "isolation": {
        "label": "Isolation — identical text on every arm",
        "isolates": "creative execution",
        "note": "Text is generated once and copied byte-identical, so any lift is "
                "attributable to the creative."},
    "bundle": {
        "label": "Bundle — text generated per asset",
        "isolates": "creative and copy as a shipped package",
        "note": "Two things vary at once. This tests the package a campaign actually "
                "ships, and makes NO claim about the creative alone."},
}

# ---------------------------------------------------------------- design
def _next_id(designs):
    used = {int(m.group(1)) for d in (designs or [])
            for m in [re.match(r"^V(\d+)$", str(d.get("design_id") or ""))] if m}
    n = 1
    while n in used:
        n += 1
    return f"V{n:02d}"

def build_bench_design(*, question, n_arms=2, text_presence="minimal",
                       attribution="isolation", existing_designs=None,
                       claims_allowed=None, title="", notes=""):
    """Write a runnable design directly. No gap, no cluster, no hypothesis."""
    n = max(2, min(int(n_arms), 5))
    tp = TEXT_PRESENCE.get(text_presence, TEXT_PRESENCE["minimal"])
    at = ATTRIBUTION.get(attribution, ATTRIBUTION["isolation"])
    return {
        "design_id": _next_id(existing_designs),
        "source": "bench",
        "experiment_id": BENCH_TEMPLATE,
        "experiment_name": title or f"Creative bench · {n} arm(s)",
        "isolates": at["isolates"],
        "mode": "A/B (fast)",              # UI label; the raw log records BENCH_MODE
        "run_mode": BENCH_MODE,
        "visual_locked": True,             # progressive would replace the manipulation
        "needs_visual": True,
        "n_visual_arms": n,
        "text_presence": text_presence,
        "attribution": attribution,
        "text_mode": f"{text_presence}/{attribution}",
        "question": question,
        "question_in_audit": True,         # verified by the caller against the audit set
        "claims_allowed": list(claims_allowed or []),
        "priority_signal": "high",
        "pillar": "Visual/Multimodal",
        "mechanism": "extraction_failure" if text_presence != "full" else "memory_deficit",
        "horizon": "retrieval",
        "hypothesis": (f"Creative variants differ in how often engines recommend the "
                       f"brand for '{question}'. {at['note']}"),
        "falsification": ("If no arm differs from any other beyond the omnibus threshold, "
                          "the creative variants are equivalent to a machine reader and "
                          "the choice between them should be made on human grounds."),
        "prediction": {"metric": "mention_rate" if text_presence != "full"
                       else "top1_unconditional", "baseline_pp": None, "lift_pp": None},
        "analyst_notes": notes,
        "include": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------- legibility pre-check
LEGIBILITY_PROMPT = """You are reading a brand's creative asset exactly as an AI assistant
would when a shopper shows it to you.

Report ONLY what is directly legible or visible IN THIS ASSET. Do not infer. Do not use
prior knowledge of any brand. Do not describe what the image implies or suggests. If
something is present but you cannot read it, say so in `what_is_unreadable` — that is the
single most useful field here, because it tells the brand what a machine cannot see.

Return STRICT JSON:
{"brand_legible": true,
 "brand_name_read": "exactly as printed, or empty string if you cannot read it",
 "product_type_read": "what kind of product this is, from the asset alone",
 "legible_text": ["every string you can actually read, verbatim, in reading order"],
 "visible_claims": ["claims a shopper could read off this asset"],
 "numbers_read": ["any quantity, percentage, size, age or duration printed"],
 "certifications_read": ["any seal, mark or certification visible"],
 "text_legibility": "clear|partial|illegible",
 "dominant_subject": "pack|person|texture|lifestyle|text|logo|other",
 "what_is_unreadable": "anything present but not readable, and why"}"""

def precheck_legibility(gateway, store, design, vision_model, module="experiment",
                        target_brand="", progress_cb=None):
    """One vision call per asset. Returns per-arm reads plus a cross-arm verdict.

    This is the visual equivalent of the pilot, and it prevents the most likely failure
    mode: a creative the model cannot read produces a flat null that reads as "this
    creative does not work" when the truth is "no creative information reached the model".
    """
    from experiment import visual as VIS
    a = VIS.assets(store, design)
    if not a:
        return {"error": "no assets attached"}

    reads, arms = {}, sorted(a)
    for i, arm in enumerate(arms):
        asset = a[arm]
        if asset.get("is_video"):
            reads[arm] = {"arm": arm, "skipped": "video",
                          "note": "Video legibility is not pre-checked — a single-frame "
                                  "read would misrepresent a moving asset. Payload "
                                  "delivery is still verified on every run."}
            continue
        try:
            r = gateway.generate(vision_model, LEGIBILITY_PROMPT,
                                 images=[(asset["bytes"], asset["mime"])],
                                 json_mode=True, module=module,
                                 task="legibility_precheck", temperature=0.0,
                                 max_tokens=1200)
            d = dict(r.get("parsed") or {})
            d.update({"arm": arm, "label": asset.get("label"),
                      "image_tokens": r.get("image_tokens"),
                      "vision_warning": r.get("vision_warning"),
                      "payload_delivered": bool(r.get("image_tokens"))})
            reads[arm] = d
            # MEASURED token count replaces the dimension-based estimate
            if r.get("image_tokens"):
                asset["measured_tokens"] = int(r["image_tokens"])
        except Exception as e:
            reads[arm] = {"arm": arm, "error": str(e)[:200], "payload_delivered": False}
        if progress_cb:
            progress_cb((i + 1) / max(len(arms), 1))

    return {**_legibility_verdict(reads, target_brand, design),
            "reads": reads, "model": vision_model,
            "checked_utc": datetime.now(timezone.utc).isoformat()}

def _sig(read):
    """A normalised signature of what the model actually read. Two assets with the same
    signature are the same asset as far as a machine reader is concerned."""
    parts = ([str(read.get("brand_name_read") or "")]
             + sorted(str(x) for x in _as_list(read.get("legible_text")))
             + sorted(str(x) for x in _as_list(read.get("visible_claims")))
             + sorted(str(x) for x in _as_list(read.get("numbers_read")))
             + sorted(str(x) for x in _as_list(read.get("certifications_read"))))
    return fold(re.sub(r"\s+", " ", " | ".join(parts))).strip(" |")

def _as_list(v):
    """LLM JSON is not shape-guaranteed. A model returning `"visible_claims": "24-hour
    hydration"` instead of a list would otherwise be iterated character by character —
    `sorted()` on a string yields letters, and `len()` reports 18 claims instead of 1."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, (list, tuple, set)):
        return [x for x in v if x is not None and str(x).strip()]
    return [v]

def _legibility_verdict(reads, target_brand, design):
    live = {k: v for k, v in reads.items()
            if not v.get("error") and not v.get("skipped")}
    skipped = sorted(k for k, v in reads.items() if v.get("skipped"))
    errored = sorted(k for k, v in reads.items() if v.get("error"))
    blockers, cautions, notes = [], [], []

    # ---- nothing readable at all ----
    if not live:
        return {"ok": False,
                "blockers": [
                    f"No arm was successfully read. "
                    + (f"{len(errored)} arm(s) errored: "
                       f"{'; '.join(str((reads[k] or {}).get('error'))[:70] for k in errored[:3])}. "
                       if errored else "")
                    + (f"{len(skipped)} arm(s) skipped (video). " if skipped else "")
                    + "Nothing can be concluded about legibility, so a run would be "
                      "uninterpretable."],
                "cautions": [], "notes": ["no_readable_arm"],
                "arms_checked": [], "arms_skipped": skipped, "arms_errored": errored,
                "distinct_machine_reads": 0, "measured_tokens": {}, "comparison": []}

    if len(live) == 1 and (len(reads) > 1):
        cautions.append(
            f"Only arm {sorted(live)[0]} was readable — the others errored or are video. "
            f"No cross-arm comparison of legibility is possible, so the run proceeds "
            f"without knowing whether the creatives differ to a machine.")

    # ---- payload delivery ----
    undelivered = [k for k, v in live.items() if not v.get("payload_delivered")]
    if undelivered and len(undelivered) == len(live):
        blockers.append(
            "No arm reported any image tokens — the vision payload is not reaching the "
            "model. Every run would be text-only, so the result would be an artifact "
            "rather than a null. Run the vision self-test in the Audit tab.")
        notes.append("payload_not_delivered")
    elif undelivered:
        cautions.append(f"Arms {undelivered} reported no image tokens — their payload did "
                        f"not arrive and they are not comparable to the others.")

    # ---- the decisive check: do the assets differ to a MACHINE? ----
    sigs = {}
    for k, v in live.items():
        s = _sig(v)
        if s:
            sigs[k] = s
    blank = sorted(set(live) - set(sigs))
    if blank:
        cautions.append(f"Arms {blank} yielded nothing readable at all — no brand, text, "
                        f"claim or number. Whatever is on those creatives is invisible to "
                        f"a machine reader.")
        notes.append("empty_machine_read")

    if len(sigs) > 1:
        uniq = set(sigs.values())
        if len(uniq) == 1:
            blockers.append(
                "Every arm reads IDENTICALLY to the model — same brand, same legible "
                "text, same claims, same numbers. The creative differences are not "
                "machine-visible, so a null is already known and the runs would only "
                "confirm it. Either the differences are purely aesthetic (which is a "
                "finding in itself) or the claims are not legible on any arm.")
            notes.append("assets_read_identically")
        elif len(uniq) < len(sigs):
            dupes = {}
            for k, s in sigs.items():
                dupes.setdefault(s, []).append(k)
            groups = [sorted(v) for v in dupes.values() if len(v) > 1]
            cautions.append(f"Arms {groups} read identically to the model — those pairs "
                            f"cannot be distinguished and will contribute noise to the "
                            f"omnibus test rather than signal.")
            notes.append("duplicate_machine_reads")

    # ---- brand legibility, judged against the text mode ----
    tp = design.get("text_presence", "minimal")
    reg = build_registry(target_brand, []) if target_brand else None
    if reg:
        unread = [k for k, v in live.items()
                  if not reg.target.pattern.search(fold(v.get("brand_name_read") or ""))]
        if unread and len(unread) == len(live):
            notes.append("brand_not_legible_in_asset")
            if tp == "none":
                cautions.append(
                    "The brand name is not legible on any arm. With text mode 'none' that "
                    "means mention rate will floor at zero on every arm and the experiment "
                    "cannot discriminate. Switch to 'minimal' so the brand is named in "
                    "text and the creative is tested on its CLAIMS instead.")
            else:
                cautions.append(
                    "The brand name is not legible on any arm, so the engine is working "
                    "from the text alone for brand identity. The creative is still tested "
                    "on its claims.")
        elif unread:
            cautions.append(f"The brand name is not legible on arms {unread}. Those arms "
                            f"depend entirely on the text for brand identity, which the "
                            f"others do not.")

    # ---- claim legibility ----
    noclaims = [k for k, v in live.items() if not _as_list(v.get("visible_claims"))]
    if noclaims and len(noclaims) == len(live) and tp != "full":
        cautions.append(
            "No arm carries a legible claim. The creative can only influence the answer "
            "through brand recognition, not through substance — which is a narrow and "
            "usually small effect.")
        notes.append("no_legible_claim")
    illegible = [k for k, v in live.items() if v.get("text_legibility") == "illegible"]
    if illegible:
        cautions.append(f"Text on arms {illegible} is illegible to the model. Whatever is "
                        f"printed there is not contributing.")

    return {
        "ok": not blockers, "blockers": blockers, "cautions": cautions, "notes": notes,
        "arms_checked": sorted(live), "arms_skipped": skipped, "arms_errored": errored,
        "distinct_machine_reads": len(set(sigs.values())),
        "measured_tokens": {k: v.get("image_tokens") for k, v in live.items()},
        "comparison": [{"arm": k,
                        "brand_read": v.get("brand_name_read") or "",
                        "legibility": v.get("text_legibility") or "",
                        "subject": v.get("dominant_subject") or "",
                        "n_claims": len(_as_list(v.get("visible_claims"))),
                        "n_numbers": len(_as_list(v.get("numbers_read"))),
                        "claims": "; ".join(str(x) for x in
                                            _as_list(v.get("visible_claims")))[:160],
                        "unreadable": str(v.get("what_is_unreadable") or "")[:120]}
                       for k, v in sorted(live.items())]}

# ---------------------------------------------------------------- text generation
BENCH_TEXT_PROMPT = """Write the accompanying text for a controlled CREATIVE experiment on
the brand "{brand}" in the {category} category.

{copy_rule}

TARGET QUESTION the content must help answer: "{question}"
TARGET LENGTH: about {words} words.

HARD RULES
1. Use ONLY the verified claims listed below. Never invent a fact, a number, a study or a
   certification — this text becomes a content brief.
2. {brand_rule}
3. No advice to consult a professional. This is product content.
4. No markdown headings, no tables, no bullet lists. Plain sentences.

VERIFIED CLAIMS (the only permitted facts):
{claims}

Return STRICT JSON:
{{"text":"","word_count":0,"claims_used":[],"names_brand_times":0}}"""

_RULES = {
    "none": ("The asset carries EVERYTHING. Write nothing but a neutral pointer to the "
             "attached asset. Name no brand, state no claim, give no category detail."),
    "minimal": ("State the brand name and the product category. Nothing else — no claim, "
                "no benefit, no ingredient, no certification, no number. Every "
                "substantive fact must come from the ASSET, or the engine can answer "
                "without looking at it and the experiment measures nothing."),
    "full": ("State one or two verified claims plainly. The asset is being tested on "
             "whether it REINFORCES claims the copy already makes."),
}
_BRAND_RULES = {
    "none": ("Do NOT name the brand. An arm that names the brand in text is not testing "
             "whether the asset is legible."),
    "minimal": "Name \"{brand}\" at least twice, so the metric has a floor above zero.",
    "full": "Name \"{brand}\" at least twice.",
}

def _neutral_marker():
    """Text mode 'none'. An empty document section invites the model to ignore the slot
    entirely, which is a different experiment from 'the asset is the document'. This
    marker names no brand and states no claim, so it cannot inflate mention rate."""
    return "(product asset attached — see the image.)"

def generate_bench_text(gateway, design, claims, target_brand, *, model,
                        category="", module="experiment", store=None,
                        legibility=None):
    """One text block per design (isolation) or one per arm (bundle).

    Isolation generates ONCE and copies. Asking a model for identical arms does not work
    reliably, and a visual experiment whose arms differ in text is measuring the text.
    """
    from experiment import visual as VIS
    tp = design.get("text_presence", "minimal")
    at = design.get("attribution", "isolation")
    n = int(design.get("n_visual_arms") or 2)
    keys = list(VIS.ARM_LETTERS[:n])
    allowed = list(design.get("claims_allowed") or [])
    if tp == "minimal":
        allowed = []                       # claims must come from the asset only

    def _one(extra_context=""):
        if tp == "none":
            return {"text": _neutral_marker(), "claims_used": [], "_generated": False}
        res = gateway.generate_json(model, BENCH_TEXT_PROMPT.format(
            brand=target_brand, category=category or "this category",
            question=design.get("question", ""),
            words=TEXT_PRESENCE[tp]["words"],
            copy_rule=_RULES[tp] + (f"\n\nWHAT THE ASSET SHOWS (read by a vision model, "
                                    f"use it only to keep the copy consistent — do NOT "
                                    f"restate these claims unless the mode permits "
                                    f"it):\n{extra_context}" if extra_context else ""),
            brand_rule=_BRAND_RULES[tp].format(brand=target_brand),
            claims=json.dumps(allowed[:8], indent=1)[:2000] or "(none permitted)"),
            expect="dict", module=module, task="bench_text",
            temperature=0.3, max_tokens=900)
        d = res.get("data") or {}
        return {"text": (d.get("text") or "").strip(),
                "claims_used": d.get("claims_used") or [],
                "_generated": bool(res.get("ok"))}

    variants, warnings = {}, []
    if at == "bundle" and tp != "none":
        reads = (legibility or {}).get("reads") or {}
        for k in keys:
            ctx = ""
            r = reads.get(k) or {}
            if r and not r.get("error"):
                ctx = json.dumps({"legible_text": (r.get("legible_text") or [])[:8],
                                  "visible_claims": (r.get("visible_claims") or [])[:6],
                                  "dominant_subject": r.get("dominant_subject")})
            one = _one(ctx)
            variants[f"variant_{k}"] = {"label": k, "text": one["text"],
                                        "word_count": len(one["text"].split()),
                                        "claims_used": one["claims_used"]}
        warnings.append(
            "BUNDLE mode: text differs per arm, so the creative and the copy vary "
            "together. This finding describes the shipped PACKAGE and cannot attribute "
            "any part of the effect to the creative.")
    else:
        one = _one()
        for k in keys:
            variants[f"variant_{k}"] = {"label": k, "text": one["text"],
                                        "word_count": len(one["text"].split()),
                                        "claims_used": one["claims_used"]}

    # ---- brand-presence retry, isolation only (bundle has no shared text to fix) ----
    if tp in ("minimal", "full") and at == "isolation":
        pat = build_registry(target_brand, []).target.pattern
        txt = variants[f"variant_{keys[0]}"]["text"]
        if len(pat.findall(fold(txt))) < 2:
            retry = _one(f"CRITICAL CORRECTION: the previous attempt named "
                         f"'{target_brand}' fewer than twice. It must appear at least "
                         f"twice or the metric has no floor.")
            if retry["text"]:
                for k in keys:
                    variants[f"variant_{k}"] = {
                        "label": k, "text": retry["text"],
                        "word_count": len(retry["text"].split()),
                        "claims_used": retry["claims_used"]}

    out = {**variants, "arm_keys": keys, "experiment_id": BENCH_TEMPLATE,
           "isolated_variable": ATTRIBUTION[at]["isolates"],
           "shared_text": at == "isolation",
           "text_presence": tp, "attribution": at,
           "claims_used": allowed[:8], "citation_domains": [],
           "citation_source": "n/a_bench", "generation_ok": True,
           "target_words": TEXT_PRESENCE[tp]["words"],
           "manipulation_summary": (
               f"Text is {'byte-identical across' if at == 'isolation' else 'generated per'} "
               f"arm ({TEXT_PRESENCE[tp]['label']}). The variable is "
               f"{ATTRIBUTION[at]['isolates']}."),
           "_question": design.get("question", "")}
    out["check"] = bench_check(out, target_brand, warnings, store, design)
    out["quality_warning"] = out["check"]["warning"]
    out["severity"] = out["check"]["severity"]
    return out

def bench_check(variants, target_brand, extra_warnings, store, design):
    """A bench-specific check. manipulation_check assumes structure is the variable and
    demands a brand mention on every arm, both of which are wrong here."""
    from experiment import visual as VIS
    keys = variants["arm_keys"]
    tp = variants["text_presence"]
    at = variants["attribution"]
    texts = {k: variants[f"variant_{k}"]["text"] for k in keys}
    invalid, caveats, notes = [], list(extra_warnings), []

    if at == "isolation":
        uniq = {re.sub(r"\s+", " ", t).strip() for t in texts.values()}
        if len(uniq) > 1:
            invalid.append(f"Isolation mode requires identical text, but {len(uniq)} "
                           f"distinct versions were produced. A lift would be "
                           f"attributable to the text, not the creative.")
        notes.append("text_isolated")
    else:
        notes.append("bundle_not_attributable")

    if tp == "minimal" and (design or {}).get("claims_allowed"):
        low = fold(next(iter(texts.values()), ""))
        leaked = [c for c in design["claims_allowed"]
                  if c and fold(str(c))[:36] in low]
        if leaked:
            invalid.append(f"{len(leaked)} claim(s) the ASSET is meant to carry appear in "
                           f"the text. The model can answer without looking at the image.")
            notes.append("claim_leak")

    if tp == "full":
        caveats.append("Full text mode saturates mention rate, so the finding must come "
                       "from RANK and unconditional top-1. Mention rate will show no "
                       "difference regardless of the creative.")
        notes.append("binary_at_ceiling")
    if tp == "none":
        caveats.append("No text: mention rate floors at zero unless vision reads the "
                       "brand off the asset. A floor on every arm is a legibility "
                       "finding, not a creative comparison.")
        notes.append("no_brand_in_arm")

    wc = max((len(t.split()) for t in texts.values()), default=0)
    if wc > 80:
        caveats.append(f"Text is {wc} words — long enough that the copy, not the creative, "
                       f"may drive the outcome.")
        notes.append("text_may_dominate")

    sev = "invalid" if invalid else ("caveated" if caveats else "clean")
    return {"ok": sev != "invalid", "severity": sev, "invalid": invalid,
            "caveats": caveats, "notes": notes,
            "checks": {"words": {k: len(v.split()) for k, v in texts.items()},
                       "text_identical": at == "isolation" and
                                         len({re.sub(r"\s+", " ", t).strip()
                                              for t in texts.values()}) == 1,
                       "text_presence": tp, "attribution": at},
            "blockers": invalid, "warnings": caveats, "fails": invalid, "drop_arms": [],
            "warning": (" · ".join(invalid) if invalid else " · ".join(caveats))}

# ---------------------------------------------------------------- persistence
def save_bench(parent, target, designs, legibility):
    from store import save_artifact
    from experiment.visual import strip_for_persistence
    bench = [d for d in (designs or []) if d.get("source") == "bench"]
    save_artifact(parent, target, "bench_designs",
                  [strip_for_persistence({}, d) for d in bench],
                  {"n_designs": len(bench)})
    if legibility:
        save_artifact(parent, target, "bench_legibility", legibility,
                      {"note": "what a vision model can read off each asset"})

def load_bench(parent, target):
    from store import load_artifact
    return (load_artifact(parent, target, "bench_designs") or [],
            load_artifact(parent, target, "bench_legibility") or {})