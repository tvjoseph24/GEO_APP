####################### new code block ##########
"""Template-driven variant generation with manipulation verification.

Design contract
---------------
E0  BUNDLED     A = plain prose, B = fully enriched (capsule+bullets+citations+table).
                Tests the GEO package. Length asymmetry is PART of the treatment, so an
                optional negative-control arm (A + equal-length irrelevant prose) rules
                out "more text wins".
E1  PROGRESSIVE Cumulative ladder: A prose -> +bullets -> +citations -> +table -> +capsule.
                Attributes the E0 lift to a specific element.
E2/E3/E4        Single variable. BOTH arms carry the SAME baseline structure, matched to
                the competitor field, so structure cannot leak into the result.
"""
from __future__ import annotations

import json
import re

from metrics_spec import fold

ARM_LETTERS = "ABCDEN"  # N = negative control

# ---- structural features, in the cumulative order used by E1 ----
FEATURES = ["prose", "bullets", "citations", "table", "capsule"]

ARM_SPEC = {
    # ---------- E0: the headline bundled test ----------
    "E0": {"variable": "the full GEO treatment (structure + specificity + corroboration)",
           "A": "PLAIN PROSE ONLY. Flowing paragraphs about {brand}. Absolutely NO "
                "bullets, NO markdown table, NO bold text, NO citations, NO headings. "
                "This represents the brand's current unstructured content.",
           "B": "FULLY ENRICHED. The SAME facts as A, restructured for machine "
                "extraction, and it MUST contain ALL FIVE of:\n"
                "   (1) a bold answer-capsule sentence at the top: **<one sentence on "
                "what {brand} is best for>**\n"
                "   (2) 4-6 bullet claims, each QUANTIFIED where the verified claims "
                "allow (numbers, named ingredients, durations)\n"
                "   (3) a markdown table with >=4 attribute rows: | Attribute | Value |\n"
                "   (4) inline citations in parentheses on the strongest claims, using "
                "ONLY these real domains: {citation_domains}\n"
                "   (5) a closing 'Sources:' line listing 2-3 of those domains",
           "N": "NEGATIVE CONTROL. Arm A's exact text, PLUS additional plain prose about "
                "{brand}'s heritage, company history and general category context, until "
                "it matches arm B's word count. Same plain-prose format as A: NO bullets, "
                "NO table, NO bold, NO citations. This arm isolates whether length alone "
                "explains any lift."},

    # ---------- E1: cumulative attribution ----------
    "E1": {"variable": "which structural element drives the lift",
           "A": "PLAIN PROSE about {brand}. No bullets, no table, no bold, no citations.",
           "B": "Arm A's content converted so the claims appear as 4-6 BULLETS. Nothing "
                "else added — no table, no bold capsule, no citations.",
           "C": "Arm B's content plus inline CITATIONS in parentheses on the strongest "
                "claims, using ONLY these real domains: {citation_domains}, plus a "
                "closing 'Sources:' line. No table, no bold capsule yet.",
           "D": "Arm C's content plus a markdown TABLE with >=4 rows: | Attribute | "
                "Value |. No bold capsule yet.",
           "E": "Arm D's content plus a bold ANSWER-CAPSULE sentence at the very top: "
                "**<one sentence on what {brand} is best for>**. Fully enriched."},

    # ---------- isolated single-variable tests ----------
    "E2": {"variable": "claim specificity",
           "A": "A document about {brand} using only VAGUE, unquantified benefit "
                "language. NO numbers, NO percentages, NO named ingredients, NO studies.",
           "B": "The SAME document — identical structure, identical length (±10%) — but "
                "every claim QUANTIFIED with the exact numbers, ingredient names and "
                "durations from the verified claims. ONLY specificity changes."},
    "E3": {"variable": "third-party corroboration",
           "A": "A document about {brand} whose claims are attributed to the brand's own "
                "materials only.",
           "B": "IDENTICAL wording to A, except each key claim additionally carries a "
                "third-party citation from these real domains: {citation_domains}. "
                "Add no new facts and no new structure."},
    "E4": {"variable": "presence of one specific missing claim",
           "A": "A document about {brand} that names {brand} and describes it factually "
                "and relevantly to the target question, but OMITS this claim entirely: "
                "{gap_claim}",
           "B": "IDENTICAL to A, plus one sentence stating: {gap_claim}. Nothing else "
                "changes."},
    "E5": {"variable": "format",
           "A": "The facts about {brand} as prose paragraphs.",
           "B": "The same facts as a markdown table only.",
           "C": "The same facts as a bullet list only."},
    "E9": {"variable": "objection rebuttal",
           "A": "A document about {brand} that does not address this objection: {objection}",
           "B": "IDENTICAL to A, plus a factual rebuttal of that objection built only "
                "from the verified claims."},
}

# Templates where structure IS the variable — do NOT force parity structure on the arms.
STRUCTURE_VARIABLE = {"E0", "E1", "E5"}

# Templates where BOTH arms must carry the competitor-matched baseline structure.
PARITY_STRUCTURE = """
 STRUCTURE — HELD CONSTANT. Every arm MUST contain ALL of the following, identically:
   1. a bold answer-capsule sentence at the top
   2. 3-5 bullet claims
   3. a markdown table with >=3 rows: | Attribute | Value |
   4. a closing 'Sources:' line
 The competitor documents in the same context all have these. Structure is NOT the
 variable here — only {variable} differs between arms.
 """

VARIANT_PROMPT = """Create a controlled GEO experiment for the brand "{brand}".

 EXPERIMENT {exp_id} — isolates: {variable}
 HYPOTHESIS: {hypothesis}
 TARGET QUESTION the content must help answer: "{question}"

 NON-NEGOTIABLE RULES
 1. FACTS: use ONLY the verified claims below. NEVER invent a fact, a number, a study or a
    certification. This is a regulated category.
 2. BRAND: every arm MUST name "{brand}" at least THREE times and read as a document
    ABOUT {brand}. A control arm is {brand}'s document minus the isolated variable —
    never generic category text.
 3. RELEVANCE: every arm must be on-topic for the target question. An off-topic control
    measures relevance, not your variable.
 4. LENGTH: arm A must be {target_words} words (±10%).{length_note}
 5. NO ADVICE: do not tell the reader to consult a professional. This is product content.
 6. CITATIONS: when an arm calls for citations, use ONLY the real domains supplied. Never
    invent a source.

 VERIFIED CLAIMS (the only permitted facts):
 {claims}

 ATTRIBUTE HINTS for the table (values must come from the verified claims):
 {attribute_hints}

 {arms}
 {parity}{emphasis}
 Return STRICT JSON:
 {{{{{arm_json},
    "manipulation_summary":"one line: exactly what differs between the arms"}}}}"""

QUANT_RX = re.compile(r"\d+(?:[.,]\d+)?\s?(?:%|mg|mcg|ml|g|kg|iu|oz|kcal|hour|hr|day|"
                      r"week|month|x)", re.I)
CITE_RX = re.compile(r"\(([^)]*\.[a-z]{2,}[^)]*)\)|\bhttps?://|sources?\s*:", re.I)
TABLE_RX = re.compile(r"^\s*\|.+\|\s*$", re.M)
BULLET_RX = re.compile(r"^\s*[-*•]\s+\S", re.M)
CAPSULE_RX = re.compile(r"\*\*[^*]{15,}\*\*")

# A citation requires a real DOMAIN. An empty "Sources:" label must not satisfy the check.
DOMAIN_RX = re.compile(
    r"\b[a-z0-9][a-z0-9\-]{1,}\.(?:com|org|net|gov|edu|io|co\.uk|uk|de|au|in|ie|ch|my)\b",
    re.I)

def _quant_capacity(claims):
    """How many quantified values the APPROVED claims can actually supply.
    Regulatory/usage claim sets often supply none — specificity is then untestable,
    which is a property of the claim register, not a generation failure."""
    return sum(len(QUANT_RX.findall(str(c or ""))) for c in (claims or []))

def _claim_coverage(text, claims, thresh=0.6):
    """Share of approved claims whose key tokens survive in the text. For E0 this
    replaces word count: restructuring must not LOSE facts."""
    if not claims:
        return 1.0
    tt = set(re.findall(r"[a-z]{4,}", fold(text)))
    hits = 0
    for c in claims:
        ct = set(re.findall(r"[a-z]{4,}", fold(str(c))))
        if ct and len(ct & tt) / len(ct) >= thresh:
            hits += 1
    return hits / len(claims)

def _has(feature, text):
    return {"prose": bool(text and text.strip()),
            "bullets": bool(BULLET_RX.search(text)),
            "citations": bool(DOMAIN_RX.search(text)),      # real domain, not a label
            "table": len(TABLE_RX.findall(text)) >= 2,
            "capsule": bool(CAPSULE_RX.search(text))}.get(feature, False)

def _features(text):
    return {f: _has(f, text) for f in FEATURES}
# NOTE: no hardcoded domain constants. Citation domains are DERIVED, in this order:
#   1. the Benchmark source hit-list (this brand's own audit — always preferred)
#   2. a scan of audit_raw.jsonl for editorially-classified cited domains
#   3. one cached LLM lookup for the category
#   4. nothing — in which case citation-dependent arms are DOWNGRADED, never faked.
CITE_TYPES = ("editorial_authority", "trade_press", "reference")

def citation_domains(hitlist=None, k=4, audit_raw=None, gateway=None, model=None,
                     category="", parent_brand=None, target_brand=None):
    """Real third-party domains, derived. Returns (domains, provenance)."""
    # ---- 1. hit-list ----
    rows = (hitlist.to_dict("records") if hasattr(hitlist, "to_dict") else (hitlist or []))
    got = [r.get("domain") for r in rows
           if r.get("source_type") in CITE_TYPES and r.get("domain")]
    out = list(dict.fromkeys(got))[:k]
    if out:
        return out, "hitlist"

    # ---- 2. broader scan of the raw audit ----
    if audit_raw:
        from collections import Counter
        from pathlib import Path
        import json as _j
        c = Counter()
        p = Path(audit_raw)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = _j.loads(line)
                    except Exception:
                        continue
                    for s in (r.get("sources") or []):
                        if s.get("source_type") in CITE_TYPES and s.get("domain"):
                            c[s["domain"]] += 1
        out = [d for d, _ in c.most_common(k)]
        if out:
            return out, "audit_scan"

    # ---- 3. cached LLM lookup for the category ----
    if gateway and model and category:
        cached = None
        if parent_brand and target_brand:
            try:
                from store import load_cache, save_cache
                cached = (load_cache(parent_brand, target_brand, "cite_domains") or {})
                if cached.get(fold(category)):
                    return cached[fold(category)][:k], "llm_cached"
            except Exception:
                cached = None
        try:
            res = gateway.generate_json(
                model,
                f'Name the {k} most authoritative INDEPENDENT sources that publish '
                f'evidence-based information about the "{category}" category — '
                f'regulators, clinical bodies, peer-reviewed repositories or established '
                f'consumer-testing publications. Real, currently-live domains only. '
                f'No brand-owned sites, no retailers, no blogs.\n'
                f'STRICT JSON: {{"domains":["example.org"]}}',
                expect="dict", module="experiment", task="cite_domains",
                temperature=0.0, max_tokens=400)
            out = [d for d in (res.get("data") or {}).get("domains", [])
                   if isinstance(d, str) and "." in d][:k]
            if out and cached is not None and parent_brand and target_brand:
                cached[fold(category)] = out
                save_cache(parent_brand, target_brand, "cite_domains", cached)
            if out:
                return out, "llm"
        except Exception:
            pass

    # ---- 4. refuse. An invented citation invalidates the experiment. ----
    return [], "unavailable"

def attribute_hints(claims, comp_docs=None, k=8):
    """Table row LABELS, derived. Competitor docs are built from this brand's own audit,
    so their attribute keys are category-correct by construction — no hardcoding."""
    hints, seen = [], set()

    # 1. whatever the competitor documents themselves tabulate
    for d in (comp_docs or {}).values():
        for a in (d.get("attributes") or {}):
            key = str(a).strip()
            if key and fold(key) not in seen:
                hints.append(key)
                seen.add(fold(key))

    # 2. generic, category-independent labels mapped from claim categories
    generic = {"ingredient": "Key ingredients", "credential": "Certification",
               "efficacy": "Primary benefit", "safety": "Safety profile",
               "feature": "Format", "other": "Notable attribute"}
    for c in (claims or []):
        lab = generic.get(c.get("category"))
        if lab and fold(lab) not in seen:
            hints.append(lab)
            seen.add(fold(lab))
    return hints[:k]



def _arm_block(exp_id, design, hypothesis, brand, cite_domains, include_negative=False):
    spec = ARM_SPEC.get(exp_id, ARM_SPEC["E4"])
    gap_claim = (design.get("gap_claim")
                 or (hypothesis.get("claims_used") or ["the missing claim"])[0])
    objection = design.get("objection") or design.get("gap") or "the stated concern"
    fmt = dict(brand=brand, gap_claim=gap_claim, objection=objection,
               citation_domains=", ".join(cite_domains) if cite_domains else "")
    keys = [k for k in "ABCDE" if k in spec]
    if include_negative and "N" in spec:
        keys.append("N")
    lines = [f"ARM {k}: " + spec[k].format(**fmt) for k in keys]
    if not cite_domains:
        lines.append(
            "\nCITATION OVERRIDE: no verified third-party domains are available for this "
            "engagement. DO NOT invent any source, domain or publication. Omit all "
            "citations and the 'Sources:' line from every arm.")
    return "\n".join(lines), spec["variable"], keys

def generate_text_variants(gateway, design, claims, target_brand,
                           model="gemini-2.5-flash", module="experiment",
                           hypothesis=None, target_words=200, hitlist=None,
                           include_negative_control=False,
                           audit_raw=None, parent_brand=None, category="",
                           comp_docs=None):
    h = hypothesis or {}
    exp_id = ((h.get("experiment") or {}).get("template")
              or design.get("experiment_id") or "E0")

    cites, cite_src = citation_domains(
        hitlist=hitlist, audit_raw=audit_raw, gateway=gateway, model=model,
        category=category or design.get("category", ""),
        parent_brand=parent_brand, target_brand=target_brand)

    arms_txt, variable, keys = _arm_block(exp_id, design, h, target_brand, cites,
                                          include_negative_control)
    allowed = h.get("claims_used") or _relevant(design, claims)
    hints = attribute_hints(claims, comp_docs)

    parity = ("" if exp_id in STRUCTURE_VARIABLE
              else PARITY_STRUCTURE.format(variable=variable))
    length_note = (" Enriched arms will naturally be longer — that is expected; do not "
                   "truncate them to match A."
                   if exp_id in ("E0", "E1") else
                   " All arms must be within 10% of each other.")
    arm_json = ",\n   ".join(
        f'"variant_{k}":{{{{"label":"{k}","text":"","word_count":0,"claims_used":[]}}}}'
        for k in keys)

    def _gen(emphasis=""):
        res = gateway.generate_json(model, VARIANT_PROMPT.format(
            brand=target_brand, exp_id=exp_id, variable=variable,
            hypothesis=h.get("hypothesis") or design.get("hypothesis", ""),
            question=design.get("question", ""), target_words=target_words,
            length_note=length_note,
            claims=json.dumps(allowed[:14], indent=1)[:4500],
            attribute_hints=json.dumps(hints),
            arms=arms_txt, parity=parity, emphasis=emphasis, arm_json=arm_json),
                                    expect="dict", module=module, task="variant_gen",
                                    temperature=0.4, max_tokens=7000)
        return res.get("data") or {}, bool(res.get("ok"))

    v, ok = _gen()

    def _txt(d, k):
        return ((d.get(f"variant_{k}") or {}).get("text") or "")

    # ---- retry on control length, missing brand, or a missing E0 feature ----
    from metrics_spec import build_registry
    _pat = build_registry(target_brand, []).target.pattern
    wa = len(_txt(v, "A").split())
    need_retry, why = False, []
    if wa and not (0.85 * target_words <= wa <= 1.15 * target_words):
        need_retry = True
        why.append(f"arm A was {wa} words; it MUST be {target_words} (±10%). The "
                   f"competitor documents are the same length.")
    for k in keys:
        if _txt(v, k) and not _pat.search(fold(_txt(v, k))):
            need_retry = True
            why.append(f"arm {k} never names '{target_brand}' — every arm must be a "
                       f"document ABOUT the brand, naming it at least three times")
    if exp_id == "E0":
        miss = [f for f in FEATURES[1:] if not _has(f, _txt(v, "B"))]
        if not cites:
            miss = [f for f in miss if f != "citations"]
        if miss:
            need_retry = True
            why.append(f"arm B is missing {', '.join(miss)}; it MUST contain a bold "
                       f"capsule, bullets, a markdown table"
                       + (" and inline citations with real domains" if cites else ""))
        if any(_has(f, _txt(v, "A")) for f in ("bullets", "table", "capsule")):
            need_retry = True
            why.append("arm A must be PLAIN PROSE — no bullets, table or bold")
    if need_retry:
        v2, ok2 = _gen(emphasis="\nCRITICAL CORRECTION — your previous attempt failed: "
                                + "; ".join(why) + ". Fix ALL of it.")
        if _txt(v2, "A"):
            v, ok = v2, ok2

    out = {f"variant_{k}": (v.get(f"variant_{k}") or {"label": k, "text": ""})
           for k in keys}
    out.update({"manipulation_summary": v.get("manipulation_summary", ""),
                "claims_used": allowed[:14], "experiment_id": exp_id,
                "isolated_variable": variable, "generation_ok": ok,
                "target_words": target_words, "citation_domains": cites,
                "citation_source": cite_src, "arm_keys": list(keys),
                "_question": design.get("question", "")})

    out["check"] = manipulation_check(
        exp_id, out, keys, allowed, brand=target_brand, target_words=target_words,
        citations_available=bool(cites),
        mechanism=(h.get("mechanism") or design.get("mechanism")))

    # prune arms the check deemed unnecessary (e.g. negative control when B <= A)
    for k in (out["check"].get("drop_arms") or []):
        out.pop(f"variant_{k}", None)
        if k in keys:
            keys.remove(k)
    out["arm_keys"] = list(keys)
    out["quality_warning"] = out["check"]["warning"]
    out["severity"] = out["check"]["severity"]
    return out


def _relevant(design, claims):
    words = set(re.findall(r"[a-z]{4,}", fold(design.get("gap", "") + " " +
                                              design.get("question", ""))))
    scored = []
    for c in (claims or []):
        ct = set(re.findall(r"[a-z]{4,}", fold(c.get("claim", ""))))
        if ct:
            scored.append((len(words & ct) / len(ct), c.get("claim")))
    scored.sort(key=lambda x: -x[0])
    top = [t for s, t in scored if s > 0.1][:14]
    return top or [c.get("claim") for c in (claims or [])][:10]

def manipulation_check(exp_id, arms, keys, allowed_claims, brand="", target_words=200,
                       citations_available=True, mechanism=None):
    """Classify design quality. Severity determines the ACTION, because the costs differ:

      legal        — variant text becomes a content brief. Unsubstantiated or invented
                     claims cannot be generated. No override.
      invalid      — no information is possible (identical arms, no treatment delta).
                     Overridable, but expect a null.
      caveated     — RUNNABLE. The confound qualifies how the FINDING is read, and
                     travels with it into Optimise as an interpretation limit.
      not_testable — not an experiment; the claim does not exist yet → Pillar 6.
      clean        — no known limitation.

    Blocking a valid experiment costs far more than running a caveated one, so only
    the first two categories stop a run.
    """
    from metrics_spec import build_registry

    def t(k):
        return (arms.get(f"variant_{k}") or {}).get("text", "") or ""

    invalid, legal, caveats, notes, drop_arms = [], [], [], [], []
    checks = {}
    variable = arms.get("isolated_variable", "the variable")

    def bad(m):
        invalid.append(m)

    def illegal(m):
        legal.append(m)

    def caveat(m):
        caveats.append(m)

    def _out():
        sev = ("legal" if legal else "invalid" if invalid
               else "caveated" if caveats else "clean")
        return {"ok": sev in ("clean", "caveated"), "severity": sev,
                "invalid": invalid, "legal": legal, "caveats": caveats, "notes": notes,
                "checks": checks, "drop_arms": drop_arms,
                # back-compat keys
                "blockers": invalid + legal, "warnings": caveats,
                "fails": invalid + legal,
                "warning": (" · ".join(legal) if legal else
                            " · ".join(invalid) if invalid else
                            " · ".join(caveats) if caveats else "")}

    # ---- 0. not an experiment at all ----
    if mechanism == "substantiation_deficit":
        notes.append("substantiation_deficit")
        return {"ok": False, "severity": "not_testable",
                "invalid": [], "legal": [], "caveats": [], "notes": notes,
                "checks": {}, "drop_arms": [],
                "blockers": ["the claim is not in the substantiated register, so it "
                             "cannot be published or tested — route to Optimise as a "
                             "Claims-Substantiation action"],
                "warnings": [], "fails": [],
                "warning": "🔬 NOT TESTABLE YET — substantiate the claim first"}

    if not t("A").strip() or not t("B").strip():
        bad("an arm is empty")
        return _out()

    checks["features"] = {k: [f for f, v in _features(t(k)).items() if v] for k in keys}
    checks["words"] = {k: len(t(k).split()) for k in keys}
    checks["words"]["target_A"] = target_words
    wa, wb = checks["words"]["A"], checks["words"]["B"]
    if not citations_available:
        checks["citations_skipped"] = "no verified third-party domains available"

    # ---- 1. identical arms → no-op ----
    ta = set(re.findall(r"[a-z]{4,}", fold(t("A"))))
    tb = set(re.findall(r"[a-z]{4,}", fold(t("B"))))
    if ta and tb:
        jac = len(ta & tb) / len(ta | tb)
        checks["arm_similarity"] = round(jac, 3)
        if jac > 0.97 and abs(wb - wa) < 5:
            bad("arms A and B are effectively identical — no variable is being tested")

    # ---- 2. brand presence (auto-retried upstream; loud caveat if it survives) ----
    if brand:
        pat = build_registry(brand, []).target.pattern
        for k in keys:
            n = len(pat.findall(fold(t(k))))
            checks.setdefault("brand_mentions", {})[k] = n
            if n < 1:
                caveat(f"arm {k} never names '{brand}' — its baseline is near zero by "
                       f"construction, so the lift measures brand presence, not {variable}")
                notes.append("no_brand_in_arm")

    # ---- 3. control length ----
    if target_words and not (0.70 * target_words <= wa <= 1.45 * target_words):
        caveat(f"arm A is {wa} words vs the {target_words} the competitor docs were "
               f"built to — absolute rates will be depressed")
        notes.append("length_asymmetry")

    # ---- 4. relevance ----
    q = arms.get("_question") or ""
    if q:
        qt = {w for w in re.findall(r"[a-z]{4,}", fold(q))} - {
            "best", "what", "which", "based", "your", "with", "good", "help", "when",
            "does", "should", "there", "than", "from", "have", "most", "more", "pick"}
        for k in ("A", "B"):
            ov = len(qt & set(re.findall(r"[a-z]{4,}", fold(t(k))))) / max(len(qt), 1)
            checks.setdefault("relevance", {})[k] = round(ov, 2)
        if checks["relevance"].get("A", 1) < 0.20:
            caveat(f"arm A shares only {checks['relevance']['A']:.0%} of the question's "
                   f"terms — an off-topic control lowers the baseline")
            notes.append("low_relevance")

    # ---- 5. quantification, CAPACITY-AWARE ----
    cap = _quant_capacity(allowed_claims)
    qa, qb = len(QUANT_RX.findall(t("A"))), len(QUANT_RX.findall(t("B")))
    checks["quantified"] = {"A": qa, "B": qb, "claim_capacity": cap}

    def _quant_rule():
        if cap <= qa:
            caveat(f"the approved claims supply only {cap} quantified value(s) and arm A "
                   f"already uses {qa} — SPECIFICITY IS UNTESTABLE with this claim set")
            notes.append("specificity_untestable")
        elif qb <= qa:
            caveat(f"arm B is no more quantified than A ({qb} vs {qa}) despite {cap} "
                   f"value(s) available — the specificity manipulation did not land")
            notes.append("specificity_not_applied")

    # ---- 6. template-specific ----
    if exp_id == "E0":
        fa, fb = _features(t("A")), _features(t("B"))
        leaked = [f for f in ("bullets", "table", "capsule") if fa[f]]
        if leaked:
            caveat(f"arm A is not pure prose — has {', '.join(leaked)}; this tests "
                   f"more-vs-less structure rather than none-vs-full")
            notes.append("control_partly_structured")
        required = [f for f in FEATURES[1:]
                    if not (f == "citations" and not citations_available)]
        missing = [f for f in required if not fb[f]]
        if missing:
            caveat(f"arm B is missing {', '.join(missing)} — the treatment is weaker "
                   f"than the full GEO package")
            notes.append("treatment_incomplete")
        if not any(fb[f] for f in FEATURES[1:]):
            bad("arm B has no structural enrichment at all — nothing to test")
        _quant_rule()

        ca, cb = (_claim_coverage(t("A"), allowed_claims),
                  _claim_coverage(t("B"), allowed_claims))
        checks["claim_coverage"] = {"A": round(ca, 2), "B": round(cb, 2)}
        if cb < ca - 0.15:
            caveat(f"arm B retains only {cb:.0%} of the approved claims vs A's {ca:.0%} — "
                   f"structuring lost facts, so content varies alongside structure")
            notes.append("facts_lost_in_treatment")

        # negative control is only meaningful when B is LONGER than A
        if "N" in keys:
            wn = checks["words"]["N"]
            checks["neg_control"] = {"N": wn, "B": wb, "A": wa,
                                     "ratio_to_B": round(wn / max(wb, 1), 2)}
            if wb <= wa * 1.05:
                drop_arms.append("N")
                notes.append("neg_control_dropped")
                caveat(f"arm B ({wb}w) is not longer than arm A ({wa}w) — enrichment "
                       f"COMPRESSED the text, so 'more text wins' is already excluded. "
                       f"Negative control dropped as unnecessary.")
            elif not (0.80 <= wn / max(wb, 1) <= 1.30):
                drop_arms.append("N")
                notes.append("neg_control_dropped")
                caveat(f"negative control is {wn}w vs B's {wb}w — cannot isolate volume; "
                       f"arm dropped. B vs A remains valid.")
            elif any(_has(f, t("N")) for f in ("bullets", "table", "capsule")):
                drop_arms.append("N")
                notes.append("neg_control_dropped")
                caveat("negative control is not plain prose — arm dropped")

    elif exp_id == "E1":
        order = [k for k in "ABCDE" if k in keys]
        broken = False
        for i, k in enumerate(order):
            got = {f for f, v in _features(t(k)).items() if v}
            exp_f = set(FEATURES[:i + 1])
            if not citations_available:
                exp_f.discard("citations")
                got.discard("citations")
            if (exp_f - got) or (got - exp_f):
                broken = True
                caveat(f"arm {k} is not strictly cumulative "
                       f"(missing {sorted(exp_f - got) or '—'}, "
                       f"extra {sorted(got - exp_f) or '—'})")
        if broken:
            notes.append("attribution_unreliable")

    elif exp_id == "E2":
        _quant_rule()
        for f in ("bullets", "table", "capsule", "citations"):
            if f == "citations" and not citations_available:
                continue
            if _has(f, t("A")) != _has(f, t("B")):
                caveat(f"{f} presence differs between arms — structure confounds "
                       f"specificity")
                notes.append("structure_confound")

    elif exp_id == "E3":
        if not citations_available:
            bad("E3 tests corroboration but no verified third-party domain is available "
                "— the treatment cannot be constructed without inventing a source")
        ca_, cb_ = len(DOMAIN_RX.findall(t("A"))), len(DOMAIN_RX.findall(t("B")))
        checks["citation_domains_found"] = {"A": ca_, "B": cb_}
        if cb_ <= ca_ and citations_available:
            bad(f"arm B has no additional cited domain ({cb_} vs {ca_}) — no corroboration "
                f"delta to measure")
        for f in ("bullets", "table", "capsule"):
            if _has(f, t("A")) != _has(f, t("B")):
                caveat(f"{f} presence differs between arms — structure confound")
                notes.append("structure_confound")

    elif exp_id in ("E4", "E9"):
        delta = wb - wa
        claim_words = max((len(str(c).split()) for c in (allowed_claims or [])),
                          default=15)
        checks["added_words"] = delta
        new = tb - ta
        checks["new_tokens_in_B"] = len(new)
        if delta <= 0 or len(new) < 3:
            bad("arm B adds nothing — the claim or rebuttal is absent")
        elif delta > 3 * claim_words + 25:
            caveat(f"arm B adds {delta} words for a ~{claim_words}-word addition — more "
                   f"than the isolated variable changed")
            notes.append("addition_oversized")
        for f in ("bullets", "table", "capsule"):
            if _has(f, t("A")) != _has(f, t("B")):
                caveat(f"{f} presence differs between arms — structure confound")
                notes.append("structure_confound")

    elif exp_id == "E5":
        fmt = {k: ("table" if _has("table", t(k)) else
                   "bullets" if _has("bullets", t(k)) else "prose") for k in keys}
        checks["formats"] = fmt
        if len(set(fmt.values())) < len(keys):
            bad(f"arms lack distinct formats: {fmt}")

    # ---- 7. parity structure for single-variable templates ----
    if exp_id not in STRUCTURE_VARIABLE:
        need = ["capsule", "bullets", "table"] + (
            ["citations"] if citations_available else [])
        for k in keys:
            miss = [f for f in need if not _has(f, t(k))]
            if miss:
                caveat(f"arm {k} missing {'+'.join(miss)} — competitor docs have them, "
                       f"so absolute rates will be low; A vs B remains valid")
                notes.append("below_field_parity")
                break

    # ---- 8. claims guardrail — REGULATORY, no override ----
    reg = {fold(c) for c in (allowed_claims or [])}
    unsourced = []
    for k in keys:
        for c in ((arms.get(f"variant_{k}") or {}).get("claims_used") or []):
            ct = set(fold(c).split())
            if fold(c) not in reg and not any(
                    len(ct & set(r.split())) / max(len(ct | set(r.split())), 1) > 0.6
                    for r in reg):
                unsourced.append(str(c)[:70])
    if unsourced:
        illegal(f"claim(s) NOT in the approved register: {unsourced[:2]} — variant text "
                f"becomes a content brief, so this cannot be generated")

    # ---- 9. invented domains — REGULATORY ----
    if citations_available:
        allowed_d = {fold(d) for d in (arms.get("citation_domains") or [])}
        found = {m.lower() for k in keys for m in DOMAIN_RX.findall(t(k))}
        invented = [d for d in found if not any(a in d or d in a for a in allowed_d)]
        if invented:
            checks["invented_domains"] = invented[:4]
            illegal(f"invented citation domain(s): {invented[:3]} — only "
                    f"{sorted(allowed_d)} were supplied")

    return _out()

def attach_visual_variant(design, image_bytes, mime):
    design["visual_asset"] = {"bytes": image_bytes, "mime": mime}
    return design

def generate_progressive_variants(gateway, design, claims, target_brand,
                                  model="gemini-2.5-flash", module="experiment",
                                  hypothesis=None, target_words=200, hitlist=None,
                                  audit_raw=None, parent_brand=None, category="",
                                  comp_docs=None):
    d = dict(design)
    d["experiment_id"] = "E1"
    h = dict(hypothesis or {})
    h["experiment"] = {**(h.get("experiment") or {}), "template": "E1"}
    v = generate_text_variants(
        gateway, d, claims, target_brand,
        model=model, module=module, hypothesis=h, target_words=target_words,
        hitlist=hitlist, include_negative_control=False,
        audit_raw=audit_raw, parent_brand=parent_brand, category=category,
        comp_docs=comp_docs)
    arms = {k: v[f"variant_{k}"] for k in "ABCDE" if f"variant_{k}" in v}
    return {"arms": arms, "claims_used": v["claims_used"], "check": v["check"],
            "quality_warning": v["quality_warning"], "feature_order": FEATURES,
            "citation_source": v.get("citation_source")}