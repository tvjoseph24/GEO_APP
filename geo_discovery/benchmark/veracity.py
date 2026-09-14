"""Veracity engine v2 — corpus-wide, per-engine, quote-grounded.

Replaces: 15 truncated responses vs a truncated claim list (which sampled the first
query on the first engine and judged ~4 responses).

Pipeline
  1  span corpus                      (spans.py)
  2  claim-conditioned retrieval      lexical + numeric must-match; zero hits ⇒ ABSENT
  3  LLM judge, verbatim-quote gated  supported/partial/distorted/contradicted/omitted
  4  REVERSE hallucination scan       what is it saying that we never claimed?
  5  aggregation                      Wilson CIs, fragility, per-engine, provenance
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from benchmark.spans import build_span_corpus
from metrics_spec import fold

_JUDGE_FAILURES: list = []

def reset_failures():
    _JUDGE_FAILURES.clear()

STOP = set("the a an and or of for for with to in on is are be been being this that "
           "from as by it its their has have had was were will can may also more most "
           "than then them these those which who whom what when where how why".split())

NUM_RX = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|mg|mcg|ml|g|kg|iu|oz|hour|hr|h|day|week|"
                    r"month|year|x|fold|kcal)?", re.I)

_WS = re.compile(r"\s+")
_MD = re.compile(r"[*_`~]+")
_TRIM = " \"'`*_·—–-…\n\t"

def _canon(s):
    return _WS.sub(" ", fold(s or "")).strip(_TRIM)

def _stratified_sample(df, by, per_group, cap, seed=7):
    """Version-proof stratified sample — no groupby.apply."""
    parts = []
    for _, g in df.groupby(by, dropna=False):
        parts.append(g.sample(min(len(g), per_group), random_state=seed))
    if not parts:
        return []
    return pd.concat(parts, ignore_index=True).head(cap).to_dict("records")

def verify_quote(quote, span):
    """Returns (ok, method). Whitespace/markdown/ellipsis tolerant — but still refuses
    paraphrase, which is the actual failure mode we care about."""
    q, s = _canon(quote), _canon(span)
    if not q:
        return False, "empty"
    if q in s:
        return True, "exact"
    q2, s2 = _MD.sub("", q), _MD.sub("", s)
    if q2 in s2:
        return True, "markdown"
    parts = [p.strip(_TRIM) for p in re.split(r"\.{3,}|…|\[\.\.\.\]", q2)]
    parts = [p for p in parts if len(p) >= 12]
    if len(parts) >= 2 and all(p in s2 for p in parts):
        return True, "elided"
    qt = q2.split()
    if len(qt) >= 6:
        st_ = set(s2.split())
        if sum(1 for w in qt if w in st_) / len(qt) >= 0.9:
            return True, "fuzzy"
    return False, "not_found"
# ------------------------------------------------------------------ 2. retrieval
def claim_terms(claim: str, drop=()):
    """drop: brand aliases — non-discriminative, they appear in every span by construction."""
    low = fold(claim)
    for d in drop:
        low = re.sub(r"(?<![a-z0-9])" + re.escape(fold(d)) + r"(?![a-z0-9])", " ", low)
    low = low.replace("'s", " ")
    nums = [n.strip() for n in NUM_RX.findall(low) if any(ch.isdigit() for ch in n)]
    words = [w for w in re.findall(r"[a-z][a-z\-]{3,}", low) if w not in STOP]
    # generic claim verbs carry no retrieval signal
    words = [w for w in words if w not in {"claims", "markets", "designed", "aimed",
                                           "supports", "support", "helps", "provide",
                                           "provides", "brand", "product", "range",
                                           "formula", "formulas", "specific", "other",
                                           "brands", "made"}]
    return nums, words

def _score_span(span_low, nums, words):
    if not words and not nums:
        return 0.0
    w_hits = sum(1 for w in set(words) if w in span_low)
    n_hits = sum(1 for n in set(nums) if n in span_low)
    w_frac = w_hits / max(len(set(words)), 1)
    return round(w_frac * 2.0 + n_hits * 1.5, 3)

def retrieve_spans_for_claim(claim, span_df, per_engine=4, min_score=0.5,
                             drop=(), rescue=True):
    """Balanced top-k per engine, with a rescue pass so a near-miss claim is never
    silently recorded as ABSENT."""
    nums, words = claim_terms(claim, drop)
    if span_df.empty or (not words and not nums):
        return []
    picked, best = [], []
    for eng, g in span_df.groupby("engine"):
        scored = sorted(((_score_span(fold(s["span"]), nums, words), s)
                         for _, s in g.iterrows()), key=lambda x: -x[0])
        hits = [s for sc, s in scored if sc >= min_score][:per_engine]
        if not hits and rescue and scored and scored[0][0] > 0:
            hits = [s for _, s in scored[:2]]       # top-2 anyway; judge decides
            best += hits
        for s in hits:
            d = s.to_dict()
            d["_retrieval_score"] = next((sc for sc, x in scored
                                          if x["span_id"] == s["span_id"]), 0)
            d["_rescued"] = s in best
            picked.append(d)
    return picked

# ------------------------------------------------------------------ 3. judge
VERACITY_JUDGE_PROMPT = """You are auditing how AI assistants describe the brand "{brand}".

VERIFIED CLAIM (ground truth, from the brand's substantiated claims register):
{claim}
Proof on record: {proof}

Below are EXCERPTS from real AI answers that mention {brand}. For EACH excerpt, judge how
it treats the verified claim.

VERDICTS
- "supported"    : states the claim accurately (paraphrase is fine)
- "partial"      : directionally right but drops the specific/quantified part
                   (e.g. says "supports immunity" where the claim specifies a named
                   ingredient at a stated level)
- "distorted"    : states the claim with WRONG specifics (wrong number/ingredient/cert)
- "contradicted" : asserts the opposite
- "omitted"      : the excerpt is on-topic for this claim but never mentions it

ATTRIBUTION
- "asserted" : stated as fact by the assistant
- "hedged"   : "may", "reportedly", "the brand claims"
- "sourced"  : attributed to a named source

RULES
- "quote" MUST be copied VERBATIM from the excerpt. Never paraphrase inside quote.
- If you cannot find supporting text, use "omitted" with quote "".
- delta_note: ONE short line naming the exact discrepancy (number, ingredient, cert).
- Judge ONLY what the excerpt says about {brand}. Ignore statements about other brands.

- "assertion" must be a full sentence stating the fact, in your own words.
  NEVER put a span_id in the assertion field.

Return STRICT JSON:
[{{"span_id":"","verdict":"supported|partial|distorted|contradicted|omitted",
   "attribution":"asserted|hedged|sourced|none","quote":"","delta_note":""}}]

EXCERPTS:
{excerpts}"""

VALID_VERDICTS = {"supported", "partial", "distorted", "contradicted", "omitted"}

def _excerpt_block(spans, max_chars=13000, per_span=1300):
    parts = []
    for s in spans:
        parts.append(f'[span_id={s["span_id"]} | engine={s["engine"]} '
                     f'| intent={s.get("intent")}]\n{s["span"][:per_span]}')
    return "\n\n".join(parts)[:max_chars]

def judge_claim(gateway, brand, claim_obj, spans, model, module="benchmark",
                cache=None):
    if not spans:
        return []
    key = None
    if cache is not None:
        key = hashlib.sha256(
            (str(claim_obj.get("claim")) + "|" + model + "|v2|" +
             ",".join(sorted(s["span_id"] for s in spans))).encode()).hexdigest()[:24]
        if key in cache:
            return cache[key]

    try:
        res = gateway.generate_json(model, VERACITY_JUDGE_PROMPT.format(
            brand=brand, claim=claim_obj.get("claim", ""),
            proof=claim_obj.get("proof_in_source") or "(none on record)",
            excerpts=_excerpt_block(spans)),
                                    expect="list", module=module, task="veracity_judge",
                                    temperature=0.0, max_tokens=3000,
                                    coerce_keys=("verdicts", "judgements", "items"))
    except Exception as e:
        _JUDGE_FAILURES.append({"claim": (claim_obj.get("claim") or "")[:120],
                                "reason": f"exception: {type(e).__name__}: {str(e)[:100]}",
                                "attempts": "n/a", "n_spans": len(spans)})
        return []
    raw = res["data"] or []
    if not res["ok"]:
        _JUDGE_FAILURES.append({"claim": (claim_obj.get("claim") or "")[:120],
                                "reason": res["failure"], "attempts": res["attempts"],
                                "finish": res.get("finish_reason"),
                                "n_spans": len(spans)})
    by_id = {s["span_id"]: s for s in spans}
    clean = []
    for v in raw:
        if not isinstance(v, dict):
            continue
        s = by_id.get(v.get("span_id"))
        if not s:
            continue
        verdict = str(v.get("verdict", "")).lower().strip()
        if verdict not in VALID_VERDICTS:
            continue
        q = (v.get("quote") or "").strip()
        ok, how = verify_quote(q, s["span"])
        if not ok and how != "empty":
            q = ""  # judge fabricated the quote
        clean.append({
            "claim": claim_obj.get("claim"), "claim_id": claim_obj.get("claim_id"),
            "claim_category": claim_obj.get("category"),
            "span_id": s["span_id"], "engine": s["engine"], "query": s["query"],
            "intent": s.get("intent"), "web_on": s.get("web_on"),
            "verdict": verdict, "attribution": v.get("attribution"),
            "quote": q, "quote_unverified": (not ok and how != "empty"),
            "quote_match": how,
            "delta_note": (v.get("delta_note") or "")[:220],
            "cited_types": s.get("cited_types"),
            "cited_domains": (s.get("cited_domains") or [])[:6],
            "retrieval_score": s.get("_retrieval_score"),
        })
    if cache is not None and key:
        cache[key] = clean
    return clean

# ------------------------------------------------------------------ 4. reverse scan
HALLUCINATION_PROMPT = """Below are excerpts from AI answers describing the brand "{brand}".

Extract EVERY factual assertion the assistant makes ABOUT {brand} — ingredients, levels,
certifications, efficacy, origin, ownership, price, format, availability, age suitability.

Then compare each assertion with the brand's REGISTERED CLAIMS and classify it:
- "registered"       : matches a registered claim
- "unsubstantiated"  : plausible, maybe true, but NOT in the register
                       (the brand should substantiate and publish it)
- "hallucinated"     : contradicts the register, or is a specific factual claim with no basis
- "competitor_bleed" : true of a DIFFERENT brand named in the excerpt, misattributed to {brand}
- "stale"            : describes a discontinued or superseded formulation

RULES
- "quote" MUST be verbatim from the excerpt.
- Only assertions about {brand}. Ignore statements about other brands.
- risk: "high" if a parent/carer could make a feeding or safety decision on it.

REGISTERED CLAIMS:
{claims}

Return STRICT JSON:
[{{"span_id":"","assertion":"","quote":"","classification":"registered|unsubstantiated|
hallucinated|competitor_bleed|stale","matched_claim":"","risk":"high|medium|low",
"why":"one line"}}]

EXCERPTS:
{excerpts}"""

HALLUC_CLASSES = {"registered", "unsubstantiated", "hallucinated",
                  "competitor_bleed", "stale"}

def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def hallucination_scan(gateway, brand, spans, claims, model, module="benchmark",
                       batch=6, cache=None, progress_cb=None):
    findings = []
    claims_txt = json.dumps([c.get("claim") for c in claims])[:6000]
    batches = list(_chunks(spans, batch))
    for bi, grp in enumerate(batches):
        key = None
        if cache is not None:
            key = hashlib.sha256(("H|" + model + "|v2|" +
                                  ",".join(sorted(s["span_id"] for s in grp))
                                  ).encode()).hexdigest()[:24]
            if key in cache:
                findings += cache[key]
                continue
        try:
            res = gateway.generate_json(model, HALLUCINATION_PROMPT.format(
                brand=brand, claims=claims_txt, excerpts=_excerpt_block(grp)),
                                        expect="list", module=module, task="hallucination_scan",
                                        temperature=0.0, max_tokens=4000,
                                        coerce_keys=("assertions", "findings", "items"))
        except Exception as e:
            _JUDGE_FAILURES.append({"claim": f"halluc_batch_{bi}",
                                    "reason": f"exception: {type(e).__name__}: {str(e)[:100]}",
                                    "attempts": "n/a", "n_spans": len(spans)})


        if not res["ok"]:
            _JUDGE_FAILURES.append({"claim": f"halluc_batch_{bi}",
                                    "reason": res["failure"],
                                    "attempts": res["attempts"], "n_spans": len(grp)})
        by_id = {s["span_id"]: s for s in grp}
        got = []
        for f in (res["data"] or []):
            if not isinstance(f, dict):
                continue
            s = by_id.get(f.get("span_id"))
            cls = str(f.get("classification", "")).lower().strip()
            if not s or cls not in HALLUC_CLASSES:
                continue
            assertion = str(f.get("assertion") or "").strip()
            if re.fullmatch(r"\d+_\d+(_\d+)?", assertion):
                assertion = (f.get("quote") or "").strip()[:300]  # recover from quote
            if not assertion:
                continue
            q = (f.get("quote") or "").strip()
            ok, how = verify_quote(q, s["span"])
            unver = (not ok and how != "empty")
            got.append({"span_id": s["span_id"], "engine": s["engine"],
                        "query": s["query"], "intent": s.get("intent"),
                        "assertion": (f.get("assertion") or "")[:300],
                        "quote": "" if unver else q, "quote_unverified": unver,
                        "quote_match": how,
                        "classification": cls,
                        "matched_claim": f.get("matched_claim"),
                        "risk": str(f.get("risk", "low")).lower(),
                        "why": (f.get("why") or "")[:200],
                        "cited_types": s.get("cited_types")})
        if cache is not None and key:
            cache[key] = got
        findings += got
        if progress_cb:
            progress_cb((bi + 1) / max(len(batches), 1))
    return findings

# ------------------------------------------------------------------ 5. aggregate
FRAGILE_HI = 0.80          # >= locked_in
FRAGILE_LO = 0.0           # >  fragile
MIN_N_FOR_FRAGILE = 8      # below this, coverage is too quantised to call "fragile"

def _status(cov, sup, par, dis, n):
    """n=4 gives coverage in {0,.25,.5,.75,1} — do not over-read mid-band values."""
    if dis > 0 and dis >= sup:
        return "distorted", "ok"
    if cov >= FRAGILE_HI:
        return "locked_in", "ok"
    if cov == 0.0:
        return ("partial_only", "ok") if par > 0 else ("absent", "ok")
    if n < MIN_N_FOR_FRAGILE:
        # 1 of 4 is weak; 3 of 4 is not "fragile", it is under-sampled
        return ("fragile", "low_n") if cov <= 0.34 else ("mostly_supported", "low_n")
    return "fragile", "ok"

def aggregate(judgements, claims, span_df):
    """Per claim × engine coverage with Wilson CIs, plus statuses that imply DIFFERENT
    remediations: absent (net-new content) vs fragile (reinforce) vs distorted (correct)."""
    rows = []
    if judgements:
        j = pd.DataFrame(judgements)
        # 'omitted' means the span was retrieved but silent → part of the denominator
        for (claim, eng), g in j.groupby(["claim", "engine"], dropna=False):
            n = len(g)
            sup = int((g.verdict == "supported").sum())
            par = int((g.verdict == "partial").sum())
            dis = int(g.verdict.isin(["distorted", "contradicted"]).sum())
            omi = int((g.verdict == "omitted").sum())
            lo, hi = proportion_confint(sup, n, alpha=0.05, method="wilson")
            cov = sup / n
            status, ev_strength = _status(cov, sup, par, dis, n)
            hedge = int((g.attribution == "hedged").sum())
            rows.append({
                "claim": claim, "claim_id": g["claim_id"].iloc[0],
                "claim_category": g["claim_category"].iloc[0],
                "engine": eng, "n_relevant_spans": n,
                "supported": sup, "partial": par, "distorted": dis, "omitted": omi,
                "hedged": hedge,
                "coverage": round(cov, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                "status": status,
                "delta_notes": [x for x in g["delta_note"].tolist() if x][:3],
                "example_quote": next((q for q in g["quote"].tolist() if q), ""),
                "evidence_strength": ev_strength,
            })
    per = pd.DataFrame(rows)

    judged = set(per["claim"]) if not per.empty else set()
    absent = [{"claim": c.get("claim"), "claim_id": c.get("claim_id"),
               "claim_category": c.get("category"), "engine": "*",
               "n_relevant_spans": 0, "supported": 0, "partial": 0, "distorted": 0,
               "omitted": 0, "hedged": 0, "coverage": 0.0, "ci_low": 0.0, "ci_high": 0.0,
               "status": "absent", "delta_notes": [], "example_quote": ""}
              for c in claims if c.get("claim") not in judged]
    if absent:
        per = pd.concat([per, pd.DataFrame(absent)], ignore_index=True)
    return per

def engine_scores(per_claim, halluc):
    """veracity_score = coverage − distortion − hallucination, per engine."""
    out = {}
    if per_claim is None or per_claim.empty:
        return out
    h = pd.DataFrame(halluc) if halluc else pd.DataFrame()
    for eng, g in per_claim[per_claim.engine != "*"].groupby("engine"):
        n = len(g)
        cov = g.coverage.mean()
        dist = (g.distorted.sum() / max(g.n_relevant_spans.sum(), 1))
        hr = 0.0
        if not h.empty and "engine" in h:
            he = h[h.engine == eng]
            hr = (0.0 if he.empty else
                  (he.classification == "hallucinated").sum() / max(len(he), 1))
        out[eng] = {"claims_evaluated": int(n),
                    "mean_coverage": round(float(cov), 3),
                    "distortion_rate": round(float(dist), 3),
                    "hallucination_rate": round(float(hr), 3),
                    "veracity_score": round(float(cov) - float(dist) - float(hr), 3),
                    "locked_in": int((g.status == "locked_in").sum()),
                    "fragile": int((g.status == "fragile").sum()),
                    "absent": int((g.status == "absent").sum()),
                    "distorted": int((g.status == "distorted").sum())}
    return out

# ------------------------------------------------------------------ orchestrator
CLAIM_PRIORITY = {"efficacy": 0, "ingredient": 1, "credential": 2, "safety": 3,
                  "feature": 4, "other": 5}

def estimate_cost(n_claims, n_spans, spans_per_engine, n_engines, halluc_sample,
                  halluc_batch=6, per_call=0.004):
    judge_calls = min(n_claims, 40)
    hall_calls = -(-min(halluc_sample, n_spans) // halluc_batch)
    total = judge_calls + hall_calls
    return total, round(total * per_call, 2)

def run_veracity_v2(gateway, df, claims, target_brand, *, judge_model,
                    competitors=(), alias_overrides=None, max_claims=40,
                    spans_per_engine=4, halluc_sample=48, halluc_batch=6,
                    cache=None, progress_cb=None):
    span_df = build_span_corpus(df, target_brand, other_brands=competitors,
                                alias_overrides=alias_overrides)
    if span_df.empty:
        return {"error": "no brand spans — the brand is never mentioned in this corpus",
                "span_count": 0, "per_claim": [], "judgements": [],
                "hallucinations": [], "engine_scores": {}}

    ranked = sorted([c for c in claims if c.get("claim")],
                    key=lambda c: (CLAIM_PRIORITY.get(c.get("category", "other"), 9),
                                   -len(str(c.get("proof_in_source") or ""))))[:max_claims]

    judgements, retrieved_any = [], 0
    for i, c in enumerate(ranked):
        reset_failures()
        drop = tuple([target_brand] + list(
            __import__("metrics_spec", fromlist=["x"]).build_registry(
                target_brand, [], alias_overrides).target.aliases))

        try:
            spans = retrieve_spans_for_claim(c.get("claim", ""), span_df,
                                             per_engine=spans_per_engine, drop=drop)
            if spans:
                retrieved_any += 1
                judgements += judge_claim(gateway, target_brand, c, spans, judge_model,
                                          cache=cache)
        except Exception as e:
            _JUDGE_FAILURES.append({"claim": (c.get("claim") or "")[:120],
                                    "reason": f"stage error: {type(e).__name__}",
                                    "n_spans": 0})
        if progress_cb:
            progress_cb(0.05 + 0.65 * (i + 1) / max(len(ranked), 1))

    # stratified span sample for the reverse scan (engine × intent)
    sample = _stratified_sample(span_df, ["engine", "intent"], 4, halluc_sample)


    halluc = hallucination_scan(gateway, target_brand, sample, ranked, judge_model,
                                batch=halluc_batch, cache=cache,
                                progress_cb=lambda x: progress_cb(0.70 + 0.28 * x)
                                if progress_cb else None)

    per = aggregate(judgements, ranked, span_df)
    hc = Counter(h["classification"] for h in halluc)
    n_h = max(len(halluc), 1)
    if progress_cb:
        progress_cb(1.0)

    return {
        "span_count": int(len(span_df)),
        "spans_by_engine": span_df.groupby("engine").size().to_dict(),
        "spans_by_intent": span_df.groupby("intent", dropna=False).size().to_dict(),
        "claims_considered": len(ranked),
        "claims_with_evidence": retrieved_any,
        "claims_absent": len(ranked) - retrieved_any,
        "per_claim": per.to_dict("records"),
        "judgements": judgements,
        "hallucinations": halluc,
        "halluc_counts": dict(hc),
        "hallucination_rate": round(hc.get("hallucinated", 0) / n_h, 3),
        "unsubstantiated_rate": round(hc.get("unsubstantiated", 0) / n_h, 3),
        "competitor_bleed_rate": round(hc.get("competitor_bleed", 0) / n_h, 3),
        "high_risk_assertions": [h for h in halluc
                                 if h.get("risk") == "high"
                                 and h.get("classification") in
                                 ("hallucinated", "competitor_bleed", "stale")][:20],
        "engine_scores": engine_scores(per, halluc),
        "quote_verification_failures": sum(1 for j in judgements
                                           if j.get("quote_unverified")),
        "quote_match_breakdown": dict(Counter(j.get("quote_match")
                                              for j in judgements)),
        "method": "veracity_v2",
        "judge_failures": list(_JUDGE_FAILURES),
        "judge_failure_count": len(_JUDGE_FAILURES),
        "claims_judged": len({j["claim"] for j in judgements}),
        "judge_yield": round(len({j["claim"] for j in judgements}) /
                             max(retrieved_any, 1), 3),
        "rescued_retrievals": sum(1 for j in judgements
                                  if j.get("retrieval_score", 1) < 0.5),
    }

# ------------------------------------------------------------------ keyword calibration
def veracity_keyword(df, claims, target_brand, span_df=None, competitors=()):
    """Kept as a CHEAP CALIBRATION CHECK, not a silent fallback.
    Now scoped to brand spans with word-boundary + numeric matching (was: any 3 long
    words anywhere in the whole concatenated corpus — which over-reported wildly)."""
    if span_df is None:
        span_df = build_span_corpus(df, target_brand, other_brands=competitors)
    corpus = [fold(s) for s in span_df["span"].tolist()] if not span_df.empty else []
    out = []
    for c in claims:
        claim = c.get("claim", "")
        nums, words = claim_terms(claim)
        key = [w for w in words if len(w) > 4][:4]
        hits = 0
        for low in corpus:
            w_ok = key and sum(1 for w in key if re.search(r"\b" + re.escape(w) + r"\b", low)) >= max(2, len(key) - 1)
            n_ok = (not nums) or any(n in low for n in nums)
            if w_ok and n_ok:
                hits += 1
        out.append({"claim": claim, "claim_id": c.get("claim_id"),
                    "verdict": "correct" if hits else "missing",
                    "spans_matched": hits, "method": "keyword_span_scoped"})
    return out

def calibration_report(llm_per_claim, keyword_rows):
    """Agreement between the cheap and expensive methods. Disagreement = investigate."""
    if not llm_per_claim or not keyword_rows:
        return {}
    llm = {}
    for r in llm_per_claim:
        if r["engine"] == "*":
            llm.setdefault(r["claim"], r["status"])
        else:
            prev = llm.get(r["claim"])
            if prev in (None, "absent"):
                llm[r["claim"]] = r["status"]
    kw = {r["claim"]: r["verdict"] for r in keyword_rows}
    agree = dis = 0
    examples = []
    for c, kv in kw.items():
        lv = llm.get(c)
        if lv is None:
            continue
        llm_pos = lv in ("locked_in", "mostly_supported", "fragile",
                         "partial_only", "distorted")
        if (kv == "correct") == llm_pos:
            agree += 1
        else:
            dis += 1
            if len(examples) < 8:
                examples.append({"claim": c[:120], "keyword": kv, "llm": lv})
    tot = agree + dis
    return {"n_compared": tot,
            "agreement": round(agree / tot, 3) if tot else None,
            "disagreements": dis, "examples": examples}