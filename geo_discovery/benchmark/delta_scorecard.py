"""Delta Scorecard — Ground Truth vs Machine Reality, with the v2 verdict vocabulary."""
import json
from datetime import datetime, timezone
# Gap detection lives in gap_detect.py (Batch 3 adds the LLM hypothesis layer on top).
# Re-exported here so `from benchmark.delta_scorecard import build_gap_register` works.
from benchmark.gap_detect import build_gap_register, detect_gaps, gap_summary  # noqa: F401

STATUS_TO_DELTA = {
    "locked_in": ("correct", "none"),
    "mostly_supported": ("correct", "none"),
    "fragile": ("fragile", "medium"),
    "partial_only": ("diluted", "medium"),
    "distorted": ("distortion", "critical"),
    "absent": ("missing", "high"),
    "no_evidence": ("no_evidence", "high"),      # W-15: retrieval found nothing
}
HALLUC_TO_DELTA = {
    "hallucinated": ("hallucination", "critical"),
    "competitor_bleed": ("misattribution", "critical"),
    "stale": ("stale", "high"),
    "unsubstantiated": ("unsubstantiated", "medium"),
}

def build_delta_scorecard(veracity, reason_rich, provenance, winnability,
                          leaderboard, target_brand, *, framing=None,
                          consistency=None, flatness=None, reason_gap=None,
                          metrics=None, qa=None):
    deltas = []
    seen = set()
    for pc in (veracity or {}).get("per_claim", []):
        key = (pc.get("claim"), pc.get("engine"))
        if key in seen:
            continue
        seen.add(key)
        dtype, sev = STATUS_TO_DELTA.get(pc.get("status"), ("missing", "high"))
        deltas.append({
            "claim": pc.get("claim"), "claim_id": pc.get("claim_id"),
            "claim_category": pc.get("claim_category"), "engine": pc.get("engine"),
            "machine_status": pc.get("status"),
            "coverage": pc.get("coverage"),
            "ci": [pc.get("ci_low"), pc.get("ci_high")],
            "n_spans": pc.get("n_relevant_spans"),
            "delta_type": dtype, "severity": sev,
            "evidence_strength": pc.get("evidence_strength"),   # ← W-17
            "delta_note": (pc.get("delta_notes") or [""])[0],
            "evidence_quote": pc.get("example_quote", "")[:280]})

    risks = []
    for h in (veracity or {}).get("hallucinations", []):
        if h.get("classification") == "registered":
            continue
        dtype, sev = HALLUC_TO_DELTA.get(h["classification"], ("unsubstantiated", "medium"))
        risks.append({"assertion": h.get("assertion"), "engine": h.get("engine"),
                      "query": h.get("query"), "delta_type": dtype,
                      "severity": "critical" if h.get("risk") == "high" else sev,
                      "risk": h.get("risk"), "why": h.get("why"),
                      "evidence_quote": (h.get("quote") or "")[:280]})

    sev_rank = {"critical": 0, "high": 1, "medium": 2, "none": 3}
    deltas.sort(key=lambda d: (sev_rank.get(d["severity"], 9), -(d.get("coverage") or 0)))
    risks.sort(key=lambda r: sev_rank.get(r["severity"], 9))

    return {
        "target_brand": target_brand,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": (veracity or {}).get("method", "unknown"),
        "evidence_base": {
            "runs": (metrics or {}).get("n_runs"),
            "engines": (metrics or {}).get("engines"),
            "brand_spans": (veracity or {}).get("span_count"),
            "spans_by_engine": (veracity or {}).get("spans_by_engine"),
            "claims_considered": (veracity or {}).get("claims_considered"),
            "claims_with_evidence": (veracity or {}).get("claims_with_evidence")},
        "headline": _headline(veracity, provenance, framing, flatness, metrics,
                              target_brand),
        "claim_deltas": deltas,
        "risk_register": risks,
        "engine_veracity": (veracity or {}).get("engine_scores", {}),
        "hallucination_summary": {
            "counts": (veracity or {}).get("halluc_counts", {}),
            "hallucination_rate": (veracity or {}).get("hallucination_rate"),
            "unsubstantiated_rate": (veracity or {}).get("unsubstantiated_rate"),
            "competitor_bleed_rate": (veracity or {}).get("competitor_bleed_rate"),
            "high_risk": (veracity or {}).get("high_risk_assertions", [])[:10]},
        "provenance": provenance,
        "framing": framing,
        "flatness": flatness,
        "reason_richness": _rec(reason_rich),
        "reason_gap": _rec(reason_gap),
        "consistency": consistency,
        "leaderboard": _rec(leaderboard),
        "winnability_summary": _winnability_summary(winnability),
        "calibration": (veracity or {}).get("calibration"),
        "qa": qa,
        "retrieval_delta": (metrics or {}).get("retrieval_delta", {}),
    }

def _rec(x):
    if x is None:
        return []
    return x.to_dict("records") if hasattr(x, "to_dict") else x

def _winnability_summary(wdf):
    if wdf is None or len(wdf) == 0:
        return {}
    return wdf["winnability"].value_counts().to_dict()

def _headline(veracity, provenance, framing, flatness, metrics, brand):
    """The 4-6 sentences a strategist reads first. Deterministic, evidence-bound."""
    out = []
    mbi = {r["intent"]: r["mention_rate"] for r in (metrics or {}).get("mention_by_intent", [])}
    vis = {k: v for k, v in mbi.items() if k not in ("branded", "comparative")}
    if vis:
        # lead with the most material category — sample size × commercial weight —
        # not simply the lowest rate
        runs = {r["intent"]: r.get("runs", 0)
                for r in (metrics or {}).get("mention_by_intent", [])}
        WEIGHT = {"need": 3.0, "persona": 2.0, "attribute": 2.0, "longtail": 1.5,
                  "negative": 1.0, "freshness": 1.0}
        lead = max(vis, key=lambda i: (1 - vis[i]) * runs.get(i, 1)
                                      * WEIGHT.get(i, 1.0))
        out.append(f"Unprompted visibility on '{lead}' queries is {vis[lead]:.1%} "
                   f"across {runs.get(lead, 0):,} runs — the most material gap. "
                   f"Weakest overall: '{min(vis, key=vis.get)}' at "
                   f"{min(vis.values()):.1%}.")
    if framing and framing.get("mentions"):
        out.append(f"Of {framing['mentions']} brand mentions, {framing.get('pos_share',0):.0%} "
                   f"are positively framed and {framing.get('neu_share',0):.0%} are flat; "
                   f"{(framing.get('advisory_share') or 0):.0%} defer to a health "
                   f"professional while still naming brands.")
    if flatness and flatness.get("spans"):
        line = (f"{flatness.get('shallow_share', 0):.0%} of brand mentions carry at most "
                f"ONE substantive reason (dose, study, certification, mechanism or cited "
                f"source); mean {flatness.get('mean_substantive_types')} per mention.")
        if flatness.get("richness_ratio_vs_leader") is not None:
            r = flatness["richness_ratio_vs_leader"]
            line += (f" Reason DENSITY is {r}× {flatness.get('leader')}'s per 100 tokens"
                     + (" — competitors argue harder in less space."
                        if r < 1 else " — density advantage."))
        out.append(line)
    if veracity and veracity.get("engine_scores"):
        worst_e = min(veracity["engine_scores"],
                      key=lambda e: veracity["engine_scores"][e]["veracity_score"])
        s = veracity["engine_scores"][worst_e]
        out.append(f"Lowest veracity on {worst_e}: mean claim coverage "
                   f"{s['mean_coverage']:.0%}, {s['absent']} claims absent, "
                   f"{s['distorted']} distorted.")
    if provenance:
        out.append(f"Citation control is weak: {provenance.get('self_referential_pct',0):.0%} "
                   f"brand-owned vs {provenance.get('grey_market_pct',0):.0%} grey-market "
                   f"resellers, {provenance.get('affiliate_pct',0):.0%} SEO/affiliate and "
                   f"{provenance.get('competitor_owned_pct',0):.0%} competitor-owned.")
    if veracity and veracity.get("halluc_counts"):
        hc = veracity["halluc_counts"]
        tot = sum(hc.values()) or 1
        out.append(
            f"Reverse scan of {tot} assertions the engines make about the brand: "
            f"{hc.get('registered', 0)} match the claims register, "
            f"{hc.get('unsubstantiated', 0)} are plausible but UNREGISTERED "
            f"({hc.get('unsubstantiated', 0) / tot:.0%} — a Pillar-6 backlog of facts the "
            f"engines already assert on the brand's behalf), "
            f"{hc.get('hallucinated', 0)} are unsupported and "
            f"{hc.get('competitor_bleed', 0)} are misattributed from a competitor.")
    return out

# ---------------------------------------------------------------- gap classifier (kept)

GAP_CLASSIFY_PROMPT = """You are a GEO (Generative Engine Optimization) strategist for the
brand "{brand}" in the "{category}" category.

Classify each diagnosed GAP by whether closing it would plausibly improve how AI assistants
DESCRIBE or RECOMMEND this brand.

TWO KINDS OF GAP — both can be actionable:
1. QUERY-LEVEL: tied to a specific user query or claim (visibility, claim absence,
   distortion, rank).
2. SYSTEMIC: not tied to one query, but shapes every answer — who the engines cite
   (provenance), whether the brand is argued for or merely listed (reason richness),
   whether uncontrolled third parties supply the evidence base.
   SYSTEMIC gaps are usually the HIGHEST-value work. Do not mark them noise merely
   because no single query asks about them.

CLASSIFICATIONS
- "actionable": closing it could realistically change how the brand is described,
  recommended, or evidenced. Default to this for critical/high severity.
- "monitor": real but low leverage, long-horizon, or dependent on something else first.
- "noise": query-irrelevant trivia — pack counts, packaging-refresh taglines, market
  slogans, founding dates. Nothing a recommendation answer would ever surface.

Be sparing with "noise". A gap backed by a measured metric is almost never noise.

Return STRICT JSON — a JSON ARRAY, one object per input gap_id, nothing else:
[{{"gap_id":"","classification":"actionable|monitor|noise","reason":"one line"}}]

QUERIES BEING AUDITED (what people actually ask):
{queries}

GAPS:
{gaps}"""

# Systemic types that shape every answer — never noise, regardless of LLM opinion.
_SYSTEMIC = {"GREY_MARKET_VOICE", "PROVENANCE_THIN", "FLAT_MENTION", "REASON_DEFICIT",
             "VISIBILITY_GAP", "RANK_GAP", "ADVISORY_DEFERRAL", "ENGINE_DIVERGENCE",
             "NEGATIVE_FRAMING", "SOURCE_ABSENCE"}
# Correctness/risk types — never noise.
_RISK = {"HALLUCINATION", "COMPETITOR_BLEED", "CLAIM_DISTORTED", "STALE_CLAIM"}

def _coerce_list(parsed):
    """Models wrap arrays inconsistently. Accept every common shape."""
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    if isinstance(parsed, dict):
        for k in ("gaps", "classifications", "results", "items", "data", "output",
                  "response"):
            v = parsed.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        if parsed and all(isinstance(v, dict) for v in parsed.values()):
            out = []
            for k, v in parsed.items():          # {gap_id: {...}} form
                d = dict(v)
                d.setdefault("gap_id", k)
                out.append(d)
            return out
    return []

def _rule_floor(gap):
    """Deterministic minimum, applied AFTER the LLM so it can't silently zero the set."""
    sev = gap.get("severity")
    gt = gap.get("type", "")
    if gt in _RISK or sev == "critical":
        return "actionable"
    if gt in _SYSTEMIC and sev in ("critical", "high"):
        return "actionable"
    if sev == "high":
        return "monitor"
    return None

_ORDER = {"actionable": 0, "monitor": 1, "noise": 2}

'''

def classify_gaps(gateway, gaps, query_master, target_brand,
                  model="gemini-2.5-flash", module="benchmark", category="",
                  batch_size=12):
    """Batched, shape-tolerant, rule-floored. Never silently yields zero actionable."""
    if not gaps:
        return gaps
    queries_txt = json.dumps([q["query"] for q in (query_master or [])][:30])[:3000]
    cls, errors = {}, []

    for i in range(0, len(gaps), batch_size):
        chunk = gaps[i:i + batch_size]
        payload = json.dumps([{
            "gap_id": g["gap_id"], "type": g.get("type"),
            "severity": g.get("severity"), "pillar": g.get("pillar"),
            "gap": g.get("gap"),
            "measured_baseline": g.get("metric_baseline"),
            "engines": g.get("engines"),
        } for g in chunk], indent=1)[:9000]
        try:
            r = gateway.generate(model, GAP_CLASSIFY_PROMPT.format(
                brand=target_brand, category=category or "this category",
                queries=queries_txt, gaps=payload),
                json_mode=True, module=module, task="gap_classify",
                temperature=0.1, max_tokens=2500)
            got = _coerce_list(r.get("parsed"))
            if not got:
                errors.append(f"batch {i//batch_size+1}: unparseable "
                              f"(finish={r.get('finish_reason')}, "
                              f"truncated={r.get('truncated')})")
            for c in got:
                gid = c.get("gap_id")
                v = str(c.get("classification", "")).lower().strip()
                if gid and v in _ORDER:
                    cls[gid] = {"classification": v,
                                "reason": (c.get("reason") or "")[:220]}
        except Exception as e:
            errors.append(f"batch {i//batch_size+1}: {str(e)[:120]}")

    for g in gaps:
        c = cls.get(g["gap_id"])
        floor = _rule_floor(g)
        if c:
            v = c["classification"]
            g["classification_source"] = "llm"
            # the LLM may not demote a critical/systemic gap to noise
            if floor == "actionable" and v == "noise":
                v = "actionable"
                g["classification_source"] = "llm_overridden_by_rule"
            elif floor == "monitor" and v == "noise":
                v = "monitor"
                g["classification_source"] = "llm_overridden_by_rule"
            g["classification"] = v
            g["classification_reason"] = c["reason"]
        else:
            g["classification"] = floor or "monitor"
            g["classification_source"] = "rule_fallback"
            g["classification_reason"] = (
                f"LLM did not classify — defaulted from severity="
                f"{g.get('severity')}, type={g.get('type')}")

    if errors:
        gaps[0].setdefault("_classify_errors", errors)

    return sorted(gaps, key=lambda g: (_ORDER.get(g["classification"], 1),
                                       -g.get("priority", 0)))
'''
def _gap_payload(chunk):
    """Only the fields that inform an actionability judgement."""
    return json.dumps([{
        "gap_id": g["gap_id"],
        "type": g.get("type"),
        "severity": g.get("severity"),
        "pillar": g.get("pillar"),
        "gap": (g.get("gap") or "")[:280],
        "measured_baseline": g.get("metric_baseline"),
        "intent": g.get("intent"),
        "query": g.get("query"),
        "domain": g.get("domain"),
        "engines": g.get("engines"),
    } for g in chunk], indent=1)[:9000]

def _call_classifier(gateway, model, module, brand, category, queries_txt, chunk):
    """One classification call. Returns (rows, error_or_None)."""
    prompt = GAP_CLASSIFY_PROMPT.format(
        brand=brand, category=category or "this category",
        queries=queries_txt, gaps=_gap_payload(chunk))

    if hasattr(gateway, "generate_json"):
        res = gateway.generate_json(
            model, prompt, expect="list", module=module, task="gap_classify",
            temperature=0.1, max_tokens=2500,
            coerce_keys=("gaps", "classifications", "results", "items", "data"))
        rows = res.get("data") or []
        if not res.get("ok"):
            return rows, (f"{res.get('failure')} after {res.get('attempts')} attempt(s), "
                          f"finish={res.get('finish_reason')}")
        return rows, None

    # ---- legacy gateway: no escalation available, so report it loudly ----
    try:
        r = gateway.generate(model, prompt, json_mode=True, module=module,
                             task="gap_classify", temperature=0.1, max_tokens=2500)
        rows = _coerce_list(r.get("parsed"))
        if not rows:
            return [], (f"unparseable/empty (finish={r.get('finish_reason')}, "
                        f"truncated={r.get('truncated')}) — gateway lacks "
                        f"generate_json(); apply the Batch-2.1 gateway patch")
        return rows, None
    except Exception as e:
        return [], f"exception: {str(e)[:160]}"
def classify_gaps(gateway, gaps, query_master, target_brand,
                  model="gemini-2.5-flash", module="benchmark", category="",
                  batch_size=8, retry_singles_cap=12):
    """Classify gaps into actionable / monitor / noise.

    Guarantees:
      1. Every gap gets a classification (LLM, or a recorded rule fallback).
      2. Critical / risk / systemic-high gaps are never dropped to noise.
      3. The actionable set is never empty (safety net promotes critical + systemic-high).
      4. Failures are recorded, never swallowed.

    Returns the gap list sorted actionable -> monitor -> noise, then by priority.
    Diagnostics land on gaps[0]['_classify_errors'] and gaps[0]['_classify_report'].
    """
    global _CLASSIFY_REPORT
    _CLASSIFY_REPORT = {}
    if not gaps:
        return gaps

    queries_txt = json.dumps([q["query"] for q in (query_master or [])][:30])[:3000]
    cls, errors, n_batches = {}, [], 0

    # ---------- pass 1: batched ----------
    for i in range(0, len(gaps), batch_size):
        chunk = gaps[i:i + batch_size]
        n_batches += 1
        rows, err = _call_classifier(gateway, model, module, target_brand, category,
                                     queries_txt, chunk)
        if err:
            errors.append(f"batch {n_batches} ({len(chunk)} gaps): {err}")
        for c in rows:
            gid = c.get("gap_id")
            v = str(c.get("classification", "")).lower().strip()
            if gid and v in _ORDER:
                cls[gid] = {"classification": v,
                            "reason": (c.get("reason") or "")[:220]}

    # ---------- pass 2: retry unclassified gaps one at a time ----------
    missing = [g for g in gaps if g["gap_id"] not in cls]
    retried = 0
    for g in missing:
        if retried >= retry_singles_cap:
            break
        retried += 1
        rows, err = _call_classifier(gateway, model, module, target_brand, category,
                                     queries_txt, [g])
        for c in rows:
            gid = c.get("gap_id") or g["gap_id"]
            v = str(c.get("classification", "")).lower().strip()
            if v in _ORDER:
                cls[gid] = {"classification": v,
                            "reason": (c.get("reason") or "")[:220], "_single": True}
    if retried:
        recovered = sum(1 for g in missing[:retried] if g["gap_id"] in cls)
        errors.append(f"single-gap retry: {recovered}/{retried} recovered")

    # ---------- assign, with rule floor ----------
    for g in gaps:
        c = cls.get(g["gap_id"])
        floor = _rule_floor(g)
        if c:
            v = c["classification"]
            src = "llm_single" if c.get("_single") else "llm"
            if floor == "actionable" and v == "noise":
                v, src = "actionable", "llm_overridden_by_rule"
            elif floor == "monitor" and v == "noise":
                v, src = "monitor", "llm_overridden_by_rule"
            g["classification"] = v
            g["classification_source"] = src
            g["classification_reason"] = c["reason"]
        else:
            g["classification"] = floor or "monitor"
            g["classification_source"] = "rule_fallback"
            g["classification_reason"] = (
                f"LLM did not classify — defaulted from severity="
                f"{g.get('severity')}, type={g.get('type')}")

    # ---------- safety net: never emit an empty actionable set ----------
    if not any(g["classification"] == "actionable" for g in gaps):
        promoted = 0
        for g in gaps:
            if g.get("severity") == "critical" or (
                    g.get("type") in (_SYSTEMIC | _RISK)
                    and g.get("severity") in ("critical", "high")):
                g["classification"] = "actionable"
                g["classification_source"] = "safety_net"
                g["classification_reason"] = (
                    "Promoted: zero actionable gaps after classification. "
                    f"severity={g.get('severity')}, type={g.get('type')}")
                promoted += 1
        errors.append(f"safety net promoted {promoted} gap(s) — the LLM left nothing "
                      f"actionable, which is almost always a call failure")

    out = sorted(gaps, key=lambda g: (_ORDER.get(g["classification"], 1),
                                      -g.get("priority", 0)))

    # ---------- diagnostics (attach AFTER sorting so the UI finds them on gaps[0]) ----------
    from collections import Counter as _Counter
    _CLASSIFY_REPORT = {
        "model": model, "n_gaps": len(out), "batch_size": batch_size,
        "batches": n_batches, "singles_retried": retried,
        "llm_classified": sum(1 for g in out if str(
            g.get("classification_source", "")).startswith("llm")),
        "rule_fallback": sum(1 for g in out
                             if g.get("classification_source") == "rule_fallback"),
        "counts": dict(_Counter(g["classification"] for g in out)),
        "sources": dict(_Counter(g.get("classification_source") for g in out)),
        "errors": errors,
    }
    if out:
        if errors:
            out[0]["_classify_errors"] = errors
        else:
            out[0].pop("_classify_errors", None)
        out[0]["_classify_report"] = _CLASSIFY_REPORT
    return out

def save_benchmark(parent_brand, target_brand, delta, gaps, hitlist, models_used):
    from store import save_artifact
    save_artifact(parent_brand, target_brand, "delta_scorecard", delta, models_used)
    save_artifact(parent_brand, target_brand, "gap_register", gaps, models_used)
    save_artifact(parent_brand, target_brand, "source_hitlist",
                  hitlist.to_dict("records") if hasattr(hitlist, "to_dict") else hitlist,
                  models_used)