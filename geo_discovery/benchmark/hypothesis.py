"""LLM hypothesis generation with evidence packets, validators and a power check.

Replaces the rule-stub hypotheses. Each cluster becomes ONE falsifiable hypothesis mapped
to an experiment template, with a prediction the planned N can actually detect.
"""
from __future__ import annotations

import json
import math
import re

from metrics_spec import fold



# exp_stats computes ONLY these. Anything else silently mis-analyses.
SUPPORTED_METRICS = {"mention_rate", "first_rec_rate"}
METRIC_ALIASES = {
    "claim_mention_rate": "mention_rate", "brand_mention_rate": "mention_rate",
    "mention": "mention_rate", "first_mention_rate": "first_rec_rate",
    "first_rate": "first_rec_rate", "first_recommendation_rate": "first_rec_rate",
}
_NA = re.compile(r"^\s*(n/?a|none|not applicable|no experiment|—|-)\s*$", re.I)

def _degenerate(exp, pred):
    """Detect designs that cannot produce a result. These must never reach the runner."""
    reasons = []
    a, b = str(exp.get("arm_A") or ""), str(exp.get("arm_B") or "")
    iv = str(exp.get("isolated_variable") or "")
    if _NA.match(a) or _NA.match(b) or _NA.match(iv):
        reasons.append("arm or isolated_variable is N/A")
    if not a.strip() or not b.strip():
        reasons.append("empty arm description")
    if fold(a) == fold(b):
        reasons.append("arm_A and arm_B are identical")
    else:
        ta = set(re.findall(r"[a-z]{4,}", fold(a)))
        tb = set(re.findall(r"[a-z]{4,}", fold(b)))
        if ta and tb and len(ta & tb) / len(ta | tb) > 0.92:
            reasons.append("arms are near-identical (>92% token overlap)")
    if re.search(r"identical to (the )?(arm a|control|a\b)", fold(b)) and \
            not re.search(r"\b(but|except|plus|with the addition|additionally)\b", fold(b)):
        reasons.append("arm_B declared identical to control with no stated difference")
    exp_pp = pred.get("expected_pp")
    base_pp = pred.get("baseline_pp")
    if exp_pp is not None and base_pp is not None:
        try:
            if abs(float(exp_pp) - float(base_pp)) < 0.5:
                reasons.append("prediction is no change — nothing to detect")
        except Exception:
            pass
    return reasons

# ---------------------------------------------------------------- power
def mde(p0, n, alpha=0.05, power=0.80):
    """Minimum detectable effect (pp, absolute) for a two-proportion test."""
    z_a, z_b = 1.959964, 0.841621
    p0 = min(max(float(p0 or 0.0), 0.01), 0.99)
    return (z_a + z_b) * math.sqrt(2 * p0 * (1 - p0) / max(int(n or 1), 1))

def n_for_effect(p0, effect_pp, alpha=0.05, power=0.80):
    z_a, z_b = 1.959964, 0.841621
    p0 = min(max(float(p0 or 0.0), 0.01), 0.99)
    e = max(float(effect_pp or 1) / 100.0, 0.005)
    return int(math.ceil(2 * p0 * (1 - p0) * ((z_a + z_b) / e) ** 2))

TEMPLATES = {
    "E0": {"name": "GEO Treatment (bundled)",
           "isolates": "the full GEO package vs plain prose",
           "arms": ["A plain prose", "B fully enriched", "N negative control (optional)"],
           "kind": "ab"},
    "E1": {"name": "Structure Isolation",
           "isolates": "which structural element drives the lift",
           "arms": ["A prose", "B +bullets", "C +citations", "D +table", "E +capsule"],
           "kind": "multi"},
    "E2": {"name": "Specificity", "isolates": "quantified vs vague claims",
           "arms": ["A vague", "B quantified"], "kind": "ab"},
    "E3": {"name": "Corroboration", "isolates": "brand-only vs third-party echo",
           "arms": ["A brand source", "B brand + editorial source"], "kind": "ab"},
    "E4": {"name": "Gap-Closure", "isolates": "adding the specific missing claim",
           "arms": ["A without claim", "B with claim"], "kind": "ab"},
    "E5": {"name": "Format", "isolates": "table vs prose vs bullets, same facts",
           "arms": ["A prose", "B table", "C bullets"], "kind": "multi"},
    "E7": {"name": "Visual Extraction", "isolates": "can vision read the pack",
           "arms": ["A conceptual pack", "B claims on-pack"], "kind": "ab"},
    "E8": {"name": "Visual+Text Align", "isolates": "aligned image + alt-text + claim",
           "arms": ["A mismatched", "B aligned"], "kind": "ab"},
    "E9": {"name": "Objection Rebuttal", "isolates": "rebuttal against negative framing",
           "arms": ["A no rebuttal", "B with rebuttal"], "kind": "ab"},
}
FAMILY_TEMPLATES = {
    "visibility": ["E0", "E1", "E4"],      # low baseline → bundled test first
    "reason":     ["E0", "E2", "E1"],
    "provenance": ["E3", "E0"],
    "claim":      ["E4", "E2", "E3"],
    "correction": ["E4", "E3"],
    "substantiation": ["E4", "E3"],
    "placement":  ["E3"],
    "objection":  ["E9", "E3"],
    "engine":     ["E1", "E5"],
    "freshness":  ["E4"],
}
MECHANISMS = {"retrieval_absence", "memory_deficit", "corroboration_deficit",
              "extraction_failure", "competitive_dominance", "substantiation_deficit",
              "format_illegibility"}
PILLARS = {"Content", "Technical/Structured-Data", "Visual/Multimodal",
           "Corroboration", "Social/Community", "Claims-Substantiation"}

BRAND_FREE_INTENTS = ("need", "attribute", "persona", "longtail", "negative", "freshness")

# ---------------------------------------------------------------- packet
def build_packet(cluster, *, brand, category, claims, metrics, delta, hitlist,
                 winnability, query_master, n_planned):
    mbi = {r["intent"]: r["mention_rate"]
           for r in (metrics or {}).get("mention_by_intent", [])}
    fbi = {r["intent"]: r.get("first_rate", 0)
           for r in (metrics or {}).get("first_mention_by_intent", [])}
    intent = (cluster.get("intents") or ["need"])[0]
    baseline = cluster.get("baseline")
    if baseline is None:
        baseline = mbi.get(intent, mbi.get("need", 0.05))
    if baseline > 1:
        baseline = None                                    # not a rate — do not use

    rr = (delta or {}).get("reason_richness") or []
    ours = next((r for r in rr if fold(r.get("brand", "")) == fold(brand)), {})
    leaders = sorted([r for r in rr if fold(r.get("brand", "")) != fold(brand)],
                     key=lambda r: -(r.get("richness_per_100t") or 0))[:2]

    rel = _relevant_claims(cluster, claims)
    bf = [q["query"] for q in (query_master or [])
          if q.get("intent") in BRAND_FREE_INTENTS
          and fold(brand) not in fold(q.get("query", ""))]
    tq = cluster.get("queries") or []
    tq = [q for q in tq if fold(brand) not in fold(q)] or bf[:6]

    hl = (hitlist.to_dict("records") if hasattr(hitlist, "to_dict") else (hitlist or []))
    win = None
    if winnability is not None and len(winnability):
        try:
            win = winnability["winnability"].value_counts().to_dict()
        except Exception:
            win = None

    p0 = baseline if baseline is not None else mbi.get(intent, 0.05)
    return {
        "brand": brand, "category": category or "this category",
        "cluster": json.dumps({k: cluster.get(k) for k in
                               ("cluster_id", "title", "family", "pillar", "severity",
                                "types", "n_gaps", "topic", "dimensions", "intents",
                                "engines", "baseline")}, indent=1),
        "member_gaps": json.dumps(cluster.get("member_gaps", [])[:10], indent=1)[:5000],
        "quotes": json.dumps(cluster.get("evidence_quotes", []), indent=1)[:3000],
        "our_position": json.dumps({
            "mention_rate_by_intent": mbi, "first_rate_by_intent": fbi,
            "our_richness": {k: ours.get(k) for k in
                             ("richness_per_100t", "reason_types", "clinical",
                              "certification", "dosed_actives", "quantified")},
            "framing": (delta or {}).get("framing", {}),
            "flatness": (delta or {}).get("flatness", {})}, indent=1)[:2500],
        "leaders": json.dumps(leaders, indent=1)[:1800],
        "reason_gaps": json.dumps((delta or {}).get("reason_gap", [])[:6], indent=1)[:1500],
        "provenance": json.dumps((delta or {}).get("provenance", {}), indent=1)[:1200],
        "claims": json.dumps([c.get("claim") for c in rel][:22], indent=1)[:4000],
        "target_queries": json.dumps(tq[:8]),
        "brand_free_queries": json.dumps(bf[:25])[:2000],
        "sources": json.dumps([{k: h.get(k) for k in
                                ("domain", "source_type", "times_cited", "bucket",
                                 "brand_present_rate", "access_route")}
                               for h in hl[:12]], indent=1)[:2000],
        "winnability": json.dumps(win or {}),
        "templates": json.dumps({k: {"name": v["name"], "isolates": v["isolates"],
                                     "arms": v["arms"]}
                                 for k, v in TEMPLATES.items()
                                 if k in FAMILY_TEMPLATES.get(cluster["family"],
                                                              list(TEMPLATES))}, indent=1),
        "n_planned": n_planned,
        "baseline_pp": round((p0 or 0) * 100, 1),
        "mde_pp": round(mde(p0, n_planned) * 100, 1),
        "_p0": p0, "_relevant_claims": rel, "_brand_free": bf, "_target_queries": tq,
    }

def _relevant_claims(cluster, claims):
    """Claims lexically related to the cluster topic; falls back to substantiated ones."""
    toks = set()
    for s in (cluster.get("claims") or []) + [cluster.get("topic", ""),
                                              cluster.get("title", "")]:
        toks |= {w for w in re.findall(r"[a-z][a-z\-]{3,}", fold(s))}
    scored = []
    for c in (claims or []):
        ct = {w for w in re.findall(r"[a-z][a-z\-]{3,}", fold(c.get("claim", "")))}
        if not ct:
            continue
        scored.append((len(toks & ct) / len(ct), c))
    scored.sort(key=lambda x: -x[0])
    top = [c for s, c in scored if s > 0.1][:22]
    return top or [c for c in (claims or [])
                   if c.get("proof_type") in ("regulatory", "clinical", "third_party")][:22]

# ---------------------------------------------------------------- prompt
HYPOTHESIS_PROMPT = """You are a GEO (Generative Engine Optimization) research lead for
"{brand}" in the {category} category. You design controlled experiments that PROVE whether
a content change causes AI assistants to describe or recommend the brand differently.

Produce ONE rigorous, falsifiable hypothesis for the diagnosed gap cluster below.

HARD CONSTRAINTS
- You may ONLY propose publishing content built from the SUBSTANTIATED CLAIMS listed.
  Never invent a fact. If no available claim can close the gap, set mechanism to
  "substantiation_deficit" and say so in root_cause.
- The experiment must ISOLATE ONE variable. Name what is held constant.
- target_query MUST be copied exactly from TARGET QUERIES or BRAND-FREE QUERIES. It must
  NOT contain the brand name (mention rate is meaningless if the prompt names the brand).
- The experiment runs with N={n_planned} runs per arm at a baseline of {baseline_pp}%,
  which can only detect effects of about {mde_pp}pp or larger. Do not predict an effect
  smaller than that.
- Experiments inject documents into context with web OFF. So a hypothesis about EARNED
  third-party placement must be tested as a PROXY: does the presence of a third-party
  citation inside the brand document change the outcome.
- If no substantiated claim can close the gap, DO NOT invent a null experiment. Instead set
    mechanism="substantiation_deficit", pillar="Claims-Substantiation", and write
    experiment.template="E4" with arm_A/arm_B describing the test that WOULD be run once the
    claim is substantiated. Never write "N/A" and never make arm_B identical to arm_A.
- prediction.metric MUST be exactly "mention_rate" or "first_rec_rate". No other metric
  can be measured.
- Never predict zero change. If you believe the change will not work, predict the lift you
  would need to see and state in falsification what a null result would prove.
- Template choice: if the brand's unprompted mention rate for this query type is below
  15%, prefer E0 (plain prose vs fully enriched). At a low baseline the bundled treatment
  is the only one with enough effect size to detect, and it answers the prior question
  ("does enrichment work here at all?") before attribution. Choose E2/E3/E4 only when the
  baseline is already healthy and one specific lever needs isolating.

GAP CLUSTER
{cluster}

MEMBER GAPS (the specific diagnosed problems)
{member_gaps}

VERBATIM EVIDENCE from real audit runs
{quotes}

OUR MEASURED POSITION
{our_position}

WHO BEATS US, and on what
{leaders}

REASON-DENSITY DEFICITS (per 100 tokens)
{reason_gaps}

CITATION PROVENANCE (who the engines actually listen to)
{provenance}

SUBSTANTIATED CLAIMS AVAILABLE
{claims}

TARGET QUERIES (prefer one of these)
{target_queries}

OTHER BRAND-FREE QUERIES
{brand_free_queries}

SOURCE ECOSYSTEM
{sources}

WINNABILITY MIX
{winnability}

EXPERIMENT TEMPLATES YOU MAY CHOOSE FROM
{templates}

Return STRICT JSON, one object:
{{
 "root_cause": "why the engines behave this way — mechanistic and specific to this brand",
 "mechanism": "retrieval_absence|memory_deficit|corroboration_deficit|extraction_failure|competitive_dominance|substantiation_deficit|format_illegibility",
 "hypothesis": "If we <specific change>, then <specific metric> will <direction> because <mechanism>",
 "prediction": {{"metric":"mention_rate|first_rec_rate","baseline_pp":0,"expected_pp":0,
                 "min_detectable_pp":0}},
 "experiment": {{"template":"E2","isolated_variable":"",
                 "arm_A":"describe the control content",
                 "arm_B":"describe the treatment content, naming the exact claims used",
                 "held_constant":["length","structure","citation count"],
                 "confounds_to_control":[],
                 "target_query":"copied verbatim from the lists above"}},
 "claims_used":["exact claim text, copied verbatim from SUBSTANTIATED CLAIMS"],
 "falsification":"what result would DISPROVE this hypothesis",
 "pillar":"Content|Technical/Structured-Data|Visual/Multimodal|Corroboration|Social/Community|Claims-Substantiation",
 "horizon":"retrieval|memory",
 "effort":"low|medium|high",
 "confidence":0.0,
 "why_this_matters":"one line a CMO would understand"
}}"""

def generate_hypotheses(gateway, clusters, *, model, brand, category, claims, metrics,
                        delta, hitlist, winnability, query_master, n_planned=100,
                        module="benchmark", progress_cb=None):
    out = []
    for i, cl in enumerate(clusters):
        pk = build_packet(cl, brand=brand, category=category, claims=claims,
                          metrics=metrics, delta=delta, hitlist=hitlist,
                          winnability=winnability, query_master=query_master,
                          n_planned=n_planned)
        prompt_args = {k: v for k, v in pk.items() if not k.startswith("_")}
        res = gateway.generate_json(model, HYPOTHESIS_PROMPT.format(**prompt_args),
                                    expect="dict", module=module, task="hypothesis_gen",
                                    temperature=0.3, max_tokens=2500)
        h = res.get("data") or {}
        h.update({"cluster_id": cl["cluster_id"], "cluster_title": cl["title"],
                  "family": cl["family"], "gap_ids": cl["member_ids"],
                  "severity": cl["severity"], "n_gaps": cl["n_gaps"],
                  "gap_types": cl["types"], "engines": cl["engines"],
                  "_packet_baseline": pk["_p0"], "_n_planned": n_planned,
                  "_relevant_claims": [c.get("claim") for c in pk["_relevant_claims"]],
                  "_brand_free": pk["_brand_free"], "_target_queries": pk["_target_queries"],
                  "generated_ok": bool(res.get("ok")),
                  "generation_failure": res.get("failure")})
        out.append(h)
        if progress_cb:
            progress_cb((i + 1) / max(len(clusters), 1))
    return out

# ---------------------------------------------------------------- validation
def validate(hyps, *, claims, brand, n_planned=100):
    reg = {fold(c.get("claim", "")) for c in (claims or [])}
    for h in hyps:
        errs, warns = [], []
        if not h.get("generated_ok"):
            errs.append(f"generation failed: {h.get('generation_failure')}")

        used = [c for c in (h.get("claims_used") or []) if str(c).strip()]
        unknown = [c for c in used if fold(c) not in reg]
        if unknown and h.get("mechanism") != "substantiation_deficit":
            # tolerate light paraphrase, reject invention
            hard = [c for c in unknown
                    if not any(_overlap(fold(c), r) > 0.6 for r in reg)]
            if hard:
                errs.append(f"claims not in register: {[str(c)[:60] for c in hard][:2]}")
            else:
                warns.append("claims paraphrased from the register — verify wording")
        if not used and h.get("mechanism") != "substantiation_deficit":
            warns.append("no claims cited — variant generation will have no material")

        exp = h.get("experiment") or {}
        tpl = exp.get("template")
        if tpl not in TEMPLATES:
            errs.append(f"invalid experiment template: {tpl}")
        # ---- metric must be one exp_stats can actually compute ----
        m = str(pred_metric := (h.get("prediction") or {}).get("metric") or "").strip()
        canon = METRIC_ALIASES.get(fold(m), fold(m))
        if canon in SUPPORTED_METRICS:
            if canon != m:
                h["prediction"]["metric"] = canon
                warns.append(f"metric '{m}' normalised to '{canon}'")
        else:
            errs.append(f"metric '{m}' is not measurable — must be one of "
                        f"{sorted(SUPPORTED_METRICS)}")

        # ---- degenerate design ----
        deg = _degenerate(exp, h.get("prediction") or {})
        if deg:
            errs += [f"degenerate design: {r}" for r in deg]
            h["runnable"] = False
        else:
            h["runnable"] = True
        if not exp.get("isolated_variable"):
            errs.append("no isolated variable named")
        if not h.get("falsification"):
            errs.append("no falsification criterion")

        tq = (exp.get("target_query") or "").strip()
        if not tq:
            errs.append("no target_query")
        else:
            if fold(brand) in fold(tq):
                errs.append("target_query names the brand — mention rate meaningless")
            allowed = {fold(q) for q in (h.get("_brand_free") or []) +
                       (h.get("_target_queries") or [])}
            if allowed and fold(tq) not in allowed:
                warns.append("target_query not copied verbatim from the audit set — "
                             "no baseline exists for it")

        # ---- power: WARN only. The audit baseline is not the experimental baseline;
        # ---- the pilot in the Experiment tab is the real gate.
        pred = h.get("prediction") or {}
        p0 = h.get("_packet_baseline") or 0.05
        need = mde(p0, n_planned) * 100
        claimed = float(pred.get("min_detectable_pp") or 0)
        expected = float(pred.get("expected_pp") or 0) - float(pred.get("baseline_pp") or 0)
        h["mde_pp"] = round(need, 1)
        h["expected_lift_pp"] = round(expected, 1)
        h["n_required_for_prediction"] = (n_for_effect(p0, expected) if expected > 0
                                          else None)
        if expected <= 0:
            errs.append("prediction is no change — nothing to detect")
        elif expected < need:
            warns.append(f"predicted +{expected:.1f}pp is below the {need:.1f}pp "
                         f"detectable at N={n_planned} on the AUDIT baseline "
                         f"({p0:.1%}); needs N≈{h['n_required_for_prediction']}. "
                         f"Re-check after the pilot.")
        if claimed and claimed < need * 0.9:
            warns.append(f"claimed MDE {claimed}pp < {need:.1f}pp achievable")

        # ---- mechanism / pillar: auto-repair, never reject ----
        if h.get("mechanism") not in MECHANISMS:
            warns.append(f"unrecognised mechanism: {h.get('mechanism')}")
        if h.get("mechanism") == "substantiation_deficit" and \
                h.get("pillar") != "Claims-Substantiation":
            warns.append(f"pillar re-routed from {h.get('pillar')} to "
                         f"Claims-Substantiation")
            h["pillar_original"] = h.get("pillar")
            h["pillar"] = "Claims-Substantiation"
        if h.get("pillar") not in PILLARS:
            warns.append(f"unrecognised pillar: {h.get('pillar')} — defaulted to Content")
            h["pillar"] = "Content"

        h["validation_errors"] = errs
        h["validation_warnings"] = warns
        if h.get("mechanism") == "substantiation_deficit":
            h["status"] = "not_testable"
            h["route"] = "optimise_pillar6"
            h["not_testable_reason"] = (
                "The claim is not in the substantiated register, so it cannot be "
                "published or tested. Substantiate first, then test with E4.")
        elif errs or not h.get("runnable", True):
            h["status"] = "needs_review"
        else:
            h["status"] = "ready"
        for k in ("_brand_free", "_relevant_claims", "_target_queries"):
            h.pop(k, None)
    return hyps



def _overlap(a, b):
    ta = set(re.findall(r"[a-z][a-z\-]{3,}", a))
    tb = set(re.findall(r"[a-z][a-z\-]{3,}", b))
    return len(ta & tb) / max(len(ta | tb), 1)

def hypothesis_summary(hyps):
    from collections import Counter
    return {"total": len(hyps),
            "ready": sum(1 for h in hyps if h.get("status") == "ready"),
            "needs_review": sum(1 for h in hyps if h.get("status") != "ready"),
            "by_template": dict(Counter((h.get("experiment") or {}).get("template")
                                        for h in hyps)),
            "by_mechanism": dict(Counter(h.get("mechanism") for h in hyps)),
            "by_pillar": dict(Counter(h.get("pillar") for h in hyps)),
            "not_testable": sum(1 for h in hyps if h.get("status") == "not_testable"),
            "by_horizon": dict(Counter(h.get("horizon") for h in hyps))}