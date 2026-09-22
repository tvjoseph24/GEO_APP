"""LLM hypothesis generation with evidence packets, validators and a power check."""
from __future__ import annotations

import json
import math
import re

from metrics_spec import fold

SUPPORTED_METRICS = {"mention_rate", "first_rec_rate"}
METRIC_ALIASES = {
    "claim_mention_rate": "mention_rate", "brand_mention_rate": "mention_rate",
    "mention": "mention_rate", "first_mention_rate": "first_rec_rate",
    "first_rate": "first_rec_rate", "first_recommendation_rate": "first_rec_rate",
    "claim_coverage": "mention_rate",
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
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def normalise_prediction(pred: dict, mde_pp: float, baseline_pp=None) -> tuple[dict, list]:
    """Repair the two systematic arithmetic failures, and record what was repaired.

    W-37/W-76, observed on 13 of 53 hypotheses:
      * expected_pp set to the MDE rather than baseline + MDE, so the prediction reads
        as "no change" and the design is rejected as degenerate
      * units inconsistent between percent and fraction, producing required-N figures
        in the hundreds of thousands

    Scale is decided ONCE for the whole object, never per value. A sub-1 baseline is
    legitimate — Aveeno's first-recommendation rate on need queries is 0.7% — so
    converting individually would inflate it 100x.

    baseline_pp: the AUTHORITATIVE baseline from measured audit data. When supplied it
    overrides whatever the model stated, because the model has no access to it.
    """
    p = dict(pred or {})
    notes = []

    base = _num(p.get("baseline_pp"))
    exp = _num(p.get("expected_pp"))
    lift = _num(p.get("lift_pp"))

    # ---- scale: one decision for the whole object ----
    vals = [v for v in (base, exp, lift) if v is not None and v != 0]
    if vals and all(abs(v) < 1.0 for v in vals):
        base = None if base is None else base * 100
        exp = None if exp is None else exp * 100
        lift = None if lift is None else lift * 100
        notes.append("prediction was stated as fractions; converted to percentage points")

    # ---- authoritative baseline wins ----
    known = _num(baseline_pp)
    if known is not None:
        if base is not None and abs(base - known) > 0.5:
            notes.append(f"stated baseline {base:.1f}pp replaced with the measured "
                         f"{known:.1f}pp")
        base = known
    elif base is None:
        base = 0.0

    # ---- lift is the field the model gets right most often ----
    if lift is not None and lift > 0:
        exp = base + lift
    elif exp is not None and exp <= base:
        if exp > 0:
            notes.append(f"expected_pp ({exp:.1f}) was at or below baseline "
                         f"({base:.1f}) — read as a lift and corrected to "
                         f"{base + exp:.1f}")
            exp = base + exp
        else:
            exp = base + max(mde_pp, 1.0)
            notes.append("expected_pp was not usable — defaulted to baseline + MDE")
    elif exp is None:
        exp = base + max(mde_pp, 1.0)
        notes.append("expected_pp missing — defaulted to baseline + MDE")

    p["baseline_pp"] = round(base, 2)
    p["expected_pp"] = round(min(exp, 99.0), 2)
    p["lift_pp"] = round(p["expected_pp"] - p["baseline_pp"], 2)
    p["min_detectable_pp"] = round(mde_pp, 1)
    if notes:
        p["_repaired"] = notes
    return p, notes

# ---------------------------------------------------------------- power
def mde(p0, n, alpha=0.05, power=0.80):
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

# Template shortlist per family. First entry is the default the prompt leads with.
FAMILY_TEMPLATES = {
    "visibility":     ["E0", "E1", "E4"],
    "rank":           ["E1", "E5", "E2"],      # present but not first — structure first
    "reason":         ["E2", "E0", "E1"],
    "provenance":     ["E3", "E0"],
    "claim":          ["E4", "E2", "E3"],
    "correction":     ["E4", "E3", "E2"],
    "substantiation": ["E4", "E3"],
    "placement":      ["E3"],
    "objection":      ["E9", "E3", "E2"],      # was unreachable — no family mapped here
    "defend":         ["E3", "E5"],
    "engine":         ["E1", "E5", "E0"],
    "freshness":      ["E4", "E0"],
    "format":         ["E5", "E1"],
    "visual":         ["E7", "E8"],
    "other":          ["E0", "E4", "E2"],
}
MECHANISMS = {"retrieval_absence", "memory_deficit", "corroboration_deficit",
              "extraction_failure", "competitive_dominance", "substantiation_deficit",
              "format_illegibility"}
PILLARS = {"Content", "Technical/Structured-Data", "Visual/Multimodal",
           "Corroboration", "Social/Community", "Claims-Substantiation"}

BRAND_FREE_INTENTS = ("need", "attribute", "persona", "longtail", "negative", "freshness")

# ---------------------------------------------------------------- packet
def choose_metric(intent, mbi, fbi, family=None, saturate=0.70, rank_floor=0.10):
    """Which metric has headroom on THIS brand and THIS intent.

    W-37: the baseline, the metric and the target query were three unrelated
    quantities. Aveeno's need-intent mention is 35.4% while first-mention is 0.7% —
    so mention_rate is the wrong target and every hypothesis chose it anyway.
    """
    m = float(mbi.get(intent, mbi.get("need", 0.05)) or 0.0)
    f = float(fbi.get(intent, fbi.get("need", 0.0)) or 0.0)
    if family in ("objection", "defend"):
        return "mention_rate", m, (
            f"Objection/defence work is about FRAMING when named, not surfacing more "
            f"often. Mention on '{intent}' is {m:.1%}.")
    if family == "rank":
        # A rank cluster's diagnosis IS "mentioned but not chosen". Targeting
        # mention_rate measures the thing that already works.
        return "first_rec_rate", f, (
            f"This is a RANK cluster — the brand is already mentioned on '{intent}' "
            f"queries ({m:.1%}) but is first in only {f:.1%}. mention_rate is not the "
            f"constraint.")
    if m >= saturate:
        return "first_rec_rate", f, (
            f"mention_rate on '{intent}' is already {m:.1%} — near saturation, little "
            f"headroom. first_rec_rate is {f:.1%} and is the metric that can move.")
    if m >= 0.25 and f <= rank_floor:
        return "first_rec_rate", f, (
            f"The brand is MENTIONED on {m:.1%} of '{intent}' queries but is first in "
            f"only {f:.1%}. This is a RANK problem, not a visibility problem. Target "
            f"first_rec_rate.")
    return "mention_rate", m, (
        f"mention_rate on '{intent}' is {m:.1%} with headroom — the correct primary "
        f"metric here.")

def build_packet(cluster, *, brand, category, claims, metrics, delta, hitlist,
                 winnability, query_master, n_planned):
    mbi = {r["intent"]: r["mention_rate"]
           for r in (metrics or {}).get("mention_by_intent", [])}
    fbi = {r["intent"]: r.get("first_rate", 0)
           for r in (metrics or {}).get("first_mention_by_intent", [])}

    # ---- target query first, because the intent follows from it ----
    bf = [q["query"] for q in (query_master or [])
          if q.get("intent") in BRAND_FREE_INTENTS
          and fold(brand) not in fold(q.get("query", ""))]
    q_intent = {q["query"]: q.get("intent") for q in (query_master or [])}
    tq = [q for q in (cluster.get("queries") or []) if fold(brand) not in fold(q)] \
         or bf[:6]
    # Comparative and branded queries NAME the brand, so mention is 100% by
    # construction and first_rec_rate measures positioning, not visibility. Also drops
    # malformed slug queries from deterministic_topup ("aveenao vs CeraVe for Health"),
    # which evade the brand-leakage check because the misspelling is not an alias.
    _named = {q["query"] for q in (query_master or [])
              if q.get("intent") in ("branded", "comparative")}
    tq = [q for q in tq if q not in _named] or [q for q in bf if q not in _named][:6]
    intent = (q_intent.get(tq[0]) if tq else None) \
             or (cluster.get("intents") or ["need"])[0]

    # ---- metric and baseline must be on the same scale ----
    metric, p0, metric_note = choose_metric(intent, mbi, fbi, cluster.get("family"))
    baseline_source = f"audit {metric} on '{intent}' queries"

    # a claim-coverage baseline is NOT a mention-rate baseline — carry it as context only
    cov = cluster.get("baseline")
    coverage_ctx = (f"Claim coverage for this cluster measured {cov:.1%} of retrieved "
                    f"brand spans (a DIFFERENT quantity from {metric} — do not use it as "
                    f"the baseline)." if isinstance(cov, (int, float)) and 0 <= cov <= 1
                    else "")

    rr = (delta or {}).get("reason_richness") or []
    ours = next((r for r in rr if fold(r.get("brand", "")) == fold(brand)), {})
    leaders = sorted([r for r in rr if fold(r.get("brand", "")) != fold(brand)],
                     key=lambda r: -(r.get("richness_per_100t") or 0))[:2]

    rel = _relevant_claims(cluster, claims)
    hl = (hitlist.to_dict("records") if hasattr(hitlist, "to_dict") else (hitlist or []))
    win = None
    if winnability is not None and len(winnability):
        try:
            win = winnability["winnability"].value_counts().to_dict()
        except Exception:
            win = None

    fam = cluster.get("family", "other")
    allowed_tpl = FAMILY_TEMPLATES.get(fam, FAMILY_TEMPLATES["other"])
    mde_pp = round(mde(p0, n_planned) * 100, 1)

    return {
        "brand": brand, "category": category or "this category",
        "cluster": json.dumps({k: cluster.get(k) for k in
                               ("cluster_id", "title", "family", "pillar", "severity",
                                "types", "n_gaps", "topic", "dimensions", "intents")},
                              indent=1),
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
        "templates": json.dumps({k: {"name": TEMPLATES[k]["name"],
                                     "isolates": TEMPLATES[k]["isolates"],
                                     "arms": TEMPLATES[k]["arms"]}
                                 for k in allowed_tpl}, indent=1),
        "default_template": allowed_tpl[0],
        "n_planned": n_planned,
        "metric": metric,
        "metric_guidance": metric_note,
        "target_intent": intent,
        "baseline_pp": round((p0 or 0) * 100, 1),
        "baseline_source": baseline_source,
        "coverage_context": coverage_ctx,
        "mde_pp": mde_pp,
        "_p0": p0, "_metric": metric, "_intent": intent,
        "_baseline_source": baseline_source, "_mde_pp": mde_pp,
        "_relevant_claims": rel, "_brand_free": bf, "_target_queries": tq,
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

=== THE METRIC IS ALREADY DECIDED — DO NOT CHANGE IT ===
prediction.metric MUST be exactly: {metric}
Why: {metric_guidance}
Measured baseline: {baseline_pp}% ({baseline_source})
{coverage_context}

=== THE PREDICTION — state the LIFT, not the end state ===
Give `lift_pp` as the ABSOLUTE percentage-point gain you expect, as a number greater
than or equal to {mde_pp}. Examples of the correct form:
    baseline 35.4%, you expect a 19-point gain  ->  "lift_pp": 19.0
    baseline  0.7%, you expect a 12-point gain  ->  "lift_pp": 12.0
NEVER express these as fractions (0.19). NEVER put the end state in lift_pp.
At N={n_planned} an effect smaller than {mde_pp} percentage points cannot be detected.

=== OTHER HARD CONSTRAINTS ===
- You may ONLY propose publishing content built from the SUBSTANTIATED CLAIMS listed.
  Never invent a fact. If no available claim can close the gap, set mechanism to
  "substantiation_deficit".
- The experiment must ISOLATE ONE variable. Name what is held constant.
- target_query MUST be copied verbatim from TARGET QUERIES or BRAND-FREE QUERIES, and
  must NOT contain the brand name.
- Prefer template {default_template} — it is the designed test for this gap family.
  Choose another from the list only if you state why in root_cause.
- Experiments inject documents into context with web OFF. A hypothesis about EARNED
  third-party placement must be tested as a PROXY: does a third-party citation inside
  the brand document change the outcome.
- Never predict zero change. If you believe it will not work, predict the lift you would
  need to see and say in falsification what a null would prove.

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

TARGET QUERIES (prefer one of these — intent '{target_intent}')
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
 "hypothesis": "If we <specific change>, then {metric} will rise by <lift_pp> points because <mechanism>",
 "prediction": {{"metric":"{metric}","baseline_pp":{baseline_pp},"lift_pp":0}},
 "experiment": {{"template":"{default_template}","isolated_variable":"",
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
        # the metric is decided from measured data, not by the model
        pred = h.get("prediction") or {}
        pred["metric"] = pk["_metric"]
        pred.setdefault("baseline_pp", pk["baseline_pp"])
        pred, repairs = normalise_prediction(pred, pk["_mde_pp"],
                                             baseline_pp=pk["baseline_pp"])
        h["prediction"] = pred

        h.update({"cluster_id": cl["cluster_id"], "cluster_title": cl["title"],
                  "family": cl["family"], "gap_ids": cl["member_ids"],
                  "severity": cl["severity"], "n_gaps": cl["n_gaps"],
                  "gap_types": cl["types"], "engines": cl["engines"],
                  "_packet_baseline": pk["_p0"], "_n_planned": n_planned,
                  "_metric": pk["_metric"], "_intent": pk["_intent"],
                  "_baseline_source": pk["_baseline_source"],
                  "_allowed_templates": FAMILY_TEMPLATES.get(
                      cl["family"], FAMILY_TEMPLATES["other"]),
                  "_relevant_claims": [c.get("claim") for c in pk["_relevant_claims"]],
                  "_brand_free": pk["_brand_free"],
                  "_target_queries": pk["_target_queries"],
                  "prediction_repairs": repairs,
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

        # The register check applies to EVERY mechanism. Exempting
        # substantiation_deficit let an invented claim ("widely available at drugstores
        # nationwide" — the model admitted no such claim exists) pass validation clean
        # and become a runnable design. Variant text becomes a content brief, so this is
        # a regulatory gate, not a preference.
        if unknown:
            hard = [c for c in unknown
                    if not any(_overlap(fold(c), r) > 0.6 for r in reg)]
            if hard:
                errs.append(f"claims not in register: "
                            f"{[str(c)[:60] for c in hard][:2]}")
            else:
                warns.append("claims paraphrased from the register — verify wording")

        # A design with no claims cannot have a treatment arm built, whatever the
        # mechanism label says. Previously only substantiation_deficit was checked.
        if not used:
            errs.append("no claims cited — the treatment arm has no material to build "
                        "from, so no manipulation is possible")

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
        # ---- power: WARN only. The pilot in the Experiment tab is the real gate. ----
        pred = h.get("prediction") or {}
        p0 = h.get("_packet_baseline") or 0.05
        need = mde(p0, n_planned) * 100
        expected = float(pred.get("lift_pp") or 0)
        h["mde_pp"] = round(need, 1)
        h["expected_lift_pp"] = round(expected, 1)
        h["n_required_for_prediction"] = (n_for_effect(p0, expected) if expected > 0
                                          else None)
        h["baseline_source"] = h.get("_baseline_source")
        if h.get("prediction_repairs"):
            warns += [f"prediction repaired: {r}" for r in h["prediction_repairs"]]
        if expected <= 0:
            errs.append("prediction is no change — nothing to detect")
        elif expected + 0.05 < need:  # tolerance: 18.9 vs 18.9 is not a failure
            warns.append(f"predicted +{expected:.1f}pp is below the {need:.1f}pp "
                         f"detectable at N={n_planned} on the AUDIT baseline "
                         f"({p0:.1%}); needs N≈{h['n_required_for_prediction']}. "
                         f"Re-check after the pilot.")
        # metric is set from measured data, so a claimed MDE cannot disagree
        tpl_allowed = h.get("_allowed_templates") or list(TEMPLATES)
        if tpl in TEMPLATES and tpl not in tpl_allowed:
            warns.append(f"template {tpl} is outside the shortlist for family "
                         f"'{h.get('family')}' ({tpl_allowed})")

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
        # ---- status ----
        # A `substantiation_deficit` mechanism previously made a hypothesis not_testable
        # UNCONDITIONALLY. That deleted 8 of 10 such clusters that had usable claims —
        # including the ONLY E9 objection design, which is the one this brand most needs.
        # The gate is now whether a publishable claim exists, not the mechanism label.
        used = [c for c in (h.get("claims_used") or []) if str(c).strip()]
        claims_ok = bool(used) and not any(
            e.startswith("claims not in register") or e.startswith("no claims cited")
            for e in errs)

        if h.get("mechanism") == "substantiation_deficit" and not claims_ok:
            h["status"] = "not_testable"
            h["route"] = "optimise_pillar6"
            h["not_testable_reason"] = (
                "No substantiated claim is available to build the treatment arm, so the "
                "change cannot be published or tested. Substantiate first, then test "
                "with E4.")
        elif not h.get("runnable", True) or errs:
            h["status"] = "needs_review"
        else:
            h["status"] = "ready"
            if h.get("mechanism") == "substantiation_deficit":
                # testable, but the diagnosis says the register is the real constraint
                h.setdefault("validation_warnings", []).append(
                    "Mechanism is substantiation_deficit but a registered claim IS "
                    "available, so this is testable. The cluster also belongs in the "
                    "Pillar-6 backlog — the engines assert more than the register holds.")
                h["also_route"] = "optimise_pillar6"
        for k in ("_brand_free", "_relevant_claims", "_target_queries",
                  "_allowed_templates", "_metric", "_intent", "_baseline_source"):
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