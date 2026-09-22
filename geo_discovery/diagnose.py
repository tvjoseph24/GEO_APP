"""diagnose.py — zero-cost diagnostics over stored artifacts and raw logs.

No LLM calls, no network, no writes. Answers the Phase-0 questions before any code is
written against them.

  D1  citation classification  — are unclassified sources genuine domains, or page titles?
  D2  specification coverage   — are "unsubstantiated" assertions already in ingested text?
  D3  retrieval-mode coverage  — was memory (web-off) ever audited?
  D4  claim denominators       — can the report print >100% of the register assessed?
  D5  artifact round-trip      — is every artifact written also loadable and reportable?
  D6  judge reliability        — calibration, cache size, what is not persisted
  D7  hit-list pollution       — how many client-facing "Source" rows are not domains?

    python diagnose.py --parent danone --target aptamil
    python diagnose.py --parent danone --target aptamil --check D1 D7
    python diagnose.py --all --json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from metrics_spec import fold
from store import ARTIFACT_NAMES, load_artifact, project_dir

CHECKS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
ICON = {"ok": "✅", "warn": "🟡", "blocker": "🔴", "n/a": "⚪", "info": "ℹ️"}

STOP = set("the a an and or of for with to in on is are be been being this that from as "
           "by it its their has have had was were will can may also more most than then "
           "them these those which who what when where how why at into over under about "
           "all any both each every other such same brand product products".split())

# ---- assertion kinds. Informs W-33 (financial claims) and W-01's inversion test. ----
_PRICE_RX = re.compile(
    r"(?:[$£€]\s?\d|\b\d+(?:\.\d+)?\s?(?:usd|gbp|eur|dollars?|pounds?|euros?)\b"
    r"|\bpric(?:e|ed|ing)\b|\bcosts?\b|\baffordab|\bcheap|\binexpensive\b"
    r"|\bexpensive\b|\bbudget[- ]friendly\b|\bvalue for money\b|\bper (?:oz|ounce|ml|100g)\b)",
    re.I)
_AVAIL_RX = re.compile(
    r"\b(?:widely available|readily available|available (?:at|in|from|online)"
    r"|sold (?:at|in|online)|stocked|in stock|drugstores?|pharmac(?:y|ies)"
    r"|supermarkets?|retailers?|nationwide|over[- ]the[- ]counter|otc\b)\b", re.I)
_SPEC_RX = re.compile(
    r"\b(?:free (?:of|from)|contains|without|fragrance[- ]free|dye[- ]free|soap[- ]free"
    r"|paraben[- ]free|sulfate[- ]free|sulphate[- ]free|non[- ]comedogenic|hypoallergenic"
    r"|ph[- ]balanced|ingredient|formulat|net\s?(?:wt|weight)"
    r"|\d+(?:\.\d+)?\s?(?:ml|l|oz|fl\s?oz|g|kg|mg|mcg|%))\b", re.I)
_CLINICAL_RX = re.compile(
    r"\b(?:dermatologist|p(?:a)?ediatrician|clinical(?:ly)?|stud(?:y|ies)|trial"
    r"|recommended by|approved by|tested|award)\b", re.I)

def classify_assertion_kind(text):
    """Which remedy applies. Price and availability are not brand claims and must not
    inflate hallucination_rate or the substantiation backlog (W-33)."""
    t = str(text or "")
    if _PRICE_RX.search(t):
        return "price"
    if _AVAIL_RX.search(t):
        return "availability"
    if _SPEC_RX.search(t):
        return "specification"
    if _CLINICAL_RX.search(t):
        return "clinical_or_credential"
    return "other"

def _res(check, title, verdict, summary, action="", **data):
    return {"check": check, "title": title, "verdict": verdict, "summary": summary,
            "action": action, "data": data}

def _na(check, title, why):
    return _res(check, title, "n/a", why)

def _toks(text, min_len=3):
    low = fold(text)
    words = {w for w in re.findall(rf"[a-z][a-z\-]{{{min_len - 1},}}", low)
             if w not in STOP}
    nums = {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)?", low)}
    return words | nums

# ====================================================================== D1
def check_D1(parent, target, top=30):
    """W-02. Split unclassified citations into genuine domains and page titles."""
    from audit_runner import build_source_classifier
    from backfill import resolve_context
    from url_resolver import domain_from_title, looks_like_domain

    path = project_dir(parent, target) / "audit_raw.jsonl"
    if not path.exists():
        return _na("D1", "Citation classification", "no audit_raw.jsonl")

    ctx = resolve_context(parent, target)
    stype = build_source_classifier(ctx["brand_domains"], ctx["competitor_domains"])

    tot = 0
    by_source, by_type = Counter(), Counter()
    unclassified_domains, unclassified_titles, recoverable = Counter(), Counter(), Counter()

    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("success"):
                continue
            for s in (r.get("sources") or []):
                d = (s.get("domain") or "").strip().lower()
                if not d:
                    continue
                tot += 1
                src = s.get("domain_source")
                if not src:
                    real = s.get("resolved_url") or s.get("url") or ""
                    if s.get("resolved_ok") and "vertexaisearch" not in str(real):
                        src = "url"
                    elif looks_like_domain(d):
                        src = "title_domain"
                    else:
                        src = "title_text"
                by_source[src] += 1
                st = s.get("source_type") or stype(d)
                by_type[st] += 1
                if st == "other":
                    if looks_like_domain(d):
                        unclassified_domains[d] += 1
                    else:
                        unclassified_titles[d[:70]] += 1
                        rec = domain_from_title(s.get("title") or d)
                        if rec:
                            recoverable[rec] += 1

    if not tot:
        return _na("D1", "Citation classification", "no citations in the log")

    n_other = by_type.get("other", 0)
    n_titles = by_source.get("title_text", 0)
    n_dom_unc = sum(unclassified_domains.values())
    n_title_unc = sum(unclassified_titles.values())
    n_rec = sum(recoverable.values())

    share_other = n_other / tot
    share_title = n_titles / tot

    if share_title >= 0.40:
        verdict = "blocker"
    elif share_title >= 0.15 or share_other >= 0.30:
        verdict = "warn"
    else:
        verdict = "ok"

    if n_title_unc > n_dom_unc:
        action = (f"The dominant cause is UNRESOLVED REDIRECTS, not a missing source-type "
                  f"mapping. {n_rec:,} of the {n_title_unc:,} title rows contain a "
                  f"recoverable domain — `python backfill.py --parent {parent} "
                  f"--target {target}` will recover those. The remainder cannot be fixed "
                  f"by any SOURCE_TYPE_MAP edit, and must be excluded from the "
                  f"client-facing hit-list (Phase 5). W-56 is LOWER priority than it "
                  f"appeared.")
    elif n_dom_unc:
        action = (f"The dominant cause is GENUINELY UNMAPPED DOMAINS ({n_dom_unc:,} "
                  f"citations across {len(unclassified_domains)} domains). W-56 is worth "
                  f"doing — the top domains below are the input.")
    else:
        action = "Citation classification is healthy."

    return _res(
        "D1", "Citation classification", verdict,
        f"{tot:,} citations · {share_other:.1%} unclassified · {share_title:.1%} carry a "
        f"page TITLE rather than a domain · {n_rec:,} domains recoverable from titles",
        action,
        total_citations=tot, by_domain_source=dict(by_source),
        by_source_type={k: round(v / tot, 4) for k, v in by_type.most_common()},
        unclassified_genuine_domains=n_dom_unc,
        unclassified_titles=n_title_unc,
        recoverable_from_title=n_rec,
        top_unclassified_domains=[{"domain": d, "citations": n,
                                   "share": round(n / tot, 4)}
                                  for d, n in unclassified_domains.most_common(top)],
        top_recoverable=[{"domain": d, "citations": n}
                         for d, n in recoverable.most_common(12)],
        sample_titles=[t for t, _ in unclassified_titles.most_common(8)])

# ====================================================================== D2
def check_D2(parent, target, min_cover=0.55):
    """W-01. Are "unsubstantiated" assertions already present in the ingested corpus?"""
    from ingestion import spec_coverage

    d = load_artifact(parent, target, "delta_scorecard") or {}
    srcs = load_artifact(parent, target, "ingested_sources") or []
    reg = load_artifact(parent, target, "claims_register") or []
    if not d:
        return _na("D2", "Specification coverage", "no delta_scorecard")

    sc = spec_coverage(srcs)
    assertions = [r for r in (d.get("risk_register") or [])
                  if r.get("delta_type") == "unsubstantiated"]
    if not assertions:
        return _res("D2", "Specification coverage", "ok",
                    f"No unsubstantiated assertions recorded. "
                    f"Specification source present: {sc['has_specification']}.",
                    "", spec_coverage=sc)

    corpus = fold(" ".join((s.get("text") or "") for s in srcs if isinstance(s, dict)))
    ctoks = _toks(corpus)
    rtoks = _toks(" ".join(c.get("claim", "") for c in reg))

    kinds, supported, near_reg, rows = Counter(), [], [], []
    for a in assertions:
        text = a.get("assertion") or ""
        at = _toks(text, 4)
        cov_src = len(at & ctoks) / max(len(at), 1)
        cov_reg = len(at & rtoks) / max(len(at), 1)
        kind = classify_assertion_kind(text)
        kinds[kind] += 1
        row = {"assertion": text[:140], "kind": kind,
               "coverage_in_ingested_text": round(cov_src, 2),
               "coverage_in_register": round(cov_reg, 2),
               "engine": a.get("engine"), "severity": a.get("severity")}
        rows.append(row)
        if cov_src >= min_cover:
            supported.append(row)
        elif cov_reg >= 0.75:
            near_reg.append(row)

    n = len(assertions)
    n_sup, n_near = len(supported), len(near_reg)
    n_commercial = kinds["price"] + kinds["availability"]

    if n_sup >= 3:
        verdict = "blocker"
        action = (f"{n_sup} assertion(s) are already present in the INGESTED TEXT but "
                  f"absent from the register. That is an EXTRACTION failure, not a "
                  f"substantiation failure — and the compliance callout in the executive "
                  f"summary is inverted. Fix extraction (or claim splitting) before any "
                  f"client-facing report. Note: persisted source text is capped at 5,000 "
                  f"characters per source, so this is a LOWER BOUND.")
    elif not sc["has_specification"] and n >= 10:
        verdict = "warn"
        action = (f"No specification source was ingested and {n} assertions are classed "
                  f"unsubstantiated. Ingest the PDP, pack artwork or ingredient list and "
                  f"re-run Discover, then re-check. Do not report a compliance exposure "
                  f"until this is excluded.")
    elif n_commercial / max(n, 1) >= 0.25:
        verdict = "warn"
        action = (f"{n_commercial} of {n} assertions are PRICE or AVAILABILITY claims, "
                  f"which a brand cannot substantiate and which should not appear in a "
                  f"compliance backlog. This is W-33 and it must exclude them from "
                  f"hallucination_rate AND unsubstantiated_rate.")
    else:
        verdict = "ok"
        action = ("The unsubstantiated set looks like genuine substantiation work.")

    return _res(
        "D2", "Specification coverage", verdict,
        f"{n} unsubstantiated assertion(s) · {n_sup} already in ingested text · "
        f"{n_near} near-match a registered claim · {n_commercial} are price/availability · "
        f"specification source present: {sc['has_specification']}",
        action,
        n_assertions=n, kinds=dict(kinds), spec_coverage=sc,
        supported_by_ingested_text=supported[:12],
        near_register_match=near_reg[:8],
        all_rows=rows[:40],
        caveat="Persisted source text is capped at 5,000 chars per source — coverage "
               "figures are a lower bound.")

# ====================================================================== D3
def check_D3(parent, target):
    """W-59. Was memory (web-off) ever audited?"""
    path = project_dir(parent, target) / "audit_raw.jsonl"
    if not path.exists():
        return _na("D3", "Retrieval-mode coverage", "no audit_raw.jsonl")

    modes, by_eng, by_intent, models = Counter(), {}, {}, Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("success"):
                continue
            m = "on" if r.get("web_on") else "off"
            modes[m] += 1
            models[r.get("model") or r.get("engine")] += 1
            by_eng.setdefault(r.get("engine"), Counter())[m] += 1
            by_intent.setdefault(r.get("intent"), Counter())[m] += 1

    n = sum(modes.values())
    if not n:
        return _na("D3", "Retrieval-mode coverage", "no successful runs")

    separated = len(modes) > 1
    incomplete = [e for e, c in by_eng.items() if len(c) < 2]

    if not separated:
        verdict, action = "warn", (
            f"Only web-{list(modes)[0].upper()} was audited across all {n:,} runs. Memory "
            f"and retrieval cannot be separated, so `by_web_mode` is empty, the "
            f"single_mode QA warning persists, and `retrieval_is_primary_constraint` in "
            f"Optimise is INFERRED rather than evidenced. Re-run the audit with "
            f"Mode='memory (web-off)' — it is resumable, so only the missing cells run. "
            f"This is pure compute and can proceed alongside every other phase.")
    elif incomplete:
        verdict, action = "warn", (
            f"Mode coverage is uneven: {incomplete} lack one mode, so cross-engine "
            f"memory comparison is not valid. Re-run those engines web-off.")
    else:
        verdict, action = "ok", "Both modes audited on every engine."

    return _res("D3", "Retrieval-mode coverage", verdict,
                f"{n:,} runs · modes {dict(modes)} · separated: {separated}",
                action, modes=dict(modes), models=dict(models),
                by_engine={k: dict(v) for k, v in by_eng.items()},
                by_intent={k: dict(v) for k, v in by_intent.items()},
                engines_missing_a_mode=incomplete)

# ====================================================================== D4
def check_D4(parent, target):
    """Can the report print >100% of the register assessed?"""
    from benchmark.claim_prep import prepare_claims
    from report.contract import coverage_declaration

    reg = load_artifact(parent, target, "claims_register") or []
    if not reg:
        return _na("D4", "Claim denominators", "no claims_register")
    cs = load_artifact(parent, target, "competitive_set") or []
    comps = [c.get("brand") for c in cs
             if isinstance(c, dict) and c.get("brand") and c.get("role") != "self"]

    kept, rep = prepare_claims(reg, brand=target, category="", competitors=comps)
    cov = coverage_declaration(parent, target)
    d = load_artifact(parent, target, "delta_scorecard") or {}
    considered = (d.get("evidence_base") or {}).get("claims_considered")

    pct = cov.get("claims_assessed_pct")
    overflow = bool(pct and pct > 100)
    mismatch = bool(considered and considered > len(reg))

    verdict = "blocker" if (overflow or mismatch) else (
        "warn" if rep["n_split"] or rep["n_dropped"] else "ok")
    action = ("The report computes coverage against the RAW register while veracity runs "
              "against the PREPARED register. Persist the prepared register and report "
              "against it (Phase 2, W-30)."
              if verdict != "ok" else "Denominators agree.")

    return _res("D4", "Claim denominators", verdict,
                f"raw {len(reg)} → prepared {rep['kept']} "
                f"({rep['n_split']} split, {rep['n_merged']} merged, "
                f"{rep['n_dropped']} dropped) · assessed {considered} · "
                f"report prints {pct}%",
                action,
                raw=len(reg), prepared=rep["kept"], considered=considered,
                reported_pct=pct, overflow=overflow,
                dropped_by_kind=rep.get("dropped_by_kind"),
                kept_by_kind=rep.get("kept_by_kind"),
                n_split=rep["n_split"], n_merged=rep["n_merged"])

# ====================================================================== D5
def check_D5(parent, target):
    """Is every artifact written also loadable, and reportable?"""
    try:
        from report.contract import CONTRACT
        contract_names = {r.artifact for r in CONTRACT}
    except Exception as e:
        contract_names = set()

    adir = project_dir(parent, target) / "artifacts"
    on_disk = sorted({p.name[: -len("_LATEST.json")]
                      for p in adir.glob("*_LATEST.json")}) if adir.exists() else []
    loadable = set(ARTIFACT_NAMES)

    not_loadable = [n for n in on_disk if n not in loadable]
    not_reportable = [n for n in on_disk if contract_names and n not in contract_names]
    declared_absent = [n for n in ARTIFACT_NAMES if n not in on_disk]

    verdict = "warn" if not_loadable else ("info" if not_reportable else "ok")
    action = ""
    if not_loadable:
        action = (f"{not_loadable} are written to disk but absent from "
                  f"store.ARTIFACT_NAMES, so recovery after a refresh silently drops "
                  f"them. Add them.")
    elif not_reportable:
        action = (f"{not_reportable} are loadable but absent from report CONTRACT, so "
                  f"nothing in the report can use them (W-32, Phase 2).")

    return _res("D5", "Artifact round-trip", verdict,
                f"{len(on_disk)} artifact(s) on disk · {len(not_loadable)} not loadable · "
                f"{len(not_reportable)} not in report CONTRACT",
                action,
                on_disk=on_disk, not_loadable=not_loadable,
                not_in_report_contract=not_reportable,
                declared_but_absent=declared_absent)

# ====================================================================== D6
def check_D6(parent, target):
    """Judge reliability, and what is not persisted."""
    d = load_artifact(parent, target, "delta_scorecard") or {}
    if not d:
        return _na("D6", "Judge reliability", "no delta_scorecard")

    cal = d.get("calibration") or {}
    eb = d.get("evidence_base") or {}
    hs = d.get("hallucination_summary") or {}
    cdir = project_dir(parent, target) / "cache"
    caches = []
    if cdir.exists():
        for p in sorted(cdir.glob("veracity_*.json")):
            try:
                caches.append({"file": p.name, "entries": len(json.loads(p.read_text()))})
            except Exception:
                caches.append({"file": p.name, "entries": "unreadable"})

    agree = cal.get("agreement")
    missing = [k for k in ("quote_match_breakdown", "judge_yield", "judge_failures",
                           "rescued_retrievals", "span_judgements")
               if k not in d]

    if agree is not None and agree < 0.60:
        verdict = "warn"
        action = (f"Keyword↔LLM agreement is only {agree:.0%} across "
                  f"{cal.get('n_compared')} claims. One method is systematically wrong. "
                  f"Inspect calibration['examples'] — usually claim-language mismatch or "
                  f"a retrieval threshold.")
    else:
        verdict = "info"
        action = (f"Span-level judgements, quote-match mix and judge yield are NOT "
                  f"persisted, so judge reliability cannot be audited after the session "
                  f"ends. The veracity cache also has no prompt-version dimension, so a "
                  f"judge-prompt fix silently does not take effect on cached claims. "
                  f"Both are Phase 2 (W-30).")

    return _res("D6", "Judge reliability", verdict,
                f"calibration agreement {agree} · brand spans {eb.get('brand_spans')} · "
                f"claims with evidence {eb.get('claims_with_evidence')}/"
                f"{eb.get('claims_considered')} · hallucination rate "
                f"{hs.get('hallucination_rate')}",
                action,
                calibration=cal, evidence_base=eb,
                hallucination_summary={k: v for k, v in hs.items() if k != "high_risk"},
                caches=caches, not_persisted=missing)

# ====================================================================== D7
def check_D7(parent, target):
    """How many client-facing 'Source' rows are not domains?"""
    from url_resolver import domain_from_title, looks_like_domain

    hl = load_artifact(parent, target, "source_hitlist") or []
    if not hl:
        return _na("D7", "Hit-list pollution", "no source_hitlist")

    bad, recoverable = [], 0
    for h in hl:
        d = (h.get("domain") or "").strip().lower()
        if d and not looks_like_domain(d):
            if domain_from_title(d):
                recoverable += 1
            bad.append({"value": d[:80], "times_cited": h.get("times_cited"),
                        "bucket": h.get("bucket"),
                        "access_route": h.get("access_route")})

    n = len(hl)
    share = len(bad) / max(n, 1)
    verdict = ("blocker" if share >= 0.25 else "warn" if bad else "ok")
    action = (f"{len(bad)} of {n} hit-list rows present a page TITLE under the column "
              f"headed 'Source', beside an 'access_route' telling the client how to pitch "
              f"it. This renders in builders.build_delta and in the executive deck. Fix: "
              f"exclude domain_is_title rows from source_hitlist and declare the excluded "
              f"share (Phase 5). {recoverable} are recoverable via backfill."
              if bad else "Every hit-list row is a valid domain.")

    return _res("D7", "Hit-list pollution", verdict,
                f"{len(bad)} of {n} rows ({share:.0%}) are not domains · "
                f"{recoverable} recoverable",
                action, n_rows=n, n_polluted=len(bad), recoverable=recoverable,
                examples=bad[:12])

# ====================================================================== driver
RUNNERS = {"D1": check_D1, "D2": check_D2, "D3": check_D3, "D4": check_D4,
           "D5": check_D5, "D6": check_D6, "D7": check_D7}

def run_all(parent, target, only=None):
    out = []
    for k in (only or CHECKS):
        try:
            out.append(RUNNERS[k](parent, target))
        except Exception as e:
            out.append(_res(k, RUNNERS[k].__doc__ or k, "warn",
                            f"check raised {type(e).__name__}: {str(e)[:160]}",
                            "This is a diagnostic bug, not a data finding."))
    return out

def print_report(parent, target, results):
    print(f"\n{'=' * 78}")
    print(f"  PHASE-0 DIAGNOSTICS · {parent} → {target}")
    print(f"{'=' * 78}")
    for r in results:
        print(f"\n{ICON.get(r['verdict'], '?')} {r['check']} · {r['title']} "
              f"[{r['verdict'].upper()}]")
        print(f"   {r['summary']}")
        if r.get("action"):
            for line in _wrap(r["action"], 74):
                print(f"   → {line}")

    print(f"\n{'-' * 78}")
    v = Counter(r["verdict"] for r in results)
    print(f"  {v.get('blocker', 0)} blocker · {v.get('warn', 0)} warn · "
          f"{v.get('ok', 0)} ok · {v.get('info', 0)} info · {v.get('n/a', 0)} n/a")
    if v.get("blocker"):
        print("  🔴 At least one finding must be resolved before a client-facing report.")
    print(f"{'-' * 78}\n")

    # the three decisions Phase 0 exists to make
    d1 = next((r for r in results if r["check"] == "D1"), None)
    d2 = next((r for r in results if r["check"] == "D2"), None)
    d3 = next((r for r in results if r["check"] == "D3"), None)
    print("  DECISIONS")
    if d1 and d1["verdict"] != "n/a":
        dd = d1["data"]
        titles = dd.get("unclassified_titles", 0)
        doms = dd.get("unclassified_genuine_domains", 0)
        print(f"    W-56 (source classification): "
              + ("LOWER priority — unclassified citations are mostly unresolved titles"
                 if titles > doms else
                 f"WORTH DOING — {doms:,} citations across genuinely unmapped domains"))
    if d2 and d2["verdict"] != "n/a":
        print(f"    W-01 (compliance finding): "
              + ("INVERTED — assertions are already in the ingested text"
                 if d2["data"].get("supported_by_ingested_text") else
                 "no specification source ingested; cannot conclude either way"
                 if not (d2["data"].get("spec_coverage") or {}).get("has_specification")
                 else "stands — genuine substantiation work"))
    if d3 and d3["verdict"] != "n/a":
        print(f"    W-59 (web-off pass): "
              + ("REQUIRED — only one retrieval mode audited"
                 if len(d3["data"].get("modes") or {}) < 2 else "already covered"))
    print()

def _wrap(text, width):
    words, line, out = str(text).split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out

def main():
    ap = argparse.ArgumentParser(description="Zero-cost GEO diagnostics.")
    ap.add_argument("--parent")
    ap.add_argument("--target")
    ap.add_argument("--all", action="store_true", help="every project in the store")
    ap.add_argument("--check", nargs="+", choices=CHECKS,
                    help="run only these checks")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.all:
        from backfill import find_projects
        projects = find_projects()
    elif a.parent and a.target:
        projects = [(a.parent, a.target)]
    else:
        ap.error("supply --parent and --target, or --all")
    if not projects:
        print("No projects found.")
        return

    payload = []
    for parent, target in projects:
        res = run_all(parent, target, a.check)
        payload.append({"parent": parent, "target": target, "checks": res})
        if not a.json:
            print_report(parent, target, res)
    if a.json:
        print(json.dumps(payload, indent=2, default=str))

if __name__ == "__main__":
    main()
