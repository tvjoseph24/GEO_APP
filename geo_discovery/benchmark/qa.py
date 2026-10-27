"""Pre-flight QA gate for Benchmark. Catches the silent-degradation failures that make
a plausible-looking scorecard wrong."""
from __future__ import annotations

import pandas as pd

def run_qa(df, claims, veracity, richness_agg, target_brand, competitors,
           manifest=None, min_runs_per_engine=20, min_spans=15, gateway=None,
           ingested_sources=None, claim_prep=None, hitlist=None):
    checks = []

    def add(level, code, msg, fix=""):
        checks.append({"level": level, "code": code, "message": msg, "fix": fix})

    tev = getattr(gateway, "transient_events", []) if gateway else []
    if tev:
        add("info", "transient_api_errors",
            f"{len(tev)} transient API error(s) were retried automatically "
            f"(e.g. {tev[0]['error'][:70]}). Results are complete; the provider was "
            f"briefly unavailable.", "")

    n = len(df)
    if n < 50:
        add("warn", "low_n", f"Only {n} successful runs — CIs will be very wide.",
            "Increase N per query in the Audit tab.")
    for eng, g in df.groupby("engine"):
        if len(g) < min_runs_per_engine:
            add("warn", "thin_engine", f"{eng}: only {len(g)} runs.",
                "Re-run the audit for this engine.")

    if "web_on" in df and df["web_on"].nunique() == 1:
        mode = "web-ON only" if df["web_on"].iloc[0] else "web-OFF only"
        add("info", "single_mode", f"Corpus is {mode} — memory vs retrieval cannot be "
            f"separated, so horizon (fast/slow) calls in Optimise are guesswork.",
            "Audit with Mode='both'.")

    if not claims:
        add("blocker", "no_claims", "Claims Register is empty — veracity is impossible.",
            "Complete Discover extraction.")
    else:
        pf = sum(1 for c in claims if c.get("_parse_note") == "parse_failed")
        if pf:
            add("warn", "claim_parse_failures",
                f"{pf} claim(s) came from failed extractions.", "Check blobs/extract_fail/.")

    sc = (veracity or {}).get("span_count", 0)
    if veracity and veracity.get("error"):
        add("blocker", "no_spans", veracity["error"],
            "The brand is absent from the corpus — Tier B cannot run.")
    elif sc < min_spans:
        add("warn", "few_spans", f"Only {sc} brand spans — per-engine veracity is thin.",
            "Increase N, or add branded/comparative intents.")
    if veracity:
        js = veracity.get("judgements") or []
        quotable = [j for j in js if j.get("verdict") != "omitted"]
        qf = sum(1 for j in quotable if j.get("quote_unverified"))
        nq = max(len(quotable), 1)
        if qf / nq > 0.15:
            add("warn", "judge_unreliable",
                f"{qf}/{nq} quotable verdicts cited a non-verbatim quote "
                f"({qf / nq:.0%}) — the judge is partly paraphrasing. "
                f"Match breakdown: {veracity.get('quote_match_breakdown')}",
                "Switch the Judge model to gemini-2.5-pro.")
        if veracity.get("claims_absent", 0) == veracity.get("claims_considered", 0):
            add("blocker", "no_claim_evidence",
                "No claim retrieved ANY span — retrieval or the register is misaligned.",
                "Check claim wording against actual response language.")

    if veracity and veracity.get("method") == "veracity_v2":
        jy = veracity.get("judge_yield")
        jf = veracity.get("judge_failure_count", 0)
        if jy is not None and jy < 0.8:
            add("blocker", "judge_empty_returns",
                f"Only {jy:.0%} of claims with retrieved evidence produced ANY verdict "
                f"({jf} failed call(s)). Veracity is under-reporting — 'absent' statuses "
                f"are unreliable.",
                "Reduce Spans-per-claim, or use gemini-2.5-flash. Check "
                "veracity['judge_failures'] for the reason codes.")
        cal = veracity.get("calibration") or {}
        if cal.get("agreement") is not None and cal["agreement"] < 0.6:
            add("warn", "low_calibration",
                f"Keyword↔LLM agreement only {cal['agreement']:.0%} — one method is "
                f"systematically wrong. Inspect calibration['examples'].",
                "Usually claim-language mismatch or retrieval threshold.")
        rr = veracity.get("rescued_retrievals", 0)
        if rr:
            add("info", "rescued_retrievals",
                f"{rr} span(s) were judged via the low-score rescue pass — "
                f"claim wording does not lexically match how engines talk.", "")
        if (veracity.get("halluc_counts") or {}) and \
                sum(veracity["halluc_counts"].values()) < 8:
            add("warn", "reverse_scan_thin",
                f"Reverse scan extracted only "
                f"{sum(veracity['halluc_counts'].values())} assertions — expect dozens.",
                "Likely empty LLM returns; check judge_failures.")

    if veracity and veracity.get("spans_by_intent"):
        sbi = veracity["spans_by_intent"]
        tot = sum(sbi.values()) or 1
        named = sbi.get("branded", 0) + sbi.get("comparative", 0)
        if named / tot > 0.8:
            add("warn", "evidence_base_skew",
                f"{named / tot:.0%} of brand spans come from prompts that NAME the brand "
                f"(branded/comparative). Veracity and richness therefore describe the "
                f"trivial case; only {tot - named} span(s) show unprompted behaviour.",
                "Raise N on need/persona/negative intents, or read need-intent "
                "findings as directional only.")

    if richness_agg is not None and not richness_agg.empty:
        if target_brand not in set(richness_agg["brand"]):
            add("warn", "no_target_richness",
                f"No richness spans for {target_brand}.", "")
        thin = richness_agg[richness_agg["spans"] < 3]["brand"].tolist()
        if thin:
            add("info", "thin_competitor_spans",
                f"Too few spans to compare fairly: {thin}", "")

    if manifest:
        vs = manifest.get("vision_selftest") or {}
        if vs.get("vision_live") is False:
            add("warn", "vision_dead",
                "Vision self-test FAILED — visual metrics are meaningless.",
                "Do not interpret visual gaps; fix the vision path.")
        elif vs.get("confabulates"):
            add("warn", "vision_confabulates",
                "Model invented text on a blank control image.",
                "E7/E8 results are untrustworthy on this model.")

    if "polarity" in df and df["polarity"].isna().all():
        add("info", "no_polarity", "No polarity data — legacy audit rows.",
            "Run backfill.py.")

    tally = {}
    for t in df.get("source_type_tally", []):
        for k, v in (t or {}).items():
            tally[k] = tally.get(k, 0) + v
    tot = sum(tally.values())
    if tot:
        other = tally.get("other", 0) / tot
        _fix = ("Generate a source pack in the Audit tab (one LLM call per 50 domains, "
                "human-reviewed), then `python backfill.py --parent P --target T` and "
                "re-run Benchmark. Do NOT extend SOURCE_TYPE_MAP by hand — the pack is "
                "versioned and re-appliable.")
        if other > 0.40:
            add("blocker", "unclassified_sources",
                f"{other:.0%} of citations are unclassified. Provenance percentages are "
                f"lower bounds, `voice_control_ratio` is withheld, and the source "
                f"hit-list is missing most of the ecosystem. A misclassified affiliate "
                f"tail has been observed to move that ratio across the 1.0 threshold — "
                f"i.e. to invert the headline conclusion about citation control.", _fix)
        elif other > 0.15:
            add("warn", "unclassified_sources",
                f"{other:.0%} of citations are unclassified — provenance is a lower "
                f"bound.", _fix)
        if not tally.get("practitioner") and tally.get("other", 0):
            add("info", "no_practitioner_bucket",
                "No citations classified as practitioner. In most service-adjacent "
                "categories the professionals who practise in the category are a material "
                "channel, and they are engaged rather than pitched — check whether they "
                "are sitting in the unclassified tail.", "")
        if not tally.get("brand_owned"):
            add("warn", "no_brand_domains",
                "No brand-owned citations detected — self_referential_pct reads 0.",
                "Add the brand's domains to competitive_set (role:'self').")

            # ---- W-02: is a "source" a domain, or a page title? ----
        from url_resolver import domain_from_title, looks_like_domain
        n_cit = n_title = n_recover = 0
        if "sources" in df.columns:
            for row in df["sources"]:
                for s in (row or []):
                    d = (s.get("domain") or "").strip().lower()
                    if not d:
                        continue
                    n_cit += 1
                    if s.get("domain_is_title") or not looks_like_domain(d):
                        n_title += 1
                        if domain_from_title(s.get("title") or d):
                            n_recover += 1
        if n_cit:
            share = n_title / n_cit
            msg = (f"{n_title:,} of {n_cit:,} citations ({share:.0%}) carry a page TITLE "
                   f"rather than a domain, because url_resolver could not follow the "
                   f"grounding redirect. These flow into provenance percentages AND into the "
                   f"client-facing source hit-list, where a title is presented as a domain "
                   f"to pitch. {n_recover:,} contain a recoverable domain.")
            fix = ("Run `python backfill.py --parent P --target T` to recover embedded "
                   "domains, then `--domains` to see what genuinely remains unclassified. "
                   "No SOURCE_TYPE_MAP edit can fix a title.")
            if share >= 0.40:
                add("blocker", "citations_are_titles", msg, fix)
            elif share >= 0.15:
                add("warn", "citations_are_titles", msg, fix)
            else:
                add("info", "citations_are_titles",
                    f"{share:.1%} of citations are unresolved titles — within tolerance.", "")

        # ---- W-01: unsubstantiated assertions vs a missing specification source ----
        if ingested_sources is not None:
            from ingestion import spec_coverage
            sc = spec_coverage(ingested_sources)
            unsub = (veracity or {}).get("halluc_counts", {}).get("unsubstantiated", 0)
            if not sc["has_specification"] and unsub >= 5:
                add("warn", "unsubstantiated_may_be_missing_source",
                    f"{unsub} assertion(s) are classed UNSUBSTANTIATED, but no product "
                    f"specification was ingested ({sc['n_sources']} source(s), none "
                    f"spec-like). Pack-level facts the engines read off the product cannot "
                    f"be in the register, so this may be an ingestion gap presented as a "
                    f"compliance exposure.",
                    "Run `python diagnose.py --parent P --target T --check D2`. If the "
                    "assertions appear in the ingested text, it is an extraction failure. "
                    "If no spec source exists, ingest the PDP or ingredient list and re-run "
                    "Discover before reporting a compliance finding.")
            elif not sc["has_specification"]:
                add("info", "no_specification_source",
                    f"No product specification among {sc['n_sources']} ingested source(s). "
                    f"The register covers positioning, not pack facts.", "")

        # ---- claim denominators: the report must not be able to print >100% ----
        if claim_prep:
            n_in = claim_prep.get("input") or 0
            n_kept = claim_prep.get("kept") or 0
            n_cons = (veracity or {}).get("claims_considered") or 0
            if n_in and n_kept > n_in:
                add("blocker", "claim_denominator_mismatch",
                    f"Claim preparation expanded the register ({n_in} → {n_kept} after "
                    f"{claim_prep.get('n_split', 0)} split(s)), and {n_cons} were assessed. "
                    f"The report computes coverage against the RAW register, so it can print "
                    f"more than 100% assessed.",
                    "The prepared register must be persisted and used as the denominator "
                    "(Phase 2, W-30).")
            elif n_cons and n_in and n_cons > n_in:
                add("blocker", "claim_denominator_mismatch",
                    f"{n_cons} claims assessed against a raw register of {n_in} — the "
                    f"coverage figure in the report will exceed 100%.",
                    "Persist the prepared register and report against it.")
        # ---- W-55: is community analysis possible, and does it say anything? ----
        _soc_types = {"community_reviews", "video_social"}
        _soc_cites = sum(v for t in df.get("source_type_tally", [])
                         for k, v in (t or {}).items() if k in _soc_types)
        _all_cites = sum(v for t in df.get("source_type_tally", [])
                         for v in (t or {}).values()) or 1
        _share = _soc_cites / _all_cites
        if _share < 0.02:
            add("info", "social_negligible",
                f"Community and social sources are {_share:.1%} of citations. Pillar 5 will "
                f"be minimal or empty, and the strategy should say why — recommending "
                f"community work where the engines do not listen is the most expensive "
                f"mistake available in that pillar.", "")
        else:
            _threads = set()
            for row in df.get("sources", []):
                for s in (row or []):
                    if s.get("source_type") in _soc_types and not s.get("domain_is_title"):
                        u = s.get("resolved_url") or s.get("url")
                        if u and "vertexaisearch" not in str(u):
                            _threads.add(u)
            if not _threads:
                add("warn", "social_urls_unresolved",
                    f"{_soc_cites:,} community citations, but no resolvable thread URL. "
                    f"Thread-level analysis is impossible and Pillar 5 can only recommend "
                    f"engaging a domain — which is not an action.",
                    "Run `python backfill.py --parent P --target T` to recover domains, then "
                    "check how many community citations still carry an unresolved redirect.")
            elif len(_threads) < 5:
                add("info", "social_thin",
                    f"Only {len(_threads)} distinct community thread(s) resolvable from "
                    f"{_soc_cites:,} citations. Thread-level findings will be thin.", "")

            # ---- a whole claim FAMILY falling outside the classifier ----
            if claim_prep:
                _in = claim_prep.get("input") or 0
                _drop = claim_prep.get("n_dropped") or 0
                _bykind = claim_prep.get("dropped_by_kind") or {}
                if _in and _drop / _in >= 0.40:
                    add("warn", "heavy_claim_exclusion",
                        f"{_drop} of {_in} register entries ({_drop / _in:.0%}) were excluded as "
                        f"unjudgeable, by reason {_bykind}. Veracity, gap detection and every "
                        f"content brief run against the remaining {claim_prep.get('kept')}. At "
                        f"this rate a whole claim FAMILY may be falling outside the classifier — "
                        f"negative attributes such as 'free of X' are a known blind spot — in "
                        f"which case the coverage figures are narrow rather than accurate.",
                        "Read the excluded entries in the normalisation detail. If they share a "
                        "form, the classifier needs that form rather than the register needing "
                        "rewriting.")
                if (claim_prep.get("kept") or 0) > _in:
                    add("info", "register_expanded_by_splitting",
                        f"Preparation SPLIT {claim_prep.get('n_split')} multi-fact entry(ies), so "
                        f"{_in} raw entries became {claim_prep.get('kept')} judgeable claims. "
                        f"Coverage must be reported against the prepared count.", "")

            # ---- has the corpus moved since a previous Benchmark? ----
            try:
                from benchmark.diagnostics import load_manifest
                _m = load_manifest(*(manifest or {}).get("_project", (None, None))) \
                    if isinstance(manifest, dict) and manifest.get("_project") else None
            except Exception:
                _m = None
            if _m and _m.get("n_audit_rows") and len(df) != _m["n_audit_rows"]:
                add("warn", "corpus_moved_since_last_benchmark",
                    f"The previous benchmark ran on {_m['n_audit_rows']:,} audit run(s); this "
                    f"corpus holds {len(df):,}. Any figure quoted from the earlier scorecard "
                    f"describes a different corpus.",
                    "Re-run Benchmark and restate anything already circulated.")

            return checks

def qa_verdict(checks):
    lv = {c["level"] for c in checks}
    if "blocker" in lv:
        return "blocked"
    if "warn" in lv:
        return "caution"
    return "ok"