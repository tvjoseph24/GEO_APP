"""Pre-flight QA gate for Benchmark. Catches the silent-degradation failures that make
a plausible-looking scorecard wrong."""
from __future__ import annotations

import pandas as pd

def run_qa(df, claims, veracity, richness_agg, target_brand, competitors,
           manifest=None, min_runs_per_engine=20, min_spans=15,gateway=None):
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
        if other > 0.30:
            add("warn", "unclassified_sources",
                f"{other:.0%} of citations are unclassified — provenance is unreliable.",
                "python backfill.py --domains, then extend SOURCE_TYPE_MAP.")
        if not tally.get("brand_owned"):
            add("warn", "no_brand_domains",
                "No brand-owned citations detected — self_referential_pct reads 0.",
                "Add the brand's domains to competitive_set (role:'self').")
    return checks

def qa_verdict(checks):
    lv = {c["level"] for c in checks}
    if "blocker" in lv:
        return "blocked"
    if "warn" in lv:
        return "caution"
    return "ok"