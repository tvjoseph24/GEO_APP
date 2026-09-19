"""Advisory entity-level coherence check for Discovery.
Flags product-brand vs corporate-parent mismatches. Non-blocking — human decides.
Skips entirely for intentional corporate-level engagements."""
import unicodedata

# Corporate holding companies — used to detect level mismatches
KNOWN_PARENTS = ["nestlé", "nestle", "pepsico", "unilever", "danone", "mars",
                 "kraft heinz", "mondelez", "general mills", "kellogg", "kellanova",
                 "procter", "procter & gamble", "p&g", "coca-cola", "l'oréal", "loreal",
                 "johnson & johnson", "j&j", "reckitt", "colgate-palmolive",
                 "kenvue", "haleon", "beiersdorf"]

def _norm(t):
    """Normalize for accent/dash-insensitive matching (Nestlé → nestle)."""
    if not t:
        return ""
    return unicodedata.normalize("NFKC", str(t)).lower()

def brand_is_product(name):
    """Heuristic: is this a product brand (True) or a corporate parent (False)?"""
    return not any(p in _norm(name) for p in KNOWN_PARENTS)

def check_entity_alignment(target_brand, query_master, competitors,
                           positioning_layers, parent_brand=None,
                           brand_level="product", ingested_sources=None):
    """Returns (warnings, errors). Errors are blocking; warnings are advisory."""
    if brand_level == "corporate":
        return [], []

    warnings, errors = [], []
    tb = _norm(target_brand)
    pb = _norm(parent_brand) if parent_brand else ""

    # ---- 0. DOCUMENT RECONCILIATION (W-69) — blocking ----
    ok, detail = verify_brand_in_sources(target_brand, ingested_sources)
    if not ok:
        errors.append(f"BRAND IDENTITY: {detail}")

    # ---- 1. SELF-PRESENCE ----
    branded_qs = [q for q in (query_master or []) if q.get("intent") == "branded"]
    if branded_qs:
        hits = sum(1 for q in branded_qs if tb in _norm(q.get("query", "")))
        rate = hits / len(branded_qs)
        if rate == 0:
            # W-69: 0% is not a warning. A branded query that never names the brand is
            # definitionally broken.
            errors.append(
                f"SELF-PRESENCE: '{target_brand}' appears in 0% of {len(branded_qs)} "
                f"branded queries. The brand name or entity scope is wrong.")
        elif rate < 0.5:
            warnings.append(
                f"SELF-PRESENCE: '{target_brand}' appears in only {rate:.0%} of "
                f"branded queries. Queries may be scoped to the wrong entity.")

    # ---- 2. PARENT-LEAKAGE: is the parent dominating instead of the target? ----
    if pb and pb != tb:
        parent_hits = sum(1 for q in (query_master or []) if pb in _norm(q.get("query", "")))
        target_hits = sum(1 for q in (query_master or []) if tb in _norm(q.get("query", "")))
        if parent_hits > target_hits:
            warnings.append(
                f"⚠️ PARENT-LEAKAGE: parent '{parent_brand}' appears in more queries "
                f"({parent_hits}) than target '{target_brand}' ({target_hits}). "
                f"Discovery may have anchored to the corporate level.")

    # ---- 3. COMPETITOR-LEVEL: are competitors corporate parents? ----
    parent_comps = [c for c in (competitors or [])
                    if any(p in _norm(c.get("brand", "")) for p in KNOWN_PARENTS)]
    if parent_comps and brand_is_product(target_brand):
        names = ", ".join(c.get("brand", "?") for c in parent_comps[:3])
        warnings.append(
            f"⚠️ COMPETITOR-LEVEL: {len(parent_comps)} competitor(s) look like corporate "
            f"parents ({names}). A product brand like '{target_brand}' should compete "
            f"with product brands, not holding companies.")

    # ---- 4. POSITIONING-COHERENCE: pillars mention target or only parent? ----
    all_pillars = []
    for layer in (positioning_layers or {}).values():
        if isinstance(layer, list):
            all_pillars += [_norm(p.get("statement", "") + " " + p.get("pillar", ""))
                            for p in layer]
    if all_pillars and pb and pb != tb:
        mentions_target = sum(1 for p in all_pillars if tb in p)
        if mentions_target == 0:
            warnings.append(
                f"⚠️ POSITIONING-LEVEL: no positioning pillar mentions '{target_brand}' "
                f"by name — positioning may describe the parent, not the product brand.")

    return warnings, errors

def verify_brand_in_sources(target_brand, ingested_sources, alias_overrides=None,
                            min_hits=3):
    """W-69: does the brand name, AS TYPED, actually appear in the ingested documents?

    Free, deterministic, and the single strongest guard against a misspelled brand name.
    Returns (ok: bool, detail: str).
    """
    from metrics_spec import build_registry, fold
    if not ingested_sources:
        return True, "no ingested sources to check against"
    pat = build_registry(target_brand, [], alias_overrides).target.pattern
    hits, checked, chars = 0, 0, 0
    for s in ingested_sources:
        txt = (s.get("text") if isinstance(s, dict) else getattr(s, "text", "")) or ""
        if not txt:
            continue
        checked += 1
        chars += len(txt)
        hits += len(pat.findall(fold(txt)))
    if checked == 0:
        return True, "no extractable text in the ingested sources"
    if hits >= min_hits:
        return True, f"'{target_brand}' found {hits}× across {checked} source(s)"
    return False, (
        f"'{target_brand}' appears {hits}× in {chars:,} characters across {checked} "
        f"ingested source(s). A brand's own documents name it constantly. This is "
        f"almost certainly a misspelling or the wrong entity — verify the brand name "
        f"before proceeding.")