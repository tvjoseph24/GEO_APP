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
                           brand_level="product"):
    """
    Validate that Discovery is coherently scoped to the TARGET (product) brand.

    Args:
        target_brand:       the product brand being audited (e.g., "Nescafé")
        query_master:       list of {query, intent, ...}
        competitors:        list of {brand, ...}
        positioning_layers: {aspirational:[...], expressed:[...], substantiated:[...]}
        parent_brand:       corporate parent for context (e.g., "Nestlé")
        brand_level:        "product" (run checks) | "corporate" (skip checks)

    Returns:
        list[str]: advisory warnings. Empty = clean. Non-blocking.
    """
    # Corporate-level engagements legitimately use corporate competitors/positioning
    if brand_level == "corporate":
        return []

    warnings = []
    tb = _norm(target_brand)
    pb = _norm(parent_brand) if parent_brand else ""

    # ---- 1. SELF-PRESENCE: target appears in its own branded queries? ----
    branded_qs = [q for q in (query_master or []) if q.get("intent") == "branded"]
    if branded_qs:
        hits = sum(1 for q in branded_qs if tb in _norm(q.get("query", "")))
        rate = hits / len(branded_qs)
        if rate < 0.5:
            warnings.append(
                f"⚠️ SELF-PRESENCE: '{target_brand}' appears in only {rate:.0%} of "
                f"branded queries. Queries may be scoped to the wrong entity "
                f"(e.g., parent company instead of the product brand).")

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

    return warnings