"""Enforce brand-naming rules: strip/flag brand from non-branded/comparative queries."""
import unicodedata

def _norm(t):
    if not t: return ""
    return unicodedata.normalize("NFKD", str(t)).encode("ascii","ignore").decode().lower()

# categories ALLOWED to name the brand
BRAND_ALLOWED = {"branded", "comparative"}

def validate_queries(queries, target_brand, parent_brand=None):
    """Flag/clean brand leakage. Returns (cleaned_queries, warnings)."""
    tb = _norm(target_brand)
    pb = _norm(parent_brand) if parent_brand else ""
    warnings = []
    cleaned = []
    for q in queries:
        intent = q.get("intent", "")
        qtext = q.get("query", "")
        low = _norm(qtext)
        has_brand = tb in low or (pb and pb in low)

        if intent in BRAND_ALLOWED:
            cleaned.append(q)                     # brand allowed here
        elif has_brand:
            # LEAKAGE: brand appears in a category that should be brand-free
            warnings.append(f"⚠️ '{intent}' query names the brand (should be brand-free): "
                            f"'{qtext}'")
            q["_brand_leakage"] = True            # flag for human review
            cleaned.append(q)                     # keep but flag (human decides)
        else:
            cleaned.append(q)                     # clean
    return cleaned, warnings

def leakage_summary(queries):
    """Count leakage flags per category."""
    from collections import Counter
    c = Counter(q.get("intent") for q in queries if q.get("_brand_leakage"))
    return dict(c)