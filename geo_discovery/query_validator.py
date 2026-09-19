"""Enforce brand-naming rules on the Query Master.

W-18: previously matched the raw brand string with a bespoke normaliser, while
measurement uses the full derived alias set from metrics_spec.build_registry. The
validator was therefore systematically WEAKER than the measurement it protects — a
query naming a model or sub-brand passed validation but registered as a mention.
Both sides now use the same registry.
"""
from metrics_spec import build_registry, fold

# categories ALLOWED to name the brand
BRAND_ALLOWED = {"branded", "comparative"}
# categories that MUST name the brand — a branded query that doesn't is broken
BRAND_REQUIRED = {"branded"}

def _norm(t):
    return fold(t)

def validate_queries(queries, target_brand, parent_brand=None, alias_overrides=None):
    """Returns (cleaned_queries, warnings, errors).

    warnings — advisory, human decides.
    errors   — structurally broken; the emit gate should block on these.
    """
    reg = build_registry(target_brand, [], alias_overrides)
    tpat = reg.target.pattern
    ppat = build_registry(parent_brand, [], None).target.pattern if parent_brand else None

    warnings, errors, cleaned = [], [], []
    n_branded = n_branded_named = 0

    for q in (queries or []):
        intent = q.get("intent", "")
        qtext = q.get("query", "") or ""
        low = fold(qtext)
        has_target = bool(tpat.search(low))
        has_parent = bool(ppat.search(low)) if ppat else False

        if intent in BRAND_REQUIRED:
            n_branded += 1
            if has_target:
                n_branded_named += 1
                q.pop("_brand_missing", None)
            else:
                # W-18/W-69: the missing half of the rule. A branded query that does
                # not name the brand measures nothing.
                errors.append(f"'{intent}' query does NOT name '{target_brand}' "
                              f"(branded queries must): '{qtext}'")
                q["_brand_missing"] = True
            cleaned.append(q)
            continue

        if intent in BRAND_ALLOWED:
            cleaned.append(q)
            continue

        if has_target or has_parent:
            which = target_brand if has_target else parent_brand
            warnings.append(f"'{intent}' query names '{which}' (should be brand-free): "
                            f"'{qtext}'")
            q["_brand_leakage"] = True
        cleaned.append(q)

    # ---- corpus-level check: 0% self-presence is an identity failure, not a nit ----
    if n_branded and n_branded_named == 0:
        errors.insert(0,
            f"IDENTITY FAILURE: '{target_brand}' appears in 0 of {n_branded} branded "
            f"queries. The brand name is almost certainly wrong, or scoped to the wrong "
            f"entity. Measurement will report 0% mention on every run. Stop and verify "
            f"the brand name before auditing.")

    return cleaned, warnings, errors

def leakage_summary(queries):
    from collections import Counter
    return {
        "leakage": dict(Counter(q.get("intent") for q in queries
                                if q.get("_brand_leakage"))),
        "branded_missing_brand": dict(Counter(q.get("intent") for q in queries
                                              if q.get("_brand_missing"))),
    }