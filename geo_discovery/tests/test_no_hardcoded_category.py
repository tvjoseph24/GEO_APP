"""Fail if any module hardcodes a domain, brand or category-specific term.

Such values must be DERIVED from artifacts (audit, claims register, hit-list) or
configured per engagement. This has leaked four times already: skincare QUANT patterns,
brand_owned domains, DEFAULT_CITE_DOMAINS, and brand tokens in stopword sets.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

# config.py holds analyst-curated infrastructure; these scripts are engagement-specific.
ALLOWED_FILES = {"config.py", "backfill.py", "diag_optimise.py", "dump_results.py",
                 "reanalyse.py", "smoke_report.py", "test_validate.py",
                 "run_commands.py"}          # gitignored local helper, not shipped

# SOURCE_TYPE_MAP and KNOWN_PARENTS are analyst-curated category infrastructure.
# variant_gen holds a JSON schema example. Flagged as WARNINGS until they move to the
# Category Profile (W-07), not hard failures.
SOFT_FILES = {"audit_runner.py", "entity_check.py", "variant_gen.py"}

DOMAIN = re.compile(r"""['"][a-z0-9][a-z0-9\-]{2,}\.(?:com|org|net|uk|gov|edu|io|de|au|"""
                    r"""in|ie|ch|my|ca|store)['"]""")
BRANDS = re.compile(
    r"\b(aptamil|aptaclub|nescafe|nescaf|garnier|aveeno|cerave|cetaphil|enfamil|similac|"
    r"kendamil|bobbie|hipp|holle|nutricia|danone|mamia|kirkland)\b", re.I)
CATEGORY_TERMS = re.compile(
    r"\b(colloidal oatmeal|ceramide|urea|scgos|lcfos|pronutra|palm[- ]oil[- ]free|"
    r"follow[- ]on milk|toddler milk|breastmilk|paediatric|pediatric)\b", re.I)

def _scan():
    hard, soft = [], []
    for f in sorted(ROOT.rglob("*.py")):
        rel = f.relative_to(ROOT)
        if f.name in ALLOWED_FILES or "tests" in rel.parts or ".venv" in rel.parts:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8",
                                             errors="ignore").splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or not s:
                continue
            hits = []
            if DOMAIN.search(line):
                hits.append("domain")
            if BRANDS.search(line):
                hits.append("brand")
            if CATEGORY_TERMS.search(line):
                hits.append("category-term")
            if hits:
                entry = f"{rel}:{i} [{','.join(hits)}] {s[:95]}"
                (soft if f.name in SOFT_FILES else hard).append(entry)
    return hard, soft

def test_no_hardcoded_brands_or_categories():
    hard, soft = _scan()
    if soft:
        print(f"\nSOFT ({len(soft)}) — category infrastructure, move to config eventually:")
        for s in soft[:15]:
            print("  " + s)
    assert not hard, (
        "\nHardcoded brand / category / domain values found. These must be derived from "
        "artifacts or configured per engagement:\n" + "\n".join("  " + h for h in hard))

def test_metrics_spec_derives_aliases():
    """Aliases must be derived from the brand string, never enumerated per brand."""
    from metrics_spec import derive_aliases
    for name in ("Aptamil", "Nescafé", "Cow & Gate", "Else Nutrition"):
        al = derive_aliases(name)
        assert al, f"no aliases derived for {name}"
        assert all(isinstance(a, str) for a in al)
    assert "nescafe" in derive_aliases("Nescafé"), "accent folding failed"
    assert "cow and gate" in derive_aliases("Cow & Gate"), "ampersand folding failed"

def test_brand_stop_is_derived():
    """Stopword sets must be built from the brand/category at runtime."""
    from metrics_spec import brand_stop
    s = brand_stop("Nescafe", [{"brand": "Folgers"}], "instant coffee")
    assert "nescafe" in s and "folgers" in s and "coffee" in s
    assert "prebiotic" not in s, "stopword set is contaminated with another category"

def test_citation_domains_refuses_to_invent():
    """With no hit-list, no audit and no gateway, citation domains must return EMPTY —
    never a hardcoded fallback. An invented citation invalidates an E3 experiment."""
    from experiment.variant_gen import citation_domains
    domains, src = citation_domains()
    assert domains == [], f"invented fallback domains: {domains}"
    assert src == "unavailable"

def test_richness_patterns_are_generic():
    """No brand or ingredient names in the reason-richness regexes."""
    from benchmark.richness import PATTERNS
    blob = " ".join(p for pats in PATTERNS.values() for p in pats).lower()
    for term in ("ceramide", "oatmeal", "aptamil", "scgos", "urea"):
        assert term not in blob, f"'{term}' hardcoded in richness PATTERNS"

def test_source_type_map_has_no_brand_buckets():
    """brand_owned and competitor_owned must be runtime-injected, never literal."""
    from audit_runner import SOURCE_TYPE_MAP
    assert SOURCE_TYPE_MAP.get("brand_owned") == [], \
        "brand_owned must be empty in the base map — inject per engagement"
    assert SOURCE_TYPE_MAP.get("competitor_owned") == [], \
        "competitor_owned must be empty in the base map — inject per engagement"