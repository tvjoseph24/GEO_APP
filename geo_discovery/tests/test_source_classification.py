"""W-56 guards. Pure functions — no LLM, no network."""
import pytest

from audit_runner import _pattern_matches, build_source_classifier
from source_profile import (SOURCE_TYPES, CORROBORATING, UNCONTROLLED, pack_lookup,
                            structural_type)

# ---------------------------------------------------------------- matching
def test_full_domain_requires_exact_or_subdomain():
    assert _pattern_matches("x.com", "x.com")
    assert _pattern_matches("x.com", "news.x.com")
    assert not _pattern_matches("x.com", "carfax.com"), \
        "substring matching made 'x.com' classify carfax.com as social"
    assert not _pattern_matches("x.com", "xx.com")

def test_fragment_patterns_still_substring_match():
    """Several base-map entries are fragments by design."""
    assert _pattern_matches("pubmed", "pubmed.ncbi.nlm.nih.gov")
    assert _pattern_matches("cdn.", "cdn.example.com")
    assert _pattern_matches("pinterest.", "pinterest.co.uk")

def test_amazon_does_not_capture_unrelated_hosts():
    f = build_source_classifier()
    assert f("amazon.com") == "retailer"
    assert f("amazon.co.uk") == "retailer"
    assert f("notamazonatall.org") != "retailer"

# ---------------------------------------------------------------- structural
@pytest.mark.parametrize("domain", [
    "fda.gov", "nih.gov", "cdc.gov", "nhtsa.gov",       # bare .gov
    "gov.uk", "health.gov.au", "canada.gc.ca",          # institutional label is leftmost
    "digital.gouv.fr", "mhlw.go.jp",
    "harvard.edu", "chop.edu", "mit.edu", "ox.ac.uk", "unsw.edu.au",
    "who.int",
])
def test_institutional_tlds_are_authority(domain):
    """Restricted registrations only — the TLD itself is the credential, in any category."""
    st, why = structural_type(domain)
    assert st == "editorial_authority", f"{domain} → {st} ({why})"

@pytest.mark.parametrize("domain", [
    "bundesbank.de",       # central bank on an open TLD
    "fca.org.uk",          # statutory regulator on an open TLD
    "nice.org.uk",
    "europa.eu",           # .eu is open
    "mayoclinic.org",
])
def test_authority_on_an_open_tld_is_declined(domain):
    """Deliberate. Recognising a statutory body on .de, .org.uk or .eu requires knowing
    the institution — same principle that keeps profession names out of the practitioner
    rule. These are base-map or pack entries, not structural rules."""
    assert structural_type(domain)[0] is None

def test_base_map_covers_the_open_tld_authorities():
    """Declining structurally is only acceptable because the fallback catches them."""
    f = build_source_classifier()
    for d in ("nhs.uk", "nice.org.uk", "mayoclinic.org", "europa.eu"):
        assert f(d) == "editorial_authority", f"{d} fell through to {f(d)}"

@pytest.mark.parametrize("domain", [
    # firm forms, concatenated — must hold across categories
    "usdermatologypartners.com",        # health
    "smithandassociates.com",           # professional services
    "mortgage-advisers.co.uk",          # financial
    "acmeconsulting.com",               # B2B
    "harcourtchambers.co.uk",           # legal
    "willisllp.com",
    # premises forms, delimited
    "westside-clinic.com",
    "north.practice.co.uk",
    "kings-workshop.com",               # automotive / trade
    "atelier.bureau.fr",
])
def test_professional_firm_forms_detected(domain):
    st, why = structural_type(domain)
    assert st == "practitioner", f"{domain} → {st} ({why})"

@pytest.mark.parametrize("domain", [
    # a profession's NAME is category knowledge — the pack decides, not a regex
    "advanceddermatologymemphis.com",
    "westsidedermatologyclinic.com",    # 'clinic' concatenated, no delimiter
    "lakesidemedicalcenter.com",
    "cityhealthgroup.com",
    # and these must never be practitioners
    "drugstore.com",                    # an honorific prefix matched this once
    "bestpractice.com",                 # 'practice' inside an ordinary word
    "dreamteam.com", "drivetime.com",
])
def test_structural_layer_declines_when_it_needs_category_knowledge(domain):
    """Deliberate. The free layer detects organisational FORM only; who practises in a
    category is derived by the source pack. Declining costs one LLM classification;
    guessing costs a wrong bucket in every category the vocabulary does not fit."""
    st, _ = structural_type(domain)
    assert st != "practitioner", f"{domain} was guessed as a practitioner"

def test_category_profile_can_extend_firm_forms():
    """W-07 hook: a category adds its own forms without a code change."""
    assert structural_type("northsidegarage.de")[0] is None
    st, _ = structural_type("northsidegarage.de",
                            extra_practitioner_forms=("garage",))
    assert st == "practitioner"

def test_structural_layer_contains_no_profession_vocabulary():
    """The guard for this defect: a profession name in the free layer is a hardcoded
    category term, and it makes the rule blind to every other category."""
    import source_profile as SP
    blob = " ".join(SP._FORM_SUFFIX + SP._FORM_DELIMITED).lower()
    forbidden = ("dermatolog", "dentist", "orthodont", "optometr", "podiatr",
                 "veterinar", "medical", "medspa", "health", "pediatr", "paediatr",
                 "physio", "chiro", "pharma", "nutrition", "derma")
    hits = [w for w in forbidden if w in blob]
    assert not hits, (f"profession vocabulary in the structural layer: {hits}. "
                      f"Move it to the source pack, where the classifier derives it "
                      f"per category.")


@pytest.mark.parametrize("domain", [
    "cdn.example.com", "static.brand.com", "d1x.cloudfront.net",
    "img.example.org", "x.imgix.net",
])
def test_cdn_hosts_detected(domain):
    st, _ = structural_type(domain)
    assert st == "cdn_asset"

def test_structural_declines_when_uncertain():
    """A generic commercial hostname must fall through to the pack, not be guessed."""
    for d in ("skinsort.com", "nbcnews.com", "ulta.com", "someblog.com"):
        st, _ = structural_type(d)
        assert st is None, f"{d} was guessed as {st} by a structural rule"

@pytest.mark.parametrize("domain,expect", [
    ("nhtsa.gov", "editorial_authority"),        # automotive safety regulator
    ("sec.gov", "editorial_authority"),          # financial regulator
    ("mit.edu", "editorial_authority"),          # academia
    ("ec.europa.eu", None),                      # open TLD → pack
    ("smithandassociates.com", "practitioner"),  # professional firm, any category
    ("kings-workshop.com", "practitioner"),      # trade premises
    ("cdn.example.com", "cdn_asset"),
    ("someblog.com", None),
])
def test_structural_rules_hold_for_unseen_categories(domain, expect):
    """The same rules across automotive, financial, academic, professional-services and
    trade — no category vocabulary anywhere, and declining where it would be needed."""
    assert structural_type(domain)[0] == expect

def test_structural_is_category_independent():
    """No category vocabulary in the structural layer — the same rules must hold for a
    category the code has never seen."""
    assert structural_type("nhtsa.gov")[0] == "editorial_automatic" or \
           structural_type("nhtsa.gov")[0] == "editorial_authority"
    assert structural_type("mit.edu")[0] == "editorial_authority"

# ---------------------------------------------------------------- precedence
def _pack(**kw):
    return {"pack_version": 1,
            "entries": {d: {"domain": d, "source_type": st, "confidence": 0.9,
                            "reviewed": False} for d, st in kw.items()}}

def test_pack_classifies_the_long_tail():
    f = build_source_classifier(source_pack=_pack(**{
        "skinsort.com": "content_affiliate", "nbcnews.com": "editorial_authority"}))
    assert f("skinsort.com") == "content_affiliate"
    assert f("nbcnews.com") == "editorial_authority"
    assert f("www.skinsort.com".replace("www.", "")) == "content_affiliate"

def test_pack_matches_subdomains():
    f = build_source_classifier(source_pack=_pack(**{"example.com": "retailer"}))
    assert f("shop.example.com") == "retailer"

def test_brand_beats_the_pack():
    """A brand domain misclassified as affiliate would understate self-reference and
    inflate uncontrolled voice — the most consequential misclassification available."""
    f = build_source_classifier(brand_domains=["mybrand.com"],
                                source_pack=_pack(**{"mybrand.com": "content_affiliate"}))
    assert f("mybrand.com") == "brand_owned"
    assert f("uk.mybrand.com") == "brand_owned"

def test_competitor_beats_the_pack():
    f = build_source_classifier(competitor_domains=["rival.com"],
                                source_pack=_pack(**{"rival.com": "retailer"}))
    assert f("rival.com") == "competitor_owned"

def test_structural_beats_the_pack():
    """A model calling a .gov site affiliate must not override a deterministic rule."""
    f = build_source_classifier(source_pack=_pack(**{"fda.gov": "content_affiliate"}))
    assert f("fda.gov") == "editorial_authority"

def test_pack_lookup_respects_confidence_and_review():
    p = {"entries": {
        "a.com": {"source_type": "content_affiliate", "confidence": 0.4,
                  "reviewed": False},
        "b.com": {"source_type": "content_affiliate", "confidence": 0.4,
                  "reviewed": True},
        "c.com": {"source_type": "other", "confidence": 1.0, "reviewed": True}}}
    lk = pack_lookup(p, min_confidence=0.6)
    assert "a.com" not in lk, "a low-confidence unreviewed entry must not be applied"
    assert lk["b.com"] == "content_affiliate", "a human decision outranks a score"
    assert "c.com" not in lk, "'other' is not a classification"

# ---------------------------------------------------------------- provenance
def test_practitioner_excluded_from_uncontrolled():
    assert "practitioner" not in UNCONTROLLED
    assert "practitioner" not in CORROBORATING

def test_bucket_vocabularies_agree():
    from audit_runner import SOURCE_TYPE_MAP
    extra = set(SOURCE_TYPE_MAP) - set(SOURCE_TYPES) - {"brand_owned", "competitor_owned"}
    assert not extra, f"base map has buckets source_profile does not define: {extra}"
    assert CORROBORATING <= set(SOURCE_TYPES)
    assert UNCONTROLLED <= set(SOURCE_TYPES)

def test_voice_control_ratio_withheld_when_unreliable():
    import pandas as pd
    from benchmark.tier_b import claim_provenance
    df = pd.DataFrame([{"engine": "e1",
                        "source_type_tally": {"brand_owned": 10, "other": 90}}])
    pv = claim_provenance(df)
    assert pv["voice_control_ratio"] is None, \
        "a ratio computed over 90% unclassified citations must not be reported"
    assert pv["voice_control_ratio_provisional"] == 10.0
    assert pv["reliability"]["reliable"] is False

def test_voice_control_ratio_reported_when_reliable():
    import pandas as pd
    from benchmark.tier_b import claim_provenance
    df = pd.DataFrame([{"engine": "e1",
                        "source_type_tally": {"brand_owned": 10,
                                              "content_affiliate": 40,
                                              "editorial_authority": 45,
                                              "other": 5}}])
    pv = claim_provenance(df)
    assert pv["reliability"]["reliable"] is True
    assert pv["voice_control_ratio"] == 0.25
    assert pv["practitioner_pct"] == 0.0
