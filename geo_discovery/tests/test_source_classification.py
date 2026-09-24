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
    "fda.gov", "nih.gov", "cdc.gov", "gov.uk", "health.gov.au",
    "harvard.edu", "chop.edu", "ox.ac.uk", "nhs.uk", "efsa.europa.eu",
])
def test_institutional_tlds_are_authority(domain):
    st, why = structural_type(domain)
    assert st == "editorial_authority", f"{domain} → {st} ({why})"

@pytest.mark.parametrize("domain", [
    "usdermatologypartners.com", "advanceddermatologymemphis.com",
    "drsmith.com", "westsidedermatologyclinic.com", "myskinpractice.co.uk",
    "cityhealthgroup.com", "lakesidemedicalcenter.com",
])
def test_practitioner_shapes_detected(domain):
    st, _ = structural_type(domain)
    assert st == "practitioner", f"{domain} classified {st}"

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
