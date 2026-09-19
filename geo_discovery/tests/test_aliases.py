"""Alias derivation and case-insensitive lookup. Covers W-12 and W-73."""
from metrics_spec import (brand_entry, brand_key, build_registry, derive_aliases,
                          brand_stop, fold, measure, same_brand)

def _has(name, wanted):
    al = derive_aliases(name)
    assert wanted in al, f"{name!r} -> {al}, missing {wanted!r}"

def test_apostrophe_insertion():
    _has("Childs Farm", "child's farm")
    _has("Child's Farm", "childs farm")

def test_parenthetical():
    _has("First Aid Beauty (FAB)", "first aid beauty")
    _has("First Aid Beauty (FAB)", "fab")

def test_slash():
    _has("Audi Q7/Q8", "audi q7")
    _has("Audi Q7/Q8", "audi q8")

def test_trailing_subbrand_only_when_two_tokens_remain():
    assert "burt's bees" in derive_aliases("Burt's Bees Baby")
    assert "range" not in derive_aliases("Range Rover")
    assert "childs" not in derive_aliases("Childs Farm")
    assert "la" not in derive_aliases("La Roche-Posay")

def test_accent_and_ampersand_preserved():
    assert "nescafe" in derive_aliases("Nescafé")
    assert "cow and gate" in derive_aliases("Cow & Gate")

def test_untracked_brands_now_match():
    for brand, text in (
            ("Childs Farm", "Child's Farm is gentle on sensitive skin."),
            ("First Aid Beauty (FAB)", "FAB Ultra Repair Cream is highly regarded."),
            ("Burt's Bees Baby", "Burt's Bees uses natural ingredients.")):
        reg = build_registry("Aveeno", [brand])
        assert reg.competitors[0].pattern.search(fold(text)), \
            f"{brand!r} did not match {text!r}"

def test_brand_key_is_case_insensitive():
    bm = {"Aveeno": {"mentioned": True, "first_index": 3}}
    assert brand_key(bm, "aveeno") == "Aveeno"
    assert brand_entry(bm, "aveeno")["mentioned"] is True
    assert brand_entry(bm, "CeraVe") == {}
    assert same_brand("Aveeno", "aveeno")
    assert not same_brand("Aveeno", "CeraVe")

def test_measure_via_slug_brand_name():
    reg = build_registry("aveeno", ["CeraVe"])
    m = measure("CeraVe is good, but Aveeno is gentler.", reg)
    assert m["mentioned"] is True
    assert m["first_brand"] == "CeraVe"
    assert m["rank"] == 2

def test_brand_stop_protects_spec_nouns():
    s = brand_stop("Range Rover", [{"brand": "BMW X7"}], "SUV")
    assert "rover" in s
    assert "range" not in s, "protected measurement noun was stopped"
    s2 = brand_stop("Aveeno", [], "skincare")
    assert "aveeno" in s2