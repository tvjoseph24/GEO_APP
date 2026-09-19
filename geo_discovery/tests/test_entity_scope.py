import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entity_check import check_entity_alignment
from store import save_artifact, list_brands

def test_detects_parent_competitor():
    w, e = check_entity_alignment(
        "Nescafé",
        [{"query": "best nescafé", "intent": "branded"}],
        [{"brand": "Unilever"}, {"brand": "PepsiCo"}],
        {"aspirational": [{"pillar": "coffee", "statement": "nescafé coffee"}]},
        parent_brand="Nestlé")
    assert any("COMPETITOR-LEVEL" in x for x in w), f"expected competitor warning, got {w}"

def test_clean_passes():
    w, e = check_entity_alignment(
        "Nescafé",
        [{"query": "best nescafé instant coffee", "intent": "branded"}],
        [{"brand": "Folgers"}, {"brand": "Starbucks"}],
        {"aspirational": [{"pillar": "c", "statement": "nescafé rich coffee"}]},
        parent_brand="Nestlé")
    assert w == [], f"expected clean, got {w}"
    assert e == [], f"expected no errors, got {e}"

def test_corporate_level_skips():
    w, e = check_entity_alignment(
        "Nestlé",
        [{"query": "nestlé sustainability", "intent": "branded"}],
        [{"brand": "Unilever"}],
        {},
        parent_brand="Nestlé", brand_level="corporate")
    assert w == [] and e == [], f"corporate should skip, got {w}, {e}"

def test_zero_self_presence_is_an_error():
    """W-69: a branded query that never names the brand is definitionally broken."""
    w, e = check_entity_alignment(
        "Aveenao",                                          # deliberate typo
        [{"query": "Does Aveeno's Active Naturals work?", "intent": "branded"}],
        [{"brand": "CeraVe"}], {}, parent_brand="Kenvue")
    assert any("SELF-PRESENCE" in x for x in e), f"expected blocking error, got {e}"

def test_document_reconciliation_catches_typo():
    from entity_check import verify_brand_in_sources
    srcs = [{"text": "Aveeno Daily Moisturising Lotion contains colloidal oatmeal. "
                     "Aveeno is dermatologist recommended. Aveeno Baby is gentle."}]
    ok, _ = verify_brand_in_sources("Aveeno", srcs)
    assert ok
    bad, detail = verify_brand_in_sources("Aveenao", srcs)
    assert not bad and "misspelling" in detail

def test_portfolio_path():
    save_artifact("Nestlé", "Nescafé", "test", {"x": 1})
    brands = list_brands("Nestlé")
    assert any("Nesca" in b for b in brands), f"expected Nescafé, got {brands}"
    print(f"✅ portfolio path: brands under Nestlé = {brands}")

if __name__ == "__main__":
    for fn in [test_detects_parent_competitor, test_clean_passes,
               test_corporate_level_skips, test_portfolio_path]:
        try: fn()
        except Exception as e: print(f"❌ {fn.__name__}: {type(e).__name__}: {e}")