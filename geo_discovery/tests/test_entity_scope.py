import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from entity_check import check_entity_alignment
from store import save_artifact, list_brands

def test_detects_parent_competitor():
    w = check_entity_alignment(
        "Nescafé",                                          # target_brand
        [{"query": "best nescafé", "intent": "branded"}],   # query_master
        [{"brand": "Unilever"}, {"brand": "PepsiCo"}],      # competitors (parents = mismatch)
        {"aspirational": [{"pillar": "coffee", "statement": "nescafé coffee"}]},  # positioning
        parent_brand="Nestlé")                              # ← keyword
    assert any("COMPETITOR-LEVEL" in x for x in w), f"expected competitor warning, got {w}"
    print("✅ detects corporate competitors for product brand")

def test_clean_passes():
    w = check_entity_alignment(
        "Nescafé",
        [{"query": "best nescafé instant coffee", "intent": "branded"}],
        [{"brand": "Folgers"}, {"brand": "Starbucks"}],     # product competitors — clean
        {"aspirational": [{"pillar": "c", "statement": "nescafé rich coffee"}]},
        parent_brand="Nestlé")
    assert w == [], f"expected clean, got {w}"
    print("✅ clean brand-scoped discovery passes")

def test_corporate_level_skips():
    w = check_entity_alignment(
        "Nestlé",
        [{"query": "nestlé sustainability", "intent": "branded"}],
        [{"brand": "Unilever"}],
        {},
        parent_brand="Nestlé", brand_level="corporate")     # ← corporate skips checks
    assert w == [], f"corporate should skip, got {w}"
    print("✅ corporate-level intentionally skips checks")

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