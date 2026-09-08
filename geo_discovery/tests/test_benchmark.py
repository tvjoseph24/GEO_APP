import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from benchmark import tier_c

def _mock_df(n_win=3, n_lose=3):
    rows = []
    # winnable query: brand present-but-buried, ownable claim
    for i in range(10):
        rows.append({"query":"best instant coffee","engine":"g","intent":"need",
            "mentioned": True, "is_first": i<2, "refusal": False,
            "brand_mentions":{"Nescafé":{"first_index":50 if i>=2 else 5},
                              "Folgers":{"first_index":10}},
            "response_text":"","sources":[],"source_type_tally":{},"num_sources":0,"web_on":True})
    return pd.DataFrame(rows)

def test_winnability_rules():
    df = _mock_df()
    claims = [{"claim":"instant coffee rich aroma"}]
    w = tier_c.classify_winnability(df, "Nescafé", ["Folgers"], claims)
    assert "winnability" in w.columns and len(w) >= 1
    assert w.iloc[0]["winnability"] in ("winnable","commoditized","unwinnable")
    assert "confidence" in w.columns and "signals" in w.columns
    print(f"✅ winnability: {w.iloc[0]['winnability']} (conf {w.iloc[0]['confidence']}) "
          f"— {w.iloc[0]['rule_reason']}")

def test_transparency():
    """Every classification carries triggering signals (design D-ii)."""
    df = _mock_df()
    w = tier_c.classify_winnability(df, "Nescafé", ["Folgers"], [{"claim":"coffee"}])
    assert all("signals" in r and "rule_reason" in r for _, r in w.iterrows())
    print("✅ transparency: all rows carry signals + reason")

if __name__ == "__main__":
    for fn in [test_winnability_rules, test_transparency]:
        try: fn()
        except Exception as e: print(f"❌ {fn.__name__}: {type(e).__name__}: {e}")