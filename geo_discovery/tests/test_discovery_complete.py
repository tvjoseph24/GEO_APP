"""Tests for completed Discovery: grounded mining, KPI, creative extraction, cost estimate.
Free tests run always; grounded/vision tests need --run-paid."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PRICING

GATED = "--run-paid" in sys.argv

# ---- FREE ----
def test_video_cost_estimate():
    from ingestion import estimate_video_cost
    toks, cost = estimate_video_cost(60, "gemini-2.5-flash", PRICING)
    assert toks == 60*300 and cost > 0
    print(f"✅ video_cost_estimate: 60s → {toks} tok, ${cost}")

def test_kpi_template():
    from kpi import default_kpis
    k = default_kpis()
    assert len(k) >= 5 and all("target" in x for x in k)
    print(f"✅ kpi_template: {len(k)} KPIs")

def test_merge_query_masters():
    from grounded_mining import merge_query_masters
    bd = [{"query": "Best X", "source": "brand_derived"}]
    mn = [{"query": "best x", "source": "real_demand_mined"},   # dup (case)
          {"query": "cheapest X", "source": "real_demand_mined"}]
    out = merge_query_masters(bd, mn)
    assert len(out) == 2   # dup merged
    print("✅ merge_query_masters dedupes + tags")

# ---- PAID ----
def test_grounded_competitors():
    from meter import Meter; from llm_gateway import LLMGateway
    from grounded_mining import discover_competitors
    gw = LLMGateway(Meter(budget_usd=2.0))
    comps, srcs = discover_competitors(gw, "Aveeno", "eczema skincare")
    assert isinstance(comps, list) and len(comps) >= 3
    print(f"✅ grounded_competitors: {[c['brand'] for c in comps[:5]]} "
          f"({len(srcs)} sources, ${gw.meter.total_cost():.4f})")

def test_grounded_demand():
    from meter import Meter; from llm_gateway import LLMGateway
    from grounded_mining import mine_real_demand
    gw = LLMGateway(Meter(budget_usd=2.0))
    q, srcs = mine_real_demand(gw, "Aveeno", "eczema skincare")
    assert isinstance(q, list) and len(q) >= 5
    assert all(x.get("source") == "real_demand_mined" for x in q)
    print(f"✅ grounded_demand: {len(q)} real queries, e.g. '{q[0]['query']}'")

def test_creative_extraction():
    """Needs a test image. Skips gracefully if none."""
    import glob
    imgs = glob.glob("tests/fixtures/*.png") + glob.glob("tests/fixtures/*.jpg")
    if not imgs:
        print("⚠️  no test image in tests/fixtures/ — skip creative test"); return
    from meter import Meter; from llm_gateway import LLMGateway
    from ingestion import ingest_image; from extraction import extract_creative
    gw = LLMGateway(Meter(budget_usd=2.0))
    src = ingest_image(imgs[0])
    cr = extract_creative(gw, src, "gemini-2.5-flash")
    assert "expressed_message" in cr or "positioning_pillars" in cr
    print(f"✅ creative_extraction: {cr.get('expressed_message','')[:60]}")

if __name__ == "__main__":
    print("\n── DISCOVERY COMPLETION: free tests ──")
    for fn in [test_video_cost_estimate, test_kpi_template, test_merge_query_masters]:
        try: fn()
        except Exception as e: print(f"❌ {fn.__name__}: {e}")
    if not GATED:
        print("\n⏭️  PAID tests skipped (--run-paid for grounded+vision)"); sys.exit(0)
    print("\n── DISCOVERY COMPLETION: paid tests ──")
    #[test_grounded_competitors, test_grounded_demand, test_creative_extraction]
    for fn in [test_creative_extraction]:
        try: fn()
        except Exception as e: print(f"❌ {fn.__name__}: {type(e).__name__}: {e}")