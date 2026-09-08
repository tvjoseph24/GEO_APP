"""Free unit tests for the cost/time meter + budget guard."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meter import Meter

def test_cost_calculation():
    from config import PRICING
    m = Meter(budget_usd=100)
    model = "gemini-2.5-flash"
    m.record(model, "vertex", 1_000_000, 1_000_000, 1.0, "test", "t")
    pin, pout = PRICING[model]                      # ← read actual configured prices
    expected = 1_000_000 * pin + 1_000_000 * pout   # ← compute expected from them
    assert abs(m.total_cost() - expected) < 1e-9, \
        f"got {m.total_cost()}, expected {expected} (prices {pin},{pout})"
    print(f"✅ cost_calculation (${expected:.4f} at current rates)")

def test_budget_guard():
    m = Meter(budget_usd=0.10)
    # force a cost above cap regardless of which model's price is configured
    m.events.append({"module":"t","task":"t","model":"x","provider":"t",
                     "tok_in":0,"tok_out":0,"seconds":0,"cost_usd":1.00})
    try:
        m.check_budget(); print("❌ budget_guard did NOT trigger")
    except RuntimeError:
        print("✅ budget_guard triggers over cap")

def test_usage_extraction_vertex():
    class FakeUM: prompt_token_count = 50; candidates_token_count = 20
    class FakeResp: usage_metadata = FakeUM()
    tin, tout = Meter().extract_usage(FakeResp(), "vertex")
    assert (tin, tout) == (50, 20)
    print("✅ usage_extraction_vertex")

def test_usage_extraction_litellm():
    class FakeResp: usage = {"prompt_tokens": 30, "completion_tokens": 12}
    tin, tout = Meter().extract_usage(FakeResp(), "litellm")
    assert (tin, tout) == (30, 12)
    print("✅ usage_extraction_litellm")

def test_summary_by_task():
    m = Meter()
    m.record("gemini-2.5-flash", "vertex", 100, 50, 1.0, "discover", "positioning")
    m.record("gemini-2.5-flash", "vertex", 200, 80, 1.5, "discover", "positioning")
    m.record("gemini-2.5-flash", "vertex", 100, 40, 0.8, "discover", "claims")
    df = m.summary_by_task()
    pos = df[df.task == "positioning"].iloc[0]
    assert pos["calls"] == 2 and pos["tok_in"] == 300
    print("✅ summary_by_task aggregates correctly")

if __name__ == "__main__":
    print("\n── METER TESTS (free) ──")
    for fn in [test_cost_calculation, test_budget_guard,
               test_usage_extraction_vertex, test_usage_extraction_litellm,
               test_summary_by_task]:
        try: fn()
        except AssertionError as e: print(f"❌ {fn.__name__}: {e}")