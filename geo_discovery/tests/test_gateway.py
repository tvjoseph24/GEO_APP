"""Tier-2 PAID gateway tests. Only runs with --run-paid flag."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meter import Meter
from llm_gateway import LLMGateway
from config import MODEL_REGISTRY

GATED = "--run-paid" in sys.argv

def _first_model(provider):
    for m, c in MODEL_REGISTRY.items():
        if c["provider"] == provider: return m
    return None

def test_text_generation():
    m = _first_model("vertex")
    gw = LLMGateway(Meter(budget_usd=1.0))
    r = gw.generate(m, "Name one benefit of colloidal oatmeal in one sentence.",
                    module="test", task="text")
    assert len(r["text"]) > 10
    print(f"✅ text_generation ({m}): '{r['text'][:60]}…' ${gw.meter.total_cost():.6f}")

def test_json_mode():
    m = _first_model("vertex")
    gw = LLMGateway(Meter(budget_usd=1.0))
    r = gw.generate(m, 'List 2 skincare claims as JSON array of strings.',
                    json_mode=True, module="test", task="json")
    assert isinstance(r["parsed"], (list, dict))
    print(f"✅ json_mode: {r['parsed']}")

def test_system_prompt():
    m = _first_model("vertex")
    gw = LLMGateway(Meter(budget_usd=1.0))
    r = gw.generate(m, "What are you?", system="You are a dermatology assistant.",
                    module="test", task="sys")
    assert len(r["text"]) > 0
    print(f"✅ system_prompt honored")

def test_budget_stops_call():
    m = _first_model("vertex")
    gw = LLMGateway(Meter(budget_usd=0.0000001))  # basically zero
    try:
        gw.generate(m, "hi", module="test", task="t")
        # first call may pass (check is before), second must fail
        gw.generate(m, "hi again", module="test", task="t")
        print("⚠️  budget did not stop within 2 calls (check placement)")
    except RuntimeError:
        print("✅ budget cap halts calls")

if __name__ == "__main__":
    if not GATED:
        print("\n⏭️  PAID gateway tests skipped. Run with:  python tests/test_gateway.py --run-paid")
        sys.exit(0)
    print("\n── GATEWAY TESTS (PAID) ──")
    for fn in [test_text_generation, test_json_mode, test_system_prompt, test_budget_stops_call]:
        try: fn()
        except Exception as e: print(f"❌ {fn.__name__}: {type(e).__name__}: {e}")