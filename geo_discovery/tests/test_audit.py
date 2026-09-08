"""Audit tests. Free logic + gated paid runs.

python tests/test_audit.py            # free logic
python tests/test_audit.py --run-paid # mini live audit + LLM classify
streamlit run audit_app.py            # full UI

"""
import os, sys, tempfile, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GATED = "--run-paid" in sys.argv

# ---- FREE ----
def test_keyword_classifier():
    from classifier import classify_keyword
    assert classify_keyword("Try Aveeno for eczema.", "Aveeno", ["CeraVe"])["outcome"] == "brand_recommended"
    assert classify_keyword("CeraVe is best.", "Aveeno", ["CeraVe"])["outcome"] == "outranked"
    assert classify_keyword("Please consult a doctor.", "Aveeno", ["CeraVe"])["outcome"] == "refusal"
    print("✅ keyword classifier: 3 outcomes correct")

def test_cost_estimate():
    from audit_runner import estimate_cost
    calls, cost = estimate_cost(3, 2, 10, ("on","off"))
    assert calls == 3*2*10*2
    print(f"✅ cost_estimate: {calls} calls ≈ ${cost}")

def test_resumability():
    from audit_runner import _already_done
    tmp = tempfile.mktemp(suffix=".jsonl")
    with open(tmp, "w") as f:
        f.write(json.dumps({"query":"q1","engine":"g","web_on":True,"run_index":0,"success":True})+"\n")
    done = _already_done(tmp)
    assert ("q1","g",True,0) in done
    os.unlink(tmp)
    print("✅ resumability: completed runs detected")

def test_url_resolver_passthrough():
    from url_resolver import _resolve_one
    url, ok = _resolve_one("https://www.reddit.com/r/eczema")  # not a redirect
    assert ok and url == "https://www.reddit.com/r/eczema"
    print("✅ url_resolver: passthrough for real URLs")

# ---- PAID ----
def test_llm_classify():
    from meter import Meter; from llm_gateway import LLMGateway
    from classifier import classify_llm
    gw = LLMGateway(Meter(budget_usd=1.0))
    d = classify_llm(gw, "CeraVe and Cetaphil are top picks; Aveeno also works.",
                     "Aveeno", "gemini-2.5-flash")
    assert "outcome" in d and "sentiment_toward_target" in d
    print(f"✅ llm_classify: {d['outcome']}, sentiment={d.get('sentiment_toward_target')}")

def test_mini_audit():
    from meter import Meter; from llm_gateway import LLMGateway
    from audit_runner import run_audit
    from baseline import compute_snapshot
    gw = LLMGateway(Meter(budget_usd=2.0))
    tmp = tempfile.mktemp(suffix=".jsonl")
    run_audit(gw, [{"query":"best moisturizer for eczema","intent":"need"}],
              [{"name":"gemini-2.5-flash","model":"gemini-2.5-flash","provider":"vertex","supports":["text"]}],
              "Aveeno", ["CeraVe","Cetaphil"], n_runs=2, web_modes=("on",), out_path=tmp)
    snap = compute_snapshot(tmp, "Aveeno")
    print(f"✅ mini_audit: {snap}")
    os.unlink(tmp)

def test_chatgpt_websearch():
    """PAID: verify ChatGPT responses path captures real source URLs."""
    from meter import Meter; from llm_gateway import LLMGateway
    gw = LLMGateway(Meter(budget_usd=2.0))
    r = gw.generate_web_litellm("azure/gpt-4o",
                                "best drugstore moisturizer for eczema",
                                module="test", task="websearch")
    assert r["text"], "empty response"
    print(f"✅ chatgpt_websearch: {len(r['sources'])} sources, "
          f"{len(r['web_search_queries'])} queries")
    for s in r["sources"][:3]:
        assert "vertexaisearch" not in (s["url"] or ""), "should be real URL"
        print(f"    • {s['domain']}: {s['url'][:60]}")

if __name__ == "__main__":
    print("\n── AUDIT: free tests ──")
    for fn in [test_keyword_classifier, test_cost_estimate, test_resumability,
               test_url_resolver_passthrough]:
        try: fn()
        except Exception as e: print(f"❌ {fn.__name__}: {e}")
    if not GATED:
        print("⏭️  paid tests skipped (--run-paid)"); sys.exit(0)
    print("\n── AUDIT: paid tests ──")
    for fn in [test_llm_classify, test_mini_audit,test_chatgpt_websearch]:
        try: fn()
        except Exception as e: print(f"❌ {fn.__name__}: {type(e).__name__}: {e}")