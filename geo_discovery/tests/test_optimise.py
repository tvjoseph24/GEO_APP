import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimise.pillar_router import route_finding, route_all
from optimise.prioritizer import score_actions
from optimise.strategy import build_strategy, build_roadmap

def test_pillar_routing():
    assert route_finding({"experiment_id":"E3"}) == "Corroboration"
    assert route_finding({"experiment_id":"E7"}) == "Visual/Multimodal"
    assert route_finding({"experiment_id":"E4"}, {"pillar":"Content"}) == "Content"
    print("✅ pillar routing")

def test_prioritization():
    findings = [
        {"gap_id":"G024","experiment_id":"E4","verdict":"engine-specific",
         "pooled_lift_pp":55,"pillar":"Content"},
        {"gap_id":"G002","experiment_id":"E4","verdict":"validated",
         "pooled_lift_pp":10,"pillar":"Corroboration"}]
    scored = score_actions(findings)
    assert all("priority_score" in f and "horizon" in f for f in scored)
    assert scored[0]["priority_score"] >= scored[-1]["priority_score"]  # sorted
    print(f"✅ prioritization: top={scored[0]['gap_id']} "
          f"(score {scored[0]['priority_score']}, {scored[0]['horizon']})")

def test_strategy_provisional():
    findings = [{"gap_id":"G024","verdict":"engine-specific","pillar":"Content",
                 "horizon":"Now","pooled_lift_pp":55,"gap_detail":{"hypothesis":"x"},
                 "per_engine":[{"engine":"azure/gpt-4o","significant":True,"lift_pp":100}]}]
    findings = score_actions(findings)
    s = build_strategy(findings, [], "Nescafé", "Nestlé")
    assert s["PROVISIONAL"] is True
    assert "azure/gpt-4o" in s["engine_overlays"]   # engine-specific → overlay
    print("✅ strategy: provisional flag + engine overlay")

def test_roadmap():
    findings = score_actions([{"gap_id":"G1","verdict":"validated","pillar":"Content",
                               "pooled_lift_pp":60}])
    rm = build_roadmap(findings)
    assert any(rm[h] for h in ["Now","Next","Later"])
    print("✅ roadmap: horizons populated")

if __name__ == "__main__":
    for fn in [test_pillar_routing, test_prioritization, test_strategy_provisional, test_roadmap]:
        try: fn()
        except Exception as e: print(f"❌ {fn.__name__}: {type(e).__name__}: {e}")