import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment.gap_mapper import map_gap_to_experiment, propose_experiments
from experiment.exp_stats import analyze_experiment, verdict

def test_gap_mapping():
    assert map_gap_to_experiment({"delta_type":"hallucination","gap":"x"}) == "E3"
    assert map_gap_to_experiment({"delta_type":"under-surfaced","gap":"winnable query camping"}) == "E4"
    assert map_gap_to_experiment({"delta_type":"missing","gap":"tastes bad"}) == "E9"
    print("✅ gap→experiment mapping")

def test_propose():
    gaps = [{"gap_id":"G019","gap":"Winnable query: camping coffee",
             "delta_type":"under-surfaced","hypothesis":"add claim"}]
    d = propose_experiments(gaps)
    assert d[0]["experiment_id"] == "E4" and d[0]["include"]
    print(f"✅ propose: {d[0]['experiment_name']}")

def test_stats():
    er = {"gap_id":"G1","experiment_id":"E4","engines":{
        "gemini":{"variant_A":{"is_first":10,"mentioned":20,"n":30},
                  "variant_B":{"is_first":25,"mentioned":30,"n":30}}}}
    rows = analyze_experiment(er, "is_first")
    assert rows[0]["lift_pp"] > 0
    v = verdict(rows)
    print(f"✅ stats: lift={rows[0]['lift_pp']}pp, p_fdr={rows[0]['p_fdr']}, verdict={v}")

if __name__ == "__main__":
    for fn in [test_gap_mapping, test_propose, test_stats]:
        try: fn()
        except Exception as e: print(f"❌ {fn.__name__}: {type(e).__name__}: {e}")