"""Free unit tests for the human-gate loop (mock regenerator, no LLM)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from human_gate import HumanGate

def make_gate():
    calls = {"n": 0}
    def regen(feedback=None):
        calls["n"] += 1
        return {"version": calls["n"], "feedback_used": feedback}
    return HumanGate("test", regen), calls

def test_generate_then_approve():
    g, _ = make_gate()
    g.generate()
    assert g.state.status == "awaiting"
    g.approve("analyst")
    assert g.is_final() and g.state.status == "approved"
    print("✅ generate→approve")

def test_edit_becomes_source_of_truth():
    g, _ = make_gate()
    g.generate()
    g.edit({"manual": True})
    assert g.state.artifact == {"manual": True} and g.is_final()
    print("✅ edit overrides artifact")

def test_reject_feedback_loop():
    g, calls = make_gate()
    g.generate()                              # v1
    g.reject_and_regenerate("add NEA seal")   # v2 with feedback
    assert g.state.artifact["version"] == 2
    assert g.state.artifact["feedback_used"] == "add NEA seal"
    assert len(g.state.feedback_history) == 1
    assert not g.is_final()                    # still awaiting
    g.approve()
    assert g.is_final()
    print("✅ reject→regenerate→approve loop")

def test_multiple_feedback_rounds():
    g, _ = make_gate()
    g.generate()
    g.reject_and_regenerate("round1")
    g.reject_and_regenerate("round2")
    assert len(g.state.feedback_history) == 2
    assert g.state.artifact["version"] == 3   # 1 gen + 2 regen
    print("✅ multiple feedback rounds tracked")

def test_audit_record():
    g, _ = make_gate()
    g.generate(); g.reject_and_regenerate("fix"); g.approve("legal_team")
    rec = g.audit_record()
    assert rec["approver"] == "legal_team" and rec["feedback_rounds"] == 1
    print("✅ audit_record captures approver + rounds")

if __name__ == "__main__":
    print("\n── HUMAN-GATE TESTS (free) ──")
    for fn in [test_generate_then_approve, test_edit_becomes_source_of_truth,
               test_reject_feedback_loop, test_multiple_feedback_rounds,
               test_audit_record]:
        try: fn()
        except AssertionError as e: print(f"❌ {fn.__name__}: {e}")