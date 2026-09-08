"""End-to-end smoke test: ingest text → extract claims → gate loop → save.
One real LLM call. Run with --run-paid."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "--run-paid" not in sys.argv:
    print("⏭️  smoke test needs --run-paid"); sys.exit(0)

from meter import Meter
from llm_gateway import LLMGateway
from ingestion import ingest_doc
from extraction import extract_claims, dedupe_claims
from human_gate import HumanGate
from store import save_artifact, save_audit_log

print("\n════ END-TO-END SMOKE TEST ════")
meter = Meter(budget_usd=2.0)
gw = LLMGateway(meter)

# 1. Ingest
src = ingest_doc(b"Aveeno Eczema Therapy: 1% colloidal oatmeal, fragrance-free, "
                 b"steroid-free, NEA accepted.", name="spec.txt")
print(f"1. Ingested: {src.origin} ({src.meta['chars']} chars)")

# 2. Extract + gate
def regen(feedback=None):
    return dedupe_claims(extract_claims(gw, src, "gemini-2.5-pro"))
gate = HumanGate("claims", regen)
gate.generate()
print(f"2. Extracted {len(gate.state.artifact)} claims (gate awaiting)")

# 3. Simulate human approve
gate.approve("test_analyst")
print(f"3. Gate {gate.state.status} by {gate.state.approver}")

# 4. Save
p = save_artifact("SmokeTest", "claims_register", gate.state.artifact,
                  {"extraction": "gemini-2.5-pro"})
save_audit_log("SmokeTest", [gate.audit_record()])
print(f"4. Saved → {p}")
print(f"\n💰 Total: ${meter.total_cost():.4f}, {meter.calls} calls, {meter.elapsed:.1f}s")
print("✅ SMOKE TEST PASSED")