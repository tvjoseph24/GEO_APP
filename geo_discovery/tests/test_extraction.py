"""Tier-2 PAID extraction tests. Only with --run-paid."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from meter import Meter
from llm_gateway import LLMGateway
from ingestion import RawSource
from extraction import extract_positioning, extract_claims, dedupe_claims
from config import MODEL_REGISTRY

GATED = "--run-paid" in sys.argv

SAMPLE = ("Aveeno Eczema Therapy Moisturizing Cream contains 1% colloidal oatmeal "
          "and ceramides. It is fragrance-free, steroid-free, and accepted by the "
          "National Eczema Association. Clinically shown to improve hydration in one day.")

def _model():
    for m, c in MODEL_REGISTRY.items():
        if c["provider"] == "vertex": return m

def test_extract_claims():
    gw = LLMGateway(Meter(budget_usd=2.0))
    src = RawSource("doc", "spec.txt", text=SAMPLE)
    claims = extract_claims(gw, src, _model())
    assert isinstance(claims, list) and len(claims) >= 3
    texts = " ".join(c.get("claim", "").lower() for c in claims)
    assert "oatmeal" in texts or "fragrance" in texts
    assert all("legal_ok" in c for c in claims)   # schema check
    print(f"✅ extract_claims: {len(claims)} claims, ${gw.meter.total_cost():.4f}")
    for c in claims[:3]:
        print(f"    - {c.get('claim')} [{c.get('proof_type')}, legal={c.get('legal_ok')}]")

def test_extract_positioning():
    gw = LLMGateway(Meter(budget_usd=2.0))
    src = RawSource("doc", "brand.txt", text=SAMPLE)
    pos = extract_positioning(gw, src, _model())

    assert isinstance(pos, dict), f"expected dict, got {type(pos)}"
    assert "pillars" in pos, f"missing 'pillars'. keys={list(pos.keys())}"
    # each pillar must carry its OWN valid layer
    for p in pos["pillars"]:
        assert p.get("layer") in ("aspirational", "expressed", "substantiated"), \
            f"pillar has bad layer: {p.get('layer')}"
    layers_found = {p.get("layer") for p in pos["pillars"]}
    print(f"✅ extract_positioning: {len(pos['pillars'])} pillars "
          f"spanning layers {layers_found}")

def test_dedupe():
    claims = [{"claim": "Fragrance-free", "_origin": "a"},
              {"claim": "fragrance-free ", "_origin": "b"},   # dup (case/space)
              {"claim": "1% colloidal oatmeal", "_origin": "a"}]
    out = dedupe_claims(claims)
    assert len(out) == 2   # the two fragrance-free merge
    merged = [c for c in out if "fragrance" in c["claim"].lower()][0]
    assert len(merged["_origins"]) == 2
    print("✅ dedupe merges cross-source duplicates")

if __name__ == "__main__":
    # dedupe is free — always run it
    print("\n── EXTRACTION: free tests ──")
    try: test_dedupe()
    except Exception as e: print(f"❌ dedupe: {e}")

    if not GATED:
        print("⏭️  PAID extraction tests skipped. Run with: --run-paid")
        sys.exit(0)
    print("\n── EXTRACTION: PAID tests ──")
    for fn in [test_extract_claims, test_extract_positioning]:
        try: fn()
        except Exception as e: print(f"❌ {fn.__name__}: {type(e).__name__}: {e}")