"""
Tier-0 CONNECTION DIAGNOSTICS — run this first.
Verifies Vertex (Gemini) and LiteLLM (GPT/Claude/DeepSeek) connectivity,
credentials, and model availability. Minimal token spend (~$0.001).

Usage:
  python tests/test_connections.py                 # test all configured
  python tests/test_connections.py --only vertex   # just Gemini
  python tests/test_connections.py --only litellm  # just LiteLLM models
"""
import os, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_REGISTRY, GCP_PROJECT, GCP_REGION

PING = "Reply with exactly the word: PONG"
GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"

def ok(msg):   print(f"{GREEN}✅ PASS{RESET}  {msg}")
def fail(msg): print(f"{RED}❌ FAIL{RESET}  {msg}")
def warn(msg): print(f"{YELLOW}⚠️  WARN{RESET}  {msg}")

# ----------------------------------------------------------------------
# 0. Pre-flight: credentials present?
# ----------------------------------------------------------------------
def check_credentials():
    print("\n── Credential pre-flight ──")
    passed = True
    cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if cred and os.path.exists(cred):
        ok(f"Vertex key file found: {cred}")
    else:
        fail(f"Vertex key missing/not found: '{cred}' "
             f"→ set GOOGLE_APPLICATION_CREDENTIALS in config.py")
        passed = False

    if os.environ.get("AZURE_API_KEY") and os.environ.get("AZURE_API_BASE"):
        ok("LiteLLM/Azure creds present (AZURE_API_KEY, AZURE_API_BASE)")
    else:
        warn("AZURE_API_KEY / AZURE_API_BASE not set → LiteLLM tests will fail")
    print(f"Project: {GCP_PROJECT} · Region: {GCP_REGION}")
    return passed

# ----------------------------------------------------------------------
# 1. Vertex (Gemini) — raw SDK call
# ----------------------------------------------------------------------
def test_vertex(model):
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_REGION)
        t0 = time.time()
        resp = client.models.generate_content(
            model=model, contents=PING,
            config=types.GenerateContentConfig(temperature=0),
        )
        dt = time.time() - t0
        text = (resp.text or "").strip()
        um = getattr(resp, "usage_metadata", None)
        tin = getattr(um, "prompt_token_count", 0) if um else 0
        tout = getattr(um, "candidates_token_count", 0) if um else 0
        if "PONG" in text.upper():
            ok(f"[vertex] {model}: '{text[:30]}' ({dt:.1f}s, in={tin} out={tout})")
            return True
        warn(f"[vertex] {model}: connected but odd reply → '{text[:40]}'")
        return True
    except Exception as e:
        fail(f"[vertex] {model}: {type(e).__name__}: {str(e)[:200]}")
        _hint_vertex(e)
        return False

def _hint_vertex(e):
    s = str(e).lower()
    if "permission" in s or "403" in s:
        print("     💡 Check: service-account has Vertex AI User role + API enabled")
    elif "not found" in s or "404" in s:
        print(f"     💡 Check: model name valid & available in region '{GCP_REGION}'")
    elif "project" in s:
        print("     💡 Check: GCP_PROJECT_ID correct & billing enabled")

# ----------------------------------------------------------------------
# 2. LiteLLM — raw call
# ----------------------------------------------------------------------
def test_litellm(model):
    try:
        import litellm
        litellm.drop_params = True
        cfg = MODEL_REGISTRY.get(model, {})
        temp = 1.0 if cfg.get("temp_policy") == "fixed_1" else 0
        max_tok = 4000 if cfg.get("reasoning") else 20  # ← reasoning models need room
        t0 = time.time()
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": PING}],
            temperature=temp, max_tokens=max_tok,
        )
        dt = time.time() - t0
        text = (resp.choices[0].message.content or "").strip()
        u = getattr(resp, "usage", None)
        tin = getattr(u, "prompt_tokens", 0) if u else 0
        tout = getattr(u, "completion_tokens", 0) if u else 0
        if "PONG" in text.upper():
            ok(f"[litellm] {model}: '{text[:30]}' ({dt:.1f}s, in={tin} out={tout})")
            return True
        warn(f"[litellm] {model}: connected but odd reply → '{text[:40]}'")
        return True
    except Exception as e:
        fail(f"[litellm] {model}: {type(e).__name__}: {str(e)[:200]}")
        _hint_litellm(e)
        return False

def _hint_litellm(e):
    s = str(e).lower()
    if "auth" in s or "401" in s or "key" in s:
        print("     💡 Check: AZURE_API_KEY / provider key correct")
    elif "not found" in s or "404" in s or "deployment" in s:
        print("     💡 Check: deployment name exists on your endpoint")
    elif "temperature" in s:
        print("     💡 Reasoning model may reject temperature (handled in gateway)")

# ----------------------------------------------------------------------
# 3. Gateway smoke — verify our wrapper works end-to-end (both providers)
# ----------------------------------------------------------------------
def test_gateway_wrapper(model):
    try:
        from meter import Meter
        from llm_gateway import LLMGateway
        gw = LLMGateway(Meter(budget_usd=1.0))
        r = gw.generate(model, PING, temperature=0, module="test", task="ping")
        cost = gw.meter.total_cost()
        if r["text"]:
            ok(f"[gateway] {model}: wrapped call ok, metered ${cost:.6f}")
            return True
        fail(f"[gateway] {model}: empty response")
        return False
    except Exception as e:
        fail(f"[gateway] {model}: {type(e).__name__}: {str(e)[:200]}")
        return False

# ----------------------------------------------------------------------
# 4. JSON mode — critical for extraction
# ----------------------------------------------------------------------
def test_json_mode(model):
    try:
        from meter import Meter
        from llm_gateway import LLMGateway
        gw = LLMGateway(Meter(budget_usd=1.0))
        r = gw.generate(model, 'Return JSON: {"status":"ok","n":1}',
                        json_mode=True, module="test", task="json")
        if r["parsed"] and isinstance(r["parsed"], dict):
            ok(f"[json] {model}: parsed → {r['parsed']}")
            return True
        warn(f"[json] {model}: json_mode returned unparseable → '{r['text'][:60]}'")
        return False
    except Exception as e:
        fail(f"[json] {model}: {type(e).__name__}: {str(e)[:150]}")
        return False

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]

    print("="*70)
    print(" GEO MODULE-1 CONNECTION DIAGNOSTICS")
    print("="*70)
    creds_ok = check_credentials()

    vertex_models  = [m for m, c in MODEL_REGISTRY.items() if c["provider"] == "vertex"]
    litellm_models = [m for m, c in MODEL_REGISTRY.items() if c["provider"] == "litellm"]

    results = {}

    if only in (None, "vertex"):
        print("\n── VERTEX (Gemini) raw SDK ──")
        for m in vertex_models:
            results[f"vertex_raw:{m}"] = test_vertex(m)

    if only in (None, "litellm"):
        print("\n── LITELLM raw ──")
        for m in litellm_models:
            results[f"litellm_raw:{m}"] = test_litellm(m)

    # Gateway wrapper + JSON only for models that passed raw
    print("\n── GATEWAY WRAPPER ──")
    for m in (vertex_models + litellm_models):
        key_raw = f"{'vertex' if MODEL_REGISTRY[m]['provider']=='vertex' else 'litellm'}_raw:{m}"
        if results.get(key_raw):
            results[f"gateway:{m}"] = test_gateway_wrapper(m)

    print("\n── JSON MODE ──")
    for m in (vertex_models + litellm_models):
        if results.get(f"gateway:{m}"):
            results[f"json:{m}"] = test_json_mode(m)

    # summary
    print("\n" + "="*70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f" SUMMARY: {passed}/{total} checks passed")
    for k, v in results.items():
        print(f"   {'✅' if v else '❌'} {k}")
    print("="*70)
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()