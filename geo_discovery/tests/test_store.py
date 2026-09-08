"""Free tests for versioned artifact persistence."""
import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
config.STORE_ROOT = config.Path("geo_projects_TEST"); config.STORE_ROOT.mkdir(exist_ok=True)
from store import save_artifact, save_audit_log, project_dir

def test_save_and_latest():
    p = save_artifact("TestBrand", "claims_register",
                      [{"claim": "colloidal oatmeal"}], {"extraction": "gemini-2.5-pro"})
    assert os.path.exists(p)
    latest = project_dir("TestBrand") / "artifacts" / "claims_register_LATEST.json"
    data = json.loads(latest.read_text())
    assert data["artifact"] == "claims_register"
    assert data["models_used"]["extraction"] == "gemini-2.5-pro"
    assert "timestamp_utc" in data          # versioning check
    print("✅ save_artifact + LATEST pointer + timestamp")

def test_audit_log():
    save_audit_log("TestBrand", [{"gate": "claims", "status": "approved"}])
    log = json.loads((project_dir("TestBrand") / "audit_log.json").read_text())
    assert log[0]["gate"] == "claims"
    print("✅ audit_log saved")

def cleanup():
    shutil.rmtree("geo_projects_TEST", ignore_errors=True)

if __name__ == "__main__":
    print("\n── STORE TESTS (free) ──")
    for fn in [test_save_and_latest, test_audit_log]:
        try: fn()
        except Exception as e: print(f"❌ {fn.__name__}: {e}")
    cleanup()