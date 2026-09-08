"""Run test tiers.
  python tests/run_tests.py            # Tier 0 + 1 (free/cheap)
  python tests/run_tests.py --paid     # + Tier 2 (costs money)
  python tests/run_tests.py --conn     # connections only
"""

import subprocess, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

def run(script, args=""):
    print(f"\n{'#'*70}\n# {script} {args}\n{'#'*70}")
    subprocess.run(f"{PY} {os.path.join(HERE, script)} {args}", shell=True)

if __name__ == "__main__":
    if "--conn" in sys.argv:
        run("test_connections.py"); sys.exit(0)

    # Tier 0
    run("test_connections.py")
    # Tier 1 (free)
    for s in ["test_meter.py", "test_human_gate.py", "test_store.py", "test_ingestion.py"]:
        run(s)
    run("test_extraction.py")   # free dedupe portion
    # Tier 2 (paid)
    if "--paid" in sys.argv:
        for s in ["test_gateway.py", "test_extraction.py", "smoke_test.py"]:
            run(s, "--run-paid")
    else:
        print("\n⏭️  Tier-2 PAID tests skipped. Add --paid to include them.")
