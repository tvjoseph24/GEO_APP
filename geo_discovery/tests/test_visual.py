"""Per-arm visual assets, identical-text enforcement, payload verification.
Covers the visual capability and W-22/W-23/W-28."""
import pytest

from experiment import visual as VIS

PNG = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
PNG2 = (b"\x89PNG\r\n\x1a\n" + b"\x01" * 200)

def _design(tpl, claims=()):
    return {"design_id": "C01", "experiment_id": tpl, "question": "best gentle cleanser",
            "claims_allowed": list(claims)}

# ---------------------------------------------------------------- per-arm assets
def test_assets_are_per_arm():
    """A single visual_asset slot could only test 'image on the treatment arm', so
    creative-vs-creative and image-presence tests were both impossible."""
    d = _design("E11")
    VIS.attach_asset(d, "A", PNG, "image/png", "creative 1")
    VIS.attach_asset(d, "B", PNG2, "image/png", "creative 2")
    a = VIS.assets(d)
    assert set(a) == {"A", "B"}
    assert a["A"]["digest"] != a["B"]["digest"]
    # the payload must survive the metadata spread — probe() previously overwrote
    # `bytes` with the byte COUNT, so the runner received an int
    assert isinstance(a["A"]["bytes"], (bytes, bytearray)), \
        f"payload is {type(a['A']['bytes'])}, not bytes"
    assert a["A"]["n_bytes"] == len(PNG)
    assert VIS.asset_for_arm(d, "A")[0] == PNG
    assert VIS.asset_for_arm(d, "variant_B")[0] == PNG2
    assert VIS.asset_for_arm(d, "C") is None

def test_legacy_single_slot_still_reads_as_arm_b():
    d = {"visual_asset": {"bytes": PNG, "mime": "image/png"}}
    assert set(VIS.assets(d)) == {"B"}

def test_attach_retires_the_legacy_slot():
    d = {"visual_asset": {"bytes": PNG, "mime": "image/png"}}
    VIS.attach_asset(d, "A", PNG2, "image/png")
    assert "visual_asset" not in d and set(VIS.assets(d)) == {"A"}

# ---------------------------------------------------------------- validation
IDENT = {"A": "Aveeno is a skincare brand. Aveeno products are widely available.",
         "B": "Aveeno is a skincare brand. Aveeno products are widely available."}

def test_identical_assets_are_invalid():
    d = _design("E11")
    VIS.attach_asset(d, "A", PNG, "image/png")
    VIS.attach_asset(d, "B", PNG, "image/png")
    r = VIS.validate(d, "E11", IDENT)
    assert not r["ok"] and "identical_assets" in r["notes"]

def test_differing_text_is_invalid_for_a_visual_test():
    d = _design("E11")
    VIS.attach_asset(d, "A", PNG, "image/png")
    VIS.attach_asset(d, "B", PNG2, "image/png")
    r = VIS.validate(d, "E11", {"A": IDENT["A"], "B": "Completely different copy here."})
    assert not r["ok"] and "text_not_identical" in r["notes"]

def test_clean_creative_bakeoff_passes():
    d = _design("E11")
    VIS.attach_asset(d, "A", PNG, "image/png")
    VIS.attach_asset(d, "B", PNG2, "image/png")
    r = VIS.validate(d, "E11", IDENT)
    assert r["ok"] and r["severity"] == "clean", r

def test_e12_arm_a_must_have_no_asset():
    d = _design("E12")
    VIS.attach_asset(d, "A", PNG, "image/png")
    VIS.attach_asset(d, "B", PNG2, "image/png")
    r = VIS.validate(d, "E12", IDENT)
    assert not r["ok"] and any("arm A must carry no asset" in x for x in r["invalid"])

def test_e12_clean_when_only_b_has_an_image():
    d = _design("E12")
    VIS.attach_asset(d, "B", PNG, "image/png")
    assert VIS.validate(d, "E12", IDENT)["ok"]

def test_missing_required_arm_is_invalid():
    d = _design("E7")
    VIS.attach_asset(d, "B", PNG, "image/png")
    r = VIS.validate(d, "E7", IDENT)
    assert not r["ok"] and any("requires an asset" in x for x in r["invalid"])

def test_e7_claim_in_text_defeats_the_experiment():
    """If the claim is in the text, the model answers without looking at the image."""
    claim = "Contains 1% colloidal oatmeal"
    d = _design("E7", claims=[claim])
    VIS.attach_asset(d, "A", PNG, "image/png")
    VIS.attach_asset(d, "B", PNG2, "image/png")
    leaky = {"A": f"Aveeno lotion. {claim}. Aveeno is gentle.",
             "B": f"Aveeno lotion. {claim}. Aveeno is gentle."}
    r = VIS.validate(d, "E7", leaky)
    assert not r["ok"] and "e7_claim_leak" in r["notes"]

def test_long_text_is_caveated_not_blocked():
    d = _design("E11")
    VIS.attach_asset(d, "A", PNG, "image/png")
    VIS.attach_asset(d, "B", PNG2, "image/png")
    long = " ".join(["Aveeno is a skincare brand with many products."] * 12)
    r = VIS.validate(d, "E11", {"A": long, "B": long})
    assert r["ok"] and "text_may_dominate" in r["notes"]

def test_non_visual_template_is_not_applicable():
    assert VIS.validate(_design("E4"), "E4", IDENT)["applicable"] is False

# ---------------------------------------------------------------- vision readiness
def test_no_vision_engine_blocks():
    ok, probs = VIS.vision_ready([{"name": "x", "supports": ["text"]}], {})
    assert not ok and any("no selected engine" in p.lower() for p in probs)

def test_failed_selftest_blocks():
    ok, probs = VIS.vision_ready([{"name": "g", "supports": ["text", "vision"]}],
                                 {"vision_selftest": {"vision_live": False}})
    assert not ok and any("never reached the model" in p for p in probs)

def test_confabulating_model_blocks():
    ok, probs = VIS.vision_ready([{"name": "g", "supports": ["text", "vision"]}],
                                 {"vision_selftest": {"vision_live": True,
                                                      "confabulates": True}})
    assert not ok and any("INVENTS" in p for p in probs)

def test_passing_selftest_is_ready():
    ok, probs = VIS.vision_ready([{"name": "g", "supports": ["text", "vision"]}],
                                 {"vision_selftest": {"vision_live": True,
                                                      "confabulates": False}})
    assert ok and probs == []

# ---------------------------------------------------------------- payload verification
from experiment.results import _cell_stats, _visual_validity

def _er(n=100, img_runs=0, tok_runs=0):
    arm = {"n": n, "mentioned": 50, "is_first": 10}
    if img_runs:
        arm["visual"] = {"runs_with_image": img_runs,
                         "runs_with_image_tokens": tok_runs,
                         "image_tokens_total": tok_runs * 258}
    return {"engines": {"g": {"A": dict(arm), "B": dict(arm)}}}

def test_zero_image_tokens_is_invalid_not_null():
    """A visual experiment where the payload never arrived produces a null that looks
    like evidence."""
    v = _visual_validity(_er(img_runs=100, tok_runs=0))
    assert v["verdict"] == "INVALID" and not v["payload_delivered"]

def test_partial_delivery_is_flagged():
    assert _visual_validity(_er(img_runs=100, tok_runs=60))["verdict"] == "PARTIAL"

def test_full_delivery_is_ok():
    v = _visual_validity(_er(img_runs=100, tok_runs=100))
    assert v["verdict"] == "OK" and v["mean_image_tokens"] > 0

def test_no_visual_returns_none():
    assert _visual_validity(_er()) is None

# ---------------------------------------------------------------- W-22
def test_runs_per_cell_is_set():
    """assemble_findings never set this — only reanalyse.py did — so every live null
    became DISPROVEN with no power check and INCONCLUSIVE was unreachable."""
    s = _cell_stats(_er(n=100))
    assert s["runs_per_cell"] == 100 and s["underpowered"] is False
    s2 = _cell_stats(_er(n=20))
    assert s2["underpowered"] is True

def test_empty_result_does_not_claim_power():
    assert _cell_stats({"engines": {}})["underpowered"] is None

# ---------------------------------------------------------------- W-28
from experiment.rag_runner import completed_runs

def test_completed_runs_is_scoped(tmp_path):
    """Counting every row in the file made `remaining` clamp to 0, so a partially
    complete run reported '0 remaining, $0.00' while arms were still missing."""
    import json
    f = tmp_path / "experiment_raw.jsonl"
    rows = ([{"design_id": "C01", "mode": "ab", "arm": "A", "model": "g", "run": i}
             for i in range(10)]
            + [{"design_id": "C99", "mode": "ab", "arm": "A", "model": "g", "run": i}
               for i in range(50)])
    f.write_text("\n".join(json.dumps(r) for r in rows))
    assert len(completed_runs(str(f))) == 60
    assert len(completed_runs(str(f), design_ids=["C01"])) == 10
    assert len(completed_runs(str(f), design_ids=["C01"], arms=["B"])) == 0