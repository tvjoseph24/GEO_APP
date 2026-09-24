"""Visual experiment semantics: E7, E8, E11, E12 and the manipulation check.

Store mechanics, persistence and media-token accounting live in
test_visual_foundations.py. This file covers what each TEMPLATE means.

API note: assets live in a store keyed by design_id, not on the design object —
"Build designs" recreates the design list wholesale and would otherwise discard
every upload (V-7). Every call therefore threads a store.
"""
import pytest

from experiment import visual as VIS

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 600
PNG2 = b"\x89PNG\r\n\x1a\n" + b"\x11" * 600
MP4 = b"\x00\x00\x00 ftypisom" + b"\x00" * 600

IDENT = {"A": "Brand X moisturiser from Brand X.",
         "B": "Brand X moisturiser from Brand X."}

def _design(tpl, did=None, **kw):
    return {"design_id": did or f"{tpl}-01", "experiment_id": tpl,
            "needs_visual": True, **kw}

def _real_png(w, h):
    """A genuine PNG, so probe() can read dimensions and megapixels."""
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()

# ---------------------------------------------------------------- per-arm assets
def test_assets_are_per_arm():
    """A single visual_asset slot could only test 'image on the treatment arm', so
    creative-vs-creative and image-presence tests were both impossible."""
    store, d = {}, _design("E11")
    VIS.attach_asset(store, d, "A", PNG, "image/png", "creative 1")
    VIS.attach_asset(store, d, "B", PNG2, "image/png", "creative 2")
    a = VIS.assets(store, d)
    assert set(a) == {"A", "B"}
    assert a["A"]["label"] == "creative 1"
    assert a["A"]["digest"] != a["B"]["digest"]
    assert VIS.asset_for_arm(store, d, "A")[0] == PNG
    assert VIS.asset_for_arm(store, d, "B")[0] == PNG2

def test_arm_key_is_normalised():
    store, d = {}, _design("E11")
    VIS.attach_asset(store, d, "variant_b", PNG, "image/png")
    assert "B" in VIS.assets(store, d)
    assert VIS.asset_for_arm(store, d, "variant_B") is not None

def test_designs_do_not_share_assets():
    """The store is keyed by design_id — a bench with several designs open at once must
    not leak creative between them."""
    store = {}
    VIS.attach_asset(store, _design("E11", "V01"), "A", PNG, "image/png")
    VIS.attach_asset(store, _design("E11", "V02"), "A", PNG2, "image/png")
    assert VIS.assets(store, _design("E11", "V01"))["A"]["digest"] != \
           VIS.assets(store, _design("E11", "V02"))["A"]["digest"]
    assert VIS.assets(store, _design("E11", "V03")) == {}

def test_attach_rejects_a_non_bytes_payload():
    """Positional drift in the six-argument signature otherwise fails deep inside
    probe() with 'string argument without an encoding'."""
    with pytest.raises(TypeError, match="raw bytes"):
        VIS.attach_asset({}, _design("E11"), "A", "image/png", "image/png")

def test_legacy_single_slot_still_reads_as_arm_b():
    d = {"design_id": "L01", "visual_asset": {"bytes": PNG, "mime": "image/png"}}
    assert set(VIS.assets({}, d)) == {"B"}

def test_legacy_design_level_assets_still_read():
    d = {"design_id": "L02",
         "visual_assets": {"A": {"bytes": PNG, "mime": "image/png", "digest": "x"}}}
    assert set(VIS.assets({}, d)) == {"A"}

def test_attach_retires_the_legacy_slot():
    """Once an asset is in the store the legacy field must go, or assets() would read
    two sources and the arm-B image could reappear after being removed."""
    store = {}
    d = {"design_id": "L03", "visual_asset": {"bytes": PNG, "mime": "image/png"}}
    VIS.attach_asset(store, d, "A", PNG2, "image/png")
    assert "visual_asset" not in d
    assert set(VIS.assets(store, d)) == {"A"}

def test_mime_inference():
    assert VIS.mime_for("hero.PNG") == "image/png"
    assert VIS.mime_for("shot.jpeg") == "image/jpeg"
    assert VIS.mime_for("film.mov") == "video/quicktime"
    assert VIS.mime_for("clip.mp4") == "video/mp4"
    assert VIS.mime_for("noextension") == "image/jpeg"

def test_probe_reads_real_image_dimensions():
    pytest.importorskip("PIL")
    store, d = {}, _design("E11")
    VIS.attach_asset(store, d, "A", _real_png(800, 600), "image/png")
    a = VIS.assets(store, d)["A"]
    assert (a["width"], a["height"]) == (800, 600)
    assert a["megapixels"] == pytest.approx(0.48, abs=0.01)
    assert a["is_video"] is False

# ---------------------------------------------------------------- manipulation check
def test_identical_assets_are_invalid():
    """Same image on every arm: nothing is being manipulated."""
    store, d = {}, _design("E11")
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    VIS.attach_asset(store, d, "B", PNG, "image/png")
    v = VIS.validate(store, d, "E11", IDENT)
    assert not v["ok"] and v["severity"] == "invalid"
    assert any("SAME image" in x for x in v["invalid"])
    assert "identical_assets" in v["notes"]

def test_differing_text_is_invalid_for_a_visual_test():
    """If text varies alongside the image, the lift is attributable to the text."""
    store, d = {}, _design("E11")
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    VIS.attach_asset(store, d, "B", PNG2, "image/png")
    v = VIS.validate(store, d, "E11",
                     {"A": "Brand X hydrating cream.",
                      "B": "Brand X cream with ceramides."})
    assert not v["ok"] and "text_not_identical" in v["notes"]

def test_whitespace_differences_do_not_count_as_different_text():
    store, d = {}, _design("E11")
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    VIS.attach_asset(store, d, "B", PNG2, "image/png")
    v = VIS.validate(store, d, "E11",
                     {"A": "Brand X moisturiser from Brand X.",
                      "B": "Brand X   moisturiser\nfrom Brand X."})
    assert v["checks"]["text_identical"] is True
    assert "text_not_identical" not in v["notes"]

def test_clean_creative_bakeoff_passes():
    store, d = {}, _design("E11")
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    VIS.attach_asset(store, d, "B", PNG2, "image/png")
    v = VIS.validate(store, d, "E11", IDENT)
    assert v["ok"] and v["severity"] == "clean"
    assert v["checks"]["arms_with_asset"] == ["A", "B"]

def test_four_way_bakeoff_passes():
    store, d = {}, _design("E11")
    for i, arm in enumerate("ABCD"):
        VIS.attach_asset(store, d, arm, PNG + bytes([i]), "image/png")
    txt = dict.fromkeys("ABCD", "Brand X moisturiser from Brand X.")
    v = VIS.validate(store, d, "E11", txt)
    assert v["ok"] and len(v["checks"]["arms_with_asset"]) == 4

def test_missing_required_arm_is_invalid():
    """E7 needs both a before and an after pack — one of them is not a comparison."""
    store, d = {}, _design("E7")
    VIS.attach_asset(store, d, "B", PNG, "image/png")
    v = VIS.validate(store, d, "E7", IDENT)
    assert not v["ok"] and any("requires an asset" in x for x in v["invalid"])

def test_e12_arm_a_must_have_no_asset():
    """E12 isolates image PRESENCE, so an image on the control destroys the contrast."""
    store, d = {}, _design("E12")
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    VIS.attach_asset(store, d, "B", PNG2, "image/png")
    v = VIS.validate(store, d, "E12", IDENT)
    assert not v["ok"] and any("arm A must carry no asset" in x for x in v["invalid"])

def test_e12_clean_when_only_b_has_an_image():
    store, d = {}, _design("E12")
    VIS.attach_asset(store, d, "B", PNG, "image/png")
    v = VIS.validate(store, d, "E12", IDENT)
    assert v["ok"] and v["severity"] == "clean"

def test_e7_claim_in_text_defeats_the_experiment():
    """If the claim is in the text, the model answers without looking at the image."""
    claim = "Contains 1% colloidal oatmeal"
    store, d = {}, _design("E7", claims_allowed=[claim])
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    VIS.attach_asset(store, d, "B", PNG2, "image/png")
    v = VIS.validate(store, d, "E7",
                     {"A": f"Brand X cream. {claim}. From Brand X.",
                      "B": f"Brand X cream. {claim}. From Brand X."})
    assert not v["ok"] and "e7_claim_leak" in v["notes"]
    assert v["checks"]["claims_leaked_into_text"] == 1

def test_e7_clean_when_text_carries_no_claim():
    store, d = {}, _design("E7", claims_allowed=["Contains 1% colloidal oatmeal"])
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    VIS.attach_asset(store, d, "B", PNG2, "image/png")
    v = VIS.validate(store, d, "E7",
                     {"A": "Brand X moisturiser. Brand X.",
                      "B": "Brand X moisturiser. Brand X."})
    assert v["ok"] and "e7_claim_leak" not in v["notes"]

def test_long_text_is_caveated_not_blocked():
    """Runnable, but the finding must say the copy may be doing the work."""
    store, d = {}, _design("E11")
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    VIS.attach_asset(store, d, "B", PNG2, "image/png")
    long = ("Brand X moisturiser from Brand X. " + "Gentle hydration all day long. " * 22)
    v = VIS.validate(store, d, "E11", {"A": long, "B": long})
    assert v["ok"] and v["severity"] == "caveated"
    assert "text_may_dominate" in v["notes"]

def test_video_is_caveated_for_cost():
    store, d = {}, _design("E11")
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    VIS.attach_asset(store, d, "B", MP4, "video/mp4")
    v = VIS.validate(store, d, "E11", IDENT)
    assert v["ok"] and "video_attached" in v["notes"]
    assert any("VIDEO" in x for x in v["caveats"])

def test_resolution_mismatch_is_caveated():
    """A legibility difference that is really a resolution artifact."""
    pytest.importorskip("PIL")
    store, d = {}, _design("E11")
    VIS.attach_asset(store, d, "A", _real_png(2000, 1500), "image/png")
    VIS.attach_asset(store, d, "B", _real_png(320, 240), "image/png")
    v = VIS.validate(store, d, "E11", IDENT)
    assert "resolution_mismatch" in v["notes"]

def test_non_visual_template_is_not_applicable():
    assert VIS.validate({}, {"design_id": "X"}, "E4", IDENT)["applicable"] is False

def test_unknown_template_with_needs_visual_falls_back_to_a_bakeoff_plan():
    """The bench writes its own templates; validate must not silently pass them."""
    store, d = {}, _design("VBENCH")
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    v = VIS.validate(store, d, "VBENCH", IDENT)
    assert v["applicable"] is True
    assert not v["ok"], "a missing arm-B asset must still be caught"

# ---------------------------------------------------------------- plan and readiness
def test_asset_plan_covers_every_visual_template():
    for tpl in VIS.VISUAL_TEMPLATES:
        plan = VIS.ASSET_PLAN[tpl]
        assert plan["required"], f"{tpl} declares no required arm"
        assert plan["note"], f"{tpl} has no explanation for the operator"
        assert set(plan["required"]) <= set(VIS.ARM_LETTERS)
        assert set(plan.get("optional") or []) <= set(VIS.ARM_LETTERS)
        assert not set(plan["required"]) & set(plan.get("optional") or [])

def test_e12_requires_only_arm_b():
    assert VIS.ASSET_PLAN["E12"]["required"] == ["B"]

def test_vision_ready_blocks_a_failed_selftest():
    engines = [{"name": "gemini-2.5-flash", "supports": ["text", "vision", "video"]}]
    ok, probs = VIS.vision_ready(engines, {"vision_selftest": {"vision_live": False}})
    assert not ok and any("never reached the model" in x for x in probs)

def test_vision_ready_blocks_a_confabulating_model():
    engines = [{"name": "gemini-2.5-flash", "supports": ["text", "vision"]}]
    ok, probs = VIS.vision_ready(
        engines, {"vision_selftest": {"vision_live": True, "confabulates": True}})
    assert not ok and any("INVENTS text" in x for x in probs)

def test_vision_ready_blocks_when_no_selftest_on_record():
    engines = [{"name": "gemini-2.5-flash", "supports": ["text", "vision"]}]
    ok, probs = VIS.vision_ready(engines, {})
    assert not ok and any("No vision self-test" in x for x in probs)

def test_vision_ready_passes_a_clean_setup():
    engines = [{"name": "gemini-2.5-flash", "supports": ["text", "vision", "video"]}]
    ok, probs = VIS.vision_ready(
        engines, {"vision_selftest": {"vision_live": True, "confabulates": False}})
    assert ok and not probs

def test_vision_ready_requires_video_capability_for_a_video_design():
    man = {"vision_selftest": {"vision_live": True, "confabulates": False}}
    vision_only = [{"name": "azure/gpt-4o", "supports": ["text", "vision"]}]
    assert VIS.vision_ready(vision_only, man, needs_video=False)[0]
    ok, probs = VIS.vision_ready(vision_only, man, needs_video=True)
    assert not ok and any("video" in x for x in probs)

# ---------------------------------------------------------------- summary
def test_summary_reports_every_arm_with_a_token_estimate():
    store, d = {}, _design("E11")
    VIS.attach_asset(store, d, "A", PNG, "image/png", "still")
    VIS.attach_asset(store, d, "B", MP4, "video/mp4", "film")
    rows = {r["arm"]: r for r in VIS.summary(store, d)}
    assert set(rows) == {"A", "B"}
    assert rows["A"]["kind"] == "image" and rows["B"]["kind"] == "video"
    assert rows["B"]["est_tokens_per_run"] > rows["A"]["est_tokens_per_run"], \
        "video payload cost must dominate, and must be visible before the run"