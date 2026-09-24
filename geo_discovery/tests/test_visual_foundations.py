"""Phase-1a guards: asset store, persistence, media tokens, video gating."""
import pytest

from experiment import visual as VIS

PNG = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 600)
MP4 = b"\x00\x00\x00 ftypisom" + b"\x00" * 600

def _design(did="V01", **kw):
    return {"design_id": did, "needs_visual": True, **kw}

# ---------------------------------------------------------------- V-7
def test_asset_survives_design_rebuild():
    """The failure this exists to prevent: 'Build designs' recreates the design list,
    and an asset held on a design object is silently discarded."""
    store, d = {}, _design()
    VIS.attach_asset(store, d, "A", PNG, "image/png", "hero")
    rebuilt = _design()                        # a fresh object, same design_id
    assert "A" in VIS.assets(store, rebuilt)
    assert VIS.asset_for_arm(store, rebuilt, "A")[1] == "image/png"

def test_assets_are_per_arm_and_independent():
    store, d = {}, _design()
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    VIS.attach_asset(store, d, "B", PNG + b"x", "image/png")
    a = VIS.assets(store, d)
    assert set(a) == {"A", "B"}
    assert a["A"]["digest"] != a["B"]["digest"]

def test_designs_do_not_share_assets():
    store = {}
    VIS.attach_asset(store, _design("V01"), "A", PNG, "image/png")
    assert VIS.assets(store, _design("V02")) == {}

def test_detach_is_scoped():
    store, d = {}, _design()
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    VIS.attach_asset(store, d, "B", PNG, "image/png")
    VIS.detach_asset(store, d, "A")
    assert set(VIS.assets(store, d)) == {"B"}

def test_metadata_cannot_overwrite_the_payload():
    """probe() returns n_bytes, not bytes — a key collision replaced the image with its
    own length."""
    store, d = {}, _design()
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    a = VIS.assets(store, d)["A"]
    assert isinstance(a["bytes"], bytes) and len(a["bytes"]) == len(PNG)
    assert a["n_bytes"] == len(PNG)

def test_legacy_design_assets_still_readable():
    d = _design()
    d["visual_assets"] = {"B": {"bytes": PNG, "mime": "image/png", "digest": "x"}}
    assert "B" in VIS.assets({}, d)

# ---------------------------------------------------------------- persistence
def test_assets_round_trip_through_blobs(tmp_path, monkeypatch):
    """save_assets wrote bytes and nothing ever read them back, so a visual experiment
    could not be re-run, extended or reported after a reload."""
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)

    src, d = {}, _design("V07")
    VIS.attach_asset(src, d, "A", PNG, "image/png", "before")
    VIS.attach_asset(src, d, "B", PNG + b"y", "image/png", "after")
    man = VIS.save_assets("p", "t", src)
    assert set(man["V07"]) == {"A", "B"}
    S.save_artifact("p", "t", "experiment_visual_assets", man, {})

    restored = VIS.load_assets("p", "t", {})
    assert set(restored["V07"]) == {"A", "B"}
    assert restored["V07"]["A"]["bytes"] == PNG
    assert restored["V07"]["A"]["label"] == "before"
    assert restored["V07"]["B"]["restored_from"]

def test_load_assets_does_not_clobber_a_session_upload(tmp_path, monkeypatch):
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    old, d = {}, _design("V08")
    VIS.attach_asset(old, d, "A", PNG, "image/png", "old")
    S.save_artifact("p", "t", "experiment_visual_assets",
                    VIS.save_assets("p", "t", old), {})
    live = {}
    VIS.attach_asset(live, d, "A", PNG + b"new", "image/png", "new")
    VIS.load_assets("p", "t", live)
    assert live["V08"]["A"]["label"] == "new"

def test_strip_for_persistence_removes_bytes_but_keeps_metadata():
    store, d = {}, _design()
    VIS.attach_asset(store, d, "A", PNG, "image/png", "hero")
    out = VIS.strip_for_persistence(store, d)
    assert "visual_assets" not in out and "visual_asset" not in out
    assert out["visual_assets_meta"]["A"]["label"] == "hero"
    import json
    json.dumps(out)                              # must be serialisable

# ---------------------------------------------------------------- media tokens
def test_video_tokens_are_counted():
    """Matching only 'IMAGE' meant every video run reported zero tokens, so
    results._visual_validity marked every video finding INVALID."""
    from meter import Meter

    class _D:
        def __init__(self, mod, n):
            self.modality, self.token_count = mod, n

    class _R:
        usage_metadata = type("U", (), {"prompt_tokens_details":
                                        [_D("VIDEO", 4500), _D("TEXT", 120)]})()

    m = Meter()
    assert m.note_image_tokens(_R(), "vertex") == 4500
    assert "VIDEO" in m.last_media_kinds

def test_image_tokens_still_counted():
    from meter import Meter

    class _D:
        def __init__(self, mod, n):
            self.modality, self.token_count = mod, n

    class _R:
        usage_metadata = type("U", (), {"prompt_tokens_details": [_D("IMAGE", 258)]})()

    assert Meter().note_image_tokens(_R(), "vertex") == 258

def test_zero_media_tokens_marks_the_finding_invalid():
    from experiment.results import _visual_validity
    er = {"engines": {"e1": {"A": {"visual": {"runs_with_image": 50,
                                              "runs_with_image_tokens": 0,
                                              "image_tokens_total": 0}}}}}
    v = _visual_validity(er)
    assert v["verdict"] == "INVALID"
    assert "artifact, not a null" in v["note"]

def test_video_skipped_everywhere_is_invalid_not_null():
    from experiment.results import _visual_validity
    er = {"engines": {"e1": {"A": {"visual": {"runs_with_image": 0,
                                              "runs_with_image_tokens": 0,
                                              "image_tokens_total": 0,
                                              "video_runs_skipped": 40}}}}}
    v = _visual_validity(er)
    assert v["verdict"] == "INVALID"
    assert "video-capable" in v["note"]

def test_delivered_video_is_valid():
    from experiment.results import _visual_validity
    er = {"engines": {"e1": {"A": {"visual": {"runs_with_image": 40,
                                              "runs_with_image_tokens": 40,
                                              "image_tokens_total": 180000}}}}}
    assert _visual_validity(er)["verdict"] == "OK"

# ---------------------------------------------------------------- video gating
def test_gateway_rejects_video_on_a_vision_only_model():
    from llm_gateway import LLMGateway
    from meter import Meter
    g = LLMGateway(Meter())
    with pytest.raises(ValueError, match="video"):
        g.generate("azure/gpt-4o", "hi", images=[(MP4, "video/mp4")])

def test_vision_ready_requires_video_capability_when_asked():
    man = {"vision_selftest": {"vision_live": True, "confabulates": False}}
    vision_only = [{"name": "azure/gpt-4o", "supports": ["text", "vision"]}]
    ok, probs = VIS.vision_ready(vision_only, man, needs_video=True)
    assert not ok and any("video" in x for x in probs)
    ok2, _ = VIS.vision_ready(vision_only, man, needs_video=False)
    assert ok2

def test_has_video_detects_a_video_arm():
    store, d = {}, _design()
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    assert not VIS.has_video(store, d)
    VIS.attach_asset(store, d, "B", MP4, "video/mp4")
    assert VIS.has_video(store, d)

# ---------------------------------------------------------------- cost
def test_video_payload_cost_dominates_and_is_visible():
    store, d = {}, _design()
    VIS.attach_asset(store, d, "A", MP4, "video/mp4")
    pc = VIS.estimate_payload_cost(store, d, n_runs=100, n_engines=1,
                                   price_per_input_token=0.30 / 1e6,
                                   video_seconds=30)
    assert pc["has_video"]
    assert pc["total_media_tokens"] == 30 * VIS.VIDEO_TOKENS_PER_SEC * 100
    assert pc["est_media_cost_usd"] > 0

def test_measured_tokens_override_the_estimate():
    store, d = {}, _design()
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    VIS.assets(store, d)["A"]["measured_tokens"] = 1234
    store["V01"]["A"]["measured_tokens"] = 1234
    pc = VIS.estimate_payload_cost(store, d, 10, 1, 1e-6)
    assert pc["total_media_tokens"] == 12340
    assert pc["measured"] is True

# ---------------------------------------------------------------- validate
def test_identical_assets_are_invalid():
    store, d = {}, _design()
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    VIS.attach_asset(store, d, "B", PNG, "image/png")
    v = VIS.validate(store, d, "E11", {"A": "Brand X cream. Brand X.",
                                       "B": "Brand X cream. Brand X."})
    assert not v["ok"] and any("SAME image" in x for x in v["invalid"])

def test_differing_text_is_invalid_for_a_visual_test():
    store, d = {}, _design()
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    VIS.attach_asset(store, d, "B", PNG + b"z", "image/png")
    v = VIS.validate(store, d, "E11", {"A": "Brand X hydrating cream.",
                                       "B": "Brand X cream with ceramides."})
    assert not v["ok"] and "text_not_identical" in v["notes"]

def test_clean_visual_design_passes():
    store, d = {}, _design()
    VIS.attach_asset(store, d, "A", PNG, "image/png")
    VIS.attach_asset(store, d, "B", PNG + b"z", "image/png")
    txt = "Brand X moisturiser from Brand X."
    v = VIS.validate(store, d, "E11", {"A": txt, "B": txt})
    assert v["ok"] and v["severity"] == "clean"
