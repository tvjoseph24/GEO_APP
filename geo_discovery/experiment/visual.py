"""Visual asset handling for multimodal experiments (E7, E8, E11, E12).

Design contract
---------------
* Assets are PER ARM. A single `visual_asset` slot could only ever test "image on the
  treatment arm", which conflates image presence with whatever else differed.
* For every visual template the TEXT is held byte-identical across arms. An LLM will not
  reliably produce identical arms, so the text is generated once and copied. The image is
  then the only variable, which is the whole point.
* `image_tokens` is verified on every run. A vision experiment where the payload never
  reached the model produces a null that looks like evidence.
* Assets are persisted as blobs so the experiment is reproducible and the report can show
  what was tested.
"""
from __future__ import annotations

import hashlib
import io

ARM_LETTERS = "ABCDE"

MIME_BY_EXT = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
               "webp": "image/webp", "gif": "image/gif",
               "mp4": "video/mp4", "mov": "video/quicktime", "webm": "video/webm"}

VISUAL_TEMPLATES = {"E7", "E8", "E11", "E12"}
VIDEO_MIMES = {"video/mp4", "video/quicktime", "video/webm"}

# How many arms each template expects to carry an asset.
ASSET_PLAN = {
    "E7":  {"required": ["A", "B"], "optional": [],
            "note": "A = creative WITHOUT legible on-pack claims · "
                    "B = same pack WITH claims printed on-pack"},
    "E8":  {"required": ["A", "B"], "optional": [],
            "note": "A = image that does NOT show the claim the text states "
                    "(mismatched) · B = image that visibly shows it (aligned)"},
    "E11": {"required": ["A", "B"], "optional": ["C", "D", "E"],
            "note": "one creative execution per arm, identical text throughout"},
    "E12": {"required": ["B"], "optional": [],
            "note": "A carries NO image · B carries one image, identical text"},
}

def mime_for(filename: str) -> str:
    ext = str(filename or "").rsplit(".", 1)[-1].lower()
    return MIME_BY_EXT.get(ext, "image/jpeg")

def digest(data: bytes) -> str:
    return hashlib.sha256(bytes(data or b"")).hexdigest()[:16]

def probe(data: bytes, mime: str) -> dict:
    """Dimensions and format, when Pillow is available. Never raises.

    NOTE: the size field is `n_bytes`, not `bytes`. Using `bytes` collided with the
    payload key when spread into the asset dict and replaced the image with its length.
    """
    out = {"n_bytes": len(data or b""), "mime": mime, "digest": digest(data),
           "is_video": mime in VIDEO_MIMES}
    if out["is_video"] or not data:
        return out
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        out.update({"width": img.width, "height": img.height,
                    "format": (img.format or "").lower(),
                    "megapixels": round(img.width * img.height / 1e6, 2)})
    except Exception:
        pass
    return out

# ---------------------------------------------------------------- attach / inspect
def attach_asset(design: dict, arm: str, data: bytes, mime: str, label: str = "") -> dict:
    """Attach an asset to ONE arm. Replaces the single-slot attach_visual_variant."""
    arm = str(arm).replace("variant_", "").upper()
    meta = probe(data, mime)
    assets_d = design.setdefault("visual_assets", {})
    # payload assigned AFTER the metadata spread — a metadata key must never be able to
    # overwrite the image itself
    assets_d[arm] = {**meta, "label": label or f"arm {arm}",
                     "mime": mime, "bytes": bytes(data or b"")}
    design.pop("visual_asset", None)
    return design

def detach_asset(design: dict, arm: str) -> dict:
    (design.get("visual_assets") or {}).pop(str(arm).upper(), None)
    return design

def assets(design: dict) -> dict:
    """Per-arm assets, with backward compatibility for the legacy single slot."""
    a = dict(design.get("visual_assets") or {})
    legacy = design.get("visual_asset")
    if legacy and not a:
        _m = probe(legacy.get("bytes"), legacy.get("mime", "image/jpeg"))
        a["B"] = {**_m, "label": "arm B (legacy single slot)",
                  "mime": legacy.get("mime", "image/jpeg"),
                  "bytes": legacy.get("bytes")}
    return a

def asset_for_arm(design: dict, arm: str):
    """Returns (bytes, mime) for the runner, or None."""
    a = assets(design).get(str(arm).replace("variant_", "").upper())
    return (a["bytes"], a["mime"]) if a and a.get("bytes") else None

def summary(design: dict) -> list:
    """Table rows for the UI."""
    return [{"arm": k, "label": v.get("label", ""),
             "kind": "video" if v.get("is_video") else "image",
             "dimensions": (f"{v.get('width')}×{v.get('height')}"
                            if v.get("width") else "—"),
             "size_kb": round(v.get("n_bytes", 0) / 1024, 1),
             "digest": v.get("digest")}
            for k, v in sorted(assets(design).items())]

# ---------------------------------------------------------------- validation
def validate(design: dict, template: str, arm_texts: dict) -> dict:
    """The visual manipulation check.

    manipulation_check tests text structure and is blind to images, so an E7 design with
    the same image on both arms — or no image at all — passed as clean.
    """
    tpl = str(template or "").upper()
    plan = ASSET_PLAN.get(tpl)
    a = assets(design)
    invalid, caveats, notes, checks = [], [], [], {}

    if tpl not in VISUAL_TEMPLATES:
        return {"applicable": False, "ok": True, "severity": "clean",
                "invalid": [], "caveats": [], "notes": [], "checks": {}}

    checks["arms_with_asset"] = sorted(a)
    checks["digests"] = {k: v.get("digest") for k, v in a.items()}

    # ---- 1. required arms present ----
    missing = [arm for arm in plan["required"] if arm not in a]
    if missing:
        invalid.append(f"{tpl} requires an asset on arm(s) {missing} — "
                       f"{plan['note']}")

    # ---- 2. E12: arm A must carry NO asset ----
    if tpl == "E12" and "A" in a:
        invalid.append("E12 isolates IMAGE PRESENCE, so arm A must carry no asset. "
                       "Remove the arm-A image or switch to E11.")

    # ---- 3. assets must actually differ ----
    digs = [v.get("digest") for v in a.values() if v.get("digest")]
    if len(digs) > 1 and len(set(digs)) == 1:
        invalid.append("every arm carries the SAME image (identical digest) — there is "
                       "no visual manipulation to measure")
        notes.append("identical_assets")

    # ---- 4. text must be identical: the image is the variable ----
    keys = [k for k in ARM_LETTERS if k in (arm_texts or {})]
    texts = {k: " ".join((arm_texts.get(k) or "").split()) for k in keys}
    uniq = {t for t in texts.values() if t}
    checks["text_identical"] = len(uniq) <= 1
    checks["words_per_arm"] = {k: len(v.split()) for k, v in texts.items()}
    if len(uniq) > 1:
        invalid.append(f"arm text differs across arms ({len(uniq)} distinct versions). "
                       f"A visual experiment must hold text byte-identical, or the "
                       f"result is attributable to the text rather than the image.")
        notes.append("text_not_identical")

    # ---- 5. E7 text must not state the claims the image is meant to carry ----
    if tpl == "E7":
        allowed = design.get("claims_allowed") or []
        leaked = [c for c in allowed
                  if c and " ".join(str(c).lower().split())[:40]
                  in " ".join(next(iter(texts.values()), "").lower().split())]
        checks["claims_leaked_into_text"] = len(leaked)
        if leaked:
            invalid.append(f"{len(leaked)} claim(s) the image is supposed to carry also "
                           f"appear in the TEXT, so the model can read them without "
                           f"looking at the image. E7 text must name the brand and "
                           f"nothing else.")
            notes.append("e7_claim_leak")

    # ---- 6. text long enough to name the brand, short enough not to dominate ----
    wc = max(checks["words_per_arm"].values() or [0])
    if wc > 80:
        caveats.append(f"arm text is {wc} words — long enough that the text, not the "
                       f"image, may drive the outcome. Under 40 words is preferable for "
                       f"a visual test.")
        notes.append("text_may_dominate")

    # ---- 7. video cost gate ----
    vids = [k for k, v in a.items() if v.get("is_video")]
    if vids:
        caveats.append(f"arm(s) {vids} carry VIDEO. Video is the most expensive modality "
                       f"in the platform — confirm the run cost before committing.")
        notes.append("video_attached")

    # ---- 8. resolution parity ----
    mp = [v.get("megapixels") for v in a.values() if v.get("megapixels")]
    if len(mp) > 1 and max(mp) > 0 and min(mp) / max(mp) < 0.5:
        caveats.append(f"asset resolution varies by more than 2× ({min(mp)}–{max(mp)} MP). "
                       f"Legibility differences may be resolution artifacts.")
        notes.append("resolution_mismatch")

    sev = "invalid" if invalid else ("caveated" if caveats else "clean")
    return {"applicable": True, "ok": sev != "invalid", "severity": sev,
            "invalid": invalid, "caveats": caveats, "notes": notes, "checks": checks,
            "plan_note": plan["note"]}

def vision_ready(engines, manifest) -> tuple[bool, list]:
    """Is it safe to run a visual experiment at all?"""
    problems = []
    vis = [e for e in engines if "vision" in (e.get("supports") or [])]
    if not vis:
        problems.append("No selected engine declares vision support — a visual "
                        "experiment cannot run.")
    vs = (manifest or {}).get("vision_selftest") or {}
    if vs.get("vision_live") is False:
        problems.append("The vision self-test FAILED: the image payload never reached "
                        "the model. Any visual result would be a false null.")
    elif vs.get("confabulates"):
        problems.append("The vision self-test found the model INVENTS text on a blank "
                        "image. Visual findings from this model are untrustworthy.")
    elif vs.get("vision_live") is not True:
        problems.append("No vision self-test on record. Run it in the Audit tab before "
                        "interpreting any visual result.")
    return (not problems), problems

# ---------------------------------------------------------------- persistence
def save_assets(parent, target, designs) -> dict:
    """Write every attached asset to blob storage. Returns a manifest."""
    from store import save_bytes
    man = {}
    for d in (designs or []):
        did = d.get("design_id") or d.get("gap_id") or "unknown"
        for arm, a in assets(d).items():
            if not a.get("bytes"):
                continue
            ext = (a.get("mime", "image/jpeg").split("/")[-1]
                   .replace("jpeg", "jpg").replace("quicktime", "mov"))
            rel = save_bytes(parent, target,
                             f"visual/{did}_{arm}_{a['digest']}.{ext}", a["bytes"])
            man.setdefault(did, {})[arm] = {
                "path": rel, "digest": a["digest"], "mime": a["mime"],
                "n_bytes": a.get("n_bytes", len(a["bytes"])), "label": a.get("label"),
                "width": a.get("width"), "height": a.get("height")}
    return man

def strip_for_persistence(design: dict) -> dict:
    """Replace raw bytes with the blob manifest so store._clean can serialise it."""
    c = {k: v for k, v in design.items()
         if k not in ("_hypothesis", "visual_asset", "visual_assets")}
    a = assets(design)
    if a:
        c["visual_assets_meta"] = {
            k: {kk: v.get(kk) for kk in
                ("mime", "digest", "n_bytes", "label", "width", "height", "is_video")}
            for k, v in a.items()}
    return c