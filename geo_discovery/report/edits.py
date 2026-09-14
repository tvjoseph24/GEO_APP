"""Human edit overlays for the report.

Edits are stored as a separate artifact keyed by (section_id, block_index) and applied
AFTER the model is built. Rebuilding the report from fresh artifacts therefore preserves
analyst edits, and every edit is auditable — the original text is retained.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from store import load_artifact, save_artifact

ARTIFACT = "report_edits"
EDITABLE_KINDS = ("prose", "bullets", "callout", "heading")

def block_key(section_id: str, index: int, block) -> str:
    """Stable key. Includes a hash of the ORIGINAL text so an edit is auto-invalidated
    when the underlying analysis changes that block — a stale edit is worse than none."""
    orig = _as_text(block.payload)
    h = hashlib.sha1(orig.encode()).hexdigest()[:8]
    return f"{section_id}:{index}:{h}"

def _as_text(payload) -> str:
    if isinstance(payload, list):
        return "\n".join(str(x) for x in payload)
    return str(payload or "")

def _from_text(text: str, kind: str):
    if kind == "bullets":
        return [l.strip().lstrip("-• ") for l in text.splitlines() if l.strip()]
    return text

def load_edits(parent, target) -> dict:
    return load_artifact(parent, target, ARTIFACT) or {}

def save_edits(parent, target, edits: dict, approver="analyst"):
    save_artifact(parent, target, ARTIFACT, edits,
                  {"approver": approver,
                   "saved_utc": datetime.now(timezone.utc).isoformat(),
                   "n_edits": sum(1 for v in edits.values() if v.get("edited_text"))})

def apply_edits(rm, edits: dict) -> dict:
    """Overlay saved edits onto the model in place. Returns a report of what applied."""
    applied, stale = 0, []
    for s in rm.sections:
        for i, b in enumerate(s.blocks):
            if b.kind not in EDITABLE_KINDS:
                continue
            key = block_key(s.id, i, b)
            e = edits.get(key)
            if e and e.get("edited_text"):
                b.payload = _from_text(e["edited_text"], b.kind)
                b.edited = True
                applied += 1
    # an edit whose key no longer exists means the source analysis changed under it
    live = {block_key(s.id, i, b) for s in rm.sections
            for i, b in enumerate(s.blocks) if b.kind in EDITABLE_KINDS}
    for k, v in edits.items():
        if v.get("edited_text") and k not in live:
            stale.append({"key": k, "section": k.split(":")[0],
                          "text": v["edited_text"][:120],
                          "reason": "the underlying analysis changed — re-apply manually"})
    return {"applied": applied, "stale": stale,
            "total_saved": sum(1 for v in edits.values() if v.get("edited_text"))}

def edit_summary(edits: dict) -> list:
    return [{"Key": k, "Section": k.split(":")[0],
             "Original": (v.get("original_text") or "")[:90],
             "Edited": (v.get("edited_text") or "")[:90],
             "By": v.get("approver", "analyst"), "When": (v.get("at") or "")[:19]}
            for k, v in edits.items() if v.get("edited_text")]