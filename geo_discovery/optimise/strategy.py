"""Assemble Overall GEO Strategy + engine overlays + roadmap. Provisional-flagged."""
import json
from datetime import datetime, timezone
from store import save_artifact
from optimise.pillar_router import pillar_summary, PILLARS

def build_strategy(findings, briefs, target_brand, parent_brand):
    """Overall (universal) + per-engine overlays."""
    universal = [f for f in findings if f.get("verdict") == "validated"]
    engine_specific = [f for f in findings if f.get("verdict") == "engine-specific"]

    # collect engine overlays
    overlays = {}
    for f in engine_specific:
        for e in f.get("per_engine", []):
            if e.get("significant"):
                overlays.setdefault(e["engine"], []).append({
                    "gap_id": f["gap_id"], "pillar": f["pillar"],
                    "lift_pp": e["lift_pp"], "horizon": f.get("horizon")})

    strategy = {
        "PROVISIONAL": True,
        "provisional_note": ("Strategy based on current experiment findings. Absolute lift "
                             "numbers are provisional pending Phase-2 competitor-content "
                             "enrichment. Directional findings and structure are sound."),
        "target_brand": target_brand, "parent_brand": parent_brand,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pillar_summary": pillar_summary(findings),
        "overall_strategy": [
            {"gap_id": f["gap_id"], "pillar": f["pillar"], "horizon": f["horizon"],
             "lift_pp": f["pooled_lift_pp"], "action": f.get("gap_detail",{}).get("hypothesis","")}
            for f in universal],
        "engine_overlays": overlays,
    }
    return strategy

def build_roadmap(findings):
    roadmap = {"Now": [], "Next": [], "Later": []}
    for f in findings:
        # for engine-specific, show the BEST engine's lift, not misleading pooled avg
        best_engine_lift = max((e["lift_pp"] for e in f.get("per_engine",[])
                                if e.get("significant")), default=f["pooled_lift_pp"])
        roadmap[f.get("horizon","Later")].append({
            "gap_id": f["gap_id"], "pillar": f["pillar"],
            "priority_score": f["priority_score"], "clock": f.get("clock"),
            "pooled_lift_pp": f["pooled_lift_pp"],          # ← labeled clearly
            "best_engine_lift_pp": best_engine_lift,        # ← the meaningful one
            "verdict": f["verdict"]})
    return roadmap

def save_strategy(parent_brand, target_brand, strategy, roadmap, briefs, models_used):
    save_artifact(parent_brand, target_brand, "geo_strategy", strategy, models_used)
    save_artifact(parent_brand, target_brand, "roadmap", roadmap, models_used)
    save_artifact(parent_brand, target_brand, "content_briefs", briefs, models_used)