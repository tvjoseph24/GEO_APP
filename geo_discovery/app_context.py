"""Shared context + pipeline-status logic for the unified GEO app."""
from dataclasses import dataclass, field
from pathlib import Path
from store import project_dir, load_artifact, list_brands

# Which artifact each stage produces (used for stage-gating + status)
STAGE_ARTIFACTS = {
    "Discover":  ["claims_register", "positioning_layers", "query_master",
                  "competitive_set", "kpi_targets"],
    "Audit":     ["__audit_raw__"],                    # special: raw jsonl, not artifact
    "Benchmark": ["delta_scorecard", "gap_register_actionable"],
    "Experiment":["validated_findings_only", "experiment_variants"],
    "Optimise":  ["geo_strategy", "roadmap", "content_briefs"],
}

# Which prior stage each stage REQUIRES (stage-gating)
STAGE_PREREQ = {
    "Discover":  None,
    "Audit":     "Discover",
    "Benchmark": "Audit",
    "Experiment":"Benchmark",
    "Optimise":  "Experiment",
}

@dataclass
class AppContext:
    parent_brand: str
    target_brand: str
    category: str
    gateway: object          # LLMGateway (shared)
    meter: object            # Meter (ONE across all modules)
    models: dict = field(default_factory=dict)   # per-task model choices

def audit_raw_exists(parent, target):
    return (project_dir(parent, target) / "audit_raw.jsonl").exists()

def stage_complete(stage, parent, target):
    """True if a stage's output artifacts all exist."""
    arts = STAGE_ARTIFACTS.get(stage, [])
    for a in arts:
        if a == "__audit_raw__":
            if not audit_raw_exists(parent, target): return False
        else:
            if load_artifact(parent, target, a) is None: return False
    return True

def prereq_met(stage, parent, target):
    """True if the stage BEFORE this one is complete (stage-gating)."""
    prev = STAGE_PREREQ.get(stage)
    if prev is None: return True
    return stage_complete(prev, parent, target)

def pipeline_status(parent, target):
    """Return {stage: 'done'|'ready'|'locked'} for the status dashboard."""
    status = {}
    for stage in ["Discover","Audit","Benchmark","Experiment","Optimise"]:
        if stage_complete(stage, parent, target):
            status[stage] = "done"
        elif prereq_met(stage, parent, target):
            status[stage] = "ready"
        else:
            status[stage] = "locked"
    return status