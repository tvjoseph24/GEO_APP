"""Human-gate controller: generate → review → (approve | edit | reject+feedback→regenerate).
Simple automation loop — NOT an agent. Reused across all gates."""
from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class GateState:
    name: str
    status: str = "generated"          # generated|awaiting|approved|edited|rejected
    artifact: object = None            # current draft (dict/list)
    feedback_history: list = field(default_factory=list)
    approver: str = ""
    decided_at: str = ""

class HumanGate:
    """Wraps a regenerator function so a human can loop until satisfied."""
    def __init__(self, name, regenerate_fn):
        self.name = name
        self.regenerate_fn = regenerate_fn  # (feedback:str|None) -> artifact
        self.state = GateState(name=name)

    def generate(self, feedback=None):
        if feedback:
            self.state.feedback_history.append(feedback)
        self.state.artifact = self.regenerate_fn(feedback)
        self.state.status = "awaiting"
        return self.state.artifact

    def approve(self, approver="analyst"):
        self.state.status = "approved"
        self.state.approver = approver
        self.state.decided_at = datetime.now(timezone.utc).isoformat()

    def edit(self, edited_artifact, approver="analyst"):
        self.state.artifact = edited_artifact
        self.state.status = "edited"
        self.state.approver = approver
        self.state.decided_at = datetime.now(timezone.utc).isoformat()

    def reject_and_regenerate(self, feedback):
        """Loop: feedback → regenerate → back to awaiting."""
        self.state.status = "rejected"
        return self.generate(feedback=feedback)

    def is_final(self):
        return self.state.status in ("approved", "edited")

    def audit_record(self):
        return {"gate": self.name, "status": self.state.status,
                "approver": self.state.approver, "decided_at": self.state.decided_at,
                "feedback_rounds": len(self.state.feedback_history),
                "feedback": self.state.feedback_history}