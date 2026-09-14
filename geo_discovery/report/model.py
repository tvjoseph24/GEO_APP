"""report/model.py — intermediate representation for the GEO report.

Design contract
---------------
* Every rendered number is a Fact carrying (source_artifact, source_path). The appendix
  traceability table is generated from the ledger, so §12 is automatic.
* Blocks carry `audience` and `redaction`, so the Word report, the practitioner appendix
  and the executive deck are FILTERS over one IR — not three documents that drift.
* The report performs NO analysis. Builders read artifacts and arrange them. Any LLM prose
  is labelled `interpretation` and must cite fact_ids.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

EVIDENCE_CLASSES = ("PROVEN", "PROVEN_ENGINE_SPECIFIC", "PROVEN_WITH_CAVEAT",
                    "DIAGNOSED", "HYPOTHESIS", "DISPROVEN", "INCONCLUSIVE",
                    "NOT_TESTABLE", "INVALID", "BLOCKED")

EVIDENCE_LABEL = {
    "PROVEN": "Proven",
    "PROVEN_ENGINE_SPECIFIC": "Proven (engine-specific)",
    "PROVEN_WITH_CAVEAT": "Proven (caveated)",
    "DIAGNOSED": "Diagnosed",
    "HYPOTHESIS": "Hypothesis",
    "DISPROVEN": "Disproven",
    "INCONCLUSIVE": "Inconclusive",
    "NOT_TESTABLE": "Not yet testable",
    "INVALID": "Invalid",
    "BLOCKED": "Blocked",
}
EVIDENCE_GLOSS = {
    "PROVEN": "Statistically significant and replicated across engines.",
    "PROVEN_ENGINE_SPECIFIC": "Statistically significant on at least one engine; did not "
                              "transfer to all engines tested.",
    "PROVEN_WITH_CAVEAT": "Statistically significant but confounded or underpowered — "
                          "direction is credible, magnitude is not.",
    "DIAGNOSED": "Strong observational evidence from the audit or benchmark; not yet "
                 "tested experimentally.",
    "HYPOTHESIS": "Reasoned but unevidenced.",
    "DISPROVEN": "Tested at adequate power and showed no effect. An actionable exclusion.",
    "INCONCLUSIVE": "Tested but underpowered — proves nothing either way.",
    "NOT_TESTABLE": "Cannot be tested until the underlying claim is substantiated.",
}

AUDIENCES = ("exec", "practitioner", "both")
REDACTIONS = ("internal", "client")

# ---------------------------------------------------------------- facts
@dataclass
class Fact:
    """An atomic traceable value. Nothing renders without one behind it."""
    value: Any
    label: str
    source_artifact: str
    source_path: str
    unit: str = ""                       # pp | pct | share | rank_positions | count | x
    ci: tuple | None = None
    n: int | None = None
    evidence_class: str = "DIAGNOSED"
    limits: list = field(default_factory=list)
    fact_id: str = ""

    def rendered(self) -> str:
        v = self.value
        try:
            if self.unit in ("share", "pct") and isinstance(v, (int, float)):
                s = f"{v:.1%}"
            elif self.unit == "pp" and isinstance(v, (int, float)):
                s = f"{v:+.1f}pp" if v < 0 or v > 0 else "0.0pp"
            elif self.unit == "x" and isinstance(v, (int, float)):
                s = f"{v:.2f}×"
            elif self.unit == "rank_positions" and isinstance(v, (int, float)):
                s = f"{v:+.2f} rank positions"
            elif isinstance(v, float):
                s = f"{v:.2f}"
            else:
                s = str(v)
        except Exception:
            s = str(v)
        if self.ci and all(isinstance(x, (int, float)) for x in self.ci):
            fmt = "{:.1%}" if self.unit in ("share", "pct") else "{:.2f}"
            s += f" (95% CI {fmt.format(self.ci[0])}–{fmt.format(self.ci[1])})"
        if self.n:
            s += f", n={self.n:,}"
        return s

class FactLedger:
    """Assigns stable IDs and backs the appendix traceability table."""

    def __init__(self):
        self._facts: list[Fact] = []

    def add(self, **kw) -> Fact:
        f = Fact(**kw)
        if f.evidence_class not in EVIDENCE_CLASSES:
            raise ValueError(f"unknown evidence_class: {f.evidence_class}")
        f.fact_id = f"F-{len(self._facts) + 1:04d}"
        self._facts.append(f)
        return f

    def all(self) -> list:
        return list(self._facts)

    def table(self) -> list:
        return [{"ID": f.fact_id, "Metric": f.label, "Value": f.rendered(),
                 "Evidence": EVIDENCE_LABEL.get(f.evidence_class, f.evidence_class),
                 "Source": f"{f.source_artifact} → {f.source_path or '(root)'}",
                 "Limits": "; ".join(f.limits)} for f in self._facts]

    def to_json(self) -> str:
        return json.dumps([asdict(f) for f in self._facts], indent=2, default=str)

# ---------------------------------------------------------------- blocks
@dataclass
class Block:
    """kind ∈ heading | prose | bullets | table | chart | callout | kpi_row |
             quote | coverage | evidence_key | pagebreak"""
    kind: str
    payload: Any = None
    facts: list = field(default_factory=list)
    interpretation: str = ""            # LLM prose; must cite fact_ids
    audience: str = "both"              # exec | practitioner | both
    redaction: str = "internal"         # internal = all profiles; client = safe for client
    caption: str = ""
    level: int = 2                      # heading level
    tone: str = "info"                   # info | warn | danger | good  (callouts)
    edited: bool = False  # True when an analyst overlay replaced the payload

    def visible(self, audience: str, profile: str) -> bool:
        if audience != "both" and self.audience not in (audience, "both"):
            return False
        # a block marked internal is hidden from a client-profile render
        if profile == "client" and self.redaction == "internal_only":
            return False
        return True

@dataclass
class Section:
    id: str
    title: str
    blocks: list = field(default_factory=list)
    coverage: dict | None = None
    provisional: bool = False
    audience: str = "both"
    subtitle: str = ""

    def add(self, block: Block):
        self.blocks.append(block)
        return block

@dataclass
class ReportModel:
    meta: dict
    sections: list = field(default_factory=list)
    ledger: FactLedger = field(default_factory=FactLedger)
    coverage: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    provisional: bool = False
    provisional_reasons: list = field(default_factory=list)

    def section(self, sid, title, **kw) -> Section:
        s = Section(id=sid, title=title, **kw)
        self.sections.append(s)
        return s

    def filtered(self, audience="both", profile="internal") -> list:
        out = []
        for s in self.sections:
            if audience != "both" and s.audience not in (audience, "both"):
                continue
            blocks = [b for b in s.blocks if b.visible(audience, profile)]
            if blocks:
                out.append(Section(id=s.id, title=s.title, blocks=blocks,
                                   coverage=s.coverage, provisional=s.provisional,
                                   audience=s.audience, subtitle=s.subtitle))
        return out

    def stats(self) -> dict:
        return {"sections": len(self.sections),
                "blocks": sum(len(s.blocks) for s in self.sections),
                "facts": len(self.ledger.all()),
                "charts": sum(1 for s in self.sections for b in s.blocks
                              if b.kind == "chart"),
                "tables": sum(1 for s in self.sections for b in s.blocks
                              if b.kind == "table"),
                "provisional": self.provisional,
                "warnings": len(self.warnings)}

# ---------------------------------------------------------------- helpers
def hash_inputs(artifacts: dict) -> dict:
    out = {}
    for name, art in artifacts.items():
        if art is None:
            continue
        blob = json.dumps(art, sort_keys=True, default=str).encode()
        out[name] = hashlib.sha256(blob).hexdigest()[:16]
    return out

def redact_domain(domain: str, profile: str, kind: str = "") -> str:
    """Grey-market reseller domains are commercially sensitive in a client-facing document.
    The aggregate percentage is always retained; only the named domain is masked."""
    if profile != "client" or not domain:
        return domain
    if kind in ("grey_market_retailer", "content_affiliate"):
        h = hashlib.sha1(domain.encode()).hexdigest()[:4].upper()
        return f"[unauthorised reseller {h}]" if kind == "grey_market_retailer" \
            else f"[affiliate site {h}]"
    return domain