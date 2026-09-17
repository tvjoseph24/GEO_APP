# GEO Framework

Generative Engine Optimisation: measure how AI assistants describe and recommend a brand,
diagnose the gap against ground truth, prove causation experimentally, and convert that
into an evidence-classed strategy and client report.

## Pipeline

Discover → Audit → Benchmark → Experiment → Optimise → Report

| Phase | Produces |
|---|---|
| Discover | Claims register, positioning layers, query master, competitive set, KPIs |
| Audit | `audit_raw.jsonl` — N runs per query per engine, resumable |
| Benchmark | Delta scorecard, typed gap register, hypotheses, source hit-list |
| Experiment | Validated findings with evidence classes (incl. well-powered nulls) |
| Optimise | Pillar strategies, engine overlays, capacity-packed roadmap, content briefs |
| Report | Word / PDF / PPT with fact ledger, editable with sign-off |

## Setup
bash pip install -r requirements.txt cp .env.example .env # fill in credentials streamlit run geo_app.py
pip install -r requirements.txt cp .env.example .env # fill in credentials streamlit run geo_app.py
GEO_STORE=geo_projects_test streamlit run geo_app.py
## Design rules

1. **No hardcoded brand, category or domain values** outside `config.py`. Everything is
   derived from artifacts or configured per engagement. Enforced by
   `tests/test_no_hardcoded_category.py`.
2. **`metrics_spec.py` is the single measurement contract.** Mention, first-mention, rank,
   refusal and polarity are defined once. No phase may reimplement them.
3. **Claims are hard-constrained to the substantiated register.** Variant text becomes
   content briefs; an unregistered claim is a compliance breach, not a style issue.
4. **Well-powered nulls are evidence.** They generate the "not doing" list, which is what
   makes a strategy defensible.
5. **Every action carries an evidence class.** `PROVEN` → `DISPROVEN`. Unevidenced work is
   capped at 15% of roadmap effort.
6. **The report performs no analysis.** It assembles artifacts. Every number traces to a
   `(artifact, json_path)` pair via the fact ledger.

## Utilities
python backfill.py --dry-run # recompute metrics on existing raw logs python backfill.py --domains # unclassified citation domains python reanalyse.py --save # rebuild findings from raw, no LLM cost python diag_optimise.py # strategy integrity checks python -m pytest tests/ -v
## Cost

Full cycle at ~600 audit runs, 4 experiments: **$25–35**. Audit dominates. Every phase has
a budget cap; Audit and Experiment are resumable.

## Validated on

`danone/aptamil` (infant formula) — 620 audit runs, 4 experiments, 1 proven
engine-specific finding, 2 well-powered nulls. See tag `v1.0-aptamil`.
