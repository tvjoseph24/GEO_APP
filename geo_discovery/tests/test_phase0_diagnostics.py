"""Phase-0 guards. Pure functions and filesystem structure — no LLM, no network."""
import json
from pathlib import Path

import pytest

from diagnose import classify_assertion_kind
from ingestion import has_spec_content, spec_coverage
from store import ARTIFACT_NAMES, STORE_ROOT
from url_resolver import attach_domain, domain_from_title, looks_like_domain

# ---------------------------------------------------------------- W-02
@pytest.mark.parametrize("value", [
    "healthline.com", "www.nhs.uk", "ncbi.nlm.nih.gov", "which.co.uk",
    "consumerreports.org", "chemistwarehouse.com.au", "sub.domain.example.io",
])
def test_looks_like_domain_accepts_real_domains(value):
    assert looks_like_domain(value.replace("www.", ""))

@pytest.mark.parametrize("value", [
    "Best Baby Formula 2024 — Top 10 Picks",
    "Aveeno Baby Eczema Therapy Moisturizing Cream Review",
    "", "   ", "reddit", "a.b",
    "How to choose a moisturiser for sensitive skin",
    "https://example.com/page",
])
def test_looks_like_domain_rejects_titles(value):
    assert not looks_like_domain(value), f"a page title passed as a domain: {value!r}"

@pytest.mark.parametrize("title,expected", [
    ("Healthline — healthline.com", "healthline.com"),
    ("nhs.uk", "nhs.uk"),
    ("Review on consumerreports.org today", "consumerreports.org"),
    ("Something at which.co.uk", "which.co.uk"),
])
def test_domain_from_title_recovers_embedded(title, expected):
    assert domain_from_title(title) == expected

@pytest.mark.parametrize("title", [
    "Best Baby Formula 2024", "Top 10 Picks for Eczema", "", "no domain here",
])
def test_domain_from_title_invents_nothing(title):
    assert domain_from_title(title) == ""

def test_attach_domain_url_path():
    s = {"resolved_url": "https://www.healthline.com/x", "resolved_ok": True,
         "title": "Some Article Title"}
    attach_domain(s)
    assert s["domain"] == "healthline.com"
    assert s["domain_source"] == "url"
    assert s["domain_is_title"] is False

def test_attach_domain_recovers_from_title():
    s = {"url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc",
         "resolved_url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc",
         "resolved_ok": False, "title": "Healthline — healthline.com"}
    attach_domain(s)
    assert s["domain"] == "healthline.com"
    assert s["domain_source"] == "title_domain"
    assert s["domain_is_title"] is False

def test_attach_domain_flags_a_title():
    s = {"url": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc",
         "resolved_ok": False, "title": "Best Baby Formula 2024 — Top 10 Picks"}
    attach_domain(s)
    assert s["domain_is_title"] is True, \
        "an unresolved page title must be flagged, or it reaches the client hit-list"
    assert s["domain_source"] == "title_text"

# ---------------------------------------------------------------- W-01
def test_spec_markers_detect_an_ingredient_list():
    ok, hits = has_spec_content(
        "Active ingredient: colloidal oatmeal 1%. Inactive ingredients: water, "
        "glycerin. Net wt 12 oz. Directions for use: apply as needed.")
    assert ok and len(hits) >= 2

def test_spec_markers_reject_positioning_copy():
    ok, _ = has_spec_content(
        "We are committed to gentle care for sensitive skin, trusted by families "
        "for over seventy years, because every child deserves comfort.")
    assert not ok, "brand narrative must not register as a product specification"

def test_spec_coverage_handles_both_shapes():
    dicts = [{"origin": "pdp.html", "text": "Ingredients: water. Net wt 100 ml."},
             {"origin": "brand.pdf", "text": "Our promise is gentle care."}]
    cov = spec_coverage(dicts)
    assert cov["n_sources"] == 2
    assert cov["has_specification"] is True
    assert cov["spec_origins"] == ["pdp.html"]
    assert spec_coverage([])["has_specification"] is False

# ---------------------------------------------------------------- W-33 precursor
@pytest.mark.parametrize("text,kind", [
    ("It costs about $10 for 11 oz.", "price"),
    ("Priced affordably compared with alternatives.", "price"),
    ("Widely available at drugstores nationwide.", "availability"),
    ("Free of fragrances, soap, and dyes.", "specification"),
    ("Contains 1% colloidal oatmeal.", "specification"),
    ("Recommended by dermatologists.", "clinical_or_credential"),
    ("Parents like the texture.", "other"),
])
def test_assertion_kind_classification(text, kind):
    assert classify_assertion_kind(text) == kind

def test_price_and_availability_are_not_brand_claims():
    """These must be excluded from hallucination_rate AND unsubstantiated_rate (W-33)."""
    commercial = {"price", "availability"}
    assert classify_assertion_kind("costs $10") in commercial
    assert classify_assertion_kind("sold at pharmacies") in commercial
    assert classify_assertion_kind("contains 1% oatmeal") not in commercial

# ---------------------------------------------------------------- round-trip
def test_artifact_names_has_no_duplicates():
    dupes = [n for n in set(ARTIFACT_NAMES) if ARTIFACT_NAMES.count(n) > 1]
    assert not dupes, f"duplicated artifact names: {dupes}"

def test_every_artifact_on_disk_is_loadable():
    """An artifact written but absent from ARTIFACT_NAMES is silently dropped on
    recovery — the W-32 failure mode."""
    if not STORE_ROOT.exists():
        pytest.skip("no local store")
    known, unknown = set(ARTIFACT_NAMES), {}
    for adir in STORE_ROOT.glob("*/*/artifacts"):
        for p in adir.glob("*_LATEST.json"):
            name = p.name[: -len("_LATEST.json")]
            if name not in known:
                unknown.setdefault(name, []).append(str(p.parent.parent))
    assert not unknown, (
        "artifacts written to disk but absent from store.ARTIFACT_NAMES — they will not "
        f"survive a session refresh:\n{json.dumps(unknown, indent=2)}")

def test_report_contract_artifacts_are_loadable():
    from report.contract import CONTRACT
    missing = sorted({r.artifact for r in CONTRACT} - set(ARTIFACT_NAMES))
    assert not missing, (
        f"report CONTRACT requires artifacts the loader does not know about: {missing}")

# ---------------------------------------------------------------- smoke
def test_diagnose_survives_an_empty_project(tmp_path, monkeypatch):
    import diagnose
    import store
    monkeypatch.setattr(store, "STORE_ROOT", tmp_path)
    monkeypatch.setattr(diagnose, "project_dir",
                        lambda p, t: (tmp_path / p / t))
    (tmp_path / "nobrand" / "nothing" / "artifacts").mkdir(parents=True)
    res = diagnose.run_all("nobrand", "nothing")
    assert len(res) == len(diagnose.CHECKS)
    assert all(r["verdict"] in ("ok", "warn", "blocker", "n/a", "info") for r in res)
