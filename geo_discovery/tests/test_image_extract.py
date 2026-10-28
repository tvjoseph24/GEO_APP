"""W-57: page-to-image extraction, enrichment, and visual analysis."""
import pandas as pd
import pytest

from benchmark.visual import collect, visual_report, visual_summary
from image_extract import (UNFETCHABLE, VIDEO_PLATFORM, _meta_images, fetch_policy,
                           host, is_direct_image, looks_like_junk)

# Markup is assembled from character codes rather than written literally — a code block
# containing raw tags gets rendered as markup somewhere between here and the file.
LT, GT = chr(60), chr(62)

def _meta(key, content, attr="property"):
    return f'{LT}meta {attr}="{key}" content="{content}"{GT}'

def _jsonld(payload):
    return (f'{LT}script type="application/ld+json"{GT}{payload}'
            f'{LT}/script{GT}')

def _img(src, w=None, h=None):
    dim = (f' width="{w}" height="{h}"' if w and h else "")
    return f'{LT}img src="{src}"{dim}{GT}'

def _page(*inner):
    return (f"{LT}html{GT}{LT}head{GT}" + "".join(inner)
            + f"{LT}/head{GT}{LT}/html{GT}")

def _body(*inner):
    return (f"{LT}html{GT}{LT}body{GT}" + "".join(inner)
            + f"{LT}/body{GT}{LT}/html{GT}")

# ---------------------------------------------------------------- direct detection
@pytest.mark.parametrize("url", [
    "https://cdn.example.com/pack.jpg",
    "https://cdn.example.com/pack.PNG",
    "https://e.com/i.webp?v=2",
    "https://e.com/i.jpeg#frag",
    "https://e.com/i.avif",
])
def test_direct_images_detected(url):
    assert is_direct_image(url)

@pytest.mark.parametrize("url", [
    "https://instagram.com/p/ABC",
    "https://retailer.com/product/123",
    "https://youtube.com/watch?v=x",
    "https://e.com/jpg-guide",
])
def test_pages_are_not_direct_images(url):
    assert not is_direct_image(url)

# ---------------------------------------------------------------- platform policy
@pytest.mark.parametrize("url", [
    "https://instagram.com/p/ABC",
    "https://www.facebook.com/brand/photos/1",
    "https://tiktok.com/@b/video/1",
    "https://x.com/b/status/1",
    "https://twitter.com/b/status/1",
])
def test_social_declared_unfetchable_not_attempted(url):
    """A login wall is a limit of the medium. Declaring it is a defensible coverage
    statement; retrying it is a scraping arms race with no analytical payoff."""
    ok, why = fetch_policy(url)
    assert not ok
    assert "authentication" in why

def test_video_previews_excluded_by_default_and_available_on_request():
    """An og:image is a single frame, which says nothing about pack legibility."""
    ok, why = fetch_policy("https://youtube.com/watch?v=x")
    assert not ok
    assert "thumbnail" in why or "frame" in why
    ok2, _ = fetch_policy("https://youtube.com/watch?v=x", allow_video_frames=True)
    assert ok2

@pytest.mark.parametrize("url", [
    "https://retailer.com/product/123",
    "https://editorial.com/best-lotions",
    "https://brand.com/our-range",
])
def test_retail_and_editorial_are_fetchable(url):
    """This is where a product photograph actually lives."""
    ok, why = fetch_policy(url)
    assert ok
    assert why == ""

def test_unfetchable_and_video_sets_are_disjoint():
    """A platform in both would make the policy evaluation order load-bearing."""
    assert not set(UNFETCHABLE) & set(VIDEO_PLATFORM)

def test_host_strips_www_and_lowercases():
    assert host("https://www.Example.COM/x") == "example.com"
    assert host("not a url") == ""
    assert host(None) == ""

# ---------------------------------------------------------------- junk filtering
@pytest.mark.parametrize("url", [
    "https://c.com/sprite.png",
    "https://c.com/1x1.gif",
    "https://c.com/icons/cart.svg",
    "https://c.com/logo-header.png",
    "https://c.com/tracking.gif",
    "https://c.com/favicon.ico",
    "https://c.com/placeholder.jpg",
])
def test_junk_images_rejected(url):
    assert looks_like_junk(url)

def test_a_cdn_hosted_product_shot_is_not_junk():
    assert not looks_like_junk("https://cdn.shopify.com/files/pack-front-800.jpg")

# ---------------------------------------------------------------- meta extraction
def test_og_image_is_extracted_and_absolutised():
    pytest.importorskip("bs4")
    got = _meta_images(_page(_meta("og:image", "/img/pack.jpg")),
                       "https://retailer.com/product/1")
    assert ("https://retailer.com/img/pack.jpg", "meta:og:image") in got

def test_twitter_image_is_extracted_from_a_name_attribute():
    pytest.importorskip("bs4")
    got = _meta_images(_page(_meta("twitter:image", "https://c.com/t.jpg",
                                   attr="name")),
                       "https://r.com/p/1")
    assert ("https://c.com/t.jpg", "meta:twitter:image") in got

def test_secure_url_is_preferred_order():
    """og:image:secure_url is listed first in META_KEYS, so it leads the candidates."""
    pytest.importorskip("bs4")
    got = _meta_images(
        _page(_meta("og:image", "https://c.com/plain.jpg"),
              _meta("og:image:secure_url", "https://c.com/secure.jpg")),
        "https://r.com/p/1")
    assert got[0][0] == "https://c.com/secure.jpg"

def test_jsonld_product_image_is_extracted():
    """Frequently the cleanest pack shot on a retail page."""
    pytest.importorskip("bs4")
    payload = '{"@type":"Product","image":["https://c.com/front.jpg"]}'
    got = _meta_images(_page(_jsonld(payload)), "https://r.com/p/1")
    assert ("https://c.com/front.jpg", "jsonld:image") in got

def test_jsonld_accepts_a_bare_string_image():
    pytest.importorskip("bs4")
    payload = '{"@type":"Product","image":"https://c.com/one.jpg"}'
    got = _meta_images(_page(_jsonld(payload)), "https://r.com/p/1")
    assert ("https://c.com/one.jpg", "jsonld:image") in got

def test_jsonld_accepts_a_list_of_nodes():
    pytest.importorskip("bs4")
    payload = ('[{"@type":"BreadcrumbList"},'
               '{"@type":"Product","image":"https://c.com/two.jpg"}]')
    got = _meta_images(_page(_jsonld(payload)), "https://r.com/p/1")
    assert ("https://c.com/two.jpg", "jsonld:image") in got

def test_malformed_jsonld_does_not_abort_extraction():
    pytest.importorskip("bs4")
    got = _meta_images(_page(_jsonld('{"@type":"Product","image":'),
                             _meta("og:image", "https://c.com/ok.jpg")),
                       "https://r.com/p/1")
    assert ("https://c.com/ok.jpg", "meta:og:image") in got

def test_meta_extraction_survives_malformed_html():
    assert _meta_images(f"{LT}html{GT}{LT}meta property=", "https://e.com") == []
    assert _meta_images("", "https://e.com") == []
    assert _meta_images(None, "https://e.com") == []

# ---------------------------------------------------------------- page patterns
def test_retailer_and_editorial_pages_are_now_captured():
    """The old pattern set covered only social and video, so a retailer product page —
    the one kind that reliably yields an image — was not even recorded."""
    import re
    from audit_runner import VISUAL_PAGE_PATTERNS
    for u in ("https://retailer.com/product/123",
              "https://shop.com/dp/B01ABC",
              "https://amazon.com/gp/product/B01",
              "https://mag.com/reviews/best-lotions",
              "https://store.com/item/55",
              "https://s.com/catalog/x?sku=991",
              "https://s.com/shop/lotions"):
        assert any(re.search(p, u, re.I) for p in VISUAL_PAGE_PATTERNS), u

def test_social_pages_are_still_captured_so_coverage_can_account_for_them():
    """'331 pages returned, 90% structurally unfetchable' is a defensible statement.
    Silently dropping them is not."""
    import re
    from audit_runner import VISUAL_PAGE_PATTERNS
    for u in ("https://instagram.com/p/ABC",
              "https://youtube.com/watch?v=x",
              "https://facebook.com/brand/photos/1"):
        assert any(re.search(p, u, re.I) for p in VISUAL_PAGE_PATTERNS), u

def test_extract_image_urls_separates_direct_from_pages():
    from audit_runner import _extract_image_urls
    srcs = [{"resolved_url": "https://cdn.com/pack.jpg"},
            {"resolved_url": "https://instagram.com/p/ABC"},
            {"resolved_url": "https://retailer.com/product/1"},
            {"resolved_url": "https://news.com/politics"}]
    imgs, pages = _extract_image_urls("see https://e.com/a.png", srcs)
    assert "https://cdn.com/pack.jpg" in imgs
    assert "https://e.com/a.png" in imgs
    assert "https://instagram.com/p/ABC" in pages
    assert "https://retailer.com/product/1" in pages
    assert "https://news.com/politics" not in pages

def test_extract_image_urls_tolerates_empty_input():
    from audit_runner import _extract_image_urls
    assert _extract_image_urls(None, None) == ([], [])
    assert _extract_image_urls("", []) == ([], [])

# ---------------------------------------------------------------- enrichment
def _raw(tmp_path, monkeypatch, rows):
    import json
    import store as S
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    d = S.project_dir("p", "t")
    with open(d / "audit_raw.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return S, str(d / "audit_raw.jsonl")

def test_collect_pages_counts_by_frequency(tmp_path, monkeypatch):
    """An image surfaced ninety times matters more than one surfaced twice, and the
    budget may not cover both."""
    from visual_enrich import collect_pages
    rows = ([{"success": True, "query": "q1",
              "visual_pages": ["https://a.com/p/1"]}] * 5
            + [{"success": True, "query": "q2",
                "visual_pages": ["https://b.com/p/2"]}] * 2)
    _, path = _raw(tmp_path, monkeypatch, rows)
    pages, meta = collect_pages(path)
    assert pages[0]["url"] == "https://a.com/p/1"
    assert pages[0]["times_returned"] == 5
    assert meta["distinct_pages"] == 2
    assert meta["total_returns"] == 7

def test_collect_pages_ignores_failed_runs(tmp_path, monkeypatch):
    from visual_enrich import collect_pages
    rows = [{"success": False, "visual_pages": ["https://a.com/p/1"]},
            {"success": True, "visual_pages": ["https://b.com/p/2"]}]
    _, path = _raw(tmp_path, monkeypatch, rows)
    pages, _ = collect_pages(path)
    assert [p["url"] for p in pages] == ["https://b.com/p/2"]

def test_collect_pages_on_a_missing_log_returns_empty(tmp_path, monkeypatch):
    import store as S
    from visual_enrich import collect_pages
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    pages, meta = collect_pages(str(tmp_path / "nothing.jsonl"))
    assert pages == [] and meta == {}

def test_plan_separates_fetchable_from_structurally_unfetchable(tmp_path, monkeypatch):
    from visual_enrich import plan
    rows = [{"success": True, "query": "q", "visual_pages": [
        "https://instagram.com/p/A",
        "https://retailer.com/product/1",
        "https://youtube.com/watch?v=x"]}]
    _, path = _raw(tmp_path, monkeypatch, rows)
    pl = plan("p", "t", path)
    assert pl["fetchable"] == 1
    assert pl["unfetchable"] == 2
    assert "cost nothing" in pl["note"]

def test_plan_reports_an_empty_corpus_without_raising(tmp_path, monkeypatch):
    from visual_enrich import plan
    _, path = _raw(tmp_path, monkeypatch, [{"success": True, "visual_pages": []}])
    pl = plan("p", "t", path)
    assert pl["distinct_pages"] == 0
    assert pl["fetchable"] == 0

def test_write_back_attaches_insights_and_backs_up(tmp_path, monkeypatch):
    """The raw log is the authority for everything downstream, so it is never rewritten
    without a timestamped copy."""
    import json
    from visual_enrich import _write_back
    rows = [{"success": True, "query": "q", "visual_pages": ["https://r.com/p/1"]},
            {"success": True, "query": "q2", "visual_pages": ["https://other.com/p/9"]}]
    S, path = _raw(tmp_path, monkeypatch, rows)
    n = _write_back(path,
                    [{"url": "https://c.com/i.jpg", "fetched": True,
                      "brand_visible": "Brand X",
                      "page_url": "https://r.com/p/1"}],
                    {"https://c.com/i.jpg": "https://r.com/p/1"})
    assert n == 1
    out = [json.loads(l) for l in open(path)]
    assert out[0]["image_insights"][0]["brand_visible"] == "Brand X"
    assert out[0]["num_images"] == 1
    assert "image_insights" not in out[1], "only the row that produced the page"
    assert list(S.project_dir("p", "t").glob("audit_raw.jsonl.bak.*")), \
        "the authority must not be rewritten without a backup"

def test_write_back_is_idempotent(tmp_path, monkeypatch):
    import json
    from visual_enrich import _write_back
    rows = [{"success": True, "visual_pages": ["https://r.com/p/1"]}]
    _, path = _raw(tmp_path, monkeypatch, rows)
    ins = [{"url": "https://c.com/i.jpg", "fetched": True,
            "page_url": "https://r.com/p/1"}]
    _write_back(path, ins, {})
    _write_back(path, ins, {})
    out = json.loads(open(path).readline())
    assert len(out["image_insights"]) == 1, "a second pass must not duplicate"

def test_write_back_preserves_an_unparseable_line(tmp_path, monkeypatch):
    """A torn line must not lose the rows around it."""
    import store as S
    from visual_enrich import _write_back
    monkeypatch.setattr(S, "STORE_ROOT", tmp_path)
    d = S.project_dir("p", "t")
    path = str(d / "audit_raw.jsonl")
    with open(path, "w") as f:
        f.write('{"success": true, "visual_pages": ["https://r.com/p/1"]}\n')
        f.write('{"success": true, "visual_pa\n')
        f.write('{"success": true, "visual_pages": []}\n')
    _write_back(path, [{"url": "https://c.com/i.jpg", "fetched": True,
                        "page_url": "https://r.com/p/1"}], {})
    assert len(open(path).readlines()) == 3

def test_write_back_with_no_matching_page_changes_nothing(tmp_path, monkeypatch):
    from visual_enrich import _write_back
    rows = [{"success": True, "visual_pages": ["https://r.com/p/1"]}]
    _, path = _raw(tmp_path, monkeypatch, rows)
    assert _write_back(path, [{"url": "https://c.com/i.jpg", "fetched": True,
                               "page_url": "https://unrelated.com/x"}], {}) == 0

# ---------------------------------------------------------------- analysis
def _read(brand="Brand X", claims=None, legibility="clear", role="product_shot",
          others=None, unreadable="", url="https://c.com/1.jpg"):
    return {"url": url, "fetched": True, "brand_visible": brand,
            "product_type": "moisturiser",
            "visible_claims": claims if claims is not None else ["24-hour hydration"],
            "legible_text": [brand] + (claims or []),
            "numbers_read": [], "certifications_read": [],
            "other_brands_visible": others or [],
            "text_legibility": legibility, "image_role": role,
            "what_is_unreadable": unreadable, "image_tokens": 258}

def _df(reads, pages=10):
    return pd.DataFrame([{"query": "q", "intent": "need", "engine": "e",
                          "num_visual_pages": pages,
                          "image_insights": reads}])

def test_no_images_reports_unmeasured_not_healthy():
    """An absence of visual findings is not the same as confirming the packaging is
    machine-readable."""
    rep = visual_report(_df([], pages=331), "Brand X")
    assert rep["verdict"] == "unmeasured"
    assert "UNMEASURED" in rep["why"]
    assert "331" in rep["why"]

def test_no_pages_at_all_says_so_differently():
    rep = visual_report(_df([], pages=0), "Brand X")
    assert rep["verdict"] == "unmeasured"
    assert "No engine returned a visual page" in rep["why"]

def test_claim_legibility_against_the_register():
    claims = [{"claim": "Contains 1% colloidal oatmeal"},
              {"claim": "Clinically proven to soothe dry skin"}]
    rep = visual_report(_df([_read(claims=["Contains 1% colloidal oatmeal"])]),
                        "Brand X", claims=claims)
    assert rep["verdict"] == "claims_legible"
    assert rep["n_claims_legible"] == 1
    assert "Clinically proven to soothe dry skin" in rep["claims_not_legible"]

def test_brand_absent_from_imagery_is_a_finding():
    reads = [_read(brand="Rival Y", claims=[], url=f"https://c.com/{i}.jpg")
             for i in range(4)]
    rep = visual_report(_df(reads), "Brand X", competitors=["Rival Y"])
    assert rep["verdict"] == "brand_absent_from_imagery"
    assert rep["brand_images"] == 0
    assert rep["competitor_images"] == 4
    assert rep["competitor_appearances"]["Rival Y"] == 4

def test_illegible_imagery_is_a_finding():
    reads = [_read(legibility="illegible", unreadable="text too small",
                   url=f"https://c.com/{i}.jpg") for i in range(4)]
    rep = visual_report(_df(reads), "Brand X")
    assert rep["verdict"] == "illegible"
    assert rep["unreadable_share"] == 1.0
    assert rep["what_is_unreadable"][0]["what"] == "text too small"

def test_recognition_without_substance_is_a_finding():
    """Imagery that establishes the brand and carries no claim is a narrow effect."""
    rep = visual_report(_df([_read(claims=[])]), "Brand X",
                        claims=[{"claim": "Contains 1% colloidal oatmeal"}])
    assert rep["verdict"] == "no_claim_legible"
    assert rep["brand_images"] == 1
    assert rep["n_claims_legible"] == 0

def test_recognition_only_when_no_register_was_supplied():
    rep = visual_report(_df([_read(claims=[])]), "Brand X")
    assert rep["verdict"] == "recognition_only"

def test_visual_share_of_voice_is_measured():
    """No text metric captures whose product the engines actually show."""
    reads = ([_read(url=f"https://c.com/b{i}.jpg") for i in range(2)]
             + [_read(brand="Rival Y", url=f"https://c.com/r{i}.jpg")
                for i in range(6)])
    rep = visual_report(_df(reads), "Brand X", competitors=["Rival Y"])
    assert rep["brand_image_share"] == 0.25
    assert rep["competitor_image_share"] == 0.75

def test_a_competitor_in_other_brands_visible_counts():
    rep = visual_report(_df([_read(brand="", claims=[], others=["Rival Y"])]),
                        "Brand X", competitors=["Rival Y"])
    assert rep["competitor_appearances"].get("Rival Y") == 1

def test_a_bare_string_claim_does_not_corrupt_the_count():
    """A model returning a string instead of a list would report eighteen claims."""
    r = _read()
    r["visible_claims"] = "24-hour hydration"
    rep = visual_report(_df([r]), "Brand X")
    assert len(rep["claims_read_from_imagery"]) == 1

def test_errored_and_unfetched_reads_are_excluded():
    reads = [_read(),
             {"url": "https://c.com/2.jpg", "fetched": False},
             {"url": "https://c.com/3.jpg", "fetched": True, "error": "boom"}]
    assert len(collect(_df(reads))) == 1

def test_duplicate_images_counted_once():
    """Engines return the same product shot repeatedly across hundreds of runs."""
    reads = [_read(url="https://c.com/same.jpg") for _ in range(5)]
    assert len(collect(_df(reads))) == 1

def test_image_roles_are_tallied():
    reads = [_read(role="product_shot", url="https://c.com/1.jpg"),
             _read(role="lifestyle", url="https://c.com/2.jpg"),
             _read(role="lifestyle", url="https://c.com/3.jpg")]
    rep = visual_report(_df(reads), "Brand X")
    assert rep["image_roles"]["lifestyle"] == 2

def test_summary_is_flat_and_json_safe():
    import json
    json.dumps(visual_summary(visual_report(_df([_read()]), "Brand X")))
    assert visual_summary(None) == {}

# ---------------------------------------------------------------- gaps
def test_unmeasured_imagery_raises_a_gap_with_the_remedy():
    from benchmark.gap_detect import detect_gaps
    gaps = detect_gaps({}, metrics={"visual": {
        "verdict": "unmeasured", "visual_pages_returned": 331, "images_read": 0,
        "why": "331 visual page(s) were returned and no readable image was recovered."}})
    g = next(x for x in gaps if x["type"] == "VISUAL_UNASSESSED")
    assert g["severity"] == "medium"
    assert "enrichment pass" in g["evidence"]["remedy"]

def test_no_pages_at_all_is_a_low_severity_gap():
    from benchmark.gap_detect import detect_gaps
    gaps = detect_gaps({}, metrics={"visual": {
        "verdict": "unmeasured", "visual_pages_returned": 0, "images_read": 0,
        "why": "No engine returned a visual page."}})
    g = next(x for x in gaps if x["type"] == "VISUAL_UNASSESSED")
    assert g["severity"] == "low"

def test_brand_absent_from_imagery_raises_a_high_gap():
    from benchmark.gap_detect import detect_gaps
    gaps = detect_gaps({}, metrics={"visual": {
        "verdict": "brand_absent_from_imagery", "images_read": 12,
        "brand_images": 0, "competitor_images": 9,
        "competitor_appearances": {"Rival Y": 9}, "image_roles": {},
        "brand_image_share": 0.0, "competitor_image_share": 0.75,
        "why": "identifiable in none"}})
    g = next(x for x in gaps if x["type"] == "VISUAL_UNREADABLE")
    assert g["severity"] == "high"
    assert g["pillar"] == "Visual/Multimodal"

def test_illegible_imagery_raises_a_high_gap():
    from benchmark.gap_detect import detect_gaps
    gaps = detect_gaps({}, metrics={"visual": {
        "verdict": "illegible", "images_read": 8, "brand_images": 6,
        "brand_image_share": 0.75, "competitor_image_share": 0.1,
        "unreadable_share": 0.75, "legibility": {"illegible": 6, "clear": 2},
        "what_is_unreadable": [{"url": "u", "what": "text too small"}],
        "why": "not reaching a machine reader"}})
    g = next(x for x in gaps if x["type"] == "VISUAL_UNREADABLE")
    assert g["severity"] == "high"

def test_visual_share_of_voice_deficit_raises_a_gap():
    from benchmark.gap_detect import detect_gaps
    gaps = detect_gaps({}, metrics={"visual": {
        "verdict": "claims_legible", "images_read": 12,
        "brand_image_share": 0.17, "competitor_image_share": 0.75,
        "competitor_appearances": {"Rival Y": 9}, "n_claims_legible": 2,
        "claims_in_register": 4, "unreadable_share": 0.1}})
    assert any("share-of-voice" in g["gap"] for g in gaps
               if g["type"] == "VISUAL_UNREADABLE")

def test_an_individually_illegible_claim_raises_a_medium_gap():
    from benchmark.gap_detect import detect_gaps
    gaps = detect_gaps({}, metrics={"visual": {
        "verdict": "claims_legible", "images_read": 8,
        "brand_image_share": 0.8, "competitor_image_share": 0.1,
        "unreadable_share": 0.1, "claims_in_register": 3,
        "n_claims_legible": 2, "n_claims_not_legible": 1,
        "claims_not_legible": ["Clinically proven to soothe dry skin"],
        "claims_legible_on_pack": ["Contains 1% colloidal oatmeal"],
        "claims_read_from_imagery": ["Contains 1% colloidal oatmeal"],
        "brand_images": 6}})
    g = next(x for x in gaps if x["type"] == "VISUAL_UNREADABLE" and x.get("claim"))
    assert g["severity"] == "medium"
    assert "Clinically proven" in g["claim"]

def test_a_failed_vision_selftest_outranks_everything():
    """No visual conclusion can be drawn from a model that cannot see."""
    from benchmark.gap_detect import detect_gaps
    gaps = detect_gaps({}, metrics={
        "_manifest": {"vision_selftest": {"vision_live": False}},
        "visual": {"verdict": "claims_legible", "images_read": 10,
                   "brand_image_share": 0.9, "competitor_image_share": 0.0,
                   "unreadable_share": 0.0}})
    g = next(x for x in gaps if x["type"] == "VISUAL_UNASSESSED")
    assert g["severity"] == "high"
    assert "self-test failed" in g["gap"]

def test_a_clean_visual_result_raises_no_visual_gap():
    from benchmark.gap_detect import detect_gaps
    gaps = detect_gaps({}, metrics={"visual": {
        "verdict": "claims_legible", "images_read": 10,
        "brand_image_share": 0.9, "competitor_image_share": 0.1,
        "unreadable_share": 0.05, "claims_in_register": 3,
        "n_claims_legible": 3, "n_claims_not_legible": 0,
        "claims_not_legible": []}})
    assert not [g for g in gaps if g["type"].startswith("VISUAL_")]