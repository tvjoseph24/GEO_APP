"""W-57: page-to-image extraction, enrichment, and visual analysis."""
import pandas as pd
import pytest

from benchmark.visual import collect, visual_report, visual_summary
from image_extract import (UNFETCHABLE, VIDEO_PLATFORM, _meta_images, fetch_policy,
                       host, is_direct_image, looks_like_junk)

# ---------------------------------------------------------------- direct detection
@pytest.mark.parametrize("url", [
"https://cdn.example.com/pack.jpg", "https://x.com/a.PNG",
"https://e.com/i.webp?v=2", "https://e.com/i.jpeg#frag",
])
def test_direct_images_detected(url):
    assert is_direct_image(url)

@pytest.mark.parametrize("url", [
"https://instagram.com/p/ABC", "https://retailer.com/product/123",
"https://youtube.com/watch?v=x", "https://e.com/jpg-guide",
])
def test_pages_are_not_direct_images(url):
    assert not is_direct_image(url)

# ---------------------------------------------------------------- policy
@pytest.mark.parametrize("url,frag", [
("https://instagram.com/p/ABC", "authentication"),
("https://www.facebook.com/brand/photos/1", "authentication"),
("https://tiktok.com/@b/video/1", "authentication"),
("https://x.com/b/status/1", "authentication"),
])
def test_social_declared_unfetchable_not_attempted(url, frag):
    """A login wall is a limit of the medium. Declaring it is a defensible coverage
    statement; retrying it is a scraping arms race with no analytical payoff."""
    ok, why = fetch_policy(url)
    assert not ok and frag in why

def test_video_previews_excluded_by_default_and_available_on_request():
    """An og:image is a single frame, which says nothing about pack legibility."""
    ok, why = fetch_policy("https://youtube.com/watch?v=x")
    assert not ok and "frame" in why or "thumbnail" in why
    ok2, _ = fetch_policy("https://youtube.com/watch?v=x", allow_video_frames=True)
    assert ok2

@pytest.mark.parametrize("url", [
"https://retailer.com/product/123", "https://editorial.com/best-lotions",
"https://brand.com/our-range",
])
def test_retail_and_editorial_are_fetchable(url):
    """This is where a product photograph actually lives."""
    ok, why = fetch_policy(url)
    assert ok and why == ""

def test_host_strips_www():
    assert host("https://www.Example.COM/x") == "example.com"
    assert host("not a url") == ""

@pytest.mark.parametrize("url", [
"https://c.com/sprite.png", "https://c.com/1x1.gif",
"https://c.com/icons/cart.svg", "https://c.com/logo-header.png",
"https://c.com/tracking.gif", "https://c.com/favicon.ico",
])
def test_junk_images_rejected(url):
    assert looks_like_junk(url)

def test_a_cdn_hosted_product_shot_is_not_junk():
    assert not looks_like_junk("https://cdn.shopify.com/files/pack-front-800.jpg")

# ---------------------------------------------------------------- meta extraction
def test_og_image_is_extracted_and_absolutised():
    pytest.importorskip("bs4")
    html = ('<html><head>''<meta property="og:image" content="/img/pack.jpg">''</head></html>')
    got = _meta_images(html, "https://retailer.com/product/1")
    assert ("https://retailer.com/img/pack.jpg", "meta:og:image") in got

def test_jsonld_product_image_is_extracted():
    """Frequently the cleanest pack shot on a retail page."""
    pytest.importorskip("bs4")
    html = ('<html><script type="application/ld+json">' \
            '{"@type":"Product","image":["https://c.com/front.jpg"]}' \
            '</script></html>')
    got = _meta_images(html, "https://r.com/p/1")
    assert ("https://c.com/front.jpg", "jsonld:image") in got

def test_jsonld_accepts_a_bare_string_image():
    pytest.importorskip("bs4")
    html = ('<html><script type="application/ld+json">' \
           '{"@type":"Product","image":"https://c.com/one.jpg"}' \
           '</script></html>')
    got = _meta_images(html, "https://r.com/p/1")
    assert ("https://c.com/one.jpg", "jsonld:image") in got