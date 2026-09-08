"""Verify web scraping: requests vs headless render, on real target URLs.
Run:  python tests/test_scraping.py
      python tests/test_scraping.py --url https://www.nestle.in/
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion import ingest_url, ingest_url_rendered, ingest_url_smart

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"

DEFAULT_URLS = ["https://example.com",          # simple (should pass requests)
                "https://www.nestle.in/"]        # JS-heavy (tests fallback)

def test_playwright_installed():
    try:
        import playwright  # noqa
        print(f"{GREEN}✅{RESET} playwright importable")
        return True
    except ImportError:
        print(f"{YELLOW}⚠️{RESET} playwright NOT installed → "
              f"pip install playwright && playwright install chromium")
        return False

def compare(url, have_pw):
    print(f"\n── {url} ──")
    fast = ingest_url(url)
    fchars = fast.meta.get("chars", 0)
    warn = fast.meta.get("warning", "")
    print(f"  requests : {fchars:>6} chars  {warn}")
    if fast.meta.get("error"):
        print(f"  {RED}requests error: {fast.meta['error'][:80]}{RESET}"); return

    if have_pw and (fast.meta.get("thin_content") or fast.meta.get("likely_js_wall")):
        rend = ingest_url_rendered(url)
        rchars = rend.meta.get("chars", 0)
        if rend.meta.get("error"):
            print(f"  render   : {RED}FAILED{RESET} {rend.meta['error'][:80]}")
        else:
            gain = rchars - fchars
            colour = GREEN if gain > 0 else YELLOW
            print(f"  render   : {colour}{rchars:>6} chars  (+{gain} vs requests){RESET}")
            if rchars < 500:
                print(f"           {YELLOW}⚠️ still thin — likely bot-blocked{RESET}")
    else:
        print("  render   : (not needed — requests content sufficient)")

def main():
    urls = DEFAULT_URLS
    if "--url" in sys.argv:
        urls = [sys.argv[sys.argv.index("--url") + 1]]
    print("="*60); print(" WEB SCRAPING DIAGNOSTIC"); print("="*60)
    have_pw = test_playwright_installed()
    for u in urls:
        compare(u, have_pw)
    print("\nInterpretation:")
    print("  • requests thin + render gains chars  → fallback WORKS ✅")
    print("  • both thin                           → site is bot-blocked; use docs/newsroom")
    print("  • example.com passes on requests      → basic scraping OK")

if __name__ == "__main__":
    main()