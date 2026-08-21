"""Render the running app in a real browser and verify the CSS actually applied.

AppTest cannot catch styling problems because it returns element source strings
and never renders. This drives headless Chromium against the live app instead.

    python scripts/screenshot_app.py [url]
"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path("evaluation/ui")
OUT.mkdir(parents=True, exist_ok=True)

CSS_LEAK_MARKERS = ["border-radius:", "font-weight:700;", ".block-container",
                    "linear-gradient(", "@import url"]

PROBE = r"""() => {
  const bar = document.querySelector('.appbar');
  const bc  = document.querySelector('.block-container');
  const out = {
    appbar: !!bar,
    style_tags: document.querySelectorAll('style').length,
    font: getComputedStyle(document.body).fontFamily.slice(0, 30),
  };
  if (bar) {
    const s = getComputedStyle(bar);
    out.gradient = s.backgroundImage.includes('gradient');
    out.radius = s.borderRadius;
    out.clipped_at_top = bar.getBoundingClientRect().top < 0;
  }
  if (bc) out.max_width = getComputedStyle(bc).maxWidth;
  return out;
}"""

failures = []

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1500, "height": 1000})
    page.goto(URL, wait_until="networkidle", timeout=90_000)
    # Wait for the styled header to exist rather than sleeping a fixed interval.
    # Streamlit binds its HTTP port well before the websocket delivers the first
    # render, so a timeout-based wait makes this check flaky.
    try:
        page.wait_for_selector(".appbar", timeout=60_000)
    except Exception:
        failures.append("timed out waiting for the styled header to render")
    page.wait_for_timeout(2500)

    leaked = [m for m in CSS_LEAK_MARKERS if m in page.inner_text("body")]
    print("1. CSS leaked as text:", leaked or "no")
    if leaked:
        failures.append(f"CSS visible as page text: {leaked}")

    d = page.evaluate(PROBE)
    for k, v in d.items():
        print(f"   {k}: {v}")
    if not d.get("appbar"):
        failures.append("appbar not in DOM")
    if not d.get("gradient"):
        failures.append("appbar gradient not applied")
    if d.get("clipped_at_top"):
        failures.append("appbar is clipped above the viewport")
    if "Inter" not in d.get("font", ""):
        failures.append(f"webfont not applied: {d.get('font')}")

    page.screenshot(path=str(OUT / "01_landing.png"), full_page=True)

    # Drive a real assessment and check the result view renders.
    print("\n2. Running the 'Struggling student' assessment")
    # Streamlit selectboxes are custom baseweb comboboxes, not native <select>.
    page.locator('[data-testid="stSelectbox"]').first.click()
    page.wait_for_timeout(900)
    page.get_by_role("option", name="Struggling student").click()
    page.wait_for_timeout(4000)
    page.get_by_role("button", name="Assess dropout risk").click()
    page.wait_for_timeout(7000)

    res = page.evaluate(r"""() => ({
        gauge: document.querySelectorAll('svg circle').length,
        factor_rows: document.querySelectorAll('.frow').length,
        bars_with_width: [...document.querySelectorAll('.ffill')]
            .filter(e => parseFloat(getComputedStyle(e).width) > 0).length,
        steps: !!document.querySelector('.steps'),
        note: !!document.querySelector('.note'),
        band_text: (document.body.innerText.match(/(High|Medium|Low) risk/) || [])[0] || null,
    })""")
    for k, v in res.items():
        print(f"   {k}: {v}")
    if res["gauge"] < 2:
        failures.append("risk gauge SVG missing")
    if res["factor_rows"] < 5:
        failures.append(f"only {res['factor_rows']} factor rows rendered")
    if res["bars_with_width"] < 5:
        failures.append("factor bars have no width, CSS not applied to them")
    if not res["steps"]:
        failures.append("suggested next steps card missing")
    if res["band_text"] != "High risk":
        failures.append(f"expected 'High risk', got {res['band_text']}")

    page.screenshot(path=str(OUT / "02_result.png"), full_page=True)

    # Batch tab
    print("\n3. Batch tab")
    page.get_by_role("tab", name="Score a whole class").click()
    page.wait_for_timeout(2500)
    page.locator('[data-testid="stCheckbox"]').first.click()
    page.wait_for_timeout(9000)
    rows = page.evaluate("() => document.querySelectorAll('[data-testid=\"stDataFrame\"]').length")
    print(f"   dataframe rendered: {rows}")
    if rows < 1:
        failures.append("batch results table did not render")
    page.screenshot(path=str(OUT / "03_batch.png"), full_page=True)

    browser.close()

print(f"\nScreenshots in {OUT}/")
if failures:
    print("\nUI CHECK FAILED:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("\nUI CHECK PASSED")
