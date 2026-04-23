"""Capture the 6 README-linked screenshots of the Experiment Designer.

Run with:

    .venv/bin/python scripts/capture_screenshots.py

It expects ``streamlit`` to already be serving ``app/experiment_designer.py``
on the port in ``APP_URL`` (default 8513). The companion script
``scripts/capture_screenshots.sh`` starts the server, runs this, and stops it.

Output: six PNG files in ``docs/screenshots/`` at the exact filenames the
repo README hard-codes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

APP_URL = "http://localhost:8513"
VIEWPORT = {"width": 1440, "height": 900}
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"


def _wait_for_streamlit(page: Page) -> None:
    """Wait for Streamlit's main content and Plotly to settle."""
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=30_000)
    page.wait_for_load_state("networkidle", timeout=30_000)
    # Plotly redraws asynchronously after hydration; 1.5s is enough on M-series.
    page.wait_for_timeout(1500)


def _click_tab(page: Page, label: str) -> None:
    """Click a top-level Streamlit tab by its visible label."""
    page.get_by_role("tab", name=label, exact=True).click()
    page.wait_for_timeout(800)


def _shoot(page: Page, filename: str) -> None:
    out = OUT_DIR / filename
    page.screenshot(path=str(out), full_page=False)
    print(f"  ✓ {out.relative_to(OUT_DIR.parent.parent)}")


def capture(page: Page) -> None:
    page.set_viewport_size(VIEWPORT)
    page.goto(APP_URL)
    _wait_for_streamlit(page)

    # --- 1. Planner hero ------------------------------------------------------
    _click_tab(page, "Planner")
    _wait_for_streamlit(page)
    _shoot(page, "01_planner_hero.png")

    # --- 2. Planner with a smaller MDE ---------------------------------------
    # Second number_input is the MDE. We nudge it down to surface the power curve.
    mde_input = page.locator('input[aria-label="Minimum detectable effect (absolute pp)"]')
    mde_input.fill("0.005")
    mde_input.press("Enter")
    page.wait_for_timeout(1500)
    _shoot(page, "02_planner_mde_sweep.png")

    # --- 3. Readout — SHIP banner + frequentist CI ---------------------------
    _click_tab(page, "Readout")
    _wait_for_streamlit(page)
    _shoot(page, "03_readout_ship.png")

    # --- 4. Readout — Bayesian sub-tab ---------------------------------------
    page.get_by_role("tab", name="Bayesian", exact=True).click()
    page.wait_for_timeout(1200)
    _shoot(page, "04_readout_bayesian.png")

    # --- 5. Peeking lab before running ---------------------------------------
    _click_tab(page, "Peeking lab")
    _wait_for_streamlit(page)
    _shoot(page, "05_peeking_before.png")

    # --- 6. Peeking lab after running ----------------------------------------
    page.get_by_role("button", name="Run peeking simulation").click()
    # Simulation takes 2-4s for 1k sims × 5 peek counts; wait for the bar chart.
    page.wait_for_selector(".js-plotly-plot", timeout=30_000)
    page.wait_for_timeout(2500)
    _shoot(page, "06_peeking_after.png")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"• capturing screenshots into {OUT_DIR}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport=VIEWPORT,
            device_scale_factor=2,  # Retina-quality
        )
        page = context.new_page()
        try:
            capture(page)
        finally:
            context.close()
            browser.close()
    print("✅ all 6 screenshots captured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
