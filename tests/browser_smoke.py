"""Check the dashboard from a checkout with the bundled aggregate files only."""
from pathlib import Path
import subprocess
import sys

from playwright.sync_api import expect, sync_playwright


root = Path(__file__).resolve().parents[1]
with subprocess.Popen(
    [sys.executable, str(root / "portfolio_demo.py"), "--port", "0"],
    stdout=subprocess.PIPE, text=True,
) as server:
    try:
        address = server.stdout.readline().strip().removeprefix("Open ")
        assert address.startswith("http://127.0.0.1:"), address
        with sync_playwright() as browser_tools:
            browser = browser_tools.chromium.launch(channel="chrome")
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(address)
            expect(page.get_by_role("heading", name="Retail Discount Response")).to_be_visible()
            expect(page.get_by_text("405,992", exact=True)).to_be_visible()
            expect(page.get_by_text("81.29%", exact=True)).to_be_visible()
            expect(page.get_by_role("slider")).to_have_count(0)
            assert not errors, errors
    finally:
        server.terminate()
