"""Check the dashboard from a checkout with the bundled aggregate files only."""
import os
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
            executable = os.environ.get("RETAIL_CHROMIUM")
            browser = browser_tools.chromium.launch(**(
                {"executable_path": executable} if executable else {"channel": "chrome"}
            ))
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(address)
            expect(page.get_by_role("heading", name="Retail Discount Response")).to_be_visible()
            expect(page.get_by_text("405,992", exact=True)).to_be_visible()
            expect(page.get_by_text("81.29%", exact=True)).to_be_visible()
            expect(page.get_by_role("slider")).to_have_count(0)
            expect(page.get_by_text("Local evidence archive required")).to_be_visible()
            expect(page.locator(".lookbook-photo img")).to_have_js_property("complete", True)
            assert page.locator(".lookbook-photo img").evaluate("(img) => img.naturalWidth > 0")
            picker = page.get_by_label("Tested markdown")
            picker.select_option("60")
            expect(page.locator("#support-detail")).to_contain_text("81.29%")
            page.set_viewport_size({"width": 390, "height": 844})
            picker.focus()
            expect(picker).to_be_focused()
            picker.select_option("0")
            expect(page.locator("#support-detail")).to_contain_text("11.95%")
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            assert not errors, errors
    finally:
        server.terminate()
