"""Capture a single-page UI (Prefect, MLflow) with headless Chrome driven over the DevTools protocol.

The page is loaded for real, optional JavaScript is run (to click a toggle or
close a side panel), the page is given time to settle, then a screenshot is
saved to ``reports/screenshots/<name>.png``.

    python scripts/capture_ui.py <name> <url> [--wait 8] [--js "<snippet>"]... [--width 1440] [--height 900]
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "screenshots"
CHROME = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")


class Page:
    def __init__(self, ws_url: str):
        self.ws = connect(ws_url, max_size=64 * 1024 * 1024)
        self.n = 0

    def call(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    def eval(self, js: str):
        return self.call("Runtime.evaluate", expression=js, awaitPromise=True, returnByValue=True).get("result", {}).get("value")


def capture(name: str, url: str, wait: float, js: list[str], width: int, height: int) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="chrome-cdp-")
    # Let Chrome pick a FREE debug port (--remote-debugging-port=0) and read the
    # chosen port back from the DevToolsActivePort file in the profile.  A fixed
    # port would silently attach to any other Chrome already debugging there.
    proc = subprocess.Popen([CHROME, "--headless=new", "--no-sandbox", "--hide-scrollbars", "--use-angle=swiftshader",
                             "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--remote-debugging-port=0",
                             f"--user-data-dir={profile}", f"--window-size={width},{height}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        port_file = Path(profile) / "DevToolsActivePort"
        port = None
        for _ in range(50):
            if port_file.exists():
                try:
                    port = int(port_file.read_text().splitlines()[0])
                    break
                except (ValueError, IndexError):
                    pass
            time.sleep(0.2)
        if port is None:
            raise RuntimeError("chrome did not report its DevTools port")
        for _ in range(50):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                page = next(t for t in targets if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        else:
            raise RuntimeError("chrome did not expose a page target")
        p = Page(page["webSocketDebuggerUrl"])
        p.call("Page.enable")
        p.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)
        p.call("Page.navigate", url=url)
        time.sleep(wait)
        for snippet in js:
            p.eval(snippet)
            time.sleep(wait / 2)
        data = p.call("Page.captureScreenshot", format="png")["data"]
        png = OUT / f"{name}.png"
        png.write_bytes(base64.b64decode(data))
        print(f"{name}: {url} -> {png.name}")
        return png
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("url")
    ap.add_argument("--wait", type=float, default=8.0)
    ap.add_argument("--js", action="append", default=[])
    ap.add_argument("--width", type=int, default=1440); ap.add_argument("--height", type=int, default=900)
    a = ap.parse_args()
    capture(a.name, a.url, a.wait, a.js, a.width, a.height)
