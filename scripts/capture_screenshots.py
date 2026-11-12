"""Capture the execution screenshots used in the final report.

For every stage the real command is run, its complete terminal output is saved
to ``reports/screenshots/<nn>_<stage>.txt`` and rendered as a terminal-styled
PNG (``.png`` beside it) with headless Chrome, so each picture in the report
is the actual output of that run.  UI pages (Prefect, MLflow) are captured
with ``capture_url``.

    python scripts/capture_screenshots.py stages          # the seven stage commands + tests
    python scripts/capture_screenshots.py url <name> <url> [width] [height] [wait_ms]
"""
from __future__ import annotations

import html
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "screenshots"
PY = sys.executable
CHROME = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")

STAGES = [
    ("01_ingestion", "python -m src.cli ingest", [PY, "-m", "src.cli", "ingest"]),
    ("02_validation", "python -m src.cli validate", [PY, "-m", "src.cli", "validate"]),
    ("03_preparation", "python -m src.cli prepare", [PY, "-m", "src.cli", "prepare"]),
    ("04_transformation", "python -m src.cli transform", [PY, "-m", "src.cli", "transform"]),
    ("05_features", "python -m src.cli features", [PY, "-m", "src.cli", "features"]),
    ("06_training", "python -m src.cli train", [PY, "-m", "src.cli", "train"]),
    ("07_evaluation", "python -m src.cli evaluate", [PY, "-m", "src.cli", "evaluate"]),
    ("08_tests", "python -m pytest -v tests", [PY, "-m", "pytest", "-v", "tests"]),
]

PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;background:#1e1e2e;padding:18px;font-family:'DejaVu Sans Mono','Menlo',monospace}
.win{background:#11111b;border-radius:10px;box-shadow:0 10px 40px rgba(0,0,0,.6);overflow:hidden;border:1px solid #313244}
.bar{background:#181825;padding:9px 14px;display:flex;align-items:center;gap:8px;border-bottom:1px solid #313244}
.dot{width:12px;height:12px;border-radius:50%%}.title{margin-left:12px;color:#a6adc8;font-size:12px}
pre{margin:0;padding:16px 18px;color:#cdd6f4;font-size:13.5px;line-height:1.5;white-space:pre-wrap;word-break:break-word}
.cmd{color:#a6e3a1}.p{color:#89b4fa}.w{color:#f9e2af}.e{color:#f38ba8}.d{color:#6c7086}
</style></head><body><div class="win"><div class="bar"><span class="dot" style="background:#f38ba8"></span><span class="dot" style="background:#f9e2af"></span><span class="dot" style="background:#a6e3a1"></span><span class="title">%(title)s</span></div><pre>%(body)s</pre></div></body></html>"""


def colorize(text: str) -> str:
    out = []
    for line in text.splitlines():
        e = html.escape(line)
        if "| WARNING |" in line or " FAIL " in line or "WARN " in line:
            e = f'<span class="w">{e}</span>'
        elif "Traceback" in line or "Error" in line or "FAILED" in line:
            e = f'<span class="e">{e}</span>'
        elif line.startswith("{"):
            e = f'<span class="d">{e}</span>'
        elif "| INFO " in line:
            e = e.replace("| INFO    |", '| <span class="p">INFO</span>    |', 1)
        out.append(e)
    return "\n".join(out)


def render(name: str, title: str, command: str, text: str, width: int = 1280) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    body = f'<span class="cmd">$ {html.escape(command)}</span>\n' + colorize(text)
    page = OUT / f"{name}.html"
    page.write_text(PAGE % {"title": html.escape(title), "body": body}, encoding="utf-8")
    png = OUT / f"{name}.png"
    # Size the window to hug the content: one row per LINE, plus the extra rows a
    # long line wraps into at this width (~150 monospace chars), plus the command
    # line and the chrome (title bar + padding).  This keeps short outputs from
    # leaving a large empty dark area below the text.
    chars_per_row = max(40, (width - 90) // 8)
    rows = 1 + sum(1 + len(l) // chars_per_row for l in text.splitlines())
    height = min(2400, 112 + round(20.5 * (rows + 1)))   # +1 row of slack so the trailing [exit code] line is never clipped
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                    f"--window-size={width},{height}", f"--screenshot={png}", page.as_uri()],
                   check=True, capture_output=True, timeout=120)
    page.unlink()
    return png


def run_stage(name: str, shown: str, argv: list[str]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "PYTHONUNBUFFERED": "1", "COLUMNS": "140"}
    t0 = time.time()
    proc = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
    text = (proc.stdout + proc.stderr).strip()
    text = "\n".join(l for l in text.splitlines() if "mlflow.store.db.utils" not in l)
    # Keep the capture free of machine-specific absolute paths: the repo root
    # becomes "." and the home directory "~", so a screenshot shows portable paths.
    text = text.replace(str(ROOT), ".").replace(str(Path.home()), "~")
    text += f"\n\n[exit code {proc.returncode}, {time.time() - t0:.1f}s]"
    (OUT / f"{name}.txt").write_text(text, encoding="utf-8")
    png = render(name, f"{shown} - recommendation-pipeline", shown, text)
    print(f"{name}: exit {proc.returncode} -> {png.name}")
    if proc.returncode != 0:
        raise SystemExit(f"{name} failed")
    return png


def capture_url(name: str, url: str, width: int = 1440, height: int = 900, wait_ms: int = 8000) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}.png"
    extra = os.environ.get("CHROME_FLAGS", "").split()   # e.g. WebGL for canvas graphs: --use-angle=swiftshader --enable-unsafe-swiftshader
    subprocess.run([CHROME, "--headless=new", "--no-sandbox", "--hide-scrollbars", *extra,
                    f"--window-size={width},{height}", f"--virtual-time-budget={wait_ms}", f"--screenshot={png}", url],
                   check=True, capture_output=True, timeout=180)
    print(f"{name}: {url} -> {png.name}")
    return png


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "stages"
    if mode == "stages":
        for name, shown, argv in STAGES:
            run_stage(name, shown, argv)
    elif mode == "url":
        _, _, name, url, *rest = sys.argv
        capture_url(name, url, *[int(x) for x in rest])
    elif mode == "cmd":   # python scripts/capture_screenshots.py cmd <name> "<shown command>" -- argv...
        _, _, name, shown, dash, *argv = sys.argv
        run_stage(name, shown, argv)
